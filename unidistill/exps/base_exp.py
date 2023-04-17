import datetime
import functools
import os
import sys
from abc import ABCMeta, abstractmethod

import numpy as np
import torch
from pytorch_lightning.core import LightningModule
from tabulate import tabulate
from torch.nn import Module

from unidistill.utils import DictAction, torch_dist
from unidistill.utils.misc import sanitize_filename

from . import global_cfg


class BaseExp(LightningModule):
    """Basic class for any experiment in unidistill.

    Args:
        batch_size_per_device (int):
            batch_size of each device

        total_devices (int):
            number of devices to use

        max_epoch (int):
            total training epochs, the reason why we need to give max_epoch
            is that lr_scheduler may need to be adapted according to max_epoch
    """

    def __init__(self, batch_size_per_device, total_devices, max_epochs, ckpt_path):
        super(BaseExp, self).__init__()
        self._batch_size_per_device = batch_size_per_device
        self._max_epochs = max_epochs
        self._total_devices = total_devices
        # ----------------------------------------------- extra configure ------------------------- #
        self.seed = None
        self.exp_name = os.path.splitext(os.path.basename(sys.argv.copy()[0]))[
            0
        ]  # entrypoint filename as exp_name
        self.print_interval = 100
        self.dump_interval = 10
        self.eval_interval = 10
        self.num_keep_latest_ckpt = 10
        self.enable_tensorboard = False
        self.eval_executor_class = None
        self.ckpt_path = ckpt_path

    @property
    def project_name(self):
        entrypoint = sys.argv[0]
        if "exps/" in entrypoint:
            proj_name = os.path.dirname(entrypoint.split("exps/")[-1])
        else:
            proj_name = "undefined"
        return f"unidistill/{proj_name}"

    @property
    def callbacks(self):
        if not hasattr(self, "_callbacks"):
            self._callbacks = self._configure_callbacks()
        return self._callbacks

    @property
    def optimizer(self):
        if "_optimizer" not in self.__dict__:
            self._optimizer = self._configure_optimizer()
        return self._optimizer

    @property
    def lr_scheduler(self):
        if "_lr_scheduler" not in self.__dict__:
            self._lr_scheduler = self._configure_lr_scheduler()
        return self._lr_scheduler

    @property
    def batch_size_per_device(self):
        return self._batch_size_per_device

    @property
    def max_epochs(self):
        return self._max_epochs

    @property
    def total_devices(self):
        return self._total_devices

    def training_step(self, *args, **kwargs):
        pass

    @abstractmethod
    def _configure_optimizer(self) -> torch.optim.Optimizer:
        pass

    @abstractmethod
    def _configure_lr_scheduler(self, **kwargs):
        pass

    def update_attr(self, options: dict) -> str:
        if options is None:
            return ""
        assert isinstance(options, dict)
        msg = ""
        for k, v in options.items():
            if k in self.__dict__:
                old_v = self.__getattribute__(k)
                if not v == old_v:
                    self.__setattr__(k, v)
                    msg = "{}\n'{}' is overriden from '{}' to '{}'".format(
                        msg, k, old_v, v
                    )
            else:
                self.__setattr__(k, v)
                msg = "{}\n'{}' is set to '{}'".format(msg, k, v)

        # update exp_name
        exp_name_suffix = "-".join(sorted([f"{k}-{v}" for k, v in options.items()]))
        self.exp_name = f"{self.exp_name}--{exp_name_suffix}"
        return msg

    def get_cfg_as_str(self) -> str:
        config_table = []
        for c, v in self.__dict__.items():
            if not isinstance(v, (int, float, str, list, tuple, dict, np.ndarray)):
                if hasattr(v, "__name__"):
                    v = v.__name__
                elif hasattr(v, "__class__"):
                    v = v.__class__
                elif type(v) == functools.partial:
                    v = v.func.__name__
            if c[0] == "_":
                c = c[1:]
            config_table.append((str(c), str(v)))

        headers = ["config key", "value"]
        config_table = tabulate(config_table, headers, tablefmt="plain")
        return config_table

    def _get_exp_output_dir(self):
        exp_dir = os.path.join(
            global_cfg.output_root_dir, sanitize_filename(self.exp_name)
        )
        os.makedirs(exp_dir, exist_ok=True)
        output_dir = None
        if self.ckpt_path:
            output_dir = os.path.dirname(
                os.path.dirname(os.path.abspath(self.ckpt_path))
            )
        elif torch_dist.get_rank() == 0:
            output_dir = os.path.join(
                exp_dir, datetime.datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
            )
            os.makedirs(output_dir, exist_ok=True)
            # make a symlink "latest"
            symlink, symlink_tmp = os.path.join(exp_dir, "latest"), os.path.join(
                exp_dir, "latest_tmp"
            )
            if os.path.exists(symlink_tmp):
                os.remove(symlink_tmp)
            os.symlink(os.path.relpath(output_dir, exp_dir), symlink_tmp)
            os.rename(symlink_tmp, symlink)
        output_dir = torch_dist.all_gather_object(output_dir)[0]
        self.output_dir = output_dir
        return output_dir

    def __str__(self):
        return self.get_cfg_as_str()

    def to_onnx(self):
        pass

    @classmethod
    def add_argparse_args(cls, parser):  # pragma: no-cover
        parser.add_argument(
            "--exp_options",
            nargs="+",
            action=DictAction,
            help="override some settings in the exp, the key-value pair in xxx=yyy format will be merged into exp. "
            'If the value to be overwritten is a list, it should be like key="[a,b]" or key=a,b '
            'It also allows nested list/tuple values, e.g. key="[(a,b),(c,d)]" '
            "Note that the quotation marks are necessary and that no white space is allowed.",
        )
        parser.add_argument("-epoch", "--max-epoch", type=int, default=None)
        return parser
