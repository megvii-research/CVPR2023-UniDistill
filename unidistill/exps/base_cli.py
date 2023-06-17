import os
import pickle
from argparse import ArgumentParser

import pytorch_lightning as pl

from unidistill.utils import torch_dist

from .base_exp import BaseExp


def run_cli(
    model_class=BaseExp,
    exp_name="base_exp",
    use_ema=False,
    extra_trainer_config_args={},
):
    parent_parser = ArgumentParser(add_help=False)
    parent_parser = pl.Trainer.add_argparse_args(parent_parser)
    parent_parser.add_argument(
        "-e",
        "--evaluate",
        dest="evaluate",
        action="store_true",
        help="evaluate model on validation set",
    )
    parent_parser.add_argument(
        "-p",
        "--predict",
        dest="predict",
        action="store_true",
        help="predict model on testing set",
    )
    parent_parser.add_argument("-b", "--batch_size_per_device", type=int)
    parent_parser.add_argument(
        "--seed", type=int, default=0, help="seed for initializing training."
    )
    parent_parser.add_argument("--ckpt_path", type=str)
    parser = BaseExp.add_argparse_args(parent_parser)
    parser.set_defaults(
        check_val_every_n_epoch=20,
        num_sanity_val_steps=0,
        gradient_clip_val=0.1,
        accelerator="ddp",
    )
    args = parser.parse_args()
    if args.seed is not None:
        pl.seed_everything(args.seed)

    model = model_class(**vars(args))
    trainer = pl.Trainer.from_argparse_args(args)

    if args.evaluate:
        trainer.validate(model, model.val_dataloader, args.ckpt_path)
    elif args.predict:
        trainer.test(model, model.test_dataloader, args.ckpt_path)
    else:
        trainer.fit(model, model.train_dataloader, model.val_dataloader)
        trainer.save_checkpoint("./example.ckpt")
