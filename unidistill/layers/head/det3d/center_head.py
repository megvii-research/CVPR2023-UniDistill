import copy
import sys
from collections import OrderedDict

import torch
import torch.nn as nn

from unidistill.layers.losses import det3d as loss_utils
from unidistill.utils.det3d_utils.initialize_utils import kaiming_init

from .generate_proposals.base_gen_proposals import BaseGenProposals
from .target_assigner.base_assigner import BaseAssigner


class CenterHead(nn.Module):
    def __init__(
        self,
        dataset_name,
        tasks,
        target_assigner: BaseAssigner,
        proposal_layer: BaseGenProposals,
        input_channels,
        grid_size,
        point_cloud_range,
        code_weights,
        loc_weight,
        share_conv_channel,
        common_heads,
        upsample_for_pedestrian=False,
        predict_boxes_when_training=False,
        mode="3d",
        init_bias=-2.19,
        distill=False,
    ):
        super().__init__()
        self.tasks = tasks
        self.in_channels = input_channels
        self.grid_size = grid_size
        self.point_cloud_range = point_cloud_range
        self.predict_boxes_when_training = predict_boxes_when_training

        self.num_classes = [len(t["class_names"]) for t in self.tasks]
        self.class_names = [t["class_names"] for t in self.tasks]

        self.code_weights = code_weights
        self.weight = loc_weight  # weight between hm loss and loc loss

        self.dataset = dataset_name
        self.box_n_dim = 9 if self.dataset == "nuscenes" else 7

        self.encode_background_as_zeros = True
        self.use_sigmoid_score = True
        self.no_log = False
        self.use_direction_classifier = False
        self.bev_only = True if mode == "bev" else False
        self.distill = distill
        # a shared convolution
        self.shared_conv = nn.Sequential(
            nn.Conv2d(
                self.in_channels,
                share_conv_channel,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
            nn.BatchNorm2d(share_conv_channel),
            nn.ReLU(inplace=True),
        )

        self.upsample_for_pedestrian = (
            upsample_for_pedestrian  # 与 output_size_factor 需要有对应关系
        )
        # upsample for detect small objects
        if self.upsample_for_pedestrian:
            self.upsample_conv = nn.Sequential(
                nn.ConvTranspose2d(
                    share_conv_channel, share_conv_channel, 2, stride=2, bias=False
                ),
                nn.BatchNorm2d(share_conv_channel),
                nn.ReLU(),
            )

        self.common_heads = common_heads
        self.init_bias = init_bias
        self.tasks = nn.ModuleList()

        for num_cls in self.num_classes:
            heads = copy.deepcopy(self.common_heads)
            heads.update(dict(hm=(num_cls, 2)))
            self.tasks.append(
                SepHead(
                    share_conv_channel,
                    heads,
                    bn=True,
                    init_bias=self.init_bias,
                    final_kernel=3,
                    directional_classifier=False,
                )
            )

        self.target_assigner = target_assigner
        self.proposal_layer = proposal_layer

        self.build_losses()

    def train(self, mode: bool = True):
        super().train(mode=mode)
        self.proposal_layer.training = mode
        return self

    def eval(self):
        self.train(False)

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, 8)
        Returns:

        """
        targets_dict = self.target_assigner.assign_targets(gt_boxes)
        return targets_dict

    def forward(self, spatial_features_2d, gt_boxes=None):
        multi_head_features = []
        forward_ret_dict = {}
        spatial_features_2d = self.shared_conv(spatial_features_2d)

        if self.upsample_for_pedestrian:
            spatial_features_2d = self.upsample_conv(spatial_features_2d)

        for task_id, task in enumerate(self.tasks):
            multi_head_features.append(task(spatial_features_2d))

        forward_ret_dict["multi_head_features"] = multi_head_features

        if self.training or self.distill:
            targets_dict = self.assign_targets(gt_boxes)
            forward_ret_dict.update(targets_dict)
            return forward_ret_dict

        if not self.training or self.predict_boxes_when_training:
            data_dict = self.proposal_layer.generate_predicted_boxes(
                forward_ret_dict, {}
            )
            return data_dict

    def build_losses(self):
        self.add_module("crit", loss_utils.CenterNetFocalLoss())
        self.add_module("crit_reg", loss_utils.CenterNetRegLoss())
        return

    def _sigmoid(self, x):
        y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
        return y

    def _transpose_and_gather_feat(self, feat, ind):
        feat = feat.permute(0, 2, 3, 1).contiguous()
        feat = feat.view(feat.size(0), -1, feat.size(3))
        feat = self._gather_feat(feat, ind)
        return feat

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def get_loss(self, forward_ret_dict):
        tb_dict = {}
        pred_dicts = forward_ret_dict["multi_head_features"]
        center_loss = []
        forward_ret_dict["pred_box_encoding"] = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            pred_dict["hm"] = self._sigmoid(pred_dict["hm"])
            hm_loss = self.crit(pred_dict["hm"], forward_ret_dict["heatmap"][task_id])

            target_box_encoding = forward_ret_dict["box_encoding"][task_id]
            # nuscense encoding format [x, y, z, w, l, h, sinr, cosr, vx, vy]

            if self.dataset == "nuscenes":
                pred_box_encoding = torch.cat(
                    [
                        pred_dict["reg"],
                        pred_dict["height"],
                        pred_dict["dim"],
                        pred_dict["rot"],
                        pred_dict["vel"],
                    ],
                    dim=1,
                ).contiguous()  # (B, 10, H, W)
            else:
                pred_box_encoding = torch.cat(
                    [
                        pred_dict["reg"],
                        pred_dict["height"],
                        pred_dict["dim"],
                        pred_dict["rot"],
                    ],
                    dim=1,
                ).contiguous()  # (B, 8, H, W)

            forward_ret_dict["pred_box_encoding"][task_id] = pred_box_encoding

            box_loss = self.crit_reg(
                pred_box_encoding,
                forward_ret_dict["mask"][task_id],
                forward_ret_dict["ind"][task_id],
                target_box_encoding,
            )

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()
            loss = hm_loss + self.weight * loc_loss

            tb_key = "task_" + str(task_id) + "/"

            if self.dataset == "nuscenes":
                tb_dict.update(
                    {
                        tb_key + "loss": loss.item(),
                        tb_key + "hm_loss": hm_loss.item(),
                        tb_key + "loc_loss": loc_loss.item(),
                        tb_key + "x_loss": box_loss[0].item(),
                        tb_key + "y_loss": box_loss[1].item(),
                        tb_key + "z_loss": box_loss[2].item(),
                        tb_key + "w_loss": box_loss[3].item(),
                        tb_key + "l_loss": box_loss[4].item(),
                        tb_key + "h_loss": box_loss[5].item(),
                        tb_key + "sin_r_loss": box_loss[6].item(),
                        tb_key + "cos_r_loss": box_loss[7].item(),
                        tb_key + "vx_loss": box_loss[8].item(),
                        tb_key + "vy_loss": box_loss[9].item(),
                        tb_key
                        + "num_positive": forward_ret_dict["mask"][task_id]
                        .float()
                        .sum(),
                    }
                )
            else:
                tb_dict.update(
                    {
                        tb_key + "loss": loss.item(),
                        tb_key + "hm_loss": hm_loss.item(),
                        tb_key + "loc_loss": loc_loss.item(),
                        tb_key + "x_loss": box_loss[0].item(),
                        tb_key + "y_loss": box_loss[1].item(),
                        tb_key + "z_loss": box_loss[2].item(),
                        tb_key + "w_loss": box_loss[3].item(),
                        tb_key + "l_loss": box_loss[4].item(),
                        tb_key + "h_loss": box_loss[5].item(),
                        tb_key + "sin_r_loss": box_loss[6].item(),
                        tb_key + "cos_r_loss": box_loss[7].item(),
                        tb_key
                        + "num_positive": forward_ret_dict["mask"][task_id]
                        .float()
                        .sum(),
                    }
                )
            center_loss.append(loss)

        return sum(center_loss), tb_dict


# BASIC BUILDING BLOCKS
class Sequential(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Sequential, self).__init__()
        if len(args) == 1 and isinstance(args[0], OrderedDict):
            for key, module in args[0].items():
                self.add_module(key, module)
        else:
            for idx, module in enumerate(args):
                self.add_module(str(idx), module)
        for name, module in kwargs.items():
            if sys.version_info < (3, 6):
                raise ValueError("kwargs only supported in py36+")
            if name in self._modules:
                raise ValueError("name exists.")
            self.add_module(name, module)

    def __getitem__(self, idx):
        if not (-len(self) <= idx < len(self)):
            raise IndexError("index {} is out of range".format(idx))
        if idx < 0:
            idx += len(self)
        it = iter(self._modules.values())
        for i in range(idx):
            next(it)
        return next(it)

    def __len__(self):
        return len(self._modules)

    def add(self, module, name=None):
        if name is None:
            name = str(len(self._modules))
            if name in self._modules:
                raise KeyError("name exists")
        self.add_module(name, module)

    def forward(self, input):
        for module in self._modules.values():
            input = module(input)
        return input


class SepHead(nn.Module):
    def __init__(
        self,
        in_channels,
        heads,
        head_conv=64,
        name="",
        final_kernel=1,
        bn=False,
        init_bias=-2.19,
        directional_classifier=False,
        **kwargs,
    ):
        super(SepHead, self).__init__(**kwargs)

        self.heads = heads
        for head in self.heads:
            classes, num_conv = self.heads[head]

            fc = Sequential()
            for i in range(num_conv - 1):
                fc.add(
                    nn.Conv2d(
                        in_channels,
                        head_conv,
                        kernel_size=final_kernel,
                        stride=1,
                        padding=final_kernel // 2,
                        bias=True,
                    )
                )
                if bn:
                    fc.add(nn.BatchNorm2d(head_conv))
                fc.add(nn.ReLU())

            fc.add(
                nn.Conv2d(
                    head_conv,
                    classes,
                    kernel_size=final_kernel,
                    stride=1,
                    padding=final_kernel // 2,
                    bias=True,
                )
            )

            if "hm" in head:
                fc[-1].bias.data.fill_(init_bias)
            else:
                for m in fc.modules():
                    if isinstance(m, nn.Conv2d):
                        kaiming_init(m)

            self.__setattr__(head, fc)

        assert (
            directional_classifier is False
        ), "Doesn't work well with nuScenes in my experiments, please open a pull request if you are able to get it work. We really appreciate contribution for this."

    def forward(self, x):
        ret_dict = dict()
        for head in self.heads:
            ret_dict[head] = self.__getattr__(head)(x)

        return ret_dict
