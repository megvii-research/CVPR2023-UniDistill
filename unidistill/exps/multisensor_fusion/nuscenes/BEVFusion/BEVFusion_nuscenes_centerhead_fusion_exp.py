from typing import Any

import mmcv
import torch
import torch.nn as nn

from unidistill.exps.base_cli import run_cli
from unidistill.exps.multisensor_fusion.nuscenes._base_.base_nuscenes_cfg import (
    CENTERPOINT_DET_HEAD_CFG,
)
from unidistill.exps.multisensor_fusion.nuscenes.BEVFusion.BEVFusion_nuscenes_base_exp import (
    BEVFusion,
)
from unidistill.exps.multisensor_fusion.nuscenes.BEVFusion.BEVFusion_nuscenes_base_exp import (
    Exp as BaseExp,
)
from unidistill.layers.head.det3d import (
    CenterHeadIouAware,
    FCOSAssigner,
    IouAwareGenProposals,
)
from unidistill.layers.losses.det3d import CenterNetRegLoss, FocalLoss

_IMG_BACKBONE_CONF = dict(
    type="ResNet",
    depth=50,
    frozen_stages=0,
    out_indices=[0, 1, 2, 3],
    norm_eval=False,
    init_cfg=dict(type="Pretrained", checkpoint="torchvision://resnet50"),
)


_IMG_NECK_CONF = dict(
    type="SECONDFPN",
    in_channels=[256, 512, 1024, 2048],
    upsample_strides=[0.25, 0.5, 1, 2],
    out_channels=[128, 128, 128, 128],
)

_DEPTH_NET_CONF = dict(in_channels=512, mid_channels=512)


class DetHead(nn.Module):
    def __init__(self, det_head_cfg: mmcv.Config, **kwargs):
        super().__init__()
        self.det_head_cfg = det_head_cfg
        self.dense_head = self.build_dense_head()

    def build_dense_head(self):
        target_cfg = self.det_head_cfg.target_assigner
        target_assigner = FCOSAssigner(
            out_size_factor=target_cfg.densehead_out_size_factor,
            tasks=target_cfg.densehead_tasks,
            dense_reg=target_cfg.target_assigner_dense_reg,
            gaussian_overlap=target_cfg.target_assigner_gaussian_overlap,
            max_objs=target_cfg.target_assigner_max_objs,
            min_radius=target_cfg.target_assigner_min_radius,
            mapping=target_cfg.target_assigner_mapping,
            grid_size=target_cfg.grid_size,
            pc_range=target_cfg.pc_range,
            voxel_size=target_cfg.voxel_size,
            assign_topk=target_cfg.target_assigner_topk,
            no_log=target_cfg.target_assigner_no_log,
            with_velocity=target_cfg.with_velocity,
        )
        proposal_cfg = self.det_head_cfg.proposal_layer
        proposal_layer = IouAwareGenProposals(
            dataset_name=proposal_cfg.densehead_dataset_name,
            class_names=[t["class_names"] for t in proposal_cfg.densehead_tasks],
            post_center_limit_range=proposal_cfg.proposal_post_center_limit_range,
            score_threshold=proposal_cfg.proposal_score_threshold,
            pc_range=proposal_cfg.proposal_pc_range,
            out_size_factor=proposal_cfg.densehead_out_size_factor,
            voxel_size=proposal_cfg.proposal_voxel_size,
            no_log=proposal_cfg.no_log,
            iou_aware_list=proposal_cfg.proposal_iou_aware_list,
            nms_iou_threshold_train=proposal_cfg.nms_iou_threshold_train,
            nms_pre_max_size_train=proposal_cfg.nms_pre_max_size_train,
            nms_post_max_size_train=proposal_cfg.nms_post_max_size_train,
            nms_iou_threshold_test=proposal_cfg.nms_iou_threshold_test,
            nms_pre_max_size_test=proposal_cfg.nms_pre_max_size_test,
            nms_post_max_size_test=proposal_cfg.nms_post_max_size_test,
        )
        head_cfg = self.det_head_cfg.dense_head
        dense_head_module = CenterHeadIouAware(
            dataset_name=head_cfg.densehead_dataset_name,
            tasks=head_cfg.densehead_tasks,
            target_assigner=target_assigner,
            proposal_layer=proposal_layer,
            out_size_factor=head_cfg.densehead_out_size_factor,
            input_channels=head_cfg.input_channels,
            grid_size=head_cfg.grid_size,
            point_cloud_range=head_cfg.point_cloud_range,
            code_weights=head_cfg.densehead_loss_code_weights,
            loc_weight=head_cfg.densehead_loss_loc_weight,
            iou_weight=head_cfg.densehead_loss_iou_weight,
            share_conv_channel=head_cfg.densehead_share_conv_channel,
            common_heads=head_cfg.densehead_common_heads,
            upsample_for_pedestrian=head_cfg.densehead_upsample_for_pedestrian,
            mode=head_cfg.densehead_mode,
            init_bias=head_cfg.densehead_init_bias,
            predict_boxes_when_training=False,
        )

        def _build_losses(m):
            m.add_module(
                "crit",
                FocalLoss(
                    self.det_head_cfg.target_assigner_alpha,
                    self.det_head_cfg.target_assigner_gamma,
                ),
            )
            m.add_module("crit_reg", CenterNetRegLoss())
            m.add_module("crit_iou_aware", CenterNetRegLoss())

        _build_losses(dense_head_module)

        return dense_head_module

    def forward(self, x: torch.tensor, gt_boxes: torch.tensor) -> Any:
        forward_ret_dict = self.dense_head(x, gt_boxes)

        if self.training:
            for _, encoding in forward_ret_dict["box_encoding"].items():
                encoding[torch.isinf(encoding)] = 0
        return forward_ret_dict


class BEVFusionCenterHead(BEVFusion):
    def __init__(self, model_cfg) -> Any:
        super().__init__(model_cfg)

    def forward(
        self,
        lidar_points,
        cameras_imgs,
        metas,
        gt_boxes,
        return_feature=False,
        **kwargs
    ):
        if self.with_lidar_encoder:
            lidar_output = self.lidar_encoder(lidar_points)
            model_output = lidar_output
        if self.with_camera_encoder:
            camera_output = self.camera_encoder(cameras_imgs, metas)
            model_output = camera_output
        if self.with_fusion_encoder:
            multimodal_output = self.fusion_encoder(lidar_output, camera_output)
            model_output = multimodal_output
        x = self.bev_encoder(model_output)
        forward_ret_dict = self.det_head(x[0], gt_boxes)
        if return_feature == True:
            return model_output, x[0], forward_ret_dict["multi_head_features"]
        if self.training:
            for task_id, encoding in forward_ret_dict["box_encoding"].items():
                encoding[torch.isinf(encoding)] = 0
            loss_rpn, tb_dict = self.det_head.dense_head.get_loss(forward_ret_dict)
            tb_dict.update({"loss_rpn": loss_rpn.item()})
            ret_dict = {"loss": loss_rpn}
            return (
                ret_dict,
                tb_dict,
                model_output,
                x[0],
                forward_ret_dict["multi_head_features"],
                {},
            )
        else:
            return forward_ret_dict

    def _configure_det_head(self):
        return DetHead(self.cfg.det_head)


class Exp(BaseExp):
    def __init__(
        self,
        batch_size_per_device=4,
        total_devices=1,
        max_epochs=20,
        ckpt_path=None,
        **kwargs
    ):
        super(Exp, self).__init__(
            batch_size_per_device, total_devices, max_epochs, ckpt_path
        )
        self.model_cfg["camera_encoder"]["img_backbone_conf"] = _IMG_BACKBONE_CONF
        self.model_cfg["camera_encoder"]["img_neck_conf"] = _IMG_NECK_CONF
        self.model_cfg["camera_encoder"]["depth_net_conf"] = _DEPTH_NET_CONF
        self.model_cfg["det_head"] = CENTERPOINT_DET_HEAD_CFG
        self._change_cfg_params()
        self.model = self._configure_model()
        self.train_dataloader = self.configure_train_dataloader()
        self.val_dataloader = self.configure_val_dataloader()
        self.test_dataloader = self.configure_test_dataloader()

    def _change_cfg_params(self):
        self.data_cfg["aug_cfg"]["gt_sampling_cfg"] = None

    def _configure_model(self):
        model = BEVFusionCenterHead(
            model_cfg=mmcv.Config(self.model_cfg),
        )
        return model


if __name__ == "__main__":
    import logging

    logging.getLogger("mmcv").disabled = True
    logging.getLogger("mmseg").disabled = True
    run_cli(Exp, "BEVFusion_nuscenes_centerhead_fusion_exp")
