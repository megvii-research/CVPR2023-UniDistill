import os
import pickle
from functools import partial
from typing import Any, Dict, List, Tuple

import mmcv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.data import DistributedSampler

from unidistill.data.det3d.preprocess.voxelization import Voxelization
from unidistill.data.multisensorfusion.nuscenes_multimodal import (
    NuscenesMultiModalData,
    collate_fn,
)
from unidistill.exps.base_exp import BaseExp
from unidistill.exps.multisensor_fusion.nuscenes._base_.base_nuscenes_cfg import (
    DATA_CFG,
    MODEL_CFG,
)
from unidistill.layers.blocks_2d.det3d import BaseBEVBackbone, HeightCompression
from unidistill.layers.blocks_3d.det3d import MeanVFE, VoxelResBackBone8x
from unidistill.layers.blocks_3d.mmdet3d.lss_fpn import LSSFPN
from unidistill.models.multisensor_fusion import BaseEncoder, BaseMultiSensorFusion
from unidistill.utils import torch_dist
from unidistill.utils import torch_dist as dist

try:
    from unidistill.layers.head.det3d import HungarianAssigner3D, TransFusionHead
    from unidistill.layers.head.det3d.bbox.coder import TransFusionBBoxCoder
except ImportError:
    import warnings

    warnings.warn("TransFusionHead and related components are not included.")


class LidarEncoder(BaseEncoder):
    def __init__(self, lidar_encoder_cfg: mmcv.Config, **kwargs) -> Any:
        super().__init__()
        self.cfg = lidar_encoder_cfg
        self.voxelizer = self.build_voxelizer()
        self.vfe = self.build_vfe()
        self.backbone_3d = self.build_backbone_3d()
        self.map_to_bev = self.build_map_to_bev()

    def build_voxelizer(self):
        return Voxelization(
            voxel_size=self.cfg.voxel_size,
            point_cloud_range=self.cfg.point_cloud_range,
            max_num_points=self.cfg.max_num_points,
            max_voxels=self.cfg.max_voxels,
            num_point_features=self.cfg.src_num_point_features,
            device=torch.device("cuda"),
        )

    def build_vfe(self):
        vfe = MeanVFE(
            num_point_features=self.cfg.use_num_point_features,
        )
        return vfe

    def build_backbone_3d(self):
        return VoxelResBackBone8x(
            input_channels=self.vfe.get_output_feature_dim(),
            grid_size=np.array(self.cfg.grid_size),
            last_pad=0,
        )

    def build_map_to_bev(self):
        return HeightCompression(num_bev_features=self.cfg.map_to_bev_num_features)

    def forward(self, lidar_points: List[torch.tensor]) -> torch.tensor:
        voxels, voxel_coords, voxel_num_points = self.voxelizer(lidar_points)
        voxel_features = self.vfe(voxels, voxel_num_points)
        encoded_spconv_tensor, encoded_spconv_tensor_stride, _ = self.backbone_3d(
            voxel_features, voxel_coords, len(lidar_points)
        )

        spatial_features, encoded_spconv_tensor_stride = self.map_to_bev(
            encoded_spconv_tensor, encoded_spconv_tensor_stride
        )
        return spatial_features


class CameraEncoder(BaseEncoder):
    def __init__(self, camera_encoder_cfg: mmcv.Config, **kwargs) -> Any:
        super().__init__()
        self.backbone = LSSFPN(**camera_encoder_cfg)

    def forward(
        self,
        imgs: torch.tensor,
        mats_dict: Dict[str, torch.tensor],
        is_return_depth=False,
    ):
        feature_map = self.backbone(
            imgs,
            mats_dict,
            is_return_depth,
        )
        return feature_map


class FusionEncoder(nn.Module):
    def __init__(
        self,
        use_elementwise: bool = True,
        input_channel: int = 512,
        output_channel: int = 256,
        reduction: int = 2,
    ) -> Any:
        super().__init__()
        self.use_elementwise = use_elementwise
        if not self.use_elementwise:
            self.att = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(input_channel, input_channel, kernel_size=1, stride=1),
                nn.Sigmoid(),
            )
            self.reduce_conv = nn.Sequential(
                nn.Conv2d(input_channel, output_channel, 3, padding=1, bias=False),
                nn.BatchNorm2d(output_channel),
                nn.ReLU(True),
            )

    def forward(self, x1: torch.tensor, x2: torch.tensor) -> torch.tensor:
        assert x1.shape == x2.shape, f"shape: {x1.shape} != {x2.shape}"
        if self.use_elementwise:
            return x1 + x2
        else:
            x = torch.cat((x1, x2), dim=1)
            return self.reduce_conv(x * self.att(x))


class BevEncoder(nn.Module):
    def __init__(self, bev_encoder_cfg: mmcv.Config, **kwargs):
        super().__init__()
        self.bev_encoder_cfg = bev_encoder_cfg
        self.backbone_2d = self.build_bev_encoder()

    def build_bev_encoder(self):
        bev_encoder = BaseBEVBackbone(
            layer_nums=self.bev_encoder_cfg.backbone2d_layer_nums,
            layer_strides=self.bev_encoder_cfg.backbone2d_layer_strides,
            num_filters=self.bev_encoder_cfg.backbone2d_num_filters,
            upsample_strides=self.bev_encoder_cfg.backbone2d_upsample_strides,
            num_upsample_filters=self.bev_encoder_cfg.backbone2d_num_upsample_filters,
            input_channels=self.bev_encoder_cfg.num_bev_features,
            use_scconv=self.bev_encoder_cfg.get("backbone2d_use_scconv", False),
            upsample_output=self.bev_encoder_cfg.get(
                "backbone2d_upsample_output", False
            ),
        )
        return bev_encoder

    def forward(self, x: torch.tensor) -> Tuple[torch.tensor, List[torch.tensor]]:
        spatial_features_2d, pyramid = self.backbone_2d(x)
        return spatial_features_2d, pyramid


class DetHead(nn.Module):
    def __init__(self, det_head_cfg: mmcv.Config, **kwargs):
        super().__init__()
        self.det_head_cfg = det_head_cfg
        self.dense_head = self.build_dense_head(det_head_cfg)

    def build_dense_head(self, det_head_cfg):
        target_assigner_cfg = det_head_cfg.pop("target_assigner")
        target_assigner = HungarianAssigner3D(**target_assigner_cfg)
        coder_cfg = det_head_cfg.pop("bbox_coder")
        bbox_coder = TransFusionBBoxCoder(**coder_cfg)
        dense_head_module = TransFusionHead(
            **det_head_cfg,
            target_assigner=target_assigner,
            bbox_coder=bbox_coder,
            predict_boxes_when_training=False,
        )

        return dense_head_module

    def forward(self, x: torch.tensor, gt_boxes: torch.tensor) -> Any:
        forward_ret_dict = self.dense_head(x, gt_boxes)
        return forward_ret_dict


class BEVFusion(BaseMultiSensorFusion):
    r"""
    `BEVFusion`: Multi-Task Multi-Sensor Fusion with Unified Bird's-Eye View Representation.

    `Reference`: https://arxiv.org/abs/2205.13542
    """

    def __init__(self, model_cfg) -> Any:
        super(BEVFusion, self).__init__()
        self.num_class = len(model_cfg.class_names)
        self.class_names = model_cfg.class_names
        self.cfg = model_cfg

        if self.cfg.get("lidar_encoder", None):
            self.lidar_encoder = self._configure_lidar_encoder()
        else:
            self.lidar_encoder = None

        if self.cfg.get("camera_encoder", None):
            self.camera_encoder = self._configure_camera_encoder()
        else:
            self.camera_encoder = None

        if self.with_lidar_encoder and self.with_camera_encoder:
            self.fusion_encoder = self._configure_fusion_encoder()
        else:
            self.fusion_encoder = None

        self.bev_encoder = self._configure_bev_encoder()
        self.det_head = self._configure_det_head()
        # self.transform_layer_1 = nn.Sequential(nn.Conv2d(256,256,3,1,1),nn.BatchNorm2d(256),nn.ReLU())
        # self.transform_layer_2 = nn.Sequential(nn.Conv2d(512,512,3,1,1),nn.BatchNorm2d(512),nn.ReLU())

    def forward(
        self,
        lidar_points: List[torch.tensor] = None,
        cameras_imgs: torch.tensor = None,
        metas: Dict[str, torch.tensor] = None,
        gt_boxes: torch.tensor = None,
        **kwargs,
    ) -> Any:
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
        if self.training:
            tb_dict = forward_ret_dict.pop("tb_dict")
            return forward_ret_dict, tb_dict, {}
        else:
            return forward_ret_dict

    def _configure_lidar_encoder(self):
        return LidarEncoder(self.cfg.lidar_encoder)

    def _configure_camera_encoder(self):
        return CameraEncoder(self.cfg.camera_encoder)

    def _configure_fusion_encoder(self):
        return FusionEncoder(use_elementwise=False)

    def _configure_bev_encoder(self):
        return BevEncoder(self.cfg.bev_encoder)

    def _configure_det_head(self):
        return DetHead(self.cfg.det_head)


def _load_data_to_gpu(data_dict):
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.cuda()
        elif isinstance(v, dict):
            _load_data_to_gpu(data_dict[k])
        else:
            data_dict[k] = v


class Exp(BaseExp):
    def __init__(
        self,
        batch_size_per_device=4,
        total_devices=1,
        max_epochs=20,
        ckpt_path=None,
        **kwargs,
    ):
        super(Exp, self).__init__(
            batch_size_per_device, total_devices, max_epochs, ckpt_path
        )
        self.save_hyperparameters()
        self.lr = 1e-3
        self.print_interval = 50
        self.num_keep_latest_ckpt = 20
        self.dump_interval = 1
        self.lr_scale_factor = {"camera_encoder": 0.1}
        self.grad_clip_value = 0.1
        self.data_cfg = DATA_CFG
        self.model_cfg = MODEL_CFG
        self.data_split = {"train": "training", "val": "validation", "test": "testing"}
        self._get_exp_output_dir()

    def _change_cfg_params(self):
        r"""
        This func is designed to change cfg temporarily. For those should be inherited, please change them in __init__
        """

    def configure_train_dataloader(self):
        train_dataset = NuscenesMultiModalData(
            **self.data_cfg,
            data_split=self.data_split["train"],
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=self.batch_size_per_device,
            num_workers=10,
            drop_last=False,
            shuffle=False,
            collate_fn=partial(collate_fn, is_return_depth=False),
            sampler=None,
            pin_memory=False,
        )
        return train_loader

    def configure_val_dataloader(self):
        val_dataset = NuscenesMultiModalData(
            **self.data_cfg,
            data_split=self.data_split["val"],
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=10,
            sampler=None,
            pin_memory=False,
        )
        return val_loader

    def configure_test_dataloader(self):
        test_dataset = NuscenesMultiModalData(
            **self.data_cfg,
            data_split=self.data_split["test"],
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=self.batch_size_per_device,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=10,
            sampler=None,
            pin_memory=False,
        )
        return test_loader

    def _configure_model(self):
        model = BEVFusion(
            model_cfg=mmcv.Config(self.model_cfg),
        )
        return model

    def forward(self, points, imgs, metas, gt_boxes):
        return self.model(points, imgs, metas, gt_boxes)

    def training_step(self, batch):
        if torch.cuda.is_available():
            _load_data_to_gpu(batch)
        if "points" in batch:
            points = [frame_point for frame_point in batch["points"]]
        else:
            points = None
        imgs = batch.get("imgs", None)
        metas = batch.get("mats_dict", None)
        gt_boxes = batch["gt_boxes"]
        gt_labels = batch["gt_labels"]

        gt_labels += 1
        gt_boxes = torch.cat([gt_boxes, gt_labels.unsqueeze(dim=2)], dim=2)
        ret_dict, tf_dict, _, _, _, _ = self(points, imgs, metas, gt_boxes)
        loss = ret_dict["loss"].mean()
        return loss

    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            _load_data_to_gpu(batch)
        if "points" in batch:
            points = [frame_point for frame_point in batch["points"]]
        else:
            points = None
        imgs = batch.get("imgs", None)
        metas = batch.get("mats_dict", None)
        ret_dict = self(points, imgs, metas, None)
        for result in ret_dict["pred_dicts"]:
            result["pred_labels"] -= 1
        pred_dicts = ret_dict["pred_dicts"]
        return self.test_dataloader.dataset.generate_prediction_dicts(
            batch, pred_dicts, self.model.class_names
        )[0]

    def test_epoch_end(self, outputs) -> None:
        torch_dist.synchronize()
        preds = sum(map(list, zip(*torch_dist.all_gather_object(outputs))), [])[
            : len(self.test_dataloader.dataset)
        ]
        if torch_dist.get_rank() == 0:
            self.test_dataloader.dataset.dump_inference_results(
                preds, output_dir=self.output_dir
            )

    def validation_epoch_end(self, outputs) -> None:
        torch_dist.synchronize()
        preds = sum(map(list, zip(*torch_dist.all_gather_object(outputs))), [])[
            : len(self.val_dataloader.dataset)
        ]
        kwargs = {}
        if torch_dist.get_rank() == 0:
            result_str, result_dict = self.val_dataloader.dataset.evaluation(
                preds, self.model.class_names, output_dir=self.output_dir, **kwargs
            )
            out_file = os.path.join(self.output_dir, "result.pkl")
            pickle.dump(preds, open(out_file, "wb"))

    def validation_step(self, batch, batch_idx):
        if torch.cuda.is_available():
            _load_data_to_gpu(batch)
        if "points" in batch:
            points = [frame_point for frame_point in batch["points"]]
        else:
            points = None
        imgs = batch.get("imgs", None)
        metas = batch.get("mats_dict", None)
        ret_dict = self(points, imgs, metas, None)
        for result in ret_dict["pred_dicts"]:
            result["pred_labels"] -= 1
        pred_dicts = ret_dict["pred_dicts"]
        return self.val_dataloader.dataset.generate_prediction_dicts(
            batch, pred_dicts, self.model.class_names
        )[0]

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.lr, weight_decay=1e-7
        )
        scheduler = MultiStepLR(optimizer, [10, 15])
        return [optimizer], [scheduler]
