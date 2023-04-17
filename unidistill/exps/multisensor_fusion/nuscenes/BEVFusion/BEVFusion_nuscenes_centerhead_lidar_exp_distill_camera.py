import copy
import os
import random
import time

import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_sample_images
from torchvision.ops import roi_align

from unidistill.exps.base_cli import run_cli
from unidistill.exps.multisensor_fusion.nuscenes._base_.base_nuscenes_cfg import (
    _GRID_SIZE,
    _OUT_SIZE_FACTOR,
    _POINT_CLOUD_RANGE,
    _VOXEL_SIZE,
)
from unidistill.exps.multisensor_fusion.nuscenes.BEVFusion.BEVFusion_nuscenes_centerhead_fusion_exp import (
    Exp as BaseExp,
)
from unidistill.utils.torch_dist import reduce_mean


def corners_nd(dims, origin=0.5):
    """Generate relative box corners based on length per dim and origin point.

    Args:
        dims (np.ndarray, shape=[N, ndim]): Array of length per dim
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5

    Returns:
        np.ndarray, shape=[N, 2 ** ndim, ndim]: Returned corners.
        point layout example: (2d) x0y0, x0y1, x1y0, x1y1;
            (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
            where x0 < x1, y0 < y1, z0 < z1.
    """
    ndim = int(dims.shape[1])
    corners_norm = np.stack(np.unravel_index(np.arange(2**ndim), [2] * ndim), axis=1)
    # now corners_norm has format: (2d) x0y0, x0y1, x1y0, x1y1
    # (3d) x0y0z0, x0y0z1, x0y1z0, x0y1z1, x1y0z0, x1y0z1, x1y1z0, x1y1z1
    # so need to convert to a format which is convenient to do other computing.
    # for 2d boxes, format is clockwise start with minimum point
    # for 3d boxes, please draw lines by your hand.
    if ndim == 2:
        # generate clockwise box corners
        corners_norm = corners_norm[[0, 1, 3, 2]]
    elif ndim == 3:
        corners_norm = corners_norm[[0, 1, 3, 2, 4, 5, 7, 6]]
    corners_norm = corners_norm - np.array(origin)
    corners = dims.reshape([-1, 1, ndim]) * corners_norm.reshape([1, 2**ndim, ndim])
    return corners


def rotation_2d_reverse(points, angles):
    """Rotation 2d points based on origin point clockwise when angle positive.

    Args:
        points (np.ndarray): Points to be rotated with shape \
            (N, point_size, 2).
        angles (np.ndarray): Rotation angle with shape (N).

    Returns:
        np.ndarray: Same shape as points.
    """
    rot_sin = np.sin(angles)
    rot_cos = np.cos(angles)
    rot_mat_T = np.stack([[rot_cos, rot_sin], [-rot_sin, rot_cos]])
    return np.einsum("aij,jka->aik", points, rot_mat_T)


def center_to_corner_box2d(centers, dims, angles=None, origin=0.5):
    """Convert kitti locations, dimensions and angles to corners.
    format: center(xy), dims(xy), angles(clockwise when positive)

    Args:
        centers (np.ndarray): Locations in kitti label file with shape (N, 2).
        dims (np.ndarray): Dimensions in kitti label file with shape (N, 2).
        angles (np.ndarray, optional): Rotation_y in kitti label file with
            shape (N). Defaults to None.
        origin (list or array or float, optional): origin point relate to
            smallest point. Defaults to 0.5.

    Returns:
        np.ndarray: Corners with the shape of (N, 4, 2).
    """
    # 'length' in kitti format is in x axis.
    # xyz(hwl)(kitti label file)<->xyz(lhw)(camera)<->z(-x)(-y)(wlh)(lidar)
    # center in kitti format is [0.5, 1.0, 0.5] in xyz.
    corners = corners_nd(dims, origin=origin)
    # corners: [N, 4, 2]
    if angles is not None:
        corners = rotation_2d_reverse(corners, angles)
    corners += centers.reshape([-1, 1, 2])
    corners = torch.from_numpy(corners)
    return corners


def calculate_box_mask_gaussian(
    preds_shape, target, pc_range, voxel_size, out_size_scale
):
    B = preds_shape[0]
    C = preds_shape[1]
    H = preds_shape[2]
    W = preds_shape[3]
    gt_mask = np.zeros((B, H, W), dtype=np.float32)  # C * H * W

    for i in range(B):
        for j in range(len(target[i])):
            if target[i][j].sum() == 0:
                break

            w, h = (
                target[i][j][3] / (voxel_size[0] * out_size_scale),
                target[i][j][4] / (voxel_size[1] * out_size_scale),
            )
            radius = gaussian_radius((w, h))
            radius = max(0, int(radius))

            center_heatmap = [
                int((target[i][j][0] - pc_range[0]) / (voxel_size[0] * out_size_scale)),
                int((target[i][j][1] - pc_range[1]) / (voxel_size[1] * out_size_scale)),
            ]
            draw_umich_gaussian(gt_mask[i], center_heatmap, radius)

    gt_mask_torch = torch.from_numpy(gt_mask).cuda()
    return gt_mask_torch


def gaussian_radius(bbox_size, min_overlap=0.7):
    height, width = bbox_size

    a1 = 1
    b1 = height + width
    c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
    sq1 = np.sqrt(b1**2 - 4 * a1 * c1)
    r1 = (b1 + sq1) / 2

    a2 = 4
    b2 = 2 * (height + width)
    c2 = (1 - min_overlap) * width * height
    sq2 = np.sqrt(b2**2 - 4 * a2 * c2)
    r2 = (b2 + sq2) / 2

    a3 = 4 * min_overlap
    b3 = -2 * min_overlap * (height + width)
    c3 = (min_overlap - 1) * width * height
    sq3 = np.sqrt(b3**2 - 4 * a3 * c3)
    r3 = (b3 + sq3) / 2
    return min(r1, r2, r3)


def gaussian2D(shape, sigma=1):
    m, n = [(ss - 1.0) / 2.0 for ss in shape]
    y, x = np.ogrid[-m : m + 1, -n : n + 1]

    h = np.exp(-(x * x + y * y) / (2 * sigma * sigma))
    h[h < np.finfo(h.dtype).eps * h.max()] = 0
    return h


def draw_umich_gaussian(heatmap, center, radius, k=1):
    diameter = 2 * radius + 1
    gaussian = gaussian2D((diameter, diameter), sigma=diameter / 6)
    x, y = int(center[0]), int(center[1])
    height, width = heatmap.shape[0:2]

    left, right = min(x, radius), min(width - x, radius + 1)
    top, bottom = min(y, radius), min(height - y, radius + 1)

    masked_heatmap = heatmap[y - top : y + bottom, x - left : x + right]
    masked_gaussian = gaussian[
        radius - top : radius + bottom, radius - left : radius + right
    ]
    if min(masked_gaussian.shape) > 0 and min(masked_heatmap.shape) > 0:  # TODO debug
        np.maximum(masked_heatmap, masked_gaussian * k, out=masked_heatmap)
    return heatmap


def _load_data_to_gpu(data_dict):
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.cuda()
        elif isinstance(v, dict):
            _load_data_to_gpu(data_dict[k])
        else:
            data_dict[k] = v


def _sigmoid(x):
    y = torch.clamp(x.sigmoid(), min=1e-4, max=1 - 1e-4)
    return y


def FeatureDistillLoss(
    feature_lidar, feature_fuse, gt_boxes_bev_coords, gt_boxes_indices
):
    h, w = feature_lidar.shape[-2:]
    gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
    gt_boxes_bev_edge_1 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_2 = torch.mean(
        gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_3 = torch.mean(
        gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_4 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_all = torch.cat(
        (
            gt_boxes_bev_coords,
            gt_boxes_bev_center,
            gt_boxes_bev_edge_1,
            gt_boxes_bev_edge_2,
            gt_boxes_bev_edge_3,
            gt_boxes_bev_edge_4,
        ),
        dim=2,
    )
    gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
    gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
    gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
    feature_lidar_sample = torch.nn.functional.grid_sample(
        feature_lidar, gt_boxes_bev_all
    )
    feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
    feature_fuse_sample = torch.nn.functional.grid_sample(
        feature_fuse, gt_boxes_bev_all
    )
    feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
    criterion = nn.L1Loss(reduce=False)
    loss_feature_distill = criterion(
        feature_lidar_sample[gt_boxes_indices], feature_fuse_sample[gt_boxes_indices]
    )
    loss_feature_distill = torch.mean(loss_feature_distill, 2)
    loss_feature_distill = torch.mean(loss_feature_distill, 1)
    loss_feature_distill = torch.sum(loss_feature_distill)
    weight = gt_boxes_indices.float().sum()
    weight = reduce_mean(weight)
    loss_feature_distill = loss_feature_distill / (weight + 1e-4)
    return loss_feature_distill


def BEVDistillLoss(bev_lidar, bev_fuse, gt_boxes_bev_coords, gt_boxes_indices):
    h, w = bev_lidar.shape[-2:]
    gt_boxes_bev_center = torch.mean(gt_boxes_bev_coords, dim=2).unsqueeze(2)
    gt_boxes_bev_edge_1 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 1], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_2 = torch.mean(
        gt_boxes_bev_coords[:, :, [1, 2], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_3 = torch.mean(
        gt_boxes_bev_coords[:, :, [2, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_edge_4 = torch.mean(
        gt_boxes_bev_coords[:, :, [0, 3], :], dim=2
    ).unsqueeze(2)
    gt_boxes_bev_all = torch.cat(
        (
            gt_boxes_bev_coords,
            gt_boxes_bev_center,
            gt_boxes_bev_edge_1,
            gt_boxes_bev_edge_2,
            gt_boxes_bev_edge_3,
            gt_boxes_bev_edge_4,
        ),
        dim=2,
    )
    gt_boxes_bev_all[:, :, :, 0] = (gt_boxes_bev_all[:, :, :, 0] - w / 2) / (w / 2)
    gt_boxes_bev_all[:, :, :, 1] = (gt_boxes_bev_all[:, :, :, 1] - h / 2) / (h / 2)
    gt_boxes_bev_all[:, :, :, [0, 1]] = gt_boxes_bev_all[:, :, :, [1, 0]]
    feature_lidar_sample = torch.nn.functional.grid_sample(bev_lidar, gt_boxes_bev_all)
    feature_lidar_sample = feature_lidar_sample.permute(0, 2, 3, 1)
    feature_fuse_sample = torch.nn.functional.grid_sample(bev_fuse, gt_boxes_bev_all)
    feature_fuse_sample = feature_fuse_sample.permute(0, 2, 3, 1)
    criterion = nn.L1Loss(reduce=False)
    weight = gt_boxes_indices.float().sum()
    weight = reduce_mean(weight)
    gt_boxes_sample_lidar_feature = feature_lidar_sample.contiguous().view(
        -1, feature_lidar_sample.shape[-2], feature_lidar_sample.shape[-1]
    )
    gt_boxes_sample_fuse_feature = feature_fuse_sample.contiguous().view(
        -1, feature_fuse_sample.shape[-2], feature_fuse_sample.shape[-1]
    )
    gt_boxes_sample_lidar_feature = gt_boxes_sample_lidar_feature / (
        torch.norm(gt_boxes_sample_lidar_feature, dim=-1, keepdim=True) + 1e-4
    )
    gt_boxes_sample_fuse_feature = gt_boxes_sample_fuse_feature / (
        torch.norm(gt_boxes_sample_fuse_feature, dim=-1, keepdim=True) + 1e-4
    )
    gt_boxes_lidar_rel = torch.bmm(
        gt_boxes_sample_lidar_feature,
        torch.transpose(gt_boxes_sample_lidar_feature, 1, 2),
    )
    gt_boxes_fuse_rel = torch.bmm(
        gt_boxes_sample_fuse_feature,
        torch.transpose(gt_boxes_sample_fuse_feature, 1, 2),
    )
    gt_boxes_lidar_rel = gt_boxes_lidar_rel.contiguous().view(
        gt_boxes_bev_coords.shape[0],
        gt_boxes_bev_coords.shape[1],
        gt_boxes_lidar_rel.shape[-2],
        gt_boxes_lidar_rel.shape[-1],
    )
    gt_boxes_fuse_rel = gt_boxes_fuse_rel.contiguous().view(
        gt_boxes_bev_coords.shape[0],
        gt_boxes_bev_coords.shape[1],
        gt_boxes_fuse_rel.shape[-2],
        gt_boxes_fuse_rel.shape[-1],
    )
    loss_rel = criterion(
        gt_boxes_lidar_rel[gt_boxes_indices], gt_boxes_fuse_rel[gt_boxes_indices]
    )
    loss_rel = torch.mean(loss_rel, 2)
    loss_rel = torch.mean(loss_rel, 1)
    loss_rel = torch.sum(loss_rel)
    loss_rel = loss_rel / (weight + 1e-4)
    return loss_rel


def ResponseDistillLoss(
    resp_lidar, resp_fuse, gt_boxes, pc_range, voxel_size, out_size_scale
):
    cls_lidar = []
    reg_lidar = []
    cls_fuse = []
    reg_fuse = []
    criterion = nn.L1Loss(reduce=False)
    for task_id, task_out in enumerate(resp_lidar):
        cls_lidar.append(task_out["hm"])
        cls_fuse.append(_sigmoid(resp_fuse[task_id]["hm"] / 2))
        reg_lidar.append(
            torch.cat(
                [
                    task_out["reg"],
                    task_out["height"],
                    task_out["dim"],
                    task_out["rot"],
                    task_out["vel"],
                    task_out["iou"],
                ],
                dim=1,
            )
        )
        reg_fuse.append(
            torch.cat(
                [
                    resp_fuse[task_id]["reg"],
                    resp_fuse[task_id]["height"],
                    resp_fuse[task_id]["dim"],
                    resp_fuse[task_id]["rot"],
                    resp_fuse[task_id]["vel"],
                    resp_fuse[task_id]["iou"],
                ],
                dim=1,
            )
        )
    cls_lidar = torch.cat(cls_lidar, dim=1)
    reg_lidar = torch.cat(reg_lidar, dim=1)
    cls_fuse = torch.cat(cls_fuse, dim=1)
    reg_fuse = torch.cat(reg_fuse, dim=1)
    cls_lidar_max, _ = torch.max(cls_lidar, dim=1)
    cls_fuse_max, _ = torch.max(cls_fuse, dim=1)
    gaussian_mask = calculate_box_mask_gaussian(
        reg_lidar.shape,
        gt_boxes.cpu().detach().numpy(),
        pc_range,
        voxel_size,
        out_size_scale,
    )
    diff_reg = criterion(reg_lidar, reg_fuse)
    diff_cls = criterion(cls_lidar_max, cls_fuse_max)
    diff_reg = torch.mean(diff_reg, dim=1)
    diff_reg = diff_reg * gaussian_mask
    diff_cls = diff_cls * gaussian_mask
    weight = gaussian_mask.sum()
    weight = reduce_mean(weight)
    loss_reg_distill = torch.sum(diff_reg) / (weight + 1e-4)
    loss_cls_distill = torch.sum(diff_cls) / (weight + 1e-4)
    return loss_cls_distill, loss_reg_distill


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
        lidar_cfg = copy.deepcopy(self.model_cfg["lidar_encoder"])
        self.model_cfg["lidar_encoder"] = None
        self.teacher_model = BaseExp._configure_model(self)
        checkpoint = torch.load(
            "unidistill/exps/multisensor_fusion/nuscenes/BEVFusion/tmp/camera_model.pth",
            map_location=torch.device("cpu"),
        )
        model_state_dict = self.teacher_model.state_dict()
        checkpoint_state_dict = checkpoint["model_state"]
        for k in list(checkpoint_state_dict.keys()):
            if k in model_state_dict:
                shape_model = tuple(model_state_dict[k].shape)
                shape_checkpoint = tuple(checkpoint_state_dict[k].shape)
                if shape_model != shape_checkpoint:
                    checkpoint_state_dict.pop(k)
        self.checkpoint_state_dict = checkpoint_state_dict
        self.teacher_model.load_state_dict(checkpoint_state_dict, strict=False)
        self.teacher_model = self.teacher_model.cuda()
        self.teacher_model.det_head.dense_head.distill = True
        self.model_cfg["lidar_encoder"] = lidar_cfg
        self.model_cfg["camera_encoder"] = None
        self.freeze_model(self.teacher_model)
        self.teacher_model.eval()
        self.model = self._configure_model()
        self.train_dataloader = self.configure_train_dataloader()
        self.val_dataloader = self.configure_val_dataloader()
        self.test_dataloader = self.configure_test_dataloader()

    def freeze_model(self, model):
        if hasattr(model, "module"):
            for p in model.module.parameters():
                p.requires_grad = False
        else:
            for p in model.parameters():
                p.requires_grad = False

    def training_step(self, batch):
        start_time = time.time()
        if torch.cuda.is_available():
            _load_data_to_gpu(batch)
        if "points" in batch:
            points = [frame_point for frame_point in batch["points"]]
        else:
            points = None
        imgs = batch.get("imgs", None)
        metas = batch.get("mats_dict", None)
        gt_boxes = batch["gt_boxes"]
        gt_boxes_indice = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1]))
        for i in range(gt_boxes.shape[0]):
            cnt = gt_boxes[i].__len__() - 1
            while cnt > 0 and gt_boxes[i][cnt].sum() == 0:
                cnt -= 1
            gt_boxes_indice[i][: cnt + 1] = 1
        gt_boxes_indice = gt_boxes_indice.bool()
        gt_labels = batch["gt_labels"]
        gt_labels += 1
        gt_boxes = torch.cat([gt_boxes, gt_labels.unsqueeze(dim=2)], dim=2)
        ret_dict, tf_dict, feature_lidar, bev_lidar, resp_lidar, _ = self(
            points, imgs, metas, gt_boxes
        )
        self.teacher_model.load_state_dict(self.checkpoint_state_dict, strict=False)
        feature_fuse, bev_fuse, resp_fuse = self.teacher_model(
            points, imgs, metas, gt_boxes, return_feature=True
        )
        gt_boxes_bev_coords = torch.zeros((gt_boxes.shape[0], gt_boxes.shape[1], 4, 2))
        for i in range(gt_boxes.shape[0]):
            gt_boxes_tmp = gt_boxes[i]
            gt_boxes_tmp_bev = center_to_corner_box2d(
                gt_boxes_tmp[:, :2].cpu().detach().numpy(),
                gt_boxes_tmp[:, 3:5].cpu().detach().numpy(),
                gt_boxes_tmp[:, 6].cpu().detach().numpy(),
                origin=(0.5, 0.5),
            )
            gt_boxes_bev_coords[i] = gt_boxes_tmp_bev
        gt_boxes_bev_coords = gt_boxes_bev_coords.cuda()
        gt_boxes_indice = gt_boxes_indice.cuda()
        gt_boxes_bev_coords[:, :, :, 0] = (
            gt_boxes_bev_coords[:, :, :, 0] - _POINT_CLOUD_RANGE[0]
        ) / (_VOXEL_SIZE[0] * _OUT_SIZE_FACTOR)
        gt_boxes_bev_coords[:, :, :, 1] = (
            gt_boxes_bev_coords[:, :, :, 1] - _POINT_CLOUD_RANGE[1]
        ) / (_VOXEL_SIZE[1] * _OUT_SIZE_FACTOR)
        loss_feature = FeatureDistillLoss(
            feature_lidar, feature_fuse, gt_boxes_bev_coords, gt_boxes_indice
        )
        loss_bev_rel = BEVDistillLoss(
            bev_lidar, bev_fuse, gt_boxes_bev_coords, gt_boxes_indice
        )
        loss_resp_cls, loss_resp_reg = ResponseDistillLoss(
            resp_lidar,
            resp_fuse,
            gt_boxes,
            _POINT_CLOUD_RANGE,
            _VOXEL_SIZE,
            _OUT_SIZE_FACTOR,
        )
        tf_dict.update(
            {
                "loss_feature": loss_feature,
                "loss_bev_rel": loss_bev_rel,
                "loss_resp_cls": loss_resp_cls,
                "loss_resp_reg": loss_resp_reg,
            }
        )
        loss = (
            ret_dict["loss"].mean()
            + 10 * (loss_feature)
            + 5 * loss_bev_rel
            + 1 * (loss_resp_cls + loss_resp_reg)
        )
        end_time = time.time()
        return loss

    def _change_cfg_params(self):
        self.data_cfg["aug_cfg"]["gt_sampling_cfg"] = None


if __name__ == "__main__":
    import logging

    logging.getLogger("mmcv").disabled = True
    logging.getLogger("mmseg").disabled = True
    run_cli(Exp, "BEVFusion_nuscenes_centerhead_lidar_exp_distill_camera")
