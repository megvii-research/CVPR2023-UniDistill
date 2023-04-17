import mmcv
import numpy as np
import torch
from PIL import Image

from .functional import bev_transform, center_to_corner_box3d, img_transform


class Compose(torch.nn.Module):
    def __init__(self, transforms):
        super().__init__()
        self.transforms = transforms

    def forward(self, data_dict):
        for t in self.transforms:
            data_dict = t(data_dict)
        return data_dict


class GlobalScaling(torch.nn.Module):
    def __init__(self, scale_range=(0.95, 1.05)):
        super().__init__()
        assert len(scale_range) == 2 and scale_range[1] - scale_range[0] > 1e-3
        self.scale_range = scale_range

    def forward(self, data_dict):
        gt_boxes = data_dict["gt_boxes"]
        points = data_dict["points"]

        if len(gt_boxes) > 0:
            noise_scale = np.random.uniform(self.scale_range[0], self.scale_range[1])
            points[:, :3] *= noise_scale
            gt_boxes[:, :6] *= noise_scale

            data_dict["gt_boxes"] = gt_boxes
            data_dict["points"] = points
        return data_dict


class RandomFlip3D(torch.nn.Module):
    def __init__(self, along_axis="X"):
        super().__init__()
        assert along_axis.upper() in ("X", "Y")
        self.along_axis = along_axis.upper()

    def forward(self, data_dict):
        gt_boxes = data_dict["gt_boxes"]
        points = data_dict["points"]
        enable = np.random.choice([False, True], replace=False, p=[0.5, 0.5])

        if enable:
            if self.along_axis == "X":
                if len(gt_boxes) > 0:
                    gt_boxes[:, 1] = -gt_boxes[:, 1]
                    gt_boxes[:, 6] = -gt_boxes[:, 6]
                points[:, 1] = -points[:, 1]

            if self.along_axis == "Y":
                if len(gt_boxes) > 0:
                    gt_boxes[:, 0] = -gt_boxes[:, 0]
                    gt_boxes[:, 6] = -(gt_boxes[:, 6] + np.pi)
                points[:, 0] = -points[:, 0]

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = (
        torch.stack((cosa, sina, zeros, -sina, cosa, zeros, zeros, zeros, ones), dim=1)
        .view(-1, 3, 3)
        .float()
    )
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot


class GlobalRotation(torch.nn.Module):
    def __init__(self, rot_range=(-np.pi / 4, np.pi / 4)):
        super().__init__()
        assert len(rot_range) == 2 and rot_range[1] - rot_range[0] > 1e-3
        self.rot_range = rot_range

    def forward(self, data_dict):
        gt_boxes = data_dict["gt_boxes"]
        points = data_dict["points"]

        noise_rotation = np.random.uniform(self.rot_range[0], self.rot_range[1])
        points = rotate_points_along_z(
            points[np.newaxis, :, :], np.array([noise_rotation])
        )[0]
        if len(gt_boxes) > 0:
            gt_boxes[:, 0:3] = rotate_points_along_z(
                gt_boxes[np.newaxis, :, 0:3], np.array([noise_rotation])
            )[0]
            gt_boxes[:, 6] += noise_rotation

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict


class GlobalTranslation(torch.nn.Module):
    def __init__(self, noise_translate_std):
        super().__init__()
        if not isinstance(noise_translate_std, (list, tuple, np.ndarray)):
            noise_translate_std = np.array(
                [noise_translate_std, noise_translate_std, noise_translate_std]
            )
        self.noise_translate_std = noise_translate_std

    def forward(self, data_dict):
        gt_boxes = data_dict["gt_boxes"]
        points = data_dict["points"]

        if all([e == 0 for e in self.noise_translate_std]):
            return data_dict

        noise_translate = np.array(
            [
                np.random.normal(0, self.noise_translate_std[0], 1),
                np.random.normal(0, self.noise_translate_std[1], 1),
                np.random.normal(0, self.noise_translate_std[2], 1),
            ]
        ).T

        points[:, :3] += noise_translate
        if len(gt_boxes) > 0:
            gt_boxes[:, :3] += noise_translate

        data_dict["gt_boxes"] = gt_boxes
        data_dict["points"] = points
        return data_dict


class PointShuffle(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        points = data_dict["points"]
        np.random.shuffle(points)
        data_dict["points"] = points
        return data_dict


class GTSampling(torch.nn.Module):
    def __init__(
        self,
        root_path,
        data_name,
        data_split,
        class_names,
        sampler_groups,
        num_point_feature,
        remove_extra_width=(0, 0, 0),
        use_road_plane=False,
        database_with_fakelidar=False,
        filter_by_min_points_cfg=None,
        removed_difficulty=None,
        limit_whole_scene=False,
        stop_epoch=None,
        logger=None,
    ):
        super().__init__()
        self.epoch = -1
        self.stop_epoch = stop_epoch
        self.db_sampler = DataBaseSampler(
            root_path=root_path,
            data_name=data_name,
            data_split=data_split,
            sampler_groups=sampler_groups,
            num_point_feature=num_point_feature,
            remove_extra_width=remove_extra_width,
            use_road_plane=use_road_plane,
            database_with_fakelidar=database_with_fakelidar,
            filter_by_min_points_cfg=filter_by_min_points_cfg,
            removed_difficulty=removed_difficulty,
            limit_whole_scene=limit_whole_scene,
            class_names=class_names,
            logger=logger,
        )

    def set_epoch(self, epoch):
        self.epoch = epoch

    def forward(self, data_dict):
        if self.stop_epoch is not None and (self.epoch + 1) >= self.stop_epoch:
            return data_dict
        if data_dict.get("points", None) is not None:
            data_dict = self.db_sampler(data_dict)
        return data_dict


class RandomJitterPoints(torch.nn.Module):
    def __init__(self, jitter_std=(0.01, 0.01, 0.01), clip_range=(-0.05, 0.05)):
        super().__init__()
        seq_types = (list, tuple, np.ndarray)
        if not isinstance(jitter_std, seq_types):
            assert isinstance(
                jitter_std, (int, float)
            ), f"unsupported jitter_std type {type(jitter_std)}"
            jitter_std = [jitter_std, jitter_std, jitter_std]
        self.jitter_std = jitter_std

        if clip_range is not None:
            if not isinstance(clip_range, seq_types):
                assert isinstance(
                    clip_range, (int, float)
                ), f"unsupported clip_range type {type(clip_range)}"
                clip_range = [-clip_range, clip_range]
        self.clip_range = clip_range

    def forward(self, data_dict):
        points = data_dict["points"]

        jitter_std = np.array(self.jitter_std, dtype=np.float32)
        jitter_noise = np.random.randn(points.shape[0], 3) * jitter_std[None, :]
        if self.clip_range is not None:
            jitter_noise = np.clip(jitter_noise, self.clip_range[0], self.clip_range[1])

        points[:, :3] += jitter_noise
        data_dict["points"] = points
        return data_dict


class ObjectRangeFilter(torch.nn.Module):
    def __init__(self, point_cloud_range):
        super().__init__()
        self.point_cloud_range = np.array(point_cloud_range, dtype=np.float32)

    @staticmethod
    def mask_points_by_range(points, limit_range):
        mask = (
            (points[:, 0] >= limit_range[0])
            & (points[:, 0] <= limit_range[3])
            & (points[:, 1] >= limit_range[1])
            & (points[:, 1] <= limit_range[4])
        )
        return mask

    @staticmethod
    def mask_boxes_outside_range_numpy(boxes, limit_range, min_num_corners=1):
        if boxes.shape[1] > 7:
            boxes = boxes[:, 0:7]
        corners = center_to_corner_box3d(
            boxes[:, :3], boxes[:, 3:6], boxes[:, 6], origin=(0.5, 0.5, 0.5), axis=2
        )
        mask = ((corners >= limit_range[0:3]) & (corners <= limit_range[3:6])).all(
            axis=2
        )
        mask = mask.sum(axis=1) >= min_num_corners

        return mask

    def forward(self, data_dict):
        if data_dict.get("points", None) is not None:
            mask = self.mask_points_by_range(
                data_dict["points"], self.point_cloud_range
            )
            data_dict["points"] = data_dict["points"][mask]

        if len(data_dict.get("gt_boxes", [])) > 0 and self.training:
            mask = self.mask_boxes_outside_range_numpy(
                data_dict["gt_boxes"], self.point_cloud_range
            )
            data_dict["gt_boxes"] = data_dict["gt_boxes"][mask]
            if data_dict.get("gt_names", None) is not None:
                data_dict["gt_names"] = data_dict["gt_names"][mask]
            if data_dict.get("gt_labels", None) is not None:
                data_dict["gt_labels"] = data_dict["gt_labels"][mask]
        return data_dict


class MultiModalGTSampling(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

    def forward(self, data_dict):
        return data_dict


class ImageAffineTransformation(torch.nn.Module):
    def __init__(self, is_train=False, **kwargs):
        super().__init__()
        self.aug_conf = kwargs
        self.is_train = is_train

    def sample_augs(self):
        H, W = self.aug_conf["H"], self.aug_conf["W"]
        fH, fW = self.aug_conf["final_dim"]
        if self.is_train:
            resize = np.random.uniform(*self.aug_conf["resize_lim"])
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = (
                int((1 - np.random.uniform(*self.aug_conf["bot_pct_lim"])) * newH) - fH
            )
            crop_w = int(np.random.uniform(0, max(0, newW - fW)))
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            if self.aug_conf["rand_flip"] and np.random.choice([0, 1]):
                flip = True
            rotate_ida = np.random.uniform(*self.aug_conf["rot_lim"])
        else:
            resize = max(fH / H, fW / W)
            resize_dims = (int(W * resize), int(H * resize))
            newW, newH = resize_dims
            crop_h = int((1 - np.mean(self.aug_conf["bot_pct_lim"])) * newH) - fH
            crop_w = int(max(0, newW - fW) / 2)
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
            flip = False
            rotate_ida = 0
        return resize, resize_dims, crop, flip, rotate_ida

    def forward(self, data_dict):
        if data_dict.get("imgs", None) is not None:
            data_dict["ida_mat"] = dict()
            for cam in data_dict["imgs"].keys():
                resize, resize_dims, crop, flip, rotate_ida = self.sample_augs()
                img = Image.fromarray(data_dict["imgs"][cam])
                img, ida_mat = img_transform(
                    img,
                    resize=resize,
                    resize_dims=resize_dims,
                    crop=crop,
                    flip=flip,
                    rotate=rotate_ida,
                )
                data_dict["imgs"][cam] = img
                data_dict["ida_mat"][cam] = ida_mat
        return data_dict


class ImageNormalize(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.aug_conf = kwargs
        self.aug_conf["img_mean"] = np.array(self.aug_conf["img_mean"])
        self.aug_conf["img_std"] = np.array(self.aug_conf["img_std"])

    def forward(self, data_dict):
        if data_dict.get("imgs", None) is not None:
            for cam in data_dict["imgs"].keys():
                img = data_dict["imgs"][cam]
                img = mmcv.imnormalize(
                    np.array(img),
                    self.aug_conf["img_mean"],
                    self.aug_conf["img_std"],
                    self.aug_conf["to_rgb"],
                )
                data_dict["imgs"][cam] = img
        return data_dict


class CollectCameraSweeps(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        return data_dict


class CollectLidarSweeps(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, data_dict):
        if data_dict.get("points", None) is not None:
            key_ego_to_global = data_dict["info"]["ego_to_global"]
            key_lidar_to_ego = data_dict["info"]["lidar_to_ego"]
            all_points = data_dict["points"].copy()
            if all_points.shape[-1] == 5:
                all_points[:, -1] = 0.0
            for swid, frame in enumerate(data_dict.pop("sweep_points")):
                sweep_ego_to_global = data_dict["info"]["sweep_lidar_infos"][swid][
                    "sweep_lidar_to_ego"
                ]
                homogeneous_point = np.ones((frame.shape[0], 4))
                homogeneous_point[:, :3] = frame[:, :3]
                sweep_on_key = (
                    np.linalg.inv(key_lidar_to_ego)
                    @ np.linalg.inv(key_ego_to_global)
                    @ sweep_ego_to_global
                    @ key_lidar_to_ego
                    @ homogeneous_point.T
                ).T
                frame[:, :3] = sweep_on_key[:, :3]
                if all_points.shape[-1] == 5:
                    frame[:, -1] = (
                        data_dict["info"]["timestamp"]
                        - data_dict["info"]["sweep_lidar_infos"][swid][
                            "sweep_lidar_timestamp"
                        ]
                    ) / 1e6
                all_points = np.concatenate([all_points, frame])
            data_dict["points"] = all_points
            data_dict["info"].pop("sweep_lidar_infos")
        return data_dict


class BevAffineTransformation(torch.nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self.aug_conf = kwargs

    def sample_augs(self):
        rotate_bda = np.random.uniform(*self.aug_conf["rot_lim"])
        scale_bda = np.random.uniform(*self.aug_conf["scale_lim"])
        trans_bda = np.random.normal(scale=self.aug_conf["trans_lim"])
        flip_dx = np.random.uniform() < self.aug_conf["flip_dx_ratio"]
        flip_dy = np.random.uniform() < self.aug_conf["flip_dy_ratio"]
        return rotate_bda, scale_bda, trans_bda, flip_dx, flip_dy

    def forward(self, data_dict):
        rotate_bda, scale_bda, trans_bda, flip_dx, flip_dy = self.sample_augs()
        gt_boxes = data_dict["gt_boxes"]
        gt_boxes, transform_mat = bev_transform(
            gt_boxes, rotate_bda, scale_bda, trans_bda, flip_dx, flip_dy
        )
        if data_dict.get("points", None) is not None:
            homogeneous_point = np.ones((data_dict["points"].shape[0], 4))
            homogeneous_point[:, :3] = data_dict["points"][:, :3]
            homogeneous_point_transform = (transform_mat @ homogeneous_point.T).T
            data_dict["points"][:, :3] = homogeneous_point_transform[:, :3]
        if data_dict.get("imgs", None) is not None:
            data_dict["bda_mat"] = transform_mat
        return data_dict
