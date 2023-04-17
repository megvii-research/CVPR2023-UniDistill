import copy
import io
import os
import pickle
from typing import Dict

import numpy as np
import pandas as pd
import pytorch_lightning as pl
from skimage import io as skimage_io
from torch.utils.data import Dataset

_map_name_from_general_to_detection = {
    "human.pedestrian.adult": "pedestrian",
    "human.pedestrian.child": "pedestrian",
    "human.pedestrian.wheelchair": "ignore",
    "human.pedestrian.stroller": "ignore",
    "human.pedestrian.personal_mobility": "ignore",
    "human.pedestrian.police_officer": "pedestrian",
    "human.pedestrian.construction_worker": "pedestrian",
    "animal": "ignore",
    "vehicle.car": "car",
    "vehicle.motorcycle": "motorcycle",
    "vehicle.bicycle": "bicycle",
    "vehicle.bus.bendy": "bus",
    "vehicle.bus.rigid": "bus",
    "vehicle.truck": "truck",
    "vehicle.construction": "construction_vehicle",
    "vehicle.emergency.ambulance": "ignore",
    "vehicle.emergency.police": "ignore",
    "vehicle.trailer": "trailer",
    "movable_object.barrier": "barrier",
    "movable_object.trafficcone": "traffic_cone",
    "movable_object.pushable_pullable": "ignore",
    "movable_object.debris": "ignore",
    "static_object.bicycle_rack": "ignore",
}


class NuscenesMultiModalDataset(Dataset):
    def __init__(
        self,
        class_names=(
            "car",
            "truck",
            "construction_vehicle",
            "bus",
            "trailer",
            "barrier",
            "motorcycle",
            "bicycle",
            "pedestrian",
            "traffic_cone",
        ),
        data_split="training",
        logger=None,
        root_path=None,
        img_key_list=[
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
        ],
        lidar_key_list=["LIDAR_TOP"],
        use_mapping_names=True,
        num_lidar_sweeps=0,
        lidar_sweeps_idx=None,
        num_cam_sweeps=0,
        cam_sweeps_idx=None,
        lidar_with_timestamp=True,
        **kwargs
    ):
        assert data_split in [
            "training",
            "validation",
            "testing",
            "trainval",
        ]
        assert set(img_key_list).issubset(
            [
                "CAM_BACK",
                "CAM_BACK_LEFT",
                "CAM_BACK_RIGHT",
                "CAM_FRONT",
                "CAM_FRONT_LEFT",
                "CAM_FRONT_RIGHT",
            ]
        ), "Illegal image keys"
        assert set(lidar_key_list).issubset(["LIDAR_TOP"]), "Illegal lidar keys"
        super().__init__()
        table = {"training": "train", "validation": "val", "testing": "test"}
        self.data_split = table[data_split]
        self.class_names = class_names
        self.root_path = root_path
        self.logger = logger

        self.img_key_list = img_key_list
        self.lidar_key_list = lidar_key_list
        self.use_mapping_names = use_mapping_names
        self.num_lidar_sweeps = num_lidar_sweeps
        self.lidar_with_timestamp = lidar_with_timestamp
        if lidar_sweeps_idx is None:
            self.lidar_sweeps_idx = list(range(num_lidar_sweeps))
        else:
            self.lidar_sweeps_idx = num_lidar_sweeps
        self.num_cam_sweeps = num_cam_sweeps
        if cam_sweeps_idx is None:
            self.cam_sweeps_idx = list(range(num_cam_sweeps))
        else:
            self.cam_sweeps_idx = num_cam_sweeps
        assert self.with_lidar or self.with_camera, "Must have one Sensor!"
        with open("/data/dataset/{}_info.pkl".format(self.data_split), "rb") as f:
            self.infos = pickle.load(f)

    @property
    def with_lidar(self):
        """bool: whether the dataset has lidar"""
        return hasattr(self, "lidar_key_list") and len(self.lidar_key_list) > 0

    @property
    def with_camera(self):
        """bool: whether the dataset has camera"""
        return hasattr(self, "img_key_list") and len(self.img_key_list) > 0

    @property
    def with_seg(self):
        """bool: whether the dataset has segmentation"""
        return False

    def __getitem__(self, idx: int):
        """
        loading data-of-all-sensors with given index

        Args:
            idx: int, Sampled index
        Returns:
            dict
        """

        item = {}
        item_info = copy.deepcopy(self.infos[idx])
        if self.use_mapping_names and item_info.get("gt_names", None) is not None:
            item_info["gt_names"] = self.get_mapping_names(
                item_info["gt_names"], mapping_dict=_map_name_from_general_to_detection
            )
        if item_info.get("ann_infos", None) is not None:
            for i in range(len(item_info["ann_infos"])):
                item_info["ann_infos"][i][
                    "gt_name"
                ] = _map_name_from_general_to_detection[
                    item_info["ann_infos"][i]["category_name"]
                ]
        if self.with_camera:
            item["imgs"] = self._get_images(idx, keys=self.img_key_list)
            self._load_camera_sweeps(item_info, item)

        if self.with_lidar:
            item["points"] = self._get_point_cloud(idx, keys=self.lidar_key_list)
            self._load_lidar_sweeps(item_info, item)

        item["info"] = item_info
        return item

    def get_mapping_names(
        self, gt_names, mapping_dict=_map_name_from_general_to_detection
    ):
        return np.array(list(map(lambda x: mapping_dict[x], gt_names)))

    def _get_images(self, idx, keys):
        result = {}
        for k in keys:
            img_file = os.path.join(
                "/data/dataset/", self.infos[idx]["cam_infos"][k]["filename"]
            )
            result[k] = skimage_io.imread(img_file)
        return result

    def _load_camera_sweeps(self, item_info: Dict, item: Dict) -> None:
        sweeps_imgs = list()
        sweeps_infos = list()
        for sw_idx in self.cam_sweeps_idx:
            sw_imgs = dict()
            sw_infos = dict()
            for cam in self.img_key_list:
                cam_sw_idx = min(sw_idx, len(item_info["cam_sweeps"]) - 1)
                while (
                    cam_sw_idx >= 0 and cam not in item_info["cam_sweeps"][cam_sw_idx]
                ):
                    cam_sw_idx -= 1
                if cam_sw_idx >= 0:
                    sw_img_file = os.path.join(
                        "/data/dataset/",
                        item_info["cam_sweeps"][cam_sw_idx][cam]["filename"],
                    )
                    sw_imgs[cam] = skimage_io.imread(io.BytesIO(sw_img_file))
                    sw_infos[cam] = item_info["cam_sweeps"][cam_sw_idx][cam]
                else:
                    sw_imgs[cam] = copy.deepcopy(item["imgs"][cam])
                    sw_infos[cam] = copy.deepcopy(item_info["cam_infos"][cam])
            sweeps_imgs.append(sw_imgs)
            sweeps_infos.append(sw_infos)
        item["sweep_imgs"] = sweeps_imgs
        item_info["sweep_img_infos"] = sweeps_infos

    def _get_point_cloud(self, idx, keys):
        load_dim = 5 if self.lidar_with_timestamp else 4
        result = {}
        for k in keys:
            point_file = os.path.join(
                "/data/dataset/", self.infos[idx]["lidar_infos"][k]["filename"]
            )
            point_cloud = np.fromfile(point_file, dtype=np.float32, count=-1).reshape(
                -1, 5
            )
            result[k] = point_cloud[:, :load_dim].copy()
        return result

    def _load_lidar_sweeps(self, item_info: Dict, item: Dict) -> None:
        load_dim = 5 if self.lidar_with_timestamp else 4
        sweep_points = list()
        sweep_points_infos = list()
        for sw_idx in self.lidar_sweeps_idx:
            sw_lidar = dict()
            sw_lidar_infos = dict()
            for lidar in self.lidar_key_list:
                lidar_sw_idx = min(sw_idx, len(item_info["lidar_sweeps"]) - 1)
                if lidar_sw_idx >= 0:
                    sw_point_file = os.path.join(
                        "/data/dataset/",
                        item_info["lidar_sweeps"][lidar_sw_idx][lidar]["filename"],
                    )
                    point_cloud = np.fromfile(
                        sw_point_file, dtype=np.float32, count=-1
                    ).reshape(-1, 5)
                    sw_lidar[lidar] = point_cloud[:, :load_dim].copy()
                    sw_lidar_infos[lidar] = item_info["lidar_sweeps"][lidar_sw_idx][
                        lidar
                    ]
                else:
                    sw_lidar[lidar] = copy.deepcopy(item["points"][lidar])
                    sw_lidar_infos[lidar] = copy.deepcopy(
                        item_info["lidar_infos"][lidar]
                    )
            sweep_points.append(sw_lidar)
            sweep_points_infos.append(sw_lidar_infos)
        item["sweep_points"] = sweep_points
        item_info["sweep_lidar_infos"] = sweep_points_infos
