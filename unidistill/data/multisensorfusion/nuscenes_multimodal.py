import os
import pickle
from typing import Dict

import numpy as np
import torch
from mmdet3d.core.bbox.structures.lidar_box3d import LiDARInstance3DBoxes
from pyquaternion import Quaternion

from . import transforms3d
from .nuScenes_multimodal import NuscenesMultiModalDataset

map_name_from_general_to_detection = {
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


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


class NuscenesMultiModalData(NuscenesMultiModalDataset):
    r"""This class is Nuscenes multi-modal dataset, such as lidar, camera."""

    def __init__(
        self,
        aug_cfg=None,
        root_path="/data/dataset",
        lidar_key_list=["LIDAR_TOP"],
        img_key_list=[
            "CAM_BACK",
            "CAM_BACK_LEFT",
            "CAM_BACK_RIGHT",
            "CAM_FRONT",
            "CAM_FRONT_LEFT",
            "CAM_FRONT_RIGHT",
        ],
        class_names=None,
        use_cbgs=False,
        data_split="training",
        num_lidar_sweeps=0,
        num_cam_sweeps=0,
        lidar_with_timestamp=True,
        filter_empty=True,
        **kwargs
    ):
        super().__init__(
            class_names=class_names,
            data_split=data_split,
            root_path=root_path,
            img_key_list=img_key_list,
            lidar_key_list=lidar_key_list,
            num_lidar_sweeps=num_lidar_sweeps,
            num_cam_sweeps=num_cam_sweeps,
            lidar_with_timestamp=lidar_with_timestamp,
        )
        self.classes = class_names
        self.is_train = data_split in ["training", "trainval"]
        self.use_cbgs = use_cbgs and self.is_train
        if self.use_cbgs:
            self.cat2id = {name: i for i, name in enumerate(self.classes)}
            self.sample_indices = self._get_sample_indices()
        self.filter_empty = filter_empty
        self.num_lidar_sweeps = num_lidar_sweeps

        self.data_processor = transforms3d.Compose(
            [
                transforms3d.CollectCameraSweeps(),
                transforms3d.CollectLidarSweeps(),
            ]
        )

        self.det_augmentor = self.get_det_augmentor(aug_cfg)

        if self.with_seg:
            # TODO:
            self.seg_augmentor = transforms3d.Compose([...])

        # TODO:
        if not self.is_train:
            meta_file = "/data/dataset/nuscenes_v1.0-trainval_meta.pkl"
            self.meta_info = load_pkl(meta_file)

    def get_det_augmentor(self, aug_cfg):
        augmentor_list = []
        if aug_cfg.get("ida_aug_cfg", None):
            augmentor_list.append(
                transforms3d.ImageAffineTransformation(
                    **aug_cfg["ida_aug_cfg"], is_train=self.is_train
                )
            )
        if self.is_train:
            if aug_cfg.get("gt_sampling_cfg", None):
                augmentor_list.append(
                    transforms3d.GTSampling(**aug_cfg["gt_sampling_cfg"])
                )
            if aug_cfg.get("bda_aug_cfg", None):
                augmentor_list.append(
                    transforms3d.BevAffineTransformation(**aug_cfg["bda_aug_cfg"])
                )
        augmentor_list.append(
            transforms3d.ObjectRangeFilter(aug_cfg["point_cloud_range"])
        )
        augmentor_list.append(transforms3d.ImageNormalize(**aug_cfg["img_norm_cfg"]))
        return transforms3d.Compose(augmentor_list)

    def set_epoch(self, epoch):
        self.epoch = epoch
        for trans in self.det_augmentor.transforms:
            if hasattr(trans, "set_epoch"):
                trans.set_epoch(epoch)

    def __len__(self):
        if self.use_cbgs:
            return len(self.sample_indices)
        else:
            return len(self.infos)

    def _get_sample_indices(self):
        """Load annotations from ann_file.

        Args:
            ann_file (str): Path of the annotation file.

        Returns:
            list[dict]: List of annotations after class sampling.
        """
        class_sample_idxs = {cat_id: [] for cat_id in self.cat2id.values()}
        for idx, info in enumerate(self.infos):
            if "ann_infos" in info:
                gt_names = [ann_info["category_name"] for ann_info in info["ann_infos"]]
            else:
                gt_names = [category for category in info["gt_names"]]
            gt_names = set(
                [map_name_from_general_to_detection[gt_name] for gt_name in gt_names]
            )
            for gt_name in gt_names:
                if gt_name not in self.classes:
                    continue
                class_sample_idxs[self.cat2id[gt_name]].append(idx)
        duplicated_samples = sum([len(v) for _, v in class_sample_idxs.items()])
        class_distribution = {
            k: len(v) / duplicated_samples for k, v in class_sample_idxs.items()
        }

        sample_indices = []

        frac = 1.0 / len(self.classes)
        ratios = [frac / v for v in class_distribution.values()]
        for cls_inds, ratio in zip(list(class_sample_idxs.values()), ratios):
            sample_indices += np.random.choice(
                cls_inds, int(len(cls_inds) * ratio)
            ).tolist()
        return sample_indices

    def _generate_data_dict(self, org_data: Dict) -> Dict:
        data_dict = dict()

        if self.is_train:
            mask = [
                org_data["info"]["gt_names"][i] in self.classes
                and (
                    org_data["info"]["num_lidar_pts"][i]
                    + org_data["info"]["num_radar_pts"][i]
                )
                > 0
                for i in range(len(org_data["info"]["gt_boxes"]))
            ]
            data_dict["gt_boxes"] = org_data["info"]["gt_boxes"][mask]
            data_dict["gt_labels"] = np.array(
                [self.classes.index(i) for i in org_data["info"]["gt_names"][mask]]
            )
        else:
            data_dict["gt_boxes"] = np.zeros((0, 9))
            data_dict["gt_labels"] = np.zeros(0)

        data_dict["info"] = dict()
        data_dict["info"]["timestamp"] = org_data["info"]["timestamp"]
        data_dict["info"]["ego_to_global"] = np.linalg.inv(
            org_data["info"]["car_from_global"]
        )

        if self.with_lidar:
            data_dict["with_lidar"] = True
            data_dict["info"]["lidar_to_ego"] = np.linalg.inv(
                org_data["info"]["ref_from_car"]
            )
            data_dict["points"] = org_data["points"]["LIDAR_TOP"]
            if "sweep_points" in org_data:
                data_dict["sweep_points"] = [
                    sweep_pts["LIDAR_TOP"] for sweep_pts in org_data["sweep_points"]
                ]
                new_sweep_infos = []
                for sweep_info in org_data["info"]["sweep_lidar_infos"]:
                    new_sweep_info = dict(
                        sweep_lidar_to_ego=np.linalg.inv(
                            sweep_info["LIDAR_TOP"]["car_from_global"]
                        ),
                        sweep_lidar_timestamp=sweep_info["LIDAR_TOP"]["timestamp"],
                    )
                    new_sweep_infos.append(new_sweep_info)
                data_dict["info"]["sweep_lidar_infos"] = new_sweep_infos
        else:
            data_dict["with_lidar"] = False

        if self.with_camera:
            data_dict["with_camera"] = True
            data_dict["imgs"] = org_data["imgs"]
            # TODO: Support multisweep imgs
            if "sweep_imgs" in org_data:
                if len(org_data["sweep_imgs"]) > 0:
                    raise NotImplementedError
        else:
            data_dict["with_camera"] = False

        return data_dict

    def format_processor(self, data_dict, data):
        """
        Data output for model forward:
            "imgs": image data, (num_cam_sweeps, num_cams, 3, h, w) tensor
            "points": point cloud data, (num_points, lidar_dims) tensor
            "mats_dict": dict of translation matrixs
                "sensor2ego_mats": (num_cam_sweeps, num_cams, 4, 4) tensor
                "intrin_mats": (num_cam_sweeps, num_cams, 4, 4) tensor
                "ida_mats": (num_cam_sweeps, num_cams, 4, 4) tensor
                "sensor2sensor_mats": (num_cam_sweeps, num_cams, 4, 4) tensor
                "bda_mat": (4, 4) tensor
            "img_metas": dict of box type and frame token, for validation
                "box_type_3d": class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'
                "token": str, token of nuScenes
                "ego2global_translation": (3) tensor
                "ego2global_rotation": (4) tensor (quaternion angle)
            "gt_boxes": (num_boxes, box_dims) tensor, (x,y,z,l,w,h,angle) or (x,y,z,l,w,h,angle,vx,vy)
            "gt_labels": (num_boxes) tensor, class id as int type
        """
        ret_data = dict()
        if self.with_camera:
            imgs = []
            sensor2ego_mats = []
            intrin_mats = []
            ida_mats = []
            sensor2sensor_mats = []
            for cam_name in self.img_key_list:
                img = torch.from_numpy(data_dict["imgs"][cam_name]).permute(2, 0, 1)
                imgs.append(img)
                sensor2ego_mat = torch.eye(4)
                sensor2ego_mat[:3, :3] = torch.Tensor(
                    Quaternion(
                        data["info"]["sensor2ego_rotations"][cam_name]
                    ).rotation_matrix
                )
                sensor2ego_mat[:3, 3] = torch.Tensor(
                    data["info"]["sensor2ego_translations"][cam_name]
                )
                sensor2ego_mat = (
                    torch.Tensor(data["info"]["ref_from_car"]).to(sensor2ego_mat.dtype)
                    @ sensor2ego_mat
                )
                sensor2ego_mats.append(sensor2ego_mat)
                intrin_mat = torch.eye(4)
                intrin_mat[:3, :3] = torch.Tensor(
                    data["info"]["cam_infos"][cam_name]["calibrated_sensor"][
                        "camera_intrinsic"
                    ]
                )
                intrin_mats.append(intrin_mat)
                ida_mats.append(torch.from_numpy(data_dict["ida_mat"][cam_name]))
                sensor2sensor_mats.append(torch.eye(4, 4))
            ret_data["imgs"] = torch.stack(imgs).unsqueeze(0)
            bda_mat = data_dict["bda_mat"] if "bda_mat" in data_dict else torch.eye(4)
            ret_data["mats_dict"] = dict(
                sensor2ego_mats=torch.stack(sensor2ego_mats).unsqueeze(0),
                intrin_mats=torch.stack(intrin_mats).unsqueeze(0),
                ida_mats=torch.stack(ida_mats).unsqueeze(0),
                sensor2sensor_mats=torch.stack(sensor2sensor_mats).unsqueeze(0),
                bda_mat=bda_mat,
            )

        if self.with_lidar:
            ret_data["points"] = torch.tensor(data_dict["points"])

        ret_data["img_metas"] = dict(
            box_type_3d=LiDARInstance3DBoxes,
            token=data["info"]["sample_token"],
            ego2global_translation=data["info"]["ego2global_translation"],
            ego2global_rotation=data["info"]["ego2global_rotation"],
        )

        ret_data["gt_boxes"] = data_dict["gt_boxes"]
        ret_data["gt_boxes"][
            np.isnan(ret_data["gt_boxes"])
        ] = 0  # replace nan velocity of cones and barriers
        ret_data["gt_labels"] = data_dict["gt_labels"]

        return ret_data

    def __getitem__(self, idx):
        if self.use_cbgs:
            idx = self.sample_indices[idx]
        data = super(NuscenesMultiModalData, self).__getitem__(idx)
        data_dict = self._generate_data_dict(data)
        data_dict = self.data_processor.forward(data_dict)
        data_dict = self.det_augmentor.forward(data_dict)
        data_dict = self.format_processor(data_dict, data)
        if self.is_train and self.filter_empty and data_dict["gt_boxes"].shape[0] == 0:
            new_idx = np.random.choice(len(self))
            return self.__getitem__(new_idx)

        return data_dict

    @staticmethod
    def generate_prediction_dicts(
        batch_dict, pred_dicts, class_names, output_path=None
    ):
        def get_template_prediction(num_samples):
            ret_dict = {
                "name": np.zeros(num_samples),
                "score": np.zeros(num_samples),
                "boxes_3d": np.zeros((num_samples, 7)),
            }
            return ret_dict

        def generate_single_sample_dict(box_dict):
            pred_scores = box_dict["pred_scores"].cpu().numpy()
            pred_boxes = box_dict["pred_boxes"].cpu().numpy()
            pred_labels = box_dict["pred_labels"].cpu().numpy()
            pred_dict = get_template_prediction(pred_scores.shape[0])
            if pred_scores.shape[0] == 0:
                return pred_dict

            pred_dict["name"] = np.array(class_names)[pred_labels]
            pred_dict["score"] = pred_scores
            pred_dict["boxes_3d"] = pred_boxes
            return pred_dict

        annos = []
        for index, box_dict in enumerate(pred_dicts):
            single_pred_dict = generate_single_sample_dict(box_dict)
            single_pred_dict["frame_id"] = -1  # Currently set up to -1 as CenterPoint
            annos.append(single_pred_dict)

            if output_path is not None:
                raise NotImplementedError
        return annos

    def evaluation(self, det_annos, class_names, **kwargs):
        from .evaluate import generate_submission_results, get_evaluation_results

        output_dir = os.path.join(str(kwargs["output_dir"]), "nuscenes")
        for frame in self.infos:
            frame["token"] = frame["sample_token"]
        generate_submission_results(
            meta_info=self.meta_info,
            gt=self.infos,
            dt=det_annos,
            result_dir=output_dir,
            meta_type_list=["use_camera", "use_lidar"],
        )

        ap_dict = get_evaluation_results(
            nusc_meta_info=self.meta_info,
            result_path=os.path.join(output_dir, "nuscenes_results.json"),
            output_dir=output_dir,
            eval_set="val",
            verbose=False,
            plot_examples=0,
            render_curves=False,
        )
        return ap_dict, None

    def dump_inference_results(self, det_annos, **kwargs):
        from .evaluate import generate_submission_results

        output_dir = os.path.join(str(kwargs["output_dir"]), "nuscenes_submission")
        for frame in self.infos:
            frame["token"] = frame["sample_token"]
        with open("/data/dataset/nuscenes_test_meta.pkl", "rb") as f:
            meta_info = pickle.load(f)
        dump_path = os.path.join(output_dir, "boxes.pkl")
        if not os.path.isdir(output_dir):
            os.makedirs(output_dir)
        with open(dump_path, "wb") as f:
            pickle.dump(det_annos, f)

        generate_submission_results(
            meta_info=meta_info,
            gt=self.infos,
            dt=det_annos,
            result_dir=output_dir,
            meta_type_list=["use_camera", "use_lidar"],
        )


def collate_fn(data, is_return_depth=False, with_points=True):
    """
    Dataset output:
        "imgs": image data, (num_cam_sweeps, num_cams, 3, h, w) array
        "points": point cloud data, (num_points, lidar_dims) array
        "mats_dict": dict of translation matrixs
            "sensor2ego_mats": (num_cam_sweeps, num_cams, 4, 4) array
            "intrin_mats": (num_cam_sweeps, num_cams, 4, 4) array
            "ida_mats": (num_cam_sweeps, num_cams, 4, 4) array
            "sensor2sensor_mats": (num_cam_sweeps, num_cams, 4, 4) array
            "bda_mat": (4, 4) array
        "img_metas": dict of box type and frame token, for validation
            "box_type_3d": class 'mmdet3d.core.bbox.structures.lidar_box3d.LiDARInstance3DBoxes'
            "token": str, token of nuScenes
            "ego2global_translation": (3) array
            "ego2global_rotation": (4) array (quaternion angle)
        "gt_boxes": (num_boxes, box_dims) array, (x,y,z,l,w,h,angle) or (x,y,z,l,w,h,angle,vx,vy)
        "gt_labels": (num_boxes) array, class id as int type
        "depth_labels": optional, for bevdepth training

    In this step, collect all array in torch tensor data batch to an extra batch size dimension
    """

    def fill_batch_tensor(batch_data: list):
        if max([len(x) for x in batch_data]) == min([len(x) for x in batch_data]):
            return torch.stack(
                [
                    data if isinstance(data, torch.Tensor) else torch.tensor(data)
                    for data in batch_data
                ]
            ).to(torch.float32)
        else:
            batch_size = len(batch_data)
            batch_length = max([len(x) for x in batch_data])
            for data in batch_data:
                if data.size != 0:
                    data_shape = data.shape
                    break
            batch_data_shape = (batch_size, batch_length, *data_shape[1:])
            batch_tensor = torch.zeros(batch_data_shape)
            for i, data in enumerate(batch_data):
                if data.size != 0:
                    batch_tensor[i, : len(data)] = (
                        data if isinstance(data, torch.Tensor) else torch.tensor(data)
                    )
            return batch_tensor.to(torch.float32)

    data_keys = ["imgs", "points", "gt_boxes", "gt_labels"]
    batch_collection = dict()
    for key in data_keys:
        if key in data[0]:
            data_list = [iter_data[key] for iter_data in data]
            batch_collection[key] = fill_batch_tensor(data_list)

    mats_keys = [
        "sensor2ego_mats",
        "intrin_mats",
        "ida_mats",
        "sensor2sensor_mats",
        "bda_mat",
    ]
    if "mats_dict" in data[0]:
        batch_collection["mats_dict"] = dict()
        for key in mats_keys:
            if key in data[0]["mats_dict"]:
                data_list = [
                    iter_data["mats_dict"][key]
                    if isinstance(iter_data["mats_dict"][key], torch.Tensor)
                    else torch.tensor(iter_data["mats_dict"][key])
                    for iter_data in data
                ]
                batch_collection["mats_dict"][key] = torch.stack(data_list).to(
                    torch.float32
                )

    batch_collection["img_metas"] = [iter_data["img_metas"] for iter_data in data]

    return batch_collection
