import json
import math
import os
import pickle
from typing import Dict, List

import numpy as np
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import Box
from pyquaternion import Quaternion

DefaultAttribute = {
    "car": "vehicle.parked",
    "pedestrian": "pedestrian.moving",
    "trailer": "vehicle.parked",
    "truck": "vehicle.parked",
    "bus": "vehicle.moving",
    "motorcycle": "cycle.without_rider",
    "construction_vehicle": "vehicle.parked",
    "bicycle": "cycle.without_rider",
    "barrier": "",
    "traffic_cone": "",
}


def dump_pkl(obj, file):
    with open(file, "wb") as f:
        pickle.dump(obj, f)


def load_pkl(file):
    with open(file, "rb") as f:
        return pickle.load(f)


def load_json(file):
    with open(file, "r") as f:
        return json.loads(f.read())


def dump_json(obj, file):
    with open(file, "w") as f:
        f.write(json.dumps(obj))


def quick_load_pkl(file):
    return load_pkl(file)


def box3d_to_nuscenesbox(
    meta_info: NuScenes,
    box_3d: List[float],
    token: str,
    box_item: Dict,
) -> Dict:
    nusc = meta_info
    translation = box_3d[:3]
    size = np.array(box_3d[3:6])[[1, 0, 2]].tolist()
    rot = box_3d[6]
    if len(box_3d) == 9:
        velocity = tuple(box_3d[7:9] + [0])
    else:
        velocity = (np.nan, np.nan, np.nan)
    nuscenesbox = Box(
        center=translation,
        size=size,
        orientation=Quaternion(math.cos(rot / 2), 0, 0, math.sin(rot / 2)),
        velocity=velocity,
    )
    sample = nusc.get("sample", token)
    ref_chan = "LIDAR_TOP"
    sample_data_token = sample["data"][ref_chan]

    sd_record = nusc.get("sample_data", sample_data_token)
    cs_record = nusc.get("calibrated_sensor", sd_record["calibrated_sensor_token"])
    pose_record = nusc.get("ego_pose", sd_record["ego_pose_token"])

    nuscenesbox.rotate(Quaternion(cs_record["rotation"]))
    nuscenesbox.translate(np.array(cs_record["translation"]))
    nuscenesbox.rotate(Quaternion(pose_record["rotation"]))
    nuscenesbox.translate(np.array(pose_record["translation"]))
    name = box_item["detection_name"]
    if np.sqrt(nuscenesbox.velocity[0] ** 2 + nuscenesbox.velocity[1] ** 2) > 0.2:
        if name in [
            "car",
            "construction_vehicle",
            "bus",
            "truck",
            "trailer",
        ]:
            attr = "vehicle.moving"
        elif name in ["bicycle", "motorcycle"]:
            attr = "cycle.with_rider"
        else:
            attr = DefaultAttribute[name]
    else:
        if name in ["pedestrian"]:
            attr = "pedestrian.standing"
        elif name in ["bus"]:
            attr = "vehicle.stopped"
        else:
            attr = DefaultAttribute[name]
    box_item.update(
        {
            "translation": nuscenesbox.center.tolist(),
            "size": nuscenesbox.wlh.tolist(),
            "rotation": list(nuscenesbox.orientation),
            "velocity": nuscenesbox.velocity.tolist()[:2],
            "attribute_name": attr,
        }
    )
    return box_item
