import json
import math
import os
from multiprocessing import Process, Queue
from typing import Dict, List

from nuscenes.eval.common.config import config_factory
from nuscenes.eval.detection.data_classes import DetectionConfig
from nuscenes.eval.detection.evaluate import DetectionEval
from nuscenes.nuscenes import NuScenes
from tqdm import tqdm

from .eval_utils import box3d_to_nuscenesbox, dump_json


def generate_submission_results(
    meta_info: NuScenes,
    gt: Dict,
    dt: Dict,
    result_dir: str,
    meta_type_list: List = ["use_lidar"],
    num_workers: int = 16,
) -> Dict:
    def worker(split_records, meta_info, result_queue):
        gt, dt = split_records
        for i in range(len(dt)):
            token = gt[i]["token"]
            names, scores, boxes_3d = dt[i]["name"], dt[i]["score"], dt[i]["boxes_3d"]
            assert len(names) == len(scores) == len(boxes_3d)
            num_dt = len(boxes_3d)
            dt_boxes = []
            for box_id in range(num_dt):
                box_item = {
                    "sample_token": token,
                    "detection_name": str(names[box_id]),
                    "detection_score": float(scores[box_id]),
                }
                box_item = box3d_to_nuscenesbox(
                    meta_info, boxes_3d[box_id].tolist(), token, box_item
                )
                dt_boxes.append(box_item)
            result_queue.put({token: dt_boxes})

    nr_records = len(dt)
    pbar = tqdm(total=nr_records)

    nr_split = math.ceil(nr_records / num_workers)
    result_queue = Queue(10000)
    procs = []
    dt_res_json = {}

    print("Generating submission results...")
    for i in range(num_workers):
        start = i * nr_split
        end = min(start + nr_split, nr_records)
        split_records = (gt[start:end], dt[start:end])
        proc = Process(target=worker, args=(split_records, meta_info, result_queue))
        print("process:%d, start:%d, end:%d" % (i, start, end))
        proc.start()
        procs.append(proc)

    for i in range(nr_records):
        dt_res_json.update(result_queue.get())
        pbar.update(1)

    for p in procs:
        p.join()

    submit_json = {
        "meta": {
            "use_camera": "use_camera" in meta_type_list,
            "use_lidar": "use_lidar" in meta_type_list,
            "use_radar": "use_radar" in meta_type_list,
            "use_map": "use_map" in meta_type_list,
            "use_external": "use_external" in meta_type_list,
        },
        "results": dt_res_json,
    }

    if not os.path.exists(result_dir):
        os.makedirs(result_dir)

    dump_json(submit_json, os.path.join(result_dir, "nuscenes_results.json"))
    return submit_json


def get_evaluation_results(
    nusc_meta_info: NuScenes,
    result_path: str,
    output_dir: str,
    config_path: str = "",
    eval_set: str = "val",
    verbose: bool = False,
    plot_examples: int = 0,
    render_curves: bool = False,
    **kwargs
) -> Dict:
    if config_path == "":
        cfg = config_factory("detection_cvpr_2019")
    else:
        with open(config_path, "r") as _f:
            cfg = DetectionConfig.deserialize(json.load(_f))

    print("Loading Nuscenes ground truths...")
    nusc_eval = DetectionEval(
        nusc_meta_info,
        config=cfg,
        result_path=result_path,
        eval_set=eval_set,
        output_dir=output_dir,
        verbose=verbose,
    )

    print("Evaluation starts...")
    eval_res = nusc_eval.main(plot_examples=plot_examples, render_curves=render_curves)

    return eval_res
