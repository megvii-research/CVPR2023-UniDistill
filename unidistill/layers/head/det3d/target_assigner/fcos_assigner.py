import math

import numpy as np
import torch

from .base_assigner import BaseAssigner


class FCOSAssigner(BaseAssigner):
    def __init__(
        self,
        out_size_factor,
        tasks,
        dense_reg,
        gaussian_overlap,
        max_objs,
        min_radius,
        mapping,
        grid_size,
        pc_range,
        voxel_size,
        assign_topk,
        no_log=False,
        with_velocity=False,
    ):
        super(FCOSAssigner, self).__init__()
        self.out_size_factor = out_size_factor
        self.tasks = tasks
        self.num_classes = sum([len(t["class_names"]) for t in self.tasks])
        self.dense_reg = dense_reg
        self.gaussian_overlap = gaussian_overlap
        self._max_objs = max_objs
        self._min_radius = min_radius
        self.class_to_idx = mapping
        self.grid_size = np.array(grid_size)
        self.pc_range = pc_range
        self.voxel_size = voxel_size
        self.no_log = no_log
        self.assign_topk = assign_topk
        self.with_velocity = with_velocity
        if self.with_velocity:
            self.default_box_dims = 10
        else:
            self.default_box_dims = 8

    def generate_anchor_grid(self, featmap_size, offsets, stride, device):
        step_x, step_y = featmap_size
        shift = offsets * stride
        grid_x = torch.linspace(
            shift, (step_x - 1) * stride + shift, steps=step_x, device=device
        )
        grid_y = torch.linspace(
            shift, (step_y - 1) * stride + shift, steps=step_y, device=device
        )
        grids_x, grids_y = torch.meshgrid(grid_y, grid_x)
        return grids_x.reshape(-1), grids_y.reshape(-1)

    def limit_period(self, val, offset=0.5, period=math.pi):
        return val - torch.floor(val / period + offset) * period

    def padding_to_maxobjs(self, tensor, max_objs):
        assert tensor.shape[0] < max_objs, "the max_objs is {} but get {}".format(
            max_objs, tensor.shape
        )
        if len(tensor.shape) == 2:
            padding = torch.zeros(max_objs - tensor.shape[0], tensor.shape[1]).to(
                tensor.device
            )
        elif len(tensor.shape) == 1:
            padding = torch.zeros(max_objs - tensor.shape[0]).to(tensor.device)
        return torch.cat([tensor, padding], dim=0)

    def assign_targets(self, gt_boxes):
        """
        Args:
            gt_boxes: (B, M, C + cls)

        Returns:

        """
        max_objs = self._max_objs * self.dense_reg
        feature_map_size = (
            self.grid_size[:2] // self.out_size_factor
        )  # grid_size WxHxD feature_map_size WxH

        grids_x, grids_y = self.generate_anchor_grid(
            feature_map_size, 0.0, self.out_size_factor, gt_boxes.device
        )
        anchor_points = torch.cat(
            [grids_y.unsqueeze(1), grids_x.unsqueeze(1)], dim=1
        )  # WxH, 2

        batch_size = gt_boxes.shape[0]
        gt_classes = gt_boxes[:, :, -1]  # begin from 1
        gt_boxes = gt_boxes[:, :, :-1]

        heatmaps = {}
        gt_inds = {}
        gt_masks = {}
        gt_box_encodings = {}
        gt_cats = {}
        for task_id, task in enumerate(self.tasks):
            heatmaps[task_id] = []
            gt_inds[task_id] = []
            gt_masks[task_id] = []
            gt_box_encodings[task_id] = []
            gt_cats[task_id] = []

        for k in range(batch_size):
            cur_gt = gt_boxes[k]
            cnt = cur_gt.__len__() - 1
            while cnt > 0 and cur_gt[cnt].sum() == 0:
                cnt -= 1
            cur_gt = cur_gt[: cnt + 1]
            cur_gt_classes = gt_classes[k][: cnt + 1].int()

            for task_id, task in enumerate(self.tasks):
                heatmap = torch.zeros(
                    (len(task.class_names), feature_map_size[1], feature_map_size[0]),
                    dtype=torch.float32,
                ).to(cur_gt.device)
                gt_ind = torch.zeros(max_objs, dtype=torch.long).to(cur_gt.device)
                gt_mask = torch.zeros(max_objs, dtype=torch.bool).to(cur_gt.device)
                gt_cat = torch.zeros(max_objs, dtype=torch.long).to(cur_gt.device)
                gt_box_encoding = torch.zeros(
                    (max_objs, self.default_box_dims), dtype=torch.float32
                ).to(cur_gt.device)

                cur_gts_of_task = []
                cur_classes_of_task = []
                class_offset = 0
                for class_name in task.class_names:
                    class_idx = self.class_to_idx[class_name]
                    class_mask = cur_gt_classes == class_idx
                    cur_gt_of_task = cur_gt[class_mask]
                    cur_class_of_task = cur_gt.new_full(
                        (cur_gt_of_task.shape[0],), class_offset
                    ).long()
                    cur_gts_of_task.append(cur_gt_of_task)
                    cur_classes_of_task.append(cur_class_of_task)
                    class_offset += 1

                cur_gts_of_task = torch.cat(cur_gts_of_task, dim=0)  # GT, 7
                cur_classes_of_task = torch.cat(cur_classes_of_task, dim=0)  # GT
                # num_boxes_of_task = cur_gts_of_task.shape[0]
                if len(cur_classes_of_task) == 0:
                    heatmaps[task_id].append(heatmap)
                    gt_inds[task_id].append(gt_ind)
                    gt_cats[task_id].append(gt_cat)
                    gt_masks[task_id].append(gt_mask)
                    gt_box_encodings[task_id].append(gt_box_encoding)
                    continue

                cur_gts_of_task[:, 0] = (
                    cur_gts_of_task[:, 0] - self.pc_range[0]
                ) / self.voxel_size[0]
                cur_gts_of_task[:, 1] = (
                    cur_gts_of_task[:, 1] - self.pc_range[1]
                ) / self.voxel_size[1]
                cur_gts_of_task[:, 3] = cur_gts_of_task[:, 3] / self.voxel_size[0]
                cur_gts_of_task[:, 4] = cur_gts_of_task[:, 4] / self.voxel_size[1]
                cur_gts_of_task[:, 6] = self.limit_period(
                    cur_gts_of_task[:, 6], offset=0.5, period=math.pi * 2
                )

                # x1y1 = cur_gts_of_task[:, 0:2] - cur_gts_of_task[:, 3:5] / 2
                # x2y2 = cur_gts_of_task[:, 0:2] + cur_gts_of_task[:, 3:5] / 2

                # 1. 在gt框内容的anchor point作为正样本（小目标掉点，故去掉）
                # ltrb = \
                #     torch.cat([
                #         anchor_points.unsqueeze(1) - x1y1.unsqueeze(0),
                #         x2y2.unsqueeze(0) - anchor_points.unsqueeze(1)
                #         ], dim=2)   # (ANC, GT, 4)

                # in_gt_matrix = ltrb.min(dim=2)[0] >= 0    # (ANC, GT)
                # in_gt_mask = in_gt_matrix.max(dim=1)[1] > 0    # ANC,

                # 2. 与gt中心点l2距离最近的k个anchor point作为正样本
                topk = self.assign_topk
                center_offsets = torch.pow(
                    anchor_points.unsqueeze(1) - cur_gts_of_task[:, :2].unsqueeze(0), 2
                ).sum(
                    dim=2
                )  # (ANC, GT)
                _, topk_inds = torch.topk(
                    center_offsets.t(), topk, largest=False
                )  # (GT, topk)
                # in_topk_mask = torch.zeros_like(in_gt_mask) # ANC,
                in_topk_mask = torch.zeros(anchor_points.shape[0]).to(
                    anchor_points.device
                )  # ANC,
                in_topk_mask[topk_inds.flatten()] = 1

                # pos_mask = (in_gt_mask * in_topk_mask).bool()   # ANC,
                pos_mask = (in_topk_mask).bool()  # ANC,

                # 3. 强制匹配top1，有topk保证其实就不需要了
                # _, top1_inds = torch.topk(center_offsets.t(), 1, largest=False)  # (GT, 1)
                # pos_mask[top1_inds.flatten()] = True

                # 4. 每个anchor point assign给最近的gt
                _, gt_ids = center_offsets.min(dim=1)  # ANC,
                pos_gt_ids = gt_ids[pos_mask]  # POS,

                # 5. 生成targets
                gt_cat = cur_classes_of_task[pos_gt_ids]  # POS,
                gt_mask = torch.ones_like(pos_gt_ids)  # POS,
                gt_ind = torch.where(pos_mask == 1)[0]  # POS,

                heatmap = torch.zeros(
                    anchor_points.shape[0], len(task.class_names), device=gt_cat.device
                )  # （ANC, num_class)
                heatmap[pos_mask] = heatmap[pos_mask].scatter_(
                    1, gt_cat.unsqueeze(1), 1
                )
                heatmap = heatmap.transpose(1, 0).reshape(
                    len(task.class_names), feature_map_size[1], feature_map_size[0]
                )

                loc_targets = cur_gts_of_task[pos_gt_ids]  # POS, 7
                pos_anchor_points = anchor_points[pos_mask]  # POS, 2
                gt_box_encoding = torch.cat(
                    [
                        (loc_targets[:, 0:2] - pos_anchor_points)
                        / self.out_size_factor,  # x, y
                        loc_targets[:, 2:3],  # z
                        torch.log(loc_targets[:, 3:4] * self.voxel_size[0]),
                        torch.log(loc_targets[:, 4:5] * self.voxel_size[1]),
                        torch.log(loc_targets[:, 5:6]),
                        torch.sin(loc_targets[:, 6:7]),
                        torch.cos(loc_targets[:, 6:7]),
                        loc_targets[:, 7:],  # velocity
                    ],
                    dim=1,
                ).to(
                    heatmap.device
                )  # POS, 7

                # padding to fixed shape
                gt_cat = self.padding_to_maxobjs(gt_cat, max_objs).long()
                gt_mask = self.padding_to_maxobjs(gt_mask, max_objs).bool()
                gt_ind = self.padding_to_maxobjs(gt_ind, max_objs).long()
                gt_box_encoding = self.padding_to_maxobjs(
                    gt_box_encoding, max_objs
                ).float()

                # 可视化label assign
                # img = np.zeros((1504, 1504, 3)).astype(np.int8)
                # for x, y in pos_anchor_points:
                #     cv2.circle(img, (int(x), int(y)), 2, (0, 0, 255), 2)
                # gtboxes = torch.cat([x1y1, x2y2], dim=1).int()
                # for box in gtboxes:
                #     x1, y1, x2, y2 = box
                #     cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

                # for x, y in cur_gts_of_task[:, :2]:
                #     cv2.circle(img, (int(x), int(y)), 2, (255, 0, 255), 2)

                # cv2.imwrite(f'./tmp_{k}.jpg', img)
                # from IPython import embed; embed()

                heatmaps[task_id].append(heatmap)
                gt_inds[task_id].append(gt_ind)
                gt_cats[task_id].append(gt_cat)
                gt_masks[task_id].append(gt_mask)
                gt_box_encodings[task_id].append(gt_box_encoding)

        for task_id, tasks in enumerate(self.tasks):
            heatmaps[task_id] = torch.stack(heatmaps[task_id], dim=0).contiguous()
            gt_inds[task_id] = torch.stack(gt_inds[task_id], dim=0).contiguous()
            gt_masks[task_id] = torch.stack(gt_masks[task_id], dim=0).contiguous()
            gt_cats[task_id] = torch.stack(gt_cats[task_id], dim=0).contiguous()
            gt_box_encodings[task_id] = torch.stack(
                gt_box_encodings[task_id], dim=0
            ).contiguous()

        target_dict = {
            "heatmap": heatmaps,
            "ind": gt_inds,
            "mask": gt_masks,
            "cat": gt_cats,
            "box_encoding": gt_box_encodings,
        }
        return target_dict
