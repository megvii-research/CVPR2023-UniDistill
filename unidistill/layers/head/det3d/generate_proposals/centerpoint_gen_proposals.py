import torch
import torch.nn as nn

from . import iou3d_nms_cuda
from .base_gen_proposals import BaseGenProposals


class CenterPointGenProposals(BaseGenProposals):
    def __init__(
        self,
        dataset_name,
        class_names,
        post_center_limit_range,
        score_threshold,
        pc_range,
        out_size_factor,
        voxel_size,
        no_log,
        nms_iou_threshold_train,
        nms_pre_max_size_train,
        nms_post_max_size_train,
        nms_iou_threshold_test,
        nms_pre_max_size_test,
        nms_post_max_size_test,
    ):
        super(CenterPointGenProposals, self).__init__()
        self.dataset_name = dataset_name
        self.class_names = class_names
        self.post_center_limit_range = post_center_limit_range
        self.score_threshold = score_threshold
        self.pc_range = pc_range
        self.out_size_factor = out_size_factor
        self.voxel_size = voxel_size
        self.no_log = no_log
        self.nms_iou_threshold_train = nms_iou_threshold_train
        self.nms_pre_max_size_train = nms_pre_max_size_train
        self.nms_post_max_size_train = nms_post_max_size_train
        self.nms_iou_threshold_test = nms_iou_threshold_test
        self.nms_pre_max_size_test = nms_pre_max_size_test
        self.nms_post_max_size_test = nms_post_max_size_test
        self.training = True

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

    def _nms(self, heat, kernel=3):
        pad = (kernel - 1) // 2

        hmax = nn.functional.max_pool2d(heat, (kernel, kernel), stride=1, padding=pad)
        keep = (hmax == heat).float()
        return heat * keep

    def _topk(self, scores, K=40):
        batch, cat, height, width = scores.size()

        topk_scores, topk_inds = torch.topk(scores.view(batch, cat, -1), K)

        topk_inds = topk_inds % (height * width)
        topk_ys = (topk_inds / width).int().float()
        topk_xs = (topk_inds % width).int().float()

        topk_score, topk_ind = torch.topk(topk_scores.view(batch, -1), K)
        topk_clses = (topk_ind / K).int()
        topk_inds = self._gather_feat(topk_inds.view(batch, -1, 1), topk_ind).view(
            batch, K
        )
        topk_ys = self._gather_feat(topk_ys.view(batch, -1, 1), topk_ind).view(batch, K)
        topk_xs = self._gather_feat(topk_xs.view(batch, -1, 1), topk_ind).view(batch, K)

        return topk_score, topk_inds, topk_clses, topk_ys, topk_xs

    def _nms_gpu_3d(self, boxes, scores, thresh, pre_maxsize=None, post_max_size=None):
        """
        :param boxes: (N, 7) [x, y, z, dx, dy, dz, heading]
        :param scores: (N)
        :param thresh:
        :return:
        """
        assert boxes.shape[1] == 7
        order = scores.sort(0, descending=True)[1]
        if pre_maxsize is not None:
            order = order[:pre_maxsize]

        boxes = boxes[order].contiguous()
        keep = torch.LongTensor(boxes.size(0))
        num_out = iou3d_nms_cuda.nms_gpu(boxes, keep, thresh)
        selected = order[keep[:num_out].cuda()].contiguous()

        if post_max_size is not None:
            selected = selected[:post_max_size]

        return selected

    def _boxes3d_to_bevboxes_lidar_torch(self, boxes3d):
        """
        :param boxes3d: (N, 7) [x, y, z, w, l, h, ry] in LiDAR coords
        :return:
            boxes_bev: (N, 5) [x1, y1, x2, y2, ry]
        """
        boxes_bev = boxes3d.new(torch.Size((boxes3d.shape[0], 5)))

        cu, cv = boxes3d[:, 0], boxes3d[:, 1]

        half_w, half_l = boxes3d[:, 3] / 2, boxes3d[:, 4] / 2
        boxes_bev[:, 0], boxes_bev[:, 1] = cu - half_w, cv - half_l
        boxes_bev[:, 2], boxes_bev[:, 3] = cu + half_w, cv + half_l
        boxes_bev[:, 4] = boxes3d[:, -1]
        return boxes_bev

    def get_nms_scores(self, scores):
        return scores

    def nms_options(self, boxes3d, labels, scores, nms_scores):
        # use_iou_3d_nms
        top_scores = nms_scores
        if top_scores.shape[0] != 0:
            selected = self._nms_gpu_3d(
                boxes3d[:, :7],
                top_scores,
                thresh=self.nms_iou_threshold_use,
                pre_maxsize=self.nms_pre_max_size_use,
                post_max_size=self.nms_post_max_size_use,
            )
        else:
            selected = []
        boxes3d = boxes3d[selected]
        labels = labels[selected]
        scores = scores[selected]
        return boxes3d, labels, scores

    @torch.no_grad()
    def proposal_layer(
        self,
        heat,
        rots,
        rotc,
        hei,
        dim,
        vel,
        reg=None,
        raw_rot=False,
        task_id=-1,
    ):
        batch, cat, _, _ = heat.size()
        K = self.nms_pre_max_size_use  # topK selected

        scores, inds, clses, ys, xs = self._topk(heat, K=K)
        nms_scores = scores
        assert reg is not None
        reg = self._transpose_and_gather_feat(reg, inds)
        reg = reg.view(batch, K, 2)
        xs = xs.view(batch, K, 1) + reg[:, :, 0:1]
        ys = ys.view(batch, K, 1) + reg[:, :, 1:2]
        assert raw_rot is False
        rots = self._transpose_and_gather_feat(rots, inds)
        rots = rots.view(batch, K, 1)
        rotc = self._transpose_and_gather_feat(rotc, inds)
        rotc = rotc.view(batch, K, 1)
        rot = torch.atan2(rots, rotc)
        # height in the bev
        hei = self._transpose_and_gather_feat(hei, inds)
        hei = hei.view(batch, K, 1)
        # dim of the box
        dim = self._transpose_and_gather_feat(dim, inds)
        dim = dim.view(batch, K, 3)
        # class label
        clses = clses.view(batch, K).float()
        scores = scores.view(batch, K)
        # center location
        pc_range = self.pc_range
        xs = (
            xs.view(batch, K, 1) * self.out_size_factor * self.voxel_size[0]
            + pc_range[0]
        )
        ys = (
            ys.view(batch, K, 1) * self.out_size_factor * self.voxel_size[1]
            + pc_range[1]
        )

        if self.dataset_name == "nuscenes":
            vel = self._transpose_and_gather_feat(vel, inds)
            vel = vel.view(batch, K, 2)
            # vel after rot
            final_box_preds = torch.cat([xs, ys, hei, dim, rot, vel], dim=2)
        else:
            final_box_preds = torch.cat([xs, ys, hei, dim, rot], dim=2)

        final_scores = scores
        final_preds = clses

        # restrict center range
        post_center_range = self.post_center_limit_range
        assert post_center_range is not None
        post_center_range = torch.tensor(post_center_range).to(final_box_preds.device)
        mask1 = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask2 = (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
        # use score threshold
        assert self.score_threshold is not None
        thresh_mask = final_scores > self.score_threshold
        mask = torch.logical_and(mask1, mask2)
        mask = torch.logical_and(mask, thresh_mask)

        predictions_dicts = []
        for i in range(batch):
            cmask = mask[i, :]
            boxes3d = final_box_preds[i, cmask]
            scores = final_scores[i, cmask]
            labels = final_preds[i, cmask]
            nms_score = nms_scores[i, cmask]

            boxes3d, labels, scores = self.nms_options(
                boxes3d, labels, scores, nms_score
            )

            predictions_dict = {
                "boxes": boxes3d,
                "scores": scores,
                "labels": labels.long(),
            }
            predictions_dicts.append(predictions_dict)
        return predictions_dicts

    @torch.no_grad()
    def generate_predicted_boxes(self, forward_ret_dict, data_dict):
        """
        Generate box predictions with decode, topk and circular_nms
        For single-stage-detector, another post-processing (nms) is needed
        For two-stage-detector, no need for proposal layer in roi_head
        Returns:
        """
        pred_dicts = forward_ret_dict["multi_head_features"]

        if self.training:
            self.nms_iou_threshold_use = self.nms_iou_threshold_train
            self.nms_pre_max_size_use = self.nms_pre_max_size_train
            self.nms_post_max_size_use = self.nms_post_max_size_train
        else:
            self.nms_iou_threshold_use = self.nms_iou_threshold_test
            self.nms_pre_max_size_use = self.nms_pre_max_size_test
            self.nms_post_max_size_use = self.nms_post_max_size_test

        task_box_preds = {}
        task_score_preds = {}
        task_label_preds = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            batch_size = pred_dict["hm"].shape[0]

            # batch_hm = pred_dict['hm'].sigmoid_() inplace may cause errors
            batch_hm = pred_dict["hm"].sigmoid()
            batch_reg = pred_dict["reg"]
            batch_hei = pred_dict["height"]

            if not self.no_log:
                batch_dim = torch.exp(pred_dict["dim"])
                # add clamp for good init, otherwise we will get inf with exp
                batch_dim = torch.clamp(batch_dim, min=0.001, max=30)
            else:
                batch_dim = pred_dict["dim"]
            batch_rots = pred_dict["rot"][:, 0].unsqueeze(1)
            batch_rotc = pred_dict["rot"][:, 1].unsqueeze(1)

            if self.dataset_name == "nuscenes":
                batch_vel = pred_dict["vel"]
            else:
                batch_vel = None

            # decode
            boxes = self.proposal_layer(
                batch_hm,
                batch_rots,
                batch_rotc,
                batch_hei,
                batch_dim,
                batch_vel,
                reg=batch_reg,
                task_id=task_id,
            )
            task_box_preds[task_id] = [box["boxes"] for box in boxes]
            task_score_preds[task_id] = [box["scores"] for box in boxes]
            task_label_preds[task_id] = [
                box["labels"] for box in boxes
            ]  # labels are local here

        pred_dicts = []
        batch_size = len(task_box_preds[0])
        rois, roi_scores, roi_labels = [], [], []
        num_rois = self.nms_post_max_size_use * len(self.class_names)
        for batch_idx in range(batch_size):
            offset = 1  # class label start from 1
            final_boxes, final_scores, final_labels = [], [], []
            for task_id, class_name in enumerate(self.class_names):
                final_boxes.append(task_box_preds[task_id][batch_idx])
                final_scores.append(task_score_preds[task_id][batch_idx])
                # convert to global labels
                final_global_label = task_label_preds[task_id][batch_idx] + offset
                offset += len(class_name)
                final_labels.append(final_global_label)

            final_boxes = torch.cat(final_boxes)
            final_scores = torch.cat(final_scores)
            final_labels = torch.cat(final_labels)

            roi = final_boxes.new_zeros(num_rois, final_boxes.shape[-1])
            roi_score = final_scores.new_zeros(num_rois)
            roi_label = final_labels.new_zeros(num_rois)
            num_boxes = final_boxes.shape[0]
            roi[:num_boxes] = final_boxes
            roi_score[:num_boxes] = final_scores
            roi_label[:num_boxes] = final_labels
            rois.append(roi)
            roi_scores.append(roi_score)
            roi_labels.append(roi_label)

            record_dict = {
                "pred_boxes": final_boxes,
                "pred_scores": final_scores,
                "pred_labels": final_labels,
            }
            pred_dicts.append(record_dict)

        data_dict["pred_dicts"] = pred_dicts
        data_dict["rois"] = torch.stack(rois, dim=0)
        data_dict["roi_scores"] = torch.stack(roi_scores, dim=0)
        data_dict["roi_labels"] = torch.stack(roi_labels, dim=0)
        data_dict["has_class_labels"] = True  # Force to be true
        data_dict.pop("batch_index", None)
        return data_dict
