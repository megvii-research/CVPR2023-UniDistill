import torch

from unidistill.layers.head.det3d.generate_proposals import CenterPointGenProposals


class IouAwareGenProposals(CenterPointGenProposals):
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
        iou_aware_list,
    ):
        super().__init__(
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
        )
        self.iou_aware_list = iou_aware_list

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
        iouhm=None,
    ):
        batch, cat, _, _ = heat.size()
        K = self.nms_pre_max_size_use  # topK selected

        scores, inds, clses, ys, xs = self._topk(heat, K=K)

        iouscore = self._transpose_and_gather_feat(iouhm, inds).view(batch, K)
        iouscore = torch.clamp(iouscore / 2 + 0.5, 0, 1)
        nms_scores = (scores ** (1 - self.iou_aware_list[task_id])).mul(
            iouscore ** self.iou_aware_list[task_id]
        )

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
        mask = (final_box_preds[..., :3] >= post_center_range[:3]).all(2)
        mask &= (final_box_preds[..., :3] <= post_center_range[3:]).all(2)
        # use score threshold
        assert self.score_threshold is not None
        thresh_mask = final_scores > self.score_threshold
        mask &= thresh_mask

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
            batch_iousocre = pred_dict["iou"]

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
                iouhm=batch_iousocre,
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
