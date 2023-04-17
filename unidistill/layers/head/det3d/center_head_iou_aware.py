import torch

from unidistill.layers.head.det3d import CenterHead
from unidistill.layers.losses.det3d import (
    AutomaticWeightedLoss,
    _transpose_and_gather_feat,
)
from unidistill.utils.det3d_utils import box_utils
from unidistill.utils.torch_dist import reduce_mean


class CenterHeadIouAware(CenterHead):
    def __init__(
        self,
        dataset_name,
        tasks,
        target_assigner,
        proposal_layer,
        out_size_factor,
        input_channels,
        grid_size,
        point_cloud_range,
        code_weights,
        loc_weight,
        iou_weight,
        share_conv_channel,
        common_heads,
        upsample_for_pedestrian=False,
        predict_boxes_when_training=False,
        mode="3d",
        init_bias=-2.19,
    ):
        super().__init__(
            dataset_name,
            tasks,
            target_assigner,
            proposal_layer,
            input_channels,
            grid_size,
            point_cloud_range,
            code_weights,
            loc_weight,
            share_conv_channel,
            common_heads,
            upsample_for_pedestrian=upsample_for_pedestrian,
            predict_boxes_when_training=predict_boxes_when_training,
            mode=mode,
            init_bias=init_bias,
        )
        num_loss = len(self.code_weights) + 2
        self.auto_loss = AutomaticWeightedLoss(num=num_loss)
        self.iou_weight = iou_weight
        self.out_size_factor = out_size_factor

    def get_loss(self, forward_ret_dict):
        tb_dict = {}
        pred_dicts = forward_ret_dict["multi_head_features"]
        center_loss = []
        forward_ret_dict["pred_box_encoding"] = {}
        for task_id, pred_dict in enumerate(pred_dicts):
            pred_dict["hm"] = self._sigmoid(pred_dict["hm"])
            hm_loss = self.crit(pred_dict["hm"], forward_ret_dict["heatmap"][task_id])

            target_box_encoding = forward_ret_dict["box_encoding"][task_id]

            if self.dataset == "nuscenes":
                pred_box_encoding = torch.cat(
                    [
                        pred_dict["reg"],
                        pred_dict["height"],
                        pred_dict["dim"],
                        pred_dict["rot"],
                        pred_dict["vel"],
                        pred_dict["iou"],
                    ],
                    dim=1,
                ).contiguous()  # (B, 11, H, W)
            else:
                pred_box_encoding = torch.cat(
                    [
                        pred_dict["reg"],
                        pred_dict["height"],
                        pred_dict["dim"],
                        pred_dict["rot"],
                        pred_dict["iou"],
                    ],
                    dim=1,
                ).contiguous()  # (B, 8+1, H, W)

            forward_ret_dict["pred_box_encoding"][task_id] = pred_box_encoding

            stride = self.out_size_factor
            voxel_size = self.proposal_layer.voxel_size

            if self.dataset == "nuscenes":
                iou_loss, iou_aware_loss = self._get_iou_loss(
                    pred_box_encoding[:, :11, :, :],
                    target_box_encoding[:, :, :10],
                    forward_ret_dict["ind"][task_id],
                    forward_ret_dict["mask"][task_id],
                    stride,
                    voxel_size,
                )

                box_loss = self.crit_reg(
                    pred_box_encoding[:, :10, :, :],
                    forward_ret_dict["mask"][task_id],
                    forward_ret_dict["ind"][task_id],
                    target_box_encoding[:, :, :10],
                )
            else:
                iou_loss, iou_aware_loss = self._get_iou_loss(
                    pred_box_encoding[:, :9, :, :],
                    target_box_encoding[:, :, :8],
                    forward_ret_dict["ind"][task_id],
                    forward_ret_dict["mask"][task_id],
                    stride,
                    voxel_size,
                )

                box_loss = self.crit_reg(
                    pred_box_encoding[:, :8, :, :],
                    forward_ret_dict["mask"][task_id],
                    forward_ret_dict["ind"][task_id],
                    target_box_encoding[:, :, :8],
                )

            loc_loss = (box_loss * box_loss.new_tensor(self.code_weights)).sum()
            # loss = hm_loss + self.weight * loc_loss + self.iou_aware_weight * iou_aware_loss
            loss = self.auto_loss(hm_loss, loc_loss, iou_aware_loss)
            # loc_loss足够小时回传iou_loss
            if loc_loss.item() < 1:
                loss = loss + iou_loss * self.iou_weight

            tb_key = "task_" + str(task_id) + "/"

            if self.dataset == "nuscenes":
                tb_dict.update(
                    {
                        tb_key + "loss": loss.item(),
                        tb_key + "hm_loss": hm_loss.item(),
                        tb_key + "loc_loss": loc_loss.item(),
                        tb_key + "x_loss": box_loss[0].item(),
                        tb_key + "y_loss": box_loss[1].item(),
                        tb_key + "z_loss": box_loss[2].item(),
                        tb_key + "w_loss": box_loss[3].item(),
                        tb_key + "l_loss": box_loss[4].item(),
                        tb_key + "h_loss": box_loss[5].item(),
                        tb_key + "sin_r_loss": box_loss[6].item(),
                        tb_key + "cos_r_loss": box_loss[7].item(),
                        tb_key + "vx_loss": box_loss[8].item(),
                        tb_key + "vy_loss": box_loss[9].item(),
                        tb_key
                        + "num_positive": forward_ret_dict["mask"][task_id]
                        .float()
                        .sum(),
                    }
                )
            else:
                tb_dict.update(
                    {
                        tb_key + "loss": loss.item(),
                        tb_key + "hm_loss": hm_loss.item(),
                        tb_key + "loc_loss": loc_loss.item(),
                        tb_key + "x_loss": box_loss[0].item(),
                        tb_key + "y_loss": box_loss[1].item(),
                        tb_key + "z_loss": box_loss[2].item(),
                        tb_key + "w_loss": box_loss[3].item(),
                        tb_key + "l_loss": box_loss[4].item(),
                        tb_key + "h_loss": box_loss[5].item(),
                        tb_key + "sin_r_loss": box_loss[6].item(),
                        tb_key + "cos_r_loss": box_loss[7].item(),
                        tb_key + "iou_aware_loss": iou_aware_loss.item(),
                        tb_key + "iou_loss": iou_loss.item(),
                        tb_key
                        + "num_positive": forward_ret_dict["mask"][task_id]
                        .float()
                        .sum(),
                    }
                )
            center_loss.append(loss)

        return sum(center_loss), tb_dict

    def _get_3d_iou(
        self,
        target_x_offset,
        target_y_offset,
        target_whl,
        hei,
        pred_x_offset,
        pred_y_offset,
        pred_whl,
        hei_p,
    ):
        # 这里还是需要每个单独clamp。否则如果两项同时小于0，反而会得到一个大于0的错误iou
        intersect_x = torch.clamp(
            torch.min(
                pred_x_offset + pred_whl[:, 0:1] / 2,
                target_x_offset + target_whl[:, 0:1] / 2,
            )
            - torch.max(
                pred_x_offset - pred_whl[:, 0:1] / 2,
                target_x_offset - target_whl[:, 0:1] / 2,
            ),
            min=1e-3,
        )
        intersect_y = torch.clamp(
            torch.min(
                pred_y_offset + pred_whl[:, 2:3] / 2,
                target_y_offset + target_whl[:, 2:3] / 2,
            )
            - torch.max(
                pred_y_offset - pred_whl[:, 2:3] / 2,
                target_y_offset - target_whl[:, 2:3] / 2,
            ),
            min=1e-3,
        )
        intersect_z = torch.clamp(
            torch.min(hei_p + pred_whl[:, 1:2] / 2, hei + target_whl[:, 1:2] / 2)
            - torch.max(hei_p - pred_whl[:, 1:2] / 2, hei - target_whl[:, 1:2] / 2),
            min=1e-3,
        )

        area_intersect = intersect_x * intersect_y * intersect_z
        area_pred = torch.clamp(
            pred_whl[:, 0:1] * pred_whl[:, 2:3] * pred_whl[:, 1:2], min=1e-3
        )
        area_tgt = torch.clamp(
            target_whl[:, 0:1] * target_whl[:, 2:3] * target_whl[:, 1:2], min=1e-3
        )

        iou = area_intersect / (area_pred + area_tgt - area_intersect)
        return iou

    def _get_iou_loss(
        self, batch_preds, batch_targets, proposal_inds, pos_mask, stride, voxel_size
    ):
        pred = _transpose_and_gather_feat(
            batch_preds, proposal_inds
        )  # bs, num_proposal, 8+1

        target_x_offset = (batch_targets[..., 0:1] * stride * voxel_size[0]).reshape(
            -1, 1
        )  # (bs*num_proposal, 1)
        target_y_offset = (batch_targets[..., 1:2] * stride * voxel_size[1]).reshape(
            -1, 1
        )
        target_whl = torch.exp(batch_targets[..., 3:6]).reshape(-1, 3)
        target_whl = torch.clamp(target_whl, min=0.001, max=30)
        targe_rot = torch.atan2(batch_targets[..., 6], batch_targets[..., 7]).reshape(
            -1, 1
        )
        hei = batch_targets[..., 2].reshape(-1, 1)
        target_bbox3d = torch.cat(
            [target_x_offset, target_y_offset, hei, target_whl, targe_rot], dim=-1
        )

        pred_x_offset = (pred[..., 0:1] * stride * voxel_size[0]).reshape(-1, 1)
        pred_y_offset = (pred[..., 1:2] * stride * voxel_size[1]).reshape(-1, 1)
        pred_whl = torch.exp(pred[..., 3:6]).reshape(-1, 3)
        pred_whl = torch.clamp(pred_whl, min=0.001, max=30)
        pred_rot = torch.atan2(pred[..., 6], pred[..., 7]).reshape(-1, 1)
        hei_p = pred[..., 2].reshape(-1, 1)
        pre_bbox3d = torch.cat(
            [pred_x_offset, pred_y_offset, hei_p, pred_whl, pred_rot], dim=-1
        )

        # iou loss
        iou = self._get_3d_iou(
            target_x_offset,
            target_y_offset,
            target_whl,
            hei,
            pred_x_offset,
            pred_y_offset,
            pred_whl,
            hei_p,
        )

        iou_pos = torch.clamp(iou[pos_mask.flatten()], 0, 1)
        iou_loss = 1 - iou_pos
        # iou_loss = iou_loss.sum() / max(1, len(iou_pos))
        num_pos = pos_mask.float().sum()
        num_pos = reduce_mean(num_pos)
        iou_loss = iou_loss.sum() / max(1, num_pos)

        # iou aware loss
        iou = box_utils.boxes3d_nearest_bev_iou(
            target_bbox3d, pre_bbox3d.clone().detach()
        )
        tril = range(len(iou))
        tar_iou_pred = 2 * (iou[tril, tril].reshape(*pos_mask.shape, 1) - 0.5)
        iou_aware_loss = self.crit_iou_aware(
            batch_preds[:, -1:, :, :], pos_mask, proposal_inds, tar_iou_pred
        )

        return iou_loss, iou_aware_loss.sum()
