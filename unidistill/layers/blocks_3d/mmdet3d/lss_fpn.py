import torch
from mmdet3d.models import build_neck
from mmdet.models import build_backbone
from torch import nn
from torch.autograd import Function

from . import voxel_pooling_ext

__all__ = ["LSSFPN"]


class VoxelPooling(Function):
    @staticmethod
    def forward(
        ctx,
        geom_xyz: torch.Tensor,
        input_features: torch.Tensor,
        voxel_num: torch.Tensor,
    ) -> torch.Tensor:
        """forward.

        Args:
            geom_xyz (Tensor): xyz coord for each voxel with the shape of [B, N, 3].
            input_features (Tensor): feature for each voxel with the shape of [B, N, C].
            voxel_num (Tensor): Number of voxels for each dim with the shape of [3].

        Returns:
            Tensor: (B, C, H, W) bev feature map.
        """
        assert geom_xyz.is_contiguous()
        assert input_features.is_contiguous()
        # no gradient for input_features and geom_feats
        ctx.mark_non_differentiable(geom_xyz)
        grad_input_features = torch.zeros_like(input_features)
        geom_xyz = geom_xyz.reshape(geom_xyz.shape[0], -1, geom_xyz.shape[-1])
        input_features = input_features.reshape(
            (geom_xyz.shape[0], -1, input_features.shape[-1])
        )
        assert geom_xyz.shape[1] == input_features.shape[1]
        batch_size = input_features.shape[0]
        num_points = input_features.shape[1]
        num_channels = input_features.shape[2]
        output_features = input_features.new_zeros(
            batch_size, voxel_num[1], voxel_num[0], num_channels
        )
        # Save the position of bev_feature_map for each input poing.
        pos_memo = geom_xyz.new_ones(batch_size, num_points, 3) * -1
        voxel_pooling_ext.voxel_pooling_forward_wrapper(
            batch_size,
            num_points,
            num_channels,
            voxel_num[0],
            voxel_num[1],
            voxel_num[2],
            geom_xyz,
            input_features,
            output_features,
            pos_memo,
        )
        # save grad_input_features and pos_memo for backward
        ctx.save_for_backward(grad_input_features, pos_memo)
        return output_features.permute(0, 3, 1, 2)

    @staticmethod
    def backward(ctx, grad_output_features):
        (grad_input_features, pos_memo) = ctx.saved_tensors
        kept = (pos_memo != -1)[..., 0]
        grad_input_features_shape = grad_input_features.shape
        grad_input_features = grad_input_features.reshape(
            grad_input_features.shape[0], -1, grad_input_features.shape[-1]
        )
        grad_input_features[kept] = grad_output_features[
            pos_memo[kept][..., 0].long(),
            :,
            pos_memo[kept][..., 1].long(),
            pos_memo[kept][..., 2].long(),
        ]
        grad_input_features = grad_input_features.reshape(grad_input_features_shape)
        return None, grad_input_features, None, None, None


voxel_pooling = VoxelPooling.apply


class LSSFPN(nn.Module):
    def __init__(
        self,
        x_bound,
        y_bound,
        z_bound,
        d_bound,
        final_dim,
        downsample_factor,
        output_channels,
        img_backbone_conf,
        img_neck_conf,
        depth_net_conf,
        timestamp_net_conf=None,
    ):
        """Modified from `https://github.com/nv-tlabs/lift-splat-shoot`.

        Args:
            x_bound (list): Boundaries for x.
            y_bound (list): Boundaries for y.
            z_bound (list): Boundaries for z.
            d_bound (list): Boundaries for d.
            final_dim (list): Dimention for input images.
            downsample_factor (int): Downsample factor between feature map and input image.
            output_channels (int): Number of channels for the output feature map.
            img_backbone_conf (dict): Config for image backbone.
            img_neck_conf (dict): Config for image neck.
            depth_net_conf (dict): Config for depth net.
        """

        super(LSSFPN, self).__init__()
        self.downsample_factor = downsample_factor
        self.d_bound = d_bound
        self.final_dim = final_dim
        self.timestamp_net_conf = timestamp_net_conf
        self.output_channels = output_channels

        self.register_buffer(
            "voxel_size", torch.Tensor([row[2] for row in [x_bound, y_bound, z_bound]])
        )
        self.register_buffer(
            "voxel_coord",
            torch.Tensor(
                [row[0] + row[2] / 2.0 for row in [x_bound, y_bound, z_bound]]
            ),
        )
        self.register_buffer(
            "voxel_num",
            torch.LongTensor(
                [
                    round((row[1] - row[0]) / row[2])
                    for row in [x_bound, y_bound, z_bound]
                ]
            ),
        )
        self.register_buffer("frustum", self.create_frustum())
        self.depth_channels, _, _, _ = self.frustum.shape

        self.img_backbone = build_backbone(img_backbone_conf)
        self.img_neck = build_neck(img_neck_conf)
        self.depth_net = self._configure_depth_net(depth_net_conf)
        self.timestamp_net = self._configure_timestamp_net()

        self.img_neck.init_weights()
        self.img_backbone.init_weights()

    def _configure_timestamp_net(self):
        if self.timestamp_net_conf is not None:
            timestamp_net = nn.Linear(
                self.timestamp_net_conf["in_channels"],
                self.timestamp_net_conf["out_channels"],
            )
        else:
            timestamp_net = None
        return timestamp_net

    def _configure_depth_net(self, depth_net_conf):
        depth_net_list = list()
        depth_out_channels = self.depth_channels + self.output_channels
        num_res_layer = depth_net_conf.get("num_res_layer", 0)
        if num_res_layer == 0:
            depth_net_list = [
                nn.Conv2d(
                    depth_net_conf["in_channels"], depth_out_channels, kernel_size=1
                )
            ]
        return nn.Sequential(*depth_net_list)

    def create_frustum(self):
        """Generate frustum"""
        # make grid in image plane
        ogfH, ogfW = self.final_dim
        fH, fW = ogfH // self.downsample_factor, ogfW // self.downsample_factor
        d_coords = (
            torch.arange(*self.d_bound, dtype=torch.float)
            .view(-1, 1, 1)
            .expand(-1, fH, fW)
        )
        D, _, _ = d_coords.shape
        x_coords = (
            torch.linspace(0, ogfW - 1, fW, dtype=torch.float)
            .view(1, 1, fW)
            .expand(D, fH, fW)
        )
        y_coords = (
            torch.linspace(0, ogfH - 1, fH, dtype=torch.float)
            .view(1, fH, 1)
            .expand(D, fH, fW)
        )
        paddings = torch.ones_like(d_coords)

        # D x H x W x 3
        frustum = torch.stack((x_coords, y_coords, d_coords, paddings), -1)
        return frustum

    def get_geometry(self, sensor2ego_mat, intrin_mat, ida_mat, bda_mat):
        """Transfer points from camera coord to ego coord.

        Args:
            rots(Tensor): Rotation matrix from camera to ego.
            trans(Tensor): Translation matrix from camera to ego.
            intrins(Tensor): Intrinsic matrix.
            post_rots_ida(Tensor): Rotation matrix for ida.
            post_trans_ida(Tensor): Translation matrix for ida
            post_rot_bda(Tensor): Rotation matrix for bda.

        Returns:
            Tensors: points ego coord.
        """
        batch_size, num_cams, _, _ = sensor2ego_mat.shape

        # undo post-transformation
        # B x N x D x H x W x 3
        points = self.frustum
        ida_mat = ida_mat.view(batch_size, num_cams, 1, 1, 1, 4, 4)
        points = ida_mat.inverse().matmul(points.unsqueeze(-1))

        # cam_to_ego
        points = torch.cat(
            (
                points[:, :, :, :, :, :2] * points[:, :, :, :, :, 2:3],
                points[:, :, :, :, :, 2:],
            ),
            5,
        )

        combine = sensor2ego_mat.matmul(torch.inverse(intrin_mat))
        points = combine.view(batch_size, num_cams, 1, 1, 1, 4, 4).matmul(points)
        if bda_mat is not None:
            bda_mat = (
                bda_mat.unsqueeze(1)
                .repeat(1, num_cams, 1, 1)
                .view(batch_size, num_cams, 1, 1, 1, 4, 4)
            )
            points = (bda_mat @ points).squeeze(-1)
        return points[..., :3]

    def get_cam_feats(self, imgs):
        """Get feature maps from images."""
        batch_size, num_sweeps, num_cams, num_channels, imH, imW = imgs.shape

        imgs = imgs.flatten().view(
            batch_size * num_sweeps * num_cams, num_channels, imH, imW
        )
        img_feats = self.img_neck(self.img_backbone(imgs))[0]
        img_feats = img_feats.reshape(
            batch_size,
            num_sweeps,
            num_cams,
            img_feats.shape[1],
            img_feats.shape[2],
            img_feats.shape[3],
        )
        return img_feats

    def _forward_depth_net(self, feat, *args, **kwargs):
        return self.depth_net(feat)

    def _forward_voxel_net(self, img_feat_with_depth):
        return img_feat_with_depth

    def _forward_single_sweep(
        self, sweep_index, sweep_imgs, mats_dict, is_return_depth=False
    ):
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape
        img_feats = self.get_cam_feats(sweep_imgs)
        source_features = img_feats[:, 0, ...]
        depth_feature = self._forward_depth_net(
            source_features.reshape(
                batch_size * num_cams,
                source_features.shape[2],
                source_features.shape[3],
                source_features.shape[4],
            ),
            mats_dict,
            sweep_index,
        )
        depth = depth_feature[:, : self.depth_channels].softmax(1)
        img_feat_with_depth = depth.unsqueeze(1) * depth_feature[
            :, self.depth_channels : (self.depth_channels + self.output_channels)
        ].unsqueeze(2)

        img_feat_with_depth = self._forward_voxel_net(img_feat_with_depth)

        img_feat_with_depth = img_feat_with_depth.reshape(
            batch_size,
            num_cams,
            img_feat_with_depth.shape[1],
            img_feat_with_depth.shape[2],
            img_feat_with_depth.shape[3],
            img_feat_with_depth.shape[4],
        )
        geom_xyz = self.get_geometry(
            mats_dict["sensor2ego_mats"][:, sweep_index, ...],
            mats_dict["intrin_mats"][:, sweep_index, ...],
            mats_dict["ida_mats"][:, sweep_index, ...],
            mats_dict.get("bda_mat", None),
        )
        img_feat_with_depth = img_feat_with_depth.permute(0, 1, 3, 4, 5, 2)
        geom_xyz = (
            (geom_xyz - (self.voxel_coord - self.voxel_size / 2.0)) / self.voxel_size
        ).int()
        feature_map = voxel_pooling(
            geom_xyz, img_feat_with_depth.contiguous(), self.voxel_num.cuda()
        )
        if is_return_depth:
            return feature_map.contiguous(), depth
        return feature_map.contiguous()

    def forward(self, sweep_imgs, mats_dict, timestamps=None, is_return_depth=False):
        """Forward function.

        Args:
            sweep_imgs(Tensor): Input images with shape of (B, num_sweeps, num_cameras, 3, H, W).
            mats_dict(dict):
                sensor2ego_mats(Tensor): Transformation matrix from camera to ego with shape of (B, num_sweeps, num_cameras, 4, 4).
                intrin_mats(Tensor): Intrinsic matrix with shape of (B, num_sweeps, num_cameras, 4, 4).
                ida_mats(Tensor): Transformation matrix for ida with shape of (B, num_sweeps, num_cameras, 4, 4).
                sensor2sensor_mats(Tensor): Transformation matrix from key frame camera to sweep frame camera with shape of (B, num_sweeps, num_cameras, 4, 4).
                bda_mat(Tensor): Rotation matrix for bda with shape of (B, 4, 4).
            timestamps(Tensor): Timestamp for all images with the shape of(B, num_sweeps, num_cameras).

        Return:
            Tensor: bev feature map.
        """
        (
            batch_size,
            num_sweeps,
            num_cams,
            num_channels,
            img_height,
            img_width,
        ) = sweep_imgs.shape

        key_frame_res = self._forward_single_sweep(
            0, sweep_imgs[:, 0:1, ...], mats_dict, is_return_depth=is_return_depth
        )
        if num_sweeps == 1:
            return key_frame_res

        key_frame_feature = key_frame_res[0] if is_return_depth else key_frame_res

        ret_feature_list = [key_frame_feature]
        for sweep_index in range(1, num_sweeps):
            with torch.no_grad():
                feature_map = self._forward_single_sweep(
                    sweep_index,
                    sweep_imgs[:, sweep_index : sweep_index + 1, ...],
                    mats_dict,
                    is_return_depth=False,
                )
                ret_feature_list.append(feature_map)

        if is_return_depth:
            return torch.cat(ret_feature_list, 1), key_frame_res[1]
        else:
            return torch.cat(ret_feature_list, 1)
