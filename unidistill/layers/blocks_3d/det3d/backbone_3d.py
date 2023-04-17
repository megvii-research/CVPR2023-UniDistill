from functools import partial

import spconv.pytorch as spconv
import torch.nn as nn

from .spconv_backbone import SparseBasicBlock, post_act_block

network_configs = {
    "VoxelExpRes18BackBone8x": {
        "exfactor": 2,
        "basic_block_list": ["SparseBasicBlock"] * 4,
        "block_num": [2, 2, 2, 2],
    },
    "VoxelRes34BackBone8x": {
        "exfactor": 1,
        "basic_block_list": ["SparseBasicBlock"] * 4,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelExpRes34BackBone8x": {
        "exfactor": 2,
        "basic_block_list": ["SparseBasicBlock"] * 4,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelRes50BackBone8x": {
        "exfactor": 1,
        "basic_block_list": ["SparseBottleNeckBlock"] * 4,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelSERes18BackBone8x": {
        "exfactor": 1,
        "basic_block_list": ["SESparseBasicBlock"] * 4,
        "block_num": [2, 2, 2, 2],
    },
    "VoxelSEExpRes18BackBone8x": {
        "exfactor": 2,
        "basic_block_list": ["SESparseBasicBlock"] * 4,
        "block_num": [2, 2, 2, 2],
    },
    "VoxelSERes34BackBone8x": {
        "exfactor": 1,
        "basic_block_list": ["SESparseBasicBlock"] * 4,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelSEExpRes34BackBone8x": {
        "exfactor": 2,
        "basic_block_list": ["SESparseBasicBlock"] * 4,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelSERes18BackBone8xV2": {
        "exfactor": 1,
        "basic_block_list": ["SparseBasicBlock"] * 2 + ["SESparseBasicBlock"] * 2,
        "block_num": [2, 2, 2, 2],
    },
    "VoxelSEExpRes18BackBone8xV2": {
        "exfactor": 2,
        "basic_block_list": ["SparseBasicBlock"] * 2 + ["SESparseBasicBlock"] * 2,
        "block_num": [2, 2, 2, 2],
    },
    "VoxelSERes34BackBone8xV2": {
        "exfactor": 1,
        "basic_block_list": ["SparseBasicBlock"] * 2 + ["SESparseBasicBlock"] * 2,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelSEExpRes34BackBone8xV2": {
        "exfactor": 2,
        "basic_block_list": ["SparseBasicBlock"] * 2 + ["SESparseBasicBlock"] * 2,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelSERes18BackBone8xV3": {
        "exfactor": 1,
        "basic_block_list": ["SparseBasicBlock"] + ["SESparseBasicBlock"] * 3,
        "block_num": [2, 2, 2, 2],
    },
    "VoxelSEExpRes18BackBone8xV3": {
        "exfactor": 2,
        "basic_block_list": ["SparseBasicBlock"] + ["SESparseBasicBlock"] * 3,
        "block_num": [2, 2, 2, 2],
    },
    "VoxelSERes34BackBone8xV3": {
        "exfactor": 1,
        "basic_block_list": ["SparseBasicBlock"] + ["SESparseBasicBlock"] * 3,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelSEExpRes34BackBone8xV3": {
        "exfactor": 2,
        "basic_block_list": ["SparseBasicBlock"] + ["SESparseBasicBlock"] * 3,
        "block_num": [3, 4, 6, 3],
    },
    "VoxelSERes18BackBone8xV4": {
        "exfactor": 1,
        "basic_block_list": ["SparseBasicBlock"] * 3 + ["SESparseBasicBlock"],
        "block_num": [2, 2, 2, 2],
    },
    "VoxelSEExpRes18BackBone8xV4": {
        "exfactor": 2,
        "basic_block_list": ["SparseBasicBlock"] * 3 + ["SESparseBasicBlock"],
        "block_num": [2, 2, 2, 2],
    },
    "VoxelSERes34BackBone8xV4": {
        "exfactor": 1,
        "basic_block_list": ["SparseBasicBlock"] * 3 + ["SESparseBasicBlock"],
        "block_num": [3, 4, 6, 3],
    },
    "VoxelSEExpRes34BackBone8xV4": {
        "exfactor": 2,
        "basic_block_list": ["SparseBasicBlock"] * 3 + ["SESparseBasicBlock"],
        "block_num": [3, 4, 6, 3],
    },
}


class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = x.permute(1, 0)
        c, _ = x.size()
        y = x.mean(dim=1)
        y = self.fc(y).view(c, 1)
        res = x * y.expand_as(x)
        return res.permute(1, 0)


class SESparseBasicBlock(spconv.SparseModule):
    expansion = 1

    def __init__(
        self, inplanes, planes, stride=1, norm_fn=None, downsample=None, indice_key=None
    ):
        super(SESparseBasicBlock, self).__init__()
        assert norm_fn is not None
        bias = norm_fn is not None
        self.conv1 = spconv.SubMConv3d(
            inplanes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(planes)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(planes)
        self.se = SELayer(planes, reduction=16)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn2(out.features))
        out = out.replace_feature(self.se(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class SparseBottleNeckBlock(spconv.SparseModule):
    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        expansion=2,
        norm_fn=None,
        downsample=None,
        indice_key=None,
    ):
        super(SparseBottleNeckBlock, self).__init__()

        assert norm_fn is not None
        bias = norm_fn is not None
        assert planes % expansion == 0
        midplane = planes // expansion
        self.conv1 = spconv.SubMConv3d(
            inplanes,
            midplane,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=bias,
            in_situ=False,
            indice_key=indice_key,
        )
        self.bn1 = norm_fn(midplane)
        self.relu = nn.ReLU()
        self.conv2 = spconv.SubMConv3d(
            midplane,
            midplane,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn2 = norm_fn(midplane)
        self.conv3 = spconv.SubMConv3d(
            midplane,
            midplane,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=bias,
            indice_key=indice_key,
        )
        self.bn3 = norm_fn(midplane)

        self.conv4 = spconv.SubMConv3d(
            midplane,
            planes,
            kernel_size=1,
            stride=1,
            padding=1,
            bias=bias,
            in_situ=False,
            indice_key=indice_key,
        )
        self.bn4 = norm_fn(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv2(out)
        out = out.replace_feature(self.bn1(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv3(out)
        out = out.replace_feature(self.bn3(out.features))
        out = out.replace_feature(self.relu(out.features))

        out = self.conv4(out)
        out = out.replace_feature(self.bn4(out.features))

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out.replace_feature(out.features + identity.features)
        out = out.replace_feature(self.relu(out.features))

        return out


class VoxelResBasicBackBone8x(nn.Module):
    def __init__(self, model_name, input_channels, grid_size, last_pad=0, **kwargs):
        super().__init__()
        norm_fn = partial(nn.BatchNorm1d, eps=1e-3, momentum=0.01)
        self.exfactor = network_configs[model_name]["exfactor"]
        basic_block_list = network_configs[model_name]["basic_block_list"]
        block_num = network_configs[model_name]["block_num"]
        basic_block_all = {
            "SparseBasicBlock": SparseBasicBlock,
            "SESparseBasicBlock": SESparseBasicBlock,
            "SparseBottleNeckBlock": SparseBottleNeckBlock,
        }

        self.sparse_shape = grid_size[::-1] + [1, 0, 0]

        self.conv_input = spconv.SparseSequential(
            spconv.SubMConv3d(
                input_channels,
                16 * self.exfactor,
                3,
                padding=1,
                bias=False,
                indice_key="subm1",
            ),
            norm_fn(16 * self.exfactor),
            nn.ReLU(),
        )
        block = post_act_block

        self.conv1 = spconv.SparseSequential(
            *[
                basic_block_all[basic_block_list[0]](
                    16 * self.exfactor,
                    16 * self.exfactor,
                    norm_fn=norm_fn,
                    indice_key="res1",
                )
                for _ in range(block_num[0])
            ]
        )

        self.conv2 = spconv.SparseSequential(
            # [1600, 1408, 41] <- [800, 704, 21]
            block(
                16 * self.exfactor,
                32 * self.exfactor,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv2",
                conv_type="spconv",
            ),
            *[
                basic_block_all[basic_block_list[1]](
                    32 * self.exfactor,
                    32 * self.exfactor,
                    norm_fn=norm_fn,
                    indice_key="res2",
                )
                for _ in range(block_num[1])
            ]
        )

        self.conv3 = spconv.SparseSequential(
            # [800, 704, 21] <- [400, 352, 11]
            block(
                32 * self.exfactor,
                64 * self.exfactor,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=1,
                indice_key="spconv3",
                conv_type="spconv",
            ),
            *[
                basic_block_all[basic_block_list[2]](
                    64 * self.exfactor,
                    64 * self.exfactor,
                    norm_fn=norm_fn,
                    indice_key="res3",
                )
                for _ in range(block_num[2])
            ]
        )

        self.conv4 = spconv.SparseSequential(
            # [400, 352, 11] <- [200, 176, 5]
            block(
                64 * self.exfactor,
                128 * self.exfactor,
                3,
                norm_fn=norm_fn,
                stride=2,
                padding=(0, 1, 1),
                indice_key="spconv4",
                conv_type="spconv",
            ),
            *[
                basic_block_all[basic_block_list[3]](
                    128 * self.exfactor,
                    128 * self.exfactor,
                    norm_fn=norm_fn,
                    indice_key="res4",
                )
                for _ in range(block_num[3])
            ]
        )

        self.conv_out = spconv.SparseSequential(
            # [200, 150, 5] -> [200, 150, 2]
            spconv.SparseConv3d(
                128 * self.exfactor,
                128 * self.exfactor,
                (3, 1, 1),
                stride=(2, 1, 1),
                padding=last_pad,
                bias=False,
                indice_key="spconv_down2",
            ),
            norm_fn(128 * self.exfactor),
            nn.ReLU(),
        )
        self.num_point_features = 128 * self.exfactor

    def forward(self, voxel_features, voxel_coords, batch_size):
        """
        Args:
            batch_dict:
                batch_size: int
                vfe_features: (num_voxels, C)
                voxel_coords: (num_voxels, 4), [batch_idx, z_idx, y_idx, x_idx]
        Returns:
            batch_dict:
                encoded_spconv_tensor: sparse tensor
        """
        input_sp_tensor = spconv.SparseConvTensor(
            features=voxel_features,
            indices=voxel_coords.int(),
            spatial_shape=self.sparse_shape,
            batch_size=batch_size,
        )
        x = self.conv_input(input_sp_tensor)

        x_conv1 = self.conv1(x)
        x_conv2 = self.conv2(x_conv1)
        x_conv3 = self.conv3(x_conv2)
        x_conv4 = self.conv4(x_conv3)

        # for detection head
        # [200, 176, 5] -> [200, 176, 2]
        out = self.conv_out(x_conv4)

        encoded_spconv_tensor = out
        encoded_spconv_tensor_stride = 8
        multi_scale_3d_features = {
            "x_conv1": x_conv1,
            "x_conv2": x_conv2,
            "x_conv3": x_conv3,
            "x_conv4": x_conv4,
        }
        return (
            encoded_spconv_tensor,
            encoded_spconv_tensor_stride,
            multi_scale_3d_features,
        )
