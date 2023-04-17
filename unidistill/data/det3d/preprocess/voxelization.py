import numpy as np
import torch
import torch.nn.functional as F
from spconv.pytorch.utils import PointToVoxel
from torch import nn


class Voxelization(nn.Module):
    def __init__(
        self,
        voxel_size,
        point_cloud_range,
        max_num_points,
        max_voxels,
        num_point_features,
        device,
    ):
        super().__init__()
        assert len(voxel_size) == 3
        assert len(point_cloud_range) == 6
        self.voxel_size = np.array(voxel_size)
        self.point_cloud_range = np.array(point_cloud_range)
        self.max_num_points = max_num_points
        self.num_point_features = num_point_features
        if isinstance(max_voxels, tuple):
            if self.training:
                max_voxels = max_voxels[0]
            else:
                max_voxels = max_voxels[1]
        self.max_voxels = max_voxels
        self.voxel_generator = PointToVoxel(
            vsize_xyz=voxel_size,
            coors_range_xyz=point_cloud_range,
            max_num_points_per_voxel=max_num_points,
            max_num_voxels=max_voxels,
            num_point_features=num_point_features,
            device=device,
        )
        grid_size = (
            self.point_cloud_range[3:6] - self.point_cloud_range[0:3]
        ) / np.array(voxel_size)
        self.grid_size = np.round(grid_size).astype(np.int64)

    def forward(self, points):
        if not isinstance(points, list):
            points = [points]

        data_dict = {}
        data_dict["voxels"] = []
        data_dict["voxel_coords"] = []
        data_dict["voxel_num_points"] = []

        for p in points:
            voxel_output = self.voxel_generator(p)
            voxels, coordinates, num_points = voxel_output
            data_dict["voxels"].append(torch.clone(voxels))
            data_dict["voxel_coords"].append(torch.clone(coordinates))
            data_dict["voxel_num_points"].append(torch.clone(num_points))

        for key, val in data_dict.items():
            if key in ["voxels", "voxel_num_points"]:
                data_dict[key] = torch.cat(val, dim=0)
            elif key in ["points", "voxel_coords"]:
                coors = []
                for i, coor in enumerate(val):
                    coor_pad = F.pad(coor, (1, 0), mode="constant", value=i)
                    coors.append(coor_pad)
                data_dict[key] = torch.cat(coors, dim=0)
        return (
            data_dict["voxels"],
            data_dict["voxel_coords"],
            data_dict["voxel_num_points"],
        )
