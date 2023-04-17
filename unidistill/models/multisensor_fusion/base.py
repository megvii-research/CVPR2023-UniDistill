from abc import ABCMeta, abstractmethod
from typing import Any, Optional

import torch
import torch.nn as nn
from torch.nn.modules.batchnorm import _BatchNorm

_TENSOR = torch.Tensor
_TORCH_NN_MODULE = nn.Module


class BaseMultiSensorFusion(nn.Module):

    """Base class for multi-sensor fusion perception."""

    def __init__(self) -> Any:
        super(BaseMultiSensorFusion, self).__init__()

    @property
    def with_lidar_encoder(self) -> Optional[_TORCH_NN_MODULE]:
        """bool: whether has a lidar encoder"""
        return hasattr(self, "lidar_encoder") and self.lidar_encoder

    @property
    def with_camera_encoder(self) -> Optional[_TORCH_NN_MODULE]:
        """bool: whether has a cameras encoder"""
        return hasattr(self, "camera_encoder") and self.camera_encoder

    @property
    def with_radar_encoder(self) -> Optional[_TORCH_NN_MODULE]:
        """bool: whether has a radar encoder"""
        return hasattr(self, "radar_encoder") and self.radar_encoder

    @property
    def with_fusion_encoder(self):
        """bool: whether has a features fusion module."""
        return hasattr(self, "fusion_encoder") and self.fusion_encoder

    @abstractmethod
    def extract_feat(self, **kwargs):
        """Extract features from all sensors."""
        raise NotImplementedError("Must be implemented by yourself!")

    def onnx_export(self, **kwargs) -> Any:
        raise NotImplementedError(
            f"{self.__class__.__name__} does " f"not support ONNX EXPORT"
        )


class BaseEncoder(nn.Module, metaclass=ABCMeta):
    def __init__(self, frozen_encoder: bool = False, frozen_bn: bool = False):
        super(BaseEncoder, self).__init__()
        self.frozen_encoder = frozen_encoder
        self.frozen_bn = frozen_bn

    def _freeze_stages(self):
        if self.frozen_encoder:
            for param in self.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        super(BaseEncoder, self).train(mode)
        self._freeze_stages()
        if mode and self.frozen_bn:
            for m in self.modules():
                if isinstance(m, _BatchNorm):
                    m.eval()
