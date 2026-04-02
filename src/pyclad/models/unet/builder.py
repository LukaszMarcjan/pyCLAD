from __future__ import annotations

from torch import nn

from pyclad.models.unet.config import UNetConfig
from pyclad.models.unet.unet import build_architecture


def build(config: UNetConfig) -> nn.Module:
    return build_architecture(config)
