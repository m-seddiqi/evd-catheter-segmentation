from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from .common import OtherBackbone
from fastvit_models.heads import MonoFPNHead


class OtherSegModel(nn.Module):
    def __init__(self, variant: str, num_classes: int, fpn_dim: int = 256, pretrained: bool = True):
        super().__init__()
        self.backbone = OtherBackbone(variant=variant, pretrained=pretrained)
        self.head = MonoFPNHead(self.backbone.out_channels, fpn_dim, num_classes)

    def forward(self, left: torch.Tensor, right: Optional[torch.Tensor] = None):
        logits = self.head(self.backbone(left), left.shape[-2:])
        return logits


def build_model(variant: str, num_classes: int, *, fpn_dim: int = 256, pretrained: bool = True) -> OtherSegModel:
    return OtherSegModel(
        variant=variant,
        num_classes=num_classes,
        fpn_dim=fpn_dim,
        pretrained=pretrained,
    )
