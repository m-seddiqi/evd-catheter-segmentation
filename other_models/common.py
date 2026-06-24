from __future__ import annotations

from typing import Dict, List, Any

import torch
import torch.nn as nn
from timm.models import create_model

MODEL_SPECS: Dict[str, Dict[str, Any]] = {
    "mobilenetv3_large_100": {
        "timm_name": "mobilenetv3_large_100",
        "out_indices": (1, 2, 3, 4),  # reductions: 4,8,16,32
        "out_channels": [24, 40, 112, 960],
    },
    "mobilenetv3_small_100": {
        "timm_name": "mobilenetv3_small_100",
        "out_indices": (1, 2, 3, 4),  # reductions: 4,8,16,32
        "out_channels": [16, 24, 48, 576],
    },
    # There is no standard "vgg18" in timm/torchvision.
    # vgg16_bn is kept as an optional legacy baseline.
    "vgg16_bn": {
        "timm_name": "vgg16_bn",
        "out_indices": (2, 3, 4, 5),  # reductions: 4,8,16,32
        "out_channels": [256, 512, 512, 512],
    },
    # Modern, stronger baseline than VGG at similar or better practical performance.
    "convnext_tiny": {
        "timm_name": "convnext_tiny",
        "out_indices": (0, 1, 2, 3),  # reductions: 4,8,16,32
        "out_channels": [96, 192, 384, 768],
    },
    # Efficient and commonly used baseline.
    "efficientnet_b0": {
        "timm_name": "efficientnet_b0",
        "out_indices": (1, 2, 3, 4),  # reductions: 4,8,16,32
        "out_channels": [24, 40, 112, 320],
    },
    # Optional robust residual baseline for papers.
    "resnet18": {
        "timm_name": "resnet18",
        "out_indices": (1, 2, 3, 4),  # reductions: 4,8,16,32
        "out_channels": [64, 128, 256, 512],
    },
}


class OtherBackbone(nn.Module):
    def __init__(self, variant: str, pretrained: bool = True):
        super().__init__()
        key = variant.lower()
        if key not in MODEL_SPECS:
            raise ValueError(
                f"Unknown variant '{variant}'. Supported: {sorted(MODEL_SPECS.keys())}"
            )

        spec = MODEL_SPECS[key]
        self.model = create_model(
            spec["timm_name"],
            pretrained=pretrained,
            features_only=True,
            out_indices=spec["out_indices"],
        )
        self.out_channels: List[int] = list(spec["out_channels"])

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        feats = self.model(x)
        return list(feats)
