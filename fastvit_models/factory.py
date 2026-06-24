import os
import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, Any
from .common import FastViTBackbone, load_backbone_init_checkpoint
from .heads import MonoFPNHead

class FastViTSegModel(nn.Module):
    def __init__(self, base_model: str, num_classes: int, fpn_dim: int = 256):
        super().__init__()
 
        self.backbone = FastViTBackbone(base_model)
        self.head = MonoFPNHead(self.backbone.out_channels, fpn_dim, num_classes)

    def initialize_backbone(self, ckpt_path: str):
        """Restored exact loading logic from the old code."""
        load_backbone_init_checkpoint(backbone_model=self.backbone, checkpoint_path=ckpt_path, strict=False)

    def forward(self, left: torch.Tensor, right: Optional[torch.Tensor] = None):        
        logits = self.head(self.backbone(left), left.shape[-2:])
        return logits

def build_model(variant: str, num_classes: int) -> FastViTSegModel:
    return FastViTSegModel(base_model=variant, num_classes=num_classes)