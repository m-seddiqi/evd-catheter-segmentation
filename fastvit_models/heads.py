import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple

class MonoFPNHead(nn.Module):
    def __init__(self, in_channels: List[int], fpn_dim: int, num_classes: int):
        super().__init__()
        self.num_stages = len(in_channels)
        self.lateral_convs = nn.ModuleList([nn.Conv2d(c, fpn_dim, 1) for c in in_channels])
        self.output_convs = nn.ModuleList([nn.Conv2d(fpn_dim, fpn_dim, 3, padding=1) for _ in in_channels])
        self.cls_head = nn.Sequential(
            nn.Conv2d(fpn_dim * self.num_stages, fpn_dim, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(fpn_dim, num_classes, 1)
        )

    def forward(self, feats: List[torch.Tensor], size: Tuple[int, int]) -> torch.Tensor:
        H, W = size
        laterals = [conv(f) for conv, f in zip(self.lateral_convs, feats)]
        for i in range(self.num_stages - 1, 0, -1):
            up = F.interpolate(laterals[i], size=laterals[i-1].shape[-2:], mode="bilinear", align_corners=False)
            laterals[i-1] = laterals[i-1] + up
        outs = [conv(lat) for conv, lat in zip(self.output_convs, laterals)]
        target_size = outs[0].shape[-2:]
        outs_up = [F.interpolate(o, size=target_size, mode="bilinear", align_corners=False) for o in outs]
        logits = self.cls_head(torch.cat(outs_up, dim=1))
        return F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)

