from __future__ import annotations

from typing import Any, Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LineNormalDistanceLoss(nn.Module):
    """
    Penalizes deviation of predicted foreground's normal-distance stats from GT line stats.
    """

    def __init__(
        self,
        *,
        foreground_index: int = 1,
        ignore_index: int = 255,
        std_weight: float = 0.5,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.foreground_index = int(foreground_index)
        self.ignore_index = int(ignore_index)
        self.std_weight = float(std_weight)
        self.eps = float(eps)
        self.uses_line_prior = True

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        line_prior: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        if line_prior is None:
            return inputs.sum() * 0.0

        # Force fp32 math for stability under AMP.
        logits_f = inputs.float()
        probs = F.softmax(logits_f, dim=1)[:, self.foreground_index]  # (B,H,W)
        valid_pix = (targets != self.ignore_index).float()

        bsz, h, w = probs.shape
        yy, xx = torch.meshgrid(
            torch.arange(h, device=inputs.device, dtype=torch.float32),
            torch.arange(w, device=inputs.device, dtype=torch.float32),
            indexing="ij",
        )
        xx = xx.unsqueeze(0).expand(bsz, -1, -1)
        yy = yy.unsqueeze(0).expand(bsz, -1, -1)

        point = line_prior["point"].to(inputs.device, dtype=torch.float32)  # (B,2)
        direc = line_prior["dir"].to(inputs.device, dtype=torch.float32)     # (B,2)
        gt_mean = line_prior["mean_dist"].to(inputs.device, dtype=torch.float32)
        gt_std = line_prior["std_dist"].to(inputs.device, dtype=torch.float32)
        prior_valid = line_prior["valid"].to(inputs.device, dtype=torch.bool)
        gt_mean = torch.nan_to_num(gt_mean, nan=0.0, posinf=0.0, neginf=0.0)
        gt_std = torch.nan_to_num(gt_std, nan=0.0, posinf=0.0, neginf=0.0)

        nx = -direc[:, 1].view(-1, 1, 1)
        ny = direc[:, 0].view(-1, 1, 1)
        px = point[:, 0].view(-1, 1, 1)
        py = point[:, 1].view(-1, 1, 1)
        dist = torch.abs((xx - px) * nx + (yy - py) * ny)  # (B,H,W)

        w_fg = probs * valid_pix
        mass = w_fg.sum(dim=(1, 2)).clamp_min(self.eps)

        pred_mean = (w_fg * dist).sum(dim=(1, 2)) / mass
        pred_var = (w_fg * (dist - pred_mean.view(-1, 1, 1)).pow(2)).sum(dim=(1, 2)) / mass
        pred_std = torch.sqrt(pred_var.clamp_min(self.eps))

        # Use relative errors (not raw pixel units) so this term does not dominate CE/Dice.
        mean_scale = torch.clamp(gt_mean.detach(), min=1.0)
        std_scale = torch.clamp(gt_std.detach(), min=1.0)
        loss_mean = F.smooth_l1_loss(pred_mean / mean_scale, gt_mean / mean_scale, reduction="none")
        loss_std = F.smooth_l1_loss(pred_std / std_scale, gt_std / std_scale, reduction="none")
        per_sample = loss_mean + self.std_weight * loss_std
        per_sample = torch.nan_to_num(per_sample, nan=0.0, posinf=1e4, neginf=0.0).clamp_max(10.0)

        sample_valid = prior_valid & (gt_mean > 0)
        if torch.any(sample_valid):
            return per_sample[sample_valid].mean()
        return inputs.sum() * 0.0


class CombinedSegLineLoss(nn.Module):
    """
    Wraps an existing segmentation criterion with an extra line-distance prior term.
    """

    def __init__(
        self,
        *,
        seg_loss: nn.Module,
        line_weight: float = 0.1,
        foreground_index: int = 1,
        ignore_index: int = 255,
        std_weight: float = 0.5,
    ):
        super().__init__()
        self.seg_loss = seg_loss
        self.line_weight = float(line_weight)
        self.line_loss = LineNormalDistanceLoss(
            foreground_index=foreground_index,
            ignore_index=ignore_index,
            std_weight=std_weight,
        )
        self.uses_line_prior = True

    def forward(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        line_prior: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        seg = self.seg_loss(inputs, targets)
        line = self.line_loss(inputs, targets, line_prior=line_prior)
        line = torch.nan_to_num(line, nan=0.0, posinf=1e4, neginf=0.0)
        return seg + self.line_weight * line


def maybe_wrap_with_line_prior(
    *,
    seg_loss: nn.Module,
    loss_cfg: Dict[str, Any],
) -> nn.Module:
    """
    Keep default behavior unless loss.line_prior.enabled=true in config.
    """
    lp_cfg = loss_cfg.get("line_prior", {}) if isinstance(loss_cfg, dict) else {}
    if not bool(lp_cfg.get("enabled", False)):
        return seg_loss

    return CombinedSegLineLoss(
        seg_loss=seg_loss,
        line_weight=float(lp_cfg.get("weight", 0.1)),
        foreground_index=int(lp_cfg.get("foreground_index", 1)),
        ignore_index=int(loss_cfg.get("ignore_index", 255)),
        std_weight=float(lp_cfg.get("std_weight", 0.5)),
    )
