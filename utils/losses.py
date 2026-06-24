# loss.py
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# =========================================================
# 0) Cross-Entropy with background weighting (multi-class)
# =========================================================

class CrossEntropyWithBackgroundWeight(nn.Module):
    """
    Cross-entropy that down/up-weights background pixels.

    inputs:  (B, C, H, W) logits
    targets: (B, H, W) int labels in [0, C-1] or ignore_index

    background_index: which class is background (default 0)
    background_weight: multiplier for loss of background pixels
                       (1.0 = normal CE; <1.0 down-weights bg)
    """
    def __init__(
        self,
        ignore_index: int = 255,
        background_weight: float = 1.0,
        background_index: int = 0,
    ):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.background_weight = float(background_weight)
        self.background_index = int(background_index)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce_loss = F.cross_entropy(
            inputs,
            targets,
            reduction="none",
            ignore_index=self.ignore_index,
        )  # (B,H,W)

        valid_mask = (targets != self.ignore_index)
        ce_loss = ce_loss[valid_mask]

        if ce_loss.numel() == 0:
            return ce_loss.sum()

        if self.background_weight != 1.0:
            t_valid = targets[valid_mask]
            w = torch.ones_like(ce_loss)
            w[t_valid == self.background_index] = self.background_weight
            ce_loss = ce_loss * w

        return ce_loss.mean()


# =========================================================
# 1) Dice Loss (multi-class, optional bg)
# =========================================================

class DiceLoss(nn.Module):
    """
    Multi-class soft Dice loss.

    By default, averages only over foreground classes (ignores background_index).
    """
    def __init__(
        self,
        ignore_index: int = 255,
        eps: float = 1e-6,
        include_background: bool = False,
        background_index: int = 0,
    ):
        super().__init__()
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.include_background = bool(include_background)
        self.background_index = int(background_index)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)      # (B,C,H,W)
        num_classes = inputs.shape[1]

        t = targets.clone()
        t[t == self.ignore_index] = self.background_index
        clamped = torch.clamp(t, 0, num_classes - 1)

        one_hot = F.one_hot(clamped, num_classes=num_classes)   # (B,H,W,C)
        one_hot = one_hot.permute(0, 3, 1, 2).float()           # (B,C,H,W)

        valid_mask = (targets != self.ignore_index).unsqueeze(1)  # (B,1,H,W)
        probs = probs * valid_mask
        one_hot = one_hot * valid_mask

        dims = (0, 2, 3)
        intersection = (probs * one_hot).sum(dim=dims)               # (C,)
        cardinality = probs.sum(dim=dims) + one_hot.sum(dim=dims)    # (C,)

        dice_per_class = (2.0 * intersection + self.eps) / (cardinality + self.eps)
        loss_per_class = 1.0 - dice_per_class                        # (C,)

        if self.include_background or num_classes == 1:
            return loss_per_class.mean()

        idx = torch.arange(num_classes, device=inputs.device)
        fg_mask = (idx != self.background_index)
        if fg_mask.sum() == 0:
            return loss_per_class.mean()
        return loss_per_class[fg_mask].mean()


# =========================================================
# 2) Tversky / Focal-Tversky
# =========================================================

class TverskyLoss(nn.Module):
    """
    Multi-class soft Tversky loss.

    TI = (TP + eps) / (TP + alpha*FP + beta*FN + eps)
    loss = 1 - TI

    Default is catheter-friendly: beta > alpha (recall-friendly).
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        ignore_index: int = 255,
        eps: float = 1e-6,
        include_background: bool = False,
        background_index: int = 0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.include_background = bool(include_background)
        self.background_index = int(background_index)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)  # (B,C,H,W)
        C = inputs.shape[1]

        t = targets.clone()
        t[t == self.ignore_index] = self.background_index
        t = torch.clamp(t, 0, C - 1)

        one_hot = F.one_hot(t, num_classes=C).permute(0, 3, 1, 2).float()

        valid = (targets != self.ignore_index).unsqueeze(1).float()
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        tp = (probs * one_hot).sum(dims)
        fp = (probs * (1.0 - one_hot)).sum(dims)
        fn = ((1.0 - probs) * one_hot).sum(dims)

        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss_per_class = 1.0 - ti

        if self.include_background or C == 1:
            return loss_per_class.mean()

        idx = torch.arange(C, device=inputs.device)
        fg_mask = (idx != self.background_index)
        if fg_mask.sum() == 0:
            return loss_per_class.mean()
        return loss_per_class[fg_mask].mean()


class FocalTverskyLoss(nn.Module):
    """
    Focal-Tversky: (1 - TI)^gamma
    """
    def __init__(
        self,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        ignore_index: int = 255,
        eps: float = 1e-6,
        include_background: bool = False,
        background_index: int = 0,
    ):
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.gamma = float(gamma)
        self.ignore_index = int(ignore_index)
        self.eps = float(eps)
        self.include_background = bool(include_background)
        self.background_index = int(background_index)

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        probs = F.softmax(inputs, dim=1)
        C = inputs.shape[1]

        t = targets.clone()
        t[t == self.ignore_index] = self.background_index
        t = torch.clamp(t, 0, C - 1)
        one_hot = F.one_hot(t, num_classes=C).permute(0, 3, 1, 2).float()

        valid = (targets != self.ignore_index).unsqueeze(1).float()
        probs = probs * valid
        one_hot = one_hot * valid

        dims = (0, 2, 3)
        tp = (probs * one_hot).sum(dims)
        fp = (probs * (1.0 - one_hot)).sum(dims)
        fn = ((1.0 - probs) * one_hot).sum(dims)

        ti = (tp + self.eps) / (tp + self.alpha * fp + self.beta * fn + self.eps)
        loss_per_class = torch.pow(1.0 - ti, self.gamma)

        if self.include_background or C == 1:
            return loss_per_class.mean()

        idx = torch.arange(C, device=inputs.device)
        fg_mask = (idx != self.background_index)
        if fg_mask.sum() == 0:
            return loss_per_class.mean()
        return loss_per_class[fg_mask].mean()


# =========================================================
# 3) Combined losses (set good defaults here)
# =========================================================

class CEDiceLoss(nn.Module):
    """
    total_loss = ce_weight * CE + (1 - ce_weight) * Dice

    Defaults:
      - ce_weight=0.35: lean a bit toward overlap for tiny foreground
      - background_weight=0.5: down-weight background CE
      - Dice is foreground-only by default
    """
    def __init__(
        self,
        ce_weight: float = 0.35,
        ignore_index: int = 255,
        dice_eps: float = 1e-6,
        include_background: bool = False,
        background_weight: float = 0.5,
        background_index: int = 0,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.ce = CrossEntropyWithBackgroundWeight(
            ignore_index=ignore_index,
            background_weight=background_weight,
            background_index=background_index,
        )
        self.dice = DiceLoss(
            ignore_index=ignore_index,
            eps=dice_eps,
            include_background=include_background,
            background_index=background_index,
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(inputs, targets) + (1.0 - self.ce_weight) * self.dice(inputs, targets)


class CETverskyLoss(nn.Module):
    """
    total_loss = ce_weight * CE + (1 - ce_weight) * Tversky

    Defaults (catheter-friendly):
      - ce_weight=0.35
      - background_weight=0.5
      - alpha=0.3, beta=0.7 (penalize FN more than FP)
    """
    def __init__(
        self,
        ce_weight: float = 0.35,
        ignore_index: int = 255,
        include_background: bool = False,
        background_weight: float = 0.5,
        background_index: int = 0,
        alpha: float = 0.3,
        beta: float = 0.7,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.ce = CrossEntropyWithBackgroundWeight(
            ignore_index=ignore_index,
            background_weight=background_weight,
            background_index=background_index,
        )
        self.tversky = TverskyLoss(
            alpha=alpha,
            beta=beta,
            ignore_index=ignore_index,
            eps=eps,
            include_background=include_background,
            background_index=background_index,
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(inputs, targets) + (1.0 - self.ce_weight) * self.tversky(inputs, targets)


class CEFocalTverskyLoss(nn.Module):
    """
    total_loss = ce_weight * CE + (1 - ce_weight) * Focal-Tversky

    Defaults:
      - ce_weight=0.35
      - background_weight=0.5
      - alpha=0.3, beta=0.7
      - gamma=1.33 (mild focal)
    """
    def __init__(
        self,
        ce_weight: float = 0.35,
        ignore_index: int = 255,
        include_background: bool = False,
        background_weight: float = 0.5,
        background_index: int = 0,
        alpha: float = 0.3,
        beta: float = 0.7,
        gamma: float = 1.33,
        eps: float = 1e-6,
    ):
        super().__init__()
        self.ce_weight = float(ce_weight)
        self.ce = CrossEntropyWithBackgroundWeight(
            ignore_index=ignore_index,
            background_weight=background_weight,
            background_index=background_index,
        )
        self.ftv = FocalTverskyLoss(
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            ignore_index=ignore_index,
            eps=eps,
            include_background=include_background,
            background_index=background_index,
        )

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return self.ce_weight * self.ce(inputs, targets) + (1.0 - self.ce_weight) * self.ftv(inputs, targets)


# =========================================================
# 4) Builder: you only want combined losses, no kwargs
# =========================================================

def build_segmentation_loss(name: str) -> nn.Module:
    """
    Create a combined segmentation loss by name, with catheter-friendly defaults.

    name in:
      - "ce_dice"
      - "ce_tversky"
      - "ce_focal_tversky"
    """
    name = name.lower().strip()

    if name in ("ce_dice", "cedice"):
        return CEDiceLoss()

    if name in ("ce_tversky", "cetversky"):
        return CETverskyLoss()

    if name in ("ce_focal_tversky", "cefocaltversky", "ce_focal_tversky_loss"):
        return CEFocalTverskyLoss()

    raise ValueError(
        f"Unknown combined segmentation loss: {name}. "
        f"Choose from: ce_dice, ce_tversky, ce_focal_tversky"
    )
