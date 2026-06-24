# utils/schedulers.py
from __future__ import annotations

from typing import Any, Dict, Optional
import torch


def _get_warmup_steps(cfg: Dict[str, Any]) -> int:
    sched_cfg = cfg.get("scheduler", {}) or {}
    w = sched_cfg.get("warmup_steps", 0)
    if w is None:
        return 0
    w = int(w)
    if w < 0:
        raise ValueError(f"warmup_steps must be >= 0, got {w}")
    return w


class WarmupPolyLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Step-based polynomial LR with optional linear warmup.

    total_steps = max_steps (including warmup)
    warmup: steps [0 .. warmup_steps-1]
      lr = base_lr * (step+1)/warmup_steps
    poly: steps [warmup_steps .. max_steps]
      lr = base_lr * (1 - t/T) ** power
      where t = step - warmup_steps, T = max_steps - warmup_steps
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        power: float = 0.9,
        min_lr: float = 0.0,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")
        self.power = float(power)
        self.min_lr = float(min_lr)
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        if self.warmup_steps >= self.max_steps:
            # allow but degenerate: all warmup, no decay
            pass
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = int(self.last_epoch)
        step = min(step, self.max_steps)

        # warmup phase
        if self.warmup_steps > 0 and step < self.warmup_steps:
            frac = float(step + 1) / float(self.warmup_steps)
            return [max(self.min_lr, base_lr * frac) for base_lr in self.base_lrs]

        # decay phase
        denom = max(1, self.max_steps - self.warmup_steps)
        t = max(0, step - self.warmup_steps)
        t = min(t, denom)
        factor = (1.0 - t / denom) ** self.power
        return [max(self.min_lr, base_lr * factor) for base_lr in self.base_lrs]


class WarmupCosineLR(torch.optim.lr_scheduler._LRScheduler):
    """
    Step-based cosine annealing with optional linear warmup.
    total_steps = max_steps (including warmup)
    warmup: lr = base_lr * (step+1)/warmup_steps
    cosine: CosineAnnealing from base_lr down to eta_min over (max_steps - warmup_steps)
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        max_steps: int,
        eta_min: float = 1e-6,
        warmup_steps: int = 0,
        last_epoch: int = -1,
    ):
        self.max_steps = int(max_steps)
        if self.max_steps <= 0:
            raise ValueError(f"max_steps must be > 0, got {self.max_steps}")
        self.eta_min = float(eta_min)
        self.warmup_steps = int(warmup_steps)
        if self.warmup_steps < 0:
            raise ValueError(f"warmup_steps must be >= 0, got {self.warmup_steps}")
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        step = int(self.last_epoch)
        step = min(step, self.max_steps)

        if self.warmup_steps > 0 and step < self.warmup_steps:
            frac = float(step + 1) / float(self.warmup_steps)
            return [base_lr * frac for base_lr in self.base_lrs]

        # cosine over remaining steps
        T = max(1, self.max_steps - self.warmup_steps)
        t = max(0, step - self.warmup_steps)
        t = min(t, T)

        # classic cosine annealing formula
        out = []
        for base_lr in self.base_lrs:
            lr = self.eta_min + (base_lr - self.eta_min) * (1.0 + torch.cos(torch.tensor(torch.pi * t / T))).item() / 2.0
            out.append(lr)
        return out


def _resolve_max_steps(cfg: Dict[str, Any], steps_per_epoch: int) -> int:
    sched_cfg = cfg.get("scheduler", {}) or {}
    if "max_steps" in sched_cfg and sched_cfg["max_steps"] is not None:
        max_steps = int(sched_cfg["max_steps"])
    else:
        epochs = int(cfg["train"]["epochs"])
        max_steps = epochs * int(steps_per_epoch)
    if max_steps <= 0:
        raise ValueError(f"Resolved max_steps must be > 0, got {max_steps}")
    return max_steps


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    cfg: Dict[str, Any],
    *,
    steps_per_epoch: int,
    warmup_steps_override: Optional[int] = None,  # NEW
) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
    sched_cfg = cfg.get("scheduler", None)
    if not sched_cfg:
        max_steps = int(cfg["train"]["epochs"]) * int(steps_per_epoch)
        # no cfg: default cosine (no warmup)
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max_steps,
            eta_min=1e-6,
        )

    name = str(sched_cfg.get("name", "cosine")).lower().strip()
    if name in ("none", "off", "disabled"):
        return None

    max_steps = _resolve_max_steps(cfg, steps_per_epoch)

    warmup_steps = int(_get_warmup_steps(cfg) if warmup_steps_override is None else warmup_steps_override)

    if name in ("cosine", "cosineannealing"):
        eta_min = float(sched_cfg.get("eta_min", 1e-6))
        # use warmup-aware scheduler
        return WarmupCosineLR(
            optimizer=optimizer,
            max_steps=max_steps,
            eta_min=eta_min,
            warmup_steps=warmup_steps,
        )

    if name in ("poly", "polynomial", "polylr"):
        power = float(sched_cfg.get("power", 0.9))
        min_lr = float(sched_cfg.get("min_lr", 0.0))
        return WarmupPolyLR(
            optimizer=optimizer,
            max_steps=max_steps,
            power=power,
            min_lr=min_lr,
            warmup_steps=warmup_steps,
        )

    raise ValueError(f"Unknown scheduler name: {name}")
