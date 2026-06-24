# utils/checkpointing.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional, Tuple

import torch

TRAIN_STATE_FNAME = "last_train.pt"
BEST_WEIGHTS_FNAME = "best_model.pth"
EFFECTIVE_CFG_FNAME = "effective_config.json"


def train_state_path(run_dir: str) -> str:
    return os.path.join(run_dir, TRAIN_STATE_FNAME)


def best_weights_path(run_dir: str) -> str:
    return os.path.join(run_dir, BEST_WEIGHTS_FNAME)


def save_train_state(
    run_dir: str,
    *,
    epoch: int,
    global_step: int,                 # NEW
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    cfg: Dict[str, Any],
    val_miou: float,
    per_class_iou: Any,
    test_miou_total: Optional[float] = None,
    test_scene_miou: Optional[Dict[str, float]] = None,
) -> None:
    payload: Dict[str, Any] = {
        "checkpoint_version": 2,      # bumped
        "epoch": int(epoch),
        "global_step": int(global_step),  # NEW
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "cfg": cfg,
        "val_miou": float(val_miou),
        "per_class_iou": per_class_iou,
    }
    if test_miou_total is not None:
        payload["test_miou_total"] = float(test_miou_total)
    if test_scene_miou is not None:
        payload["test_scene_miou"] = dict(test_scene_miou)
    if scheduler is not None:
        payload["scheduler"] = scheduler.state_dict()
        payload["scheduler_name"] = type(scheduler).__name__
    if scaler is not None:
        payload["scaler"] = scaler.state_dict()

    torch.save(payload, train_state_path(run_dir))


def load_train_state(
    from_dir: str,
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: Optional[Any],
    map_location: str = "cpu",
    steps_per_epoch: Optional[int] = None,   # NEW: fallback only
) -> Tuple[int, int, Dict[str, Any]]:        # NEW: returns global_step
    path = train_state_path(from_dir)
    if not os.path.exists(path):
        raise FileNotFoundError(f"Resume enabled but train state not found: {path}")

    ckpt = torch.load(path, map_location=map_location)
    if "model" not in ckpt or "optimizer" not in ckpt or "epoch" not in ckpt:
        raise ValueError(f"Invalid train checkpoint format: {path}")

    model.load_state_dict(ckpt["model"], strict=True)
    optimizer.load_state_dict(ckpt["optimizer"])

    if scheduler is not None and "scheduler" in ckpt:
        scheduler.load_state_dict(ckpt["scheduler"])

    if scaler is not None and "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])

    start_epoch = int(ckpt["epoch"]) + 1

    if "global_step" in ckpt:
        global_step = int(ckpt["global_step"])
    else:
        # fallback for older checkpoints
        if steps_per_epoch is None:
            raise ValueError("Old checkpoint has no global_step; pass steps_per_epoch to load_train_state().")
        global_step = int(ckpt["epoch"]) * int(steps_per_epoch)

    return start_epoch, global_step, ckpt


def save_best_weights(run_dir: str, model: torch.nn.Module) -> None:
    torch.save(model.state_dict(), best_weights_path(run_dir))

def load_best_weights(run_dir, model, map_location=None, strict=True):
    model.load_state_dict(
        torch.load(best_weights_path(run_dir), map_location=map_location),
        strict=strict,
    )
