from __future__ import annotations

import logging
from typing import Any, Dict


def setup_logger(log_path: str, verbose: bool = True) -> logging.Logger:
    logger = logging.getLogger("train")
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        fmt = logging.Formatter("%(message)s")

        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.INFO)
        fh.setFormatter(fmt)
        logger.addHandler(fh)

        if verbose:
            ch = logging.StreamHandler()
            ch.setLevel(logging.INFO)
            ch.setFormatter(fmt)
            logger.addHandler(ch)

    return logger


def log_run_header(
    log_fn,
    *,
    run_dir: str,
    device: Any,
    amp: bool,
    resume_enabled: bool,
    resume_from_dir: Any,
    cfg: Dict[str, Any],
    backbone_init_requested: bool,
    backbone_init_effective: bool,
) -> None:
    log_fn("========== RUN ==========")
    log_fn(f"run_dir:     {run_dir}")
    log_fn(f"device:      {device}")
    log_fn(f"amp:         {amp}")
    log_fn(f"resume:      {resume_enabled} ({resume_from_dir})")
    log_fn(f"variant:     {cfg['model']['variant']}")
    log_fn(f"backbone_init_requested: {backbone_init_requested}")
    log_fn(f"backbone_init_effective: {backbone_init_effective}")
    log_fn("=========================\n")


def format_epoch_line(
    *,
    epoch: int,
    epochs: int,
    global_step: int,
    lr: float,
    train_loss: float,
    val_loss: float,
    val_miou: float,
    test_miou: float,
    dt_sec: float,
    evaluated: bool = True,
) -> str:
    eval_part = (
        f"val_loss={val_loss:.4f} "
        f"val_mIoU={val_miou:.4f} "
        f"test_mIoU={test_miou:.4f}"
        if evaluated
        else "eval=skipped"
    )
    return (
        f"Epoch [{epoch:03d}/{epochs:03d}] "
        f"step={global_step} "
        f"lr={lr:.3e} "
        f"train_loss={train_loss:.4f} "
        f"{eval_part} "
        f"time={dt_sec:.1f}s"
    )


def format_test_scene_line(scene_miou: Dict[str, float]) -> str:
    if not scene_miou:
        return "test_scene_mIoU: <empty>"
    parts = [f"{scene}={miou:.4f}" for scene, miou in sorted(scene_miou.items())]
    return "test_scene_mIoU: " + " | ".join(parts)


def get_lr(optimizer) -> float:
    return float(optimizer.param_groups[0]["lr"])
