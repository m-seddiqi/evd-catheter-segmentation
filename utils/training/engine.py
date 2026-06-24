from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.optim as optim

from utils.eval_iou import evaluate_iou_mono, evaluate_iou_mono_by_scene


def _unwrap_logits(model_out: Any) -> Any:
    """
    For training loss: accept logits tensor OR dict like {"out": logits} / {"output_0": logits}.
    Returns the logits tensor or the single dict value.
    """
    if isinstance(model_out, dict):
        if "out" in model_out:
            return model_out["out"]
        if len(model_out) == 1:
            return next(iter(model_out.values()))
        raise KeyError(f"Model output dict missing 'out'. Keys={list(model_out.keys())}")
    return model_out


def _compute_loss(
    criterion: nn.Module,
    logits: torch.Tensor,
    mask: torch.Tensor,
    batch: Optional[Dict[str, Any]] = None,
) -> torch.Tensor:
    if bool(getattr(criterion, "uses_line_prior", False)):
        line_prior = None if batch is None else batch.get("line_prior")
        return criterion(logits, mask, line_prior=line_prior)
    return criterion(logits, mask)


def _grads_are_finite(model: nn.Module) -> bool:
    for param in model.parameters():
        if param.grad is not None and not torch.isfinite(param.grad).all():
            return False
    return True


@dataclass
class EpochMetrics:
    train_loss: float
    val_loss: float
    val_miou: float
    val_per_class_iou: Any
    val_component_miou: Dict[str, float]
    test_miou: float
    test_per_class_iou: Any
    test_scene_miou: Dict[str, float]
    evaluated: bool
    dt_sec: float


class SegmentationTrainer:
    """
    Owns the per-epoch logic only (MONO):
      - train_one_epoch
      - validate
    No checkpointing or 'best' logic inside.
    """
    def __init__(
        self,
        *,
        cfg: Dict[str, Any],
        model: nn.Module,
        train_loader: Any,
        val_loader: Any,
        test_loader: Any,
        optimizer: optim.Optimizer,
        criterion: nn.Module,
        scheduler: Optional[Any],
        device: torch.device,
        amp_setup: Tuple[bool, Any, Any, dict],
        real_val_loader: Any = None,
        synthetic_val_loader: Any = None,
        val_real_weight: Optional[float] = None,
    ):
        self.cfg = cfg
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.real_val_loader = real_val_loader
        self.synthetic_val_loader = synthetic_val_loader
        self.val_real_weight = None if val_real_weight is None else max(0.0, min(1.0, float(val_real_weight)))
        self.optimizer = optimizer
        self.criterion = criterion
        self.scheduler = scheduler
        self.device = device

        # AMP setup
        self.amp_enabled, self.scaler, self.autocast_ctx, self.autocast_kwargs = amp_setup

        # eval params
        self.num_classes = int(cfg["model"]["num_classes"])
        self.ignore_index = int(cfg["loss"]["ignore_index"])
        self.include_background = bool(cfg["loss"]["eval_include_background"])
        self.background_index = int(cfg["loss"]["background_index"])
        self.compute_val_loss = bool((cfg.get("eval", {}) or {}).get("compute_val_loss", False))

    def train_one_epoch(self, *, epoch: int, global_step: int) -> Tuple[float, int]:
        self.model.train()
        running_loss = 0.0
        n_samples = 0

        for batch in self.train_loader:
            self.optimizer.zero_grad(set_to_none=True)

            img = batch["image"].to(self.device, non_blocking=True)
            mask = batch["mask"].to(self.device, non_blocking=True)
            batch_size = int(img.size(0))

            with self.autocast_ctx(enabled=self.amp_enabled, **self.autocast_kwargs):
                out = self.model(img)
                logits = _unwrap_logits(out)
                loss = _compute_loss(self.criterion, logits, mask, batch=batch)

                # safety: if criterion returns per-sample/per-pixel tensor
                if getattr(loss, "ndim", 0) > 0:
                    loss = loss.mean()

            if not torch.isfinite(loss):
                # Skip numerically bad batches so running averages and weights stay valid.
                continue

            if self.amp_enabled:
                scale_before = self.scaler.get_scale()
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                optimizer_stepped = self.scaler.get_scale() >= scale_before
            else:
                loss.backward()
                if not _grads_are_finite(self.model):
                    self.optimizer.zero_grad(set_to_none=True)
                    continue
                self.optimizer.step()
                optimizer_stepped = True

            if self.scheduler is not None and optimizer_stepped:
                self.scheduler.step()

            global_step += 1
            running_loss += float(loss.item()) * batch_size
            n_samples += batch_size

        return running_loss / max(1, n_samples), global_step

    @torch.no_grad()
    def validate(self):
        criterion = self.criterion if self.compute_val_loss else None
        if self.real_val_loader is not None and self.synthetic_val_loader is not None:
            real_loss, real_miou, real_per_class = evaluate_iou_mono(
                model=self.model,
                loader=self.real_val_loader,
                device=self.device,
                num_classes=self.num_classes,
                criterion=criterion,
                ignore_index=self.ignore_index,
                include_background=self.include_background,
                background_index=self.background_index,
            )
            synthetic_loss, synthetic_miou, synthetic_per_class = evaluate_iou_mono(
                model=self.model,
                loader=self.synthetic_val_loader,
                device=self.device,
                num_classes=self.num_classes,
                criterion=criterion,
                ignore_index=self.ignore_index,
                include_background=self.include_background,
                background_index=self.background_index,
            )
            real_weight = 0.5 if self.val_real_weight is None else self.val_real_weight
            synthetic_weight = 1.0 - real_weight
            val_miou = real_weight * float(real_miou) + synthetic_weight * float(synthetic_miou)
            val_loss = None
            if real_loss is not None and synthetic_loss is not None:
                val_loss = real_weight * float(real_loss) + synthetic_weight * float(synthetic_loss)
            per_class = [
                real_weight * float(r) + synthetic_weight * float(s)
                for r, s in zip(real_per_class, synthetic_per_class)
            ]
            components = {
                "real": float(real_miou),
                "synthetic": float(synthetic_miou),
                "real_weight": float(real_weight),
                "synthetic_weight": float(synthetic_weight),
            }
            return val_loss, val_miou, per_class, components

        val_loss, val_miou, val_per_class = evaluate_iou_mono(
            model=self.model,
            loader=self.val_loader,
            device=self.device,
            num_classes=self.num_classes,
            criterion=criterion,
            ignore_index=self.ignore_index,
            include_background=self.include_background,
            background_index=self.background_index,
        )
        return val_loss, val_miou, val_per_class, {}

    @torch.no_grad()
    def test(self):
        _, test_miou, test_per_class, test_scene_miou = evaluate_iou_mono_by_scene(
            model=self.model,
            loader=self.test_loader,
            device=self.device,
            num_classes=self.num_classes,
            criterion=None,
            ignore_index=self.ignore_index,
            include_background=self.include_background,
            background_index=self.background_index,
        )
        return test_miou, test_per_class, test_scene_miou

    def run_epoch(self, *, epoch: int, global_step: int, evaluate: bool = True) -> Tuple[EpochMetrics, int]:
        t0 = time.time()

        train_loss, global_step = self.train_one_epoch(epoch=epoch, global_step=global_step)
        if evaluate:
            val_loss, val_miou, val_per_class, val_component_miou = self.validate()
            test_miou, test_per_class, test_scene_miou = self.test()
        else:
            val_loss = float("nan")
            val_miou = float("nan")
            val_per_class = None
            val_component_miou = {}
            test_miou = float("nan")
            test_per_class = None
            test_scene_miou = {}

        dt = time.time() - t0
        return EpochMetrics(
            train_loss=float(train_loss),
            val_loss=float(val_loss if val_loss is not None else 0.0),
            val_miou=float(val_miou),
            val_per_class_iou=val_per_class,
            val_component_miou=val_component_miou,
            test_miou=float(test_miou),
            test_per_class_iou=test_per_class,
            test_scene_miou=test_scene_miou,
            evaluated=bool(evaluate),
            dt_sec=float(dt),
        ), global_step
