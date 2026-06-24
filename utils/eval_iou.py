# utils/seg/eval_iou.py
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import torch


def _get_logits(model_out: Any) -> torch.Tensor:
    # torchvision-style: {"out": logits}
    if isinstance(model_out, dict):
        if "out" in model_out:
            return model_out["out"]
        if len(model_out) == 1:
            return next(iter(model_out.values()))
        raise KeyError(f"Model output dict missing 'out'. Keys={list(model_out.keys())}")
    if torch.is_tensor(model_out):
        return model_out
    raise TypeError(f"Unexpected model output type: {type(model_out)}")


def _compute_loss(
    criterion: callable,
    logits: torch.Tensor,
    masks: torch.Tensor,
    batch: Dict[str, Any],
) -> torch.Tensor:
    if bool(getattr(criterion, "uses_line_prior", False)):
        return criterion(logits, masks, line_prior=batch.get("line_prior"))
    return criterion(logits, masks)


@torch.no_grad()
def evaluate_iou_mono(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    num_classes: int,
    criterion: Optional[callable] = None,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    background_index: int = 0,
) -> Tuple[Optional[float], float, List[float]]:
    """
    Mono loader batch format:
      batch["image"]: (B,C,H,W)
      batch["mask"]:  (B,H,W)
      batch["meta"]:  list[dict] (unused here)

    Returns:
      avg_loss (or None), mean_iou, per_class_iou
    """
    model.eval()
    compute_loss = criterion is not None
    running_loss = 0.0
    counted_samples = 0

    total_intersection = torch.zeros(num_classes, dtype=torch.float64)
    total_union        = torch.zeros(num_classes, dtype=torch.float64)

    eps = 1e-6

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks  = batch["mask"].to(device, non_blocking=True)

        logits = _get_logits(model(images))

        if compute_loss:
            loss = _compute_loss(criterion, logits, masks, batch)
            bs = int(images.size(0))
            running_loss += float(loss.item()) * bs
            counted_samples += bs

        preds = logits.argmax(dim=1)

        if ignore_index is not None:
            valid = (masks != ignore_index)
            preds = preds[valid]
            masks = masks[valid]
            if masks.numel() == 0:
                continue
        else:
            preds = preds.reshape(-1)
            masks = masks.reshape(-1)

        for c in range(num_classes):
            pred_c = (preds == c)
            gt_c   = (masks == c)

            inter = (pred_c & gt_c).sum().item()
            union = (pred_c | gt_c).sum().item()

            total_intersection[c] += inter
            total_union[c]        += union

    per_class_iou = (total_intersection / (total_union + eps)).tolist()

    valid = (total_union > 0)
    if not include_background:
        valid[background_index] = False

    mean_iou = (
        total_intersection[valid] / (total_union[valid] + eps)
    ).mean().item() if valid.any() else 0.0

    avg_loss = (running_loss / max(counted_samples, 1)) if compute_loss else None
    return avg_loss, mean_iou, per_class_iou


def _mean_iou_from_stats(
    intersection: torch.Tensor,
    union: torch.Tensor,
    *,
    include_background: bool,
    background_index: int,
    eps: float = 1e-6,
) -> float:
    valid = (union > 0)
    if not include_background and 0 <= background_index < int(valid.numel()):
        valid[background_index] = False
    if not valid.any():
        return 0.0
    return (intersection[valid] / (union[valid] + eps)).mean().item()


@torch.no_grad()
def evaluate_iou_mono_by_scene(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    *,
    num_classes: int,
    criterion: Optional[callable] = None,
    ignore_index: Optional[int] = None,
    include_background: bool = True,
    background_index: int = 0,
) -> Tuple[Optional[float], float, List[float], Dict[str, float]]:
    """
    Same as evaluate_iou_mono, plus per-scene mIoU from batch["meta"][i]["scene_id"].
    Returns:
      avg_loss, total_mean_iou, total_per_class_iou, scene_miou
    """
    model.eval()
    compute_loss = criterion is not None
    running_loss = 0.0
    counted_samples = 0
    eps = 1e-6

    total_intersection = torch.zeros(num_classes, dtype=torch.float64)
    total_union = torch.zeros(num_classes, dtype=torch.float64)

    scene_intersection: Dict[str, torch.Tensor] = {}
    scene_union: Dict[str, torch.Tensor] = {}

    for batch in loader:
        images = batch["image"].to(device, non_blocking=True)
        masks = batch["mask"].to(device, non_blocking=True)
        metas = batch.get("meta", [{} for _ in range(int(images.size(0)))])

        logits = _get_logits(model(images))

        if compute_loss:
            loss = _compute_loss(criterion, logits, masks, batch)
            bs = int(images.size(0))
            running_loss += float(loss.item()) * bs
            counted_samples += bs

        preds = logits.argmax(dim=1)
        bs = int(preds.size(0))

        for i in range(bs):
            scene_id = str(metas[i].get("scene_id", "unknown"))
            if scene_id not in scene_intersection:
                scene_intersection[scene_id] = torch.zeros(num_classes, dtype=torch.float64)
                scene_union[scene_id] = torch.zeros(num_classes, dtype=torch.float64)

            pred_i = preds[i]
            mask_i = masks[i]

            if ignore_index is not None:
                valid = (mask_i != ignore_index)
                pred_i = pred_i[valid]
                mask_i = mask_i[valid]
                if mask_i.numel() == 0:
                    continue
            else:
                pred_i = pred_i.reshape(-1)
                mask_i = mask_i.reshape(-1)

            for c in range(num_classes):
                pred_c = (pred_i == c)
                gt_c = (mask_i == c)
                inter = (pred_c & gt_c).sum().item()
                uni = (pred_c | gt_c).sum().item()

                total_intersection[c] += inter
                total_union[c] += uni
                scene_intersection[scene_id][c] += inter
                scene_union[scene_id][c] += uni

    total_per_class_iou = (total_intersection / (total_union + eps)).tolist()
    total_miou = _mean_iou_from_stats(
        total_intersection,
        total_union,
        include_background=include_background,
        background_index=background_index,
        eps=eps,
    )

    scene_miou: Dict[str, float] = {}
    for scene_id in sorted(scene_intersection):
        scene_miou[scene_id] = _mean_iou_from_stats(
            scene_intersection[scene_id],
            scene_union[scene_id],
            include_background=include_background,
            background_index=background_index,
            eps=eps,
        )

    avg_loss = (running_loss / max(counted_samples, 1)) if compute_loss else None
    return avg_loss, total_miou, total_per_class_iou, scene_miou


