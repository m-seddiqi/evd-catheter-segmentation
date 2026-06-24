from __future__ import annotations

import csv
import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Literal, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.transforms.functional import normalize

from utils.data.loaders import IMAGENET_MEAN, IMAGENET_STD
from utils.data.pipeline import MonoEvalPipeline, MonoPipelineConfig
from utils.metrics import dice, miou_from_stats, nsd_from_points, surface_points


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
MASK_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class FolderSample:
    scene_id: str
    frame_id: str
    img_path: Path
    mask_path: Path | None


@dataclass(frozen=True)
class FolderInferenceSettings:
    crop_sizes: tuple[tuple[int, int], ...]
    num_classes: int = 2
    img_pad_value: int = 0
    mask_pad_value: int = 0
    include_background: bool = False
    background_index: int = 0


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_state_dict(path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _image_files(image_dir: Path) -> List[Path]:
    return sorted(p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTS)


def _scene_image_dirs(root: Path) -> List[tuple[str, Path, Path]]:
    if (root / "images").is_dir():
        return [(root.name, root, root / "images")]

    scene_dirs = [
        (p.name, p, p / "images")
        for p in sorted(root.iterdir())
        if p.is_dir() and (p / "images").is_dir()
    ]
    if scene_dirs:
        return scene_dirs

    return [(root.name, root, root)]


def _mask_dir_for(image_dir: Path, scene_dir: Path, mask_dirname: str, mask_root: Path | None) -> Path:
    if mask_root is not None:
        scene_mask_dir = mask_root / scene_dir.name
        if scene_mask_dir.is_dir():
            return scene_mask_dir
        return mask_root
    if (scene_dir / mask_dirname).is_dir():
        return scene_dir / mask_dirname
    if image_dir.name == "images" and (image_dir.parent / mask_dirname).is_dir():
        return image_dir.parent / mask_dirname
    return image_dir / mask_dirname


def _mask_by_stem(mask_dir: Path) -> Dict[str, Path]:
    if not mask_dir.is_dir():
        return {}
    out: Dict[str, Path] = {}
    for p in sorted(mask_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in MASK_EXTS:
            out[p.stem] = p
    return out


def collect_samples(
    image_root: Path,
    *,
    compute_metrics: bool,
    mask_dirname: str,
    mask_root: Path | None,
) -> List[FolderSample]:
    samples: List[FolderSample] = []
    for scene_id, scene_dir, image_dir in _scene_image_dirs(image_root):
        masks = _mask_by_stem(_mask_dir_for(image_dir, scene_dir, mask_dirname, mask_root)) if compute_metrics else {}
        for img_path in _image_files(image_dir):
            mask_path = masks.get(img_path.stem)
            if compute_metrics and mask_path is None:
                raise FileNotFoundError(f"No matching mask for image: {img_path}")
            samples.append(FolderSample(scene_id, img_path.stem, img_path, mask_path))
    if not samples:
        raise RuntimeError(f"No images found in {image_root}")
    return samples


def _pil_to_normalized_tensor(img: Image.Image) -> torch.Tensor:
    arr = torch.from_numpy(np.array(img.convert("RGB")))
    tensor = arr.permute(2, 0, 1).float() / 255.0
    return normalize(tensor, mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _mask_to_labels(mask: Image.Image, target_hw: tuple[int, int]) -> torch.Tensor:
    target_h, target_w = target_hw
    if mask.size != (target_w, target_h):
        mask = mask.resize((target_w, target_h), Image.NEAREST)
    arr = np.array(mask.convert("L"), dtype=np.uint8)
    return torch.from_numpy((arr > 0).astype(np.int64))


def _unletterbox_prediction(pred: torch.Tensor, *, orig_size: tuple[int, int], crop_size: tuple[int, int]) -> torch.Tensor:
    orig_w, orig_h = orig_size
    crop_h, crop_w = crop_size
    scale = min(crop_w / orig_w, crop_h / orig_h)
    new_w = int(round(orig_w * scale))
    new_h = int(round(orig_h * scale))
    pad_left = (crop_w - new_w) // 2
    pad_top = (crop_h - new_h) // 2
    cropped = pred[pad_top : pad_top + new_h, pad_left : pad_left + new_w].float()[None, None]
    restored = F.interpolate(cropped, size=(orig_h, orig_w), mode="nearest")[0, 0]
    return restored.to(torch.uint8)


MetricSpace = Literal["eval", "original"]


def _prepare_batch(
    samples: Sequence[FolderSample],
    pipeline: MonoEvalPipeline,
    *,
    load_eval_masks: bool,
) -> List[tuple[FolderSample, torch.Tensor, Dict[str, Any], torch.Tensor | None]]:
    prepared: List[tuple[FolderSample, torch.Tensor, Dict[str, Any], torch.Tensor | None]] = []
    for sample in samples:
        img = Image.open(sample.img_path).convert("RGB")
        mask = Image.open(sample.mask_path).convert("L") if load_eval_masks and sample.mask_path is not None else None
        meta: Dict[str, Any] = {
            "scene_id": sample.scene_id,
            "frame_id": sample.frame_id,
            "img_path": str(sample.img_path),
            "mask_path": str(sample.mask_path) if sample.mask_path else None,
            "orig_size": img.size,
        }
        out = pipeline(img=img, mask=mask, meta=meta)
        tensor = _pil_to_normalized_tensor(out["img"])
        eval_mask = _mask_to_labels(out["mask"], target_hw=out["img"].size[::-1]) if out["mask"] is not None else None
        prepared.append((sample, tensor, out["meta"], eval_mask))
    return prepared


def _group_prepared_batch(
    prepared: Sequence[tuple[FolderSample, torch.Tensor, Dict[str, Any], torch.Tensor | None]],
) -> List[tuple[List[FolderSample], torch.Tensor, List[Dict[str, Any]], List[torch.Tensor | None]]]:
    groups: Dict[tuple[int, int], List[tuple[FolderSample, torch.Tensor, Dict[str, Any], torch.Tensor | None]]] = {}
    for item in prepared:
        _, tensor, _, _ = item
        key = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        groups.setdefault(key, []).append(item)

    out: List[tuple[List[FolderSample], torch.Tensor, List[Dict[str, Any]], List[torch.Tensor | None]]] = []
    for items in groups.values():
        group_samples = [item[0] for item in items]
        images = torch.stack([item[1] for item in items], dim=0)
        metas = [item[2] for item in items]
        eval_masks = [item[3] for item in items]
        out.append((group_samples, images, metas, eval_masks))
    return out


def _empty_metrics(num_classes: int) -> Dict[str, Any]:
    return {
        "inter": np.zeros((num_classes,), dtype=np.float64),
        "union": np.zeros((num_classes,), dtype=np.float64),
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "nsd": [],
        "n_frames": 0,
    }


def _update_metrics(
    state: Dict[str, Any],
    pred: torch.Tensor,
    gt: torch.Tensor,
    *,
    settings: FolderInferenceSettings,
    surface_stride: int,
    nsd_tolerance_px: float,
) -> None:
    pred_np = pred.to(torch.uint8).cpu().numpy()
    gt_np = gt.to(torch.uint8).cpu().numpy()

    for c in range(settings.num_classes):
        pred_c = pred_np == c
        gt_c = gt_np == c
        state["inter"][c] += float(np.count_nonzero(pred_c & gt_c))
        state["union"][c] += float(np.count_nonzero(pred_c | gt_c))

    pred_fg = pred_np == 1
    gt_fg = gt_np == 1
    state["tp"] += int(np.count_nonzero(pred_fg & gt_fg))
    state["fp"] += int(np.count_nonzero(pred_fg & ~gt_fg))
    state["fn"] += int(np.count_nonzero(~pred_fg & gt_fg))
    state["nsd"].append(
        nsd_from_points(
            surface_points(gt_np == 1, stride=surface_stride),
            surface_points(pred_np == 1, stride=surface_stride),
            tolerance_px=nsd_tolerance_px,
        )
    )
    state["n_frames"] += 1


def _final_metrics(state: Dict[str, Any], settings: FolderInferenceSettings) -> Dict[str, float | int]:
    return {
        "miou": miou_from_stats(
            state["inter"],
            state["union"],
            settings.include_background,
            settings.background_index,
        ),
        "dice": dice(int(state["tp"]), int(state["fp"]), int(state["fn"])),
        "nsd": float(np.mean(state["nsd"])) if state["nsd"] else 0.0,
        "n_frames": int(state["n_frames"]),
    }


def _macro_scene_metrics(per_scene: Sequence[Dict[str, float | int | str]]) -> Dict[str, float | int]:
    if not per_scene:
        return {"miou": 0.0, "dice": 0.0, "nsd": 0.0, "n_frames": 0}
    return {
        "miou": float(np.mean([float(row["miou"]) for row in per_scene])),
        "dice": float(np.mean([float(row["dice"]) for row in per_scene])),
        "nsd": float(np.mean([float(row["nsd"]) for row in per_scene])),
        "n_frames": int(sum(int(row["n_frames"]) for row in per_scene)),
    }


def _write_mask(path: Path, pred: torch.Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((pred.to(torch.uint8).cpu().numpy() * 255), mode="L").save(path)


def _mask_yt3_bottom(images: torch.Tensor, metas: List[Dict[str, Any]]) -> torch.Tensor:
    if not any(str(m.get("scene_id")) == "scene_yt3_003" for m in metas):
        return images
    out = images.clone()
    y0 = int(out.shape[-2] * 0.85)
    for i, meta in enumerate(metas):
        if str(meta.get("scene_id")) == "scene_yt3_003":
            out[i, :, y0:, :] = 0.0
    return out


def _write_metric_files(out_dir: Path, result: Dict[str, Any]) -> None:
    (out_dir / "metrics.json").write_text(json.dumps(result, indent=2))
    with (out_dir / "metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "miou", "dice", "nsd", "n_frames"])
        writer.writeheader()
        writer.writerow({"scene_id": "macro_scene_mean", **result["metrics"]})
        if "micro_metrics" in result:
            writer.writerow({"scene_id": "micro_all_frames", **result["micro_metrics"]})
        writer.writerows(result["per_scene"])


def run_folder_inference(
    *,
    model: torch.nn.Module,
    settings: FolderInferenceSettings,
    samples: Sequence[FolderSample],
    output_dir: Path,
    device: torch.device,
    batch_size: int,
    compute_metrics: bool,
    copy_gt: bool,
    surface_stride: int,
    nsd_tolerance_px: float,
    metadata: Dict[str, Any],
    mask_yt3_bottom: bool = False,
    metric_space: MetricSpace = "original",
) -> Dict[str, Any]:
    if metric_space not in ("eval", "original"):
        raise ValueError(f"metric_space must be 'eval' or 'original', got: {metric_space}")

    output_dir.mkdir(parents=True, exist_ok=True)
    pred_root = output_dir / "pred_masks"
    gt_root = output_dir / "gt_masks"
    pipeline = MonoEvalPipeline(
        MonoPipelineConfig(
            crop_sizes=list(settings.crop_sizes),
            img_pad_value=settings.img_pad_value,
            mask_pad_value=settings.mask_pad_value,
        )
    )

    model.eval()
    total = _empty_metrics(settings.num_classes)
    by_scene: Dict[str, Dict[str, Any]] = {}
    prediction_rows: List[Dict[str, Any]] = []
    batch_size = max(1, int(batch_size))

    with torch.no_grad():
        for start in range(0, len(samples), batch_size):
            batch_samples = samples[start : start + batch_size]
            prepared = _prepare_batch(
                batch_samples,
                pipeline,
                load_eval_masks=bool(compute_metrics and metric_space == "eval"),
            )
            for group_samples, images, metas, eval_masks in _group_prepared_batch(prepared):
                images = images.to(device, non_blocking=True)
                if mask_yt3_bottom:
                    images = _mask_yt3_bottom(images, metas)
                preds = model(images).argmax(dim=1).cpu()

                for pred, meta, sample, eval_mask in zip(preds, metas, group_samples, eval_masks):
                    pred_orig = _unletterbox_prediction(
                        pred,
                        orig_size=tuple(meta["orig_size"]),
                        crop_size=tuple(int(x) for x in meta["crop_size_used"]),
                    )
                    pred_path = pred_root / sample.scene_id / f"{sample.frame_id}.png"
                    _write_mask(pred_path, pred_orig)

                    row = {
                        "scene_id": sample.scene_id,
                        "frame_id": sample.frame_id,
                        "image_path": str(sample.img_path.resolve()),
                        "pred_mask_path": str(pred_path.resolve()),
                        "mask_path": str(sample.mask_path.resolve()) if sample.mask_path else "",
                    }
                    prediction_rows.append(row)

                    if compute_metrics and sample.mask_path is not None:
                        if metric_space == "eval":
                            if eval_mask is None:
                                raise RuntimeError(f"Eval-space mask was not prepared for: {sample.mask_path}")
                            metric_pred = pred
                            metric_mask = eval_mask
                        else:
                            metric_pred = pred_orig
                            metric_mask = _mask_to_labels(Image.open(sample.mask_path), target_hw=tuple(pred_orig.shape))

                        by_scene.setdefault(sample.scene_id, _empty_metrics(settings.num_classes))
                        _update_metrics(
                            total,
                            metric_pred,
                            metric_mask,
                            settings=settings,
                            surface_stride=surface_stride,
                            nsd_tolerance_px=nsd_tolerance_px,
                        )
                        _update_metrics(
                            by_scene[sample.scene_id],
                            metric_pred,
                            metric_mask,
                            settings=settings,
                            surface_stride=surface_stride,
                            nsd_tolerance_px=nsd_tolerance_px,
                        )
                        if copy_gt:
                            gt_dst = gt_root / sample.scene_id / f"{sample.frame_id}.png"
                            gt_dst.parent.mkdir(parents=True, exist_ok=True)
                            shutil.copyfile(sample.mask_path, gt_dst)

    with (output_dir / "predictions.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["scene_id", "frame_id", "image_path", "pred_mask_path", "mask_path"])
        writer.writeheader()
        writer.writerows(prediction_rows)

    result: Dict[str, Any] = {
        "generated_at_utc": now_utc(),
        "output_dir": str(output_dir.resolve()),
        "num_images": len(prediction_rows),
        "compute_metrics": bool(compute_metrics),
        "metric_space": metric_space if compute_metrics else "",
        **metadata,
    }
    if compute_metrics:
        per_scene = [
            {"scene_id": scene_id, **_final_metrics(state, settings)}
            for scene_id, state in sorted(by_scene.items())
        ]
        result["metric_average"] = "macro_scene_mean"
        result["metrics"] = _macro_scene_metrics(per_scene)
        result["micro_metrics"] = _final_metrics(total, settings)
        result["per_scene"] = per_scene
        _write_metric_files(output_dir, result)

    (output_dir / "metadata.json").write_text(json.dumps(result, indent=2))
    return result
