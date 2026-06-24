# utils/data/loaders.py
from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import Any, Dict, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.transforms.functional import normalize
from PIL import Image

from utils.data.dataset import CatheterMonoDataset
from utils.data.indexing import MonoSample
from utils.data.pipeline import MonoTrainPipeline, MonoEvalPipeline


# -----------------------------
# NORMALIZATION CONSTANTS
# -----------------------------
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def mask_pil_to_labels(mask_pil: Image.Image) -> torch.Tensor:
    """Converts mask PIL {0, 255} to LongTensor {0, 1}."""
    m = np.array(mask_pil, dtype=np.uint8)   # (H,W) in {0,255}
    m = (m > 0).astype(np.int64)             # (H,W) in {0,1}
    return torch.from_numpy(m)


def _fit_line_prior_from_mask(mask: torch.Tensor) -> Dict[str, Any]:
    """
    Fit a principal axis line from foreground pixels in a single label mask.
    Returns line point/dir and GT normal-distance moments.
    """
    fg = (mask == 1)
    ys, xs = torch.where(fg)
    if xs.numel() < 2:
        return {
            "valid": False,
            "point": torch.zeros(2, dtype=torch.float32),
            "dir": torch.tensor([1.0, 0.0], dtype=torch.float32),
            "mean_dist": torch.tensor(0.0, dtype=torch.float32),
            "std_dist": torch.tensor(0.0, dtype=torch.float32),
        }

    coords = torch.stack([xs.float(), ys.float()], dim=1)  # (N,2)
    center = coords.mean(dim=0)
    centered = coords - center[None, :]
    cov = (centered.t() @ centered) / max(1, int(coords.shape[0] - 1))
    eigvals, eigvecs = torch.linalg.eigh(cov)
    direction = eigvecs[:, int(torch.argmax(eigvals))]
    direction = direction / torch.clamp(torch.linalg.norm(direction), min=1e-6)

    nx = -direction[1]
    ny = direction[0]
    d = torch.abs((coords[:, 0] - center[0]) * nx + (coords[:, 1] - center[1]) * ny)

    return {
        "valid": True,
        "point": center.to(torch.float32),
        "dir": direction.to(torch.float32),
        "mean_dist": d.mean().to(torch.float32),
        "std_dist": d.std(unbiased=False).to(torch.float32),
    }


# ============================================================
# Robust conversions: PIL/numpy/torch -> torch tensors
# ============================================================

def _is_pil(x: Any) -> bool:
    return x.__class__.__module__.startswith("PIL.")


def _to_chw_float_tensor(img: Any) -> torch.Tensor:
    """
    Accepts PIL/numpy/torch.
    Returns torch.FloatTensor (C,H,W) scaled to [0, 1].
    """
    if isinstance(img, torch.Tensor):
        t = img
    elif isinstance(img, np.ndarray):
        t = torch.from_numpy(img)
    elif _is_pil(img):
        t = torch.from_numpy(np.array(img))
    else:
        raise TypeError(f"Unsupported image type: {type(img)}")

    if t.ndim == 2:
        t = t.unsqueeze(0)
    elif t.ndim == 3:
        # if HWC, convert -> CHW
        if t.shape[-1] in (1, 3, 4) and t.shape[0] not in (1, 3, 4):
            t = t.permute(2, 0, 1)

    # Scale to [0, 1] if uint8
    if t.dtype == torch.uint8:
        t = t.float() / 255.0
    elif t.dtype != torch.float32:
        t = t.float()

    return t.contiguous()


# ============================================================
# Collate (MONO)
# ============================================================

def collate_mono(
    batch: List[Dict[str, Any]],
    *,
    compute_line_prior: bool = True,
) -> Dict[str, Any]:
    # dataset/pipeline returns "img" + "mask" + "meta"
    image_list = [_to_chw_float_tensor(b["img"]) for b in batch]
    mask_list = [mask_pil_to_labels(b["mask"]) for b in batch]

    max_h = max(int(x.shape[-2]) for x in image_list)
    max_w = max(int(x.shape[-1]) for x in image_list)

    # Mixed crop buckets can produce variable H/W in the same batch.
    # Pad to per-batch max size so tensors can be stacked.
    padded_images = []
    for img in image_list:
        ph = max_h - int(img.shape[-2])
        pw = max_w - int(img.shape[-1])
        if ph > 0 or pw > 0:
            img = F.pad(img, (0, pw, 0, ph), mode="constant", value=0.0)
        padded_images.append(img)
    images = torch.stack(padded_images, dim=0)

    # Apply ImageNet normalization
    if images.shape[1] == 3:
        images = normalize(images, mean=IMAGENET_MEAN, std=IMAGENET_STD)

    # Pad mask with ignore_index=255 so padded pixels are excluded from loss/metrics.
    padded_masks = []
    for m in mask_list:
        ph = max_h - int(m.shape[-2])
        pw = max_w - int(m.shape[-1])
        if ph > 0 or pw > 0:
            m = F.pad(m, (0, pw, 0, ph), mode="constant", value=255)
        padded_masks.append(m)
    masks = torch.stack(padded_masks, dim=0)

    line_prior = None
    if compute_line_prior:
        priors = [_fit_line_prior_from_mask(m) for m in padded_masks]
        line_prior = {
            "valid": torch.tensor([bool(p["valid"]) for p in priors], dtype=torch.bool),
            "point": torch.stack([p["point"] for p in priors], dim=0),        # (B,2) [x,y]
            "dir": torch.stack([p["dir"] for p in priors], dim=0),            # (B,2) [vx,vy]
            "mean_dist": torch.stack([p["mean_dist"] for p in priors], dim=0),# (B,)
            "std_dist": torch.stack([p["std_dist"] for p in priors], dim=0),  # (B,)
        }

    return {
        "image": images,   # keep training contract: "image"
        "mask": masks,
        "line_prior": line_prior,
        "meta": [b.get("meta", {}) for b in batch],
    }


# ============================================================
# Public helpers (MONO)
# ============================================================

@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 8
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True
    drop_last_train: bool = True


def make_mono_dataset(
    samples: Sequence[MonoSample],
    *,
    pipeline: MonoTrainPipeline | MonoEvalPipeline,
    return_paths: bool = False,
) -> CatheterMonoDataset:
    return CatheterMonoDataset(samples, pipeline=pipeline, return_paths=return_paths)


def make_mono_loader(
    ds: CatheterMonoDataset,
    *,
    cfg: LoaderConfig,
    shuffle: bool,
    drop_last: bool,
    compute_line_prior: bool,
) -> DataLoader:
    return DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
        persistent_workers=(cfg.persistent_workers and cfg.num_workers > 0),
        collate_fn=partial(collate_mono, compute_line_prior=compute_line_prior),
    )


def build_train_val_loaders(
    train_samples: Sequence[MonoSample],
    val_samples: Sequence[MonoSample],
    *,
    train_pipeline: MonoTrainPipeline,
    eval_pipeline: MonoEvalPipeline,
    cfg: LoaderConfig = LoaderConfig(),
    return_paths: bool = False,
    compute_line_prior_train: bool = True,
    compute_line_prior_eval: bool = False,
) -> Dict[str, DataLoader]:
    loaders = build_train_val_test_loaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=[],
        train_pipeline=train_pipeline,
        eval_pipeline=eval_pipeline,
        cfg=cfg,
        return_paths=return_paths,
        compute_line_prior_train=compute_line_prior_train,
        compute_line_prior_eval=compute_line_prior_eval,
    )
    return {"train": loaders["train"], "val": loaders["val"]}


def build_train_val_test_loaders(
    train_samples: Sequence[MonoSample],
    val_samples: Sequence[MonoSample],
    test_samples: Sequence[MonoSample],
    *,
    train_pipeline: MonoTrainPipeline,
    eval_pipeline: MonoEvalPipeline,
    cfg: LoaderConfig = LoaderConfig(),
    return_paths: bool = False,
    compute_line_prior_train: bool = True,
    compute_line_prior_eval: bool = False,
) -> Dict[str, DataLoader]:
    train_ds = make_mono_dataset(train_samples, pipeline=train_pipeline, return_paths=return_paths)
    val_ds = make_mono_dataset(val_samples, pipeline=eval_pipeline, return_paths=return_paths)
    test_ds = make_mono_dataset(test_samples, pipeline=eval_pipeline, return_paths=return_paths)

    return {
        "train": make_mono_loader(
            train_ds,
            cfg=cfg,
            shuffle=True,
            drop_last=cfg.drop_last_train,
            compute_line_prior=compute_line_prior_train,
        ),
        "val": make_mono_loader(
            val_ds,
            cfg=cfg,
            shuffle=False,
            drop_last=False,
            compute_line_prior=compute_line_prior_eval,
        ),
        "test": make_mono_loader(
            test_ds,
            cfg=cfg,
            shuffle=False,
            drop_last=False,
            compute_line_prior=compute_line_prior_eval,
        ),
    }
