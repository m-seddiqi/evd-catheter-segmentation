from __future__ import annotations

from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import re


# ----------------------------
# Data model (MONO)
# ----------------------------

@dataclass(frozen=True)
class MonoSample:
    video_dir: Path
    scene_id: str          # e.g. "scene_001"
    frame_id: str          # basename without extension, e.g. "000123"

    img: Path
    mask: Path

    def __iter__(self) -> Iterator[Tuple[str, Any]]:
        for f in fields(self):
            yield f.name, getattr(self, f.name)


# ----------------------------
# Helpers
# ----------------------------

_SCENE_NAME_RE = re.compile(r"^scene_.+")
# RGB inputs may be stored as JPEG to reduce synthetic dataset size.
_IMAGE_EXTS = ("*.jpg", "*.jpeg", "*.png")
# Masks are label images and should stay lossless.
_MASK_EXTS = ("*.png",)

def _is_scene_dir(p: Path) -> bool:
    return p.is_dir() and _SCENE_NAME_RE.match(p.name) is not None

def _list_by_patterns(root: Path, patterns: Sequence[str]) -> List[Path]:
    paths: List[Path] = []
    for pat in patterns:
        paths.extend(root.glob(pat))
    return sorted(paths)

def _index_by_stem(paths: Iterable[Path]) -> Dict[str, Path]:
    out: Dict[str, Path] = {}
    for p in paths:
        out[p.stem] = p
    return out


# ----------------------------
# Core scanning (MONO, scene-based only)
# ----------------------------

def _scan_scene_dir(
    scene_dir: Path,
    *,
    mask_subdir: str = "masks",
) -> List[MonoSample]:
    """
    Scan a single scene directory:
      scene_*/images/*
      scene_*/<mask_subdir>/*
    Pairs by common filename stem.
    """
    img_dir = scene_dir / "images"
    mask_dir = scene_dir / mask_subdir
    if not (img_dir.is_dir() and mask_dir.is_dir()):
        return []

    imgs = _index_by_stem(_list_by_patterns(img_dir, _IMAGE_EXTS))
    masks = _index_by_stem(_list_by_patterns(mask_dir, _MASK_EXTS))
    keys = sorted(set(imgs) & set(masks))

    return [
        MonoSample(
            video_dir=scene_dir.parent,
            scene_id=scene_dir.name,
            frame_id=frame_id,
            img=imgs[frame_id],
            mask=masks[frame_id],
        )
        for frame_id in keys
    ]


def scan_video_dir(video_dir: Path, *, mask_subdir: str = "masks") -> List[MonoSample]:
    """
    Scan one scene folder with:
      scene_*/images
      scene_*/<mask_subdir>
    """
    return _scan_scene_dir(video_dir, mask_subdir=mask_subdir)


def build_splits(
    root_dir: str | Path,
    *,
    val_scene_names: Optional[Sequence[str]] = None,
) -> Tuple[List[MonoSample], List[MonoSample]]:
    """
    Build train/val samples from scene-centric layout only.
    """
    return build_splits_custom(
        root_dir=root_dir,
        mask_subdir="masks",
        val_scene_names=val_scene_names,
        val_fraction=None,
        split_seed=42,
    )


def build_splits_custom(
    root_dir: str | Path,
    *,
    mask_subdir: str = "masks",
    val_scene_names: Optional[Sequence[str]] = None,
    val_fraction: Optional[float] = None,
    split_seed: int = 42,
) -> Tuple[List[MonoSample], List[MonoSample]]:
    """
    Build train/val samples from scene-centric layout.

    Split policy (in order):
      1) val_scene_names if provided and non-empty
      2) val_fraction if provided (>0)
      3) fallback to no validation split
    """
    root = Path(root_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"Root not found: {root}")

    all_scene_dirs = sorted([p for p in root.iterdir() if _is_scene_dir(p) and (p / "images").is_dir()])

    val_scene_set = set(val_scene_names or [])
    if val_scene_set:
        val_scene_dirs = [p for p in all_scene_dirs if p.name in val_scene_set]
        train_scene_dirs = [p for p in all_scene_dirs if p.name not in val_scene_set]
    elif val_fraction is not None and float(val_fraction) > 0.0 and len(all_scene_dirs) > 0:
        frac = float(max(0.0, min(1.0, val_fraction)))
        n_val = int(round(len(all_scene_dirs) * frac))
        if len(all_scene_dirs) > 1:
            n_val = max(1, min(n_val, len(all_scene_dirs) - 1))
        else:
            n_val = 0
        import random

        shuffled = list(all_scene_dirs)
        random.Random(int(split_seed)).shuffle(shuffled)
        val_scene_dirs = sorted(shuffled[:n_val])
        train_scene_dirs = sorted(shuffled[n_val:])
    else:
        val_scene_dirs = []
        train_scene_dirs = all_scene_dirs

    train_samples: List[MonoSample] = []
    val_samples: List[MonoSample] = []
    for sd in train_scene_dirs:
        train_samples.extend(scan_video_dir(sd, mask_subdir=mask_subdir))
    for sd in val_scene_dirs:
        val_samples.extend(scan_video_dir(sd, mask_subdir=mask_subdir))

    return train_samples, val_samples


def build_test_samples(
    test_root_dir: str | Path,
    *,
    mask_subdir: str = "masks",
) -> List[MonoSample]:
    """
    Build test samples from scene-centric test dataset
    (e.g. datasets/external_test_catheter) using the requested mask subdir.
    """
    test_root = Path(test_root_dir)
    if not test_root.is_dir():
        raise FileNotFoundError(f"Test root not found: {test_root}")

    scene_dirs = sorted([p for p in test_root.iterdir() if _is_scene_dir(p) and (p / "images").is_dir()])
    out: List[MonoSample] = []
    for sd in scene_dirs:
        out.extend(scan_video_dir(sd, mask_subdir=mask_subdir))
    return out


# ----------------------------
# Quick sanity report
# ----------------------------

def summarize(samples: Sequence[MonoSample]) -> Dict[str, int]:
    by_video: Dict[str, int] = {}
    by_scene: Dict[str, int] = {}
    for s in samples:
        by_video[s.video_dir.name] = by_video.get(s.video_dir.name, 0) + 1
        key = f"{s.video_dir.name}/{s.scene_id}"
        by_scene[key] = by_scene.get(key, 0) + 1
    return {
        "num_samples": len(samples),
        "num_videos": len(by_video),
        "num_scenes": len(by_scene),
    }


if __name__ == "__main__":
    ROOT = "datasets/real_dataset"
    TEST_ROOT = "datasets/external_test_catheter"

    train_samples, val_samples = build_splits(ROOT)
    test_samples = build_test_samples(TEST_ROOT, mask_subdir="masks")

    print("train:", summarize(train_samples))
    print("val:  ", summarize(val_samples))
    print("test: ", summarize(test_samples))

    # Peek one sample from each split
    if train_samples:
        s = train_samples[0]
        print("train sample:", s.img, s.mask)
    if val_samples:
        s = val_samples[0]
        print("val sample:  ", s.img, s.mask)
    if test_samples:
        s = test_samples[0]
        print("test sample: ", s.img, s.mask)
