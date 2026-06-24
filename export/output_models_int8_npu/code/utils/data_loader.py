from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision.transforms.functional import normalize


@dataclass(frozen=True)
class FrameRef:
    scene_id: str
    frame_id: str
    img_path: Path
    gt_path: Path


def build_upload_payload(
    dataset_cfg: Dict[str, Any],
    train_cfg: Dict[str, Any],
    test_root: Path,
    channel_first: bool = True,
):
    from utils.data.indexing import build_test_samples
    from utils.data.pipeline import MonoEvalPipeline, MonoPipelineConfig, parse_crop_sizes
    from utils.data.loaders import IMAGENET_MEAN, IMAGENET_STD

    mask_subdir = str(dataset_cfg.get("mask_subdir", "masks"))
    scene_id = dataset_cfg.get("scene_id")
    num_samples = int(dataset_cfg.get("num_samples", 1000))
    input_h, input_w = [int(v) for v in dataset_cfg["input_hw"]]

    crop_sizes = parse_crop_sizes(train_cfg["data"])
    pipe_cfg = MonoPipelineConfig(
        crop_sizes=crop_sizes,
        img_pad_value=int(train_cfg["data"].get("img_pad_value", 0)),
        mask_pad_value=int(train_cfg["data"].get("mask_pad_value", 0)),
    )
    eval_pipeline = MonoEvalPipeline(pipe_cfg)

    all_samples = build_test_samples(test_root, mask_subdir=mask_subdir)
    if scene_id is not None:
        all_samples = [s for s in all_samples if s.scene_id == scene_id]
    if not all_samples:
        raise RuntimeError("No test samples found for current dataset settings.")

    samples = all_samples[:num_samples]
    hub_inputs = {"image": []}
    frame_keys: List[Tuple[str, str]] = []

    for s in samples:
        img = Image.open(s.img).convert("RGB")
        out = eval_pipeline(
            img=img,
            mask=None,
            meta={"scene_id": s.scene_id, "frame_id": s.frame_id},
        )

        crop_used = tuple(int(v) for v in out["meta"]["crop_size_used"])
        if crop_used != (input_h, input_w):
            raise ValueError(f"Expected transformed size {(input_h, input_w)}, got {crop_used}.")

        img_arr = np.asarray(out["img"], dtype=np.uint8)
        img_t = torch.from_numpy(img_arr).permute(2, 0, 1).float().div(255.0)
        img_t = normalize(img_t, mean=IMAGENET_MEAN, std=IMAGENET_STD)

        if tuple(int(v) for v in img_t.shape) != (3, input_h, input_w):
            raise ValueError(f"Unexpected transformed tensor shape: {tuple(img_t.shape)}")

        input_arr = img_t.unsqueeze(0).cpu().numpy().astype(np.float32)
        if not channel_first:
            input_arr = np.transpose(input_arr, (0, 2, 3, 1))
        hub_inputs["image"].append(input_arr)
        frame_keys.append((s.scene_id, s.frame_id))

    return hub_inputs, frame_keys


def build_manifest(
    dataset_cfg: Dict[str, Any],
    train_cfg_path: Path,
    test_root: Path,
    frame_keys: List[Tuple[str, str]],
    channel_first: bool = True,
) -> Dict[str, Any]:
    input_h, input_w = [int(v) for v in dataset_cfg["input_hw"]]
    return {
        "config_path": str(train_cfg_path),
        "test_root": str(test_root),
        "mask_subdir": str(dataset_cfg.get("mask_subdir", "masks")),
        "scene_id": dataset_cfg.get("scene_id"),
        "num_samples": len(frame_keys),
        "input_hw": [input_h, input_w],
        "dataset_name": str(dataset_cfg["dataset_name"]),
        "channel_first": bool(channel_first),
        "frames": [{"scene_id": s, "frame_id": f} for s, f in frame_keys],
    }


def build_refs_from_manifest(manifest: Dict[str, Any]) -> List[FrameRef]:
    test_root = Path(str(manifest["test_root"]))
    mask_subdir = str(manifest["mask_subdir"])
    refs: List[FrameRef] = []
    for item in manifest.get("frames", []):
        scene_id = str(item["scene_id"])
        frame_id = str(item["frame_id"])
        img_dir = test_root / scene_id / "images"
        img_path = img_dir / f"{frame_id}.jpg"
        if not img_path.exists():
            for ext in (".png", ".jpeg", ".bmp", ".tif", ".tiff"):
                candidate = img_dir / f"{frame_id}{ext}"
                if candidate.exists():
                    img_path = candidate
                    break
        refs.append(
            FrameRef(
                scene_id=scene_id,
                frame_id=frame_id,
                img_path=img_path,
                gt_path=test_root / scene_id / mask_subdir / f"{frame_id}.png",
            )
        )
    return refs
