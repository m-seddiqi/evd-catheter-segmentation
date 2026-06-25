#!/usr/bin/env python3
"""Generate prompted SAM3 masks for the reconstructed external-test frames.

The script reads `selected_points.json` by default and writes one PNG mask per
frame under `<dataset-root>/<scene>/masks/`. The SAM3 source tree and
checkpoint are external inputs and should be placed under `third_party/sam3/`.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Iterable, Optional

try:
    import numpy as np
    from PIL import Image
except ModuleNotFoundError as exc:
    np = None
    Image = None
    _RUNTIME_IMPORT_ERROR = exc
else:
    _RUNTIME_IMPORT_ERROR = None


def _default_dataset_root() -> Path:
    return Path(__file__).resolve().parents[1] / "external_test_catheter"


def _default_points_json() -> Path:
    return Path(__file__).resolve().parent / "selected_points.json"


def _default_sam3_root() -> Path:
    return Path(__file__).resolve().parent / "third_party" / "sam3"


def _require_runtime_deps() -> None:
    if _RUNTIME_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Mask generation requires numpy and Pillow. Install the dataset/SAM3 "
            "runtime dependencies in the environment used to run this script."
        ) from _RUNTIME_IMPORT_ERROR


def _frame_sort_key(name: str) -> int:
    match = re.search(r"(\d+)$", Path(name).stem)
    if match is None:
        raise ValueError(f"Frame name '{name}' has no trailing digits.")
    return int(match.group(1))


def _list_scene_frames(images_dir: Path) -> list[str]:
    frame_names = [
        p.name
        for p in images_dir.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg"}
    ]
    if not frame_names:
        raise RuntimeError(f"No JPG frames found in {images_dir}")
    return sorted(frame_names, key=_frame_sort_key)


def _build_numeric_tmp_dir(images_dir: Path, frame_names: list[str]) -> tempfile.TemporaryDirectory:
    tmp_dir = tempfile.TemporaryDirectory(prefix="sam3_video_frames_")
    tmp_path = Path(tmp_dir.name)
    for idx, frame_name in enumerate(frame_names):
        src = (images_dir / frame_name).resolve()
        dst = tmp_path / f"{idx:06d}.jpg"
        try:
            os.symlink(src, dst)
        except OSError:
            shutil.copy2(src, dst)
    return tmp_dir


def _save_mask_png(path: Path, mask_bool) -> None:
    mask = mask_bool.astype(np.uint8) * 255
    Image.fromarray(mask, mode="L").save(path)


def _as_numpy(value):
    if hasattr(value, "detach"):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _collect_points_by_frame(scene_prompts: list[dict]):
    by_frame: dict[str, list[list[int]]] = {}
    for entry in scene_prompts:
        frame_name = str(entry["frame_name"])
        by_frame.setdefault(frame_name, []).extend(entry.get("points", []))

    out = {}
    for frame_name, pts in by_frame.items():
        points = np.array([[p[0], p[1]] for p in pts], dtype=np.float32)
        labels = np.array([p[2] for p in pts], dtype=np.int32)
        out[frame_name] = (points, labels)
    return out


def _selected_scenes(points_cfg: dict, scenes: Optional[Iterable[str]]) -> list[str]:
    if not scenes:
        return list(points_cfg.keys())
    requested = list(scenes)
    missing = [name for name in requested if name not in points_cfg]
    if missing:
        raise KeyError(f"Requested scenes missing from points JSON: {missing}")
    return requested


def _prepare_scene(scene_name: str, scene_cfg: dict, scene_dir: Path):
    images_dir = scene_dir / "images"
    frame_names = _list_scene_frames(images_dir)
    name_to_idx = {name: idx for idx, name in enumerate(frame_names)}

    prompts_by_frame = _collect_points_by_frame(scene_cfg["prompts"])
    prompt_indices = []
    for frame_name in prompts_by_frame:
        if frame_name not in name_to_idx:
            raise RuntimeError(f"{scene_name}: prompt frame '{frame_name}' not found in {images_dir}")
        prompt_indices.append(name_to_idx[frame_name])
    if not prompt_indices:
        raise RuntimeError(f"{scene_name}: no prompts found")

    return images_dir, frame_names, name_to_idx, prompts_by_frame, min(prompt_indices)


def _extract_sam3_mask(outputs: dict, obj_id: int, height: int, width: int):
    out_obj_ids = outputs.get("out_obj_ids", [])
    out_masks = outputs.get("out_binary_masks", None)
    if out_masks is None or len(out_obj_ids) == 0:
        return np.zeros((height, width), dtype=bool)

    for i, oid in enumerate(out_obj_ids):
        if int(oid) == int(obj_id):
            mask = np.squeeze(_as_numpy(out_masks[i]))
            if mask.ndim != 2:
                raise RuntimeError(f"SAM3 returned mask with unsupported shape {mask.shape}")
            return mask.astype(bool)
    return np.zeros((height, width), dtype=bool)


def _write_scene_masks(
    scene_dir: Path,
    output_dirname: str,
    frame_names: list[str],
    segments,
    height: int,
    width: int,
) -> None:
    out_dir = scene_dir / output_dirname
    out_dir.mkdir(parents=True, exist_ok=True)
    zero_mask = np.zeros((height, width), dtype=bool)
    for idx, frame_name in enumerate(frame_names):
        out_name = Path(frame_name).with_suffix(".png").name
        _save_mask_png(out_dir / out_name, segments.get(idx, zero_mask))


def run_scene_sam3(*, model, scene_name: str, scene_cfg: dict, scene_dir: Path, output_dirname: str) -> None:
    images_dir, frame_names, name_to_idx, prompts_by_frame, start_idx = _prepare_scene(
        scene_name=scene_name,
        scene_cfg=scene_cfg,
        scene_dir=scene_dir,
    )
    obj_id = int(scene_cfg.get("obj_id", 1))

    tmp_dir = _build_numeric_tmp_dir(images_dir, frame_names)
    try:
        state = model.init_state(
            resource_path=tmp_dir.name,
            async_loading_frames=False,
            video_loader_type="cv2",
        )
        for frame_idx in range(int(state["num_frames"])):
            state["cached_frame_outputs"].setdefault(frame_idx, {})

        width = int(state["orig_width"])
        height = int(state["orig_height"])

        for frame_name, (points_abs, labels) in prompts_by_frame.items():
            frame_idx = name_to_idx[frame_name]
            points_rel = points_abs.copy()
            points_rel[:, 0] = points_rel[:, 0] / max(width, 1)
            points_rel[:, 1] = points_rel[:, 1] / max(height, 1)
            model.add_prompt(
                inference_state=state,
                frame_idx=frame_idx,
                points=points_rel,
                point_labels=labels,
                obj_id=obj_id,
                rel_coordinates=True,
            )

        segments = {}
        for reverse in (False, True):
            for out_frame_idx, outputs in model.propagate_in_video(
                inference_state=state,
                start_frame_idx=start_idx,
                reverse=reverse,
            ):
                segments[int(out_frame_idx)] = _extract_sam3_mask(
                    outputs=outputs,
                    obj_id=obj_id,
                    height=height,
                    width=width,
                )

        _write_scene_masks(scene_dir, output_dirname, frame_names, segments, height, width)
    finally:
        tmp_dir.cleanup()


def _build_sam3_model(args: argparse.Namespace):
    sam3_root = args.sam3_root.resolve()
    checkpoint = args.checkpoint.resolve() if args.checkpoint else sam3_root / "checkpoints" / "sam3.pt"
    if not sam3_root.exists():
        raise FileNotFoundError(f"SAM3 source tree not found: {sam3_root}")
    if not (sam3_root / "sam3" / "model_builder.py").exists():
        raise FileNotFoundError(
            f"SAM3 checkout at {sam3_root} does not contain sam3/model_builder.py. "
            "Clone https://github.com/facebookresearch/sam3.git into this directory."
        )
    if not checkpoint.exists():
        raise FileNotFoundError(f"SAM3 checkpoint not found: {checkpoint}")

    if str(sam3_root) not in sys.path:
        sys.path.insert(0, str(sam3_root))
    from sam3.model_builder import build_sam3_video_model

    model_kwargs = {"checkpoint_path": str(checkpoint), "load_from_HF": False}
    if args.device is not None:
        model_kwargs["device"] = args.device
    print(f"Building SAM3 video model from checkpoint={checkpoint}")
    return build_sam3_video_model(**model_kwargs)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate prompted SAM3 masks for external-test scenes.")
    parser.add_argument("--dataset-root", type=Path, default=_default_dataset_root())
    parser.add_argument("--points-json", type=Path, default=_default_points_json())
    parser.add_argument("--checkpoint", type=Path, default=None, help="SAM3 checkpoint path.")
    parser.add_argument("--output-dirname", default="masks")
    parser.add_argument("--device", default=None, help="Optional device override, e.g. cuda, cuda:0, or cpu.")
    parser.add_argument("--scenes", nargs="*", default=None, help="Optional subset of scene ids.")
    parser.add_argument("--sam3-root", type=Path, default=_default_sam3_root())
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    _require_runtime_deps()

    dataset_root = args.dataset_root.resolve()
    points_json = args.points_json.resolve()

    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not points_json.exists():
        raise FileNotFoundError(f"Points JSON not found: {points_json}")

    with points_json.open("r", encoding="utf-8") as handle:
        points_cfg = json.load(handle)

    model = _build_sam3_model(args)
    for scene_name in _selected_scenes(points_cfg, args.scenes):
        scene_dir = dataset_root / scene_name
        if not scene_dir.exists():
            raise FileNotFoundError(f"Scene directory not found: {scene_dir}")
        print(f"[{scene_name}] Generating masks -> {scene_dir / args.output_dirname}")
        run_scene_sam3(
            model=model,
            scene_name=scene_name,
            scene_cfg=points_cfg[scene_name],
            scene_dir=scene_dir,
            output_dirname=args.output_dirname,
        )

    print("Done.")


if __name__ == "__main__":
    main()
