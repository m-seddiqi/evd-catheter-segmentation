#!/usr/bin/env python3
"""Extract image windows from user-provided videos.

Define one or more scenes in a segment JSON file, place the corresponding
videos under ``source_videos/`` or pass ``--video-dir``, and this script writes
JPEG frames in the dataset layout expected by the evaluation code.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    video_file: str
    start_frame: int | None = None
    end_frame: int | None = None
    start_time: float | None = None
    end_time: float | None = None


def _default_video_dir() -> Path:
    return Path(__file__).resolve().parent / "source_videos"


def _default_segments_json() -> Path:
    return Path(__file__).resolve().parent / "video_segments.json"


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "external_test_catheter"


def _parse_time_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if not isinstance(value, str):
        raise TypeError(f"Time values must be seconds or HH:MM:SS strings, got {type(value)!r}")

    parts = value.strip().split(":")
    if not 1 <= len(parts) <= 3:
        raise ValueError(f"Invalid time value: {value!r}")
    total = 0.0
    for part in parts:
        total = total * 60.0 + float(part)
    return total


def _format_time(seconds: float | None) -> str:
    if seconds is None:
        return "-"
    whole = int(seconds)
    frac = seconds - whole
    minute, second = divmod(whole, 60)
    hour, minute = divmod(minute, 60)
    suffix = f"{frac:.3f}"[1:] if frac else ""
    if hour:
        return f"{hour:02d}:{minute:02d}:{second:02d}{suffix}"
    return f"{minute:02d}:{second:02d}{suffix}"


def _load_scene_specs(path: Path) -> list[SceneSpec]:
    if not path.exists():
        raise FileNotFoundError(
            f"Segment JSON not found: {path}\n"
            "Create it from video_segments.example.json or pass --segments-json."
        )

    with path.open("r", encoding="utf-8") as handle:
        cfg = json.load(handle)

    scenes = cfg.get("scenes", cfg) if isinstance(cfg, dict) else cfg
    if not isinstance(scenes, list) or not scenes:
        raise ValueError("Segment JSON must contain a non-empty 'scenes' list.")

    specs: list[SceneSpec] = []
    for idx, raw_scene in enumerate(scenes):
        if not isinstance(raw_scene, dict):
            raise TypeError(f"Scene entry {idx} must be an object.")

        scene_id = str(raw_scene.get("scene_id", "")).strip()
        video_file = str(raw_scene.get("video_file", "")).strip()
        if not scene_id:
            raise ValueError(f"Scene entry {idx} is missing 'scene_id'.")
        if not video_file:
            raise ValueError(f"Scene entry {idx} is missing 'video_file'.")

        start_frame = raw_scene.get("start_frame")
        end_frame = raw_scene.get("end_frame")
        start_time = raw_scene.get("start_time")
        end_time = raw_scene.get("end_time")

        specs.append(
            SceneSpec(
                scene_id=scene_id,
                video_file=video_file,
                start_frame=int(start_frame) if start_frame is not None else None,
                end_frame=int(end_frame) if end_frame is not None else None,
                start_time=_parse_time_seconds(start_time) if start_time is not None else None,
                end_time=_parse_time_seconds(end_time) if end_time is not None else None,
            )
        )
    return specs


def _resolve_frame_window(spec: SceneSpec, fps: float) -> tuple[int, int]:
    if spec.start_frame is not None or spec.end_frame is not None:
        if spec.start_frame is None or spec.end_frame is None:
            raise ValueError(f"{spec.scene_id}: provide both start_frame and end_frame.")
        start_frame = spec.start_frame
        end_frame = spec.end_frame
    else:
        if spec.start_time is None or spec.end_time is None:
            raise ValueError(
                f"{spec.scene_id}: provide start_frame/end_frame or start_time/end_time."
            )
        start_frame = int(math.floor(spec.start_time * fps))
        end_frame = int(math.ceil(spec.end_time * fps))

    if start_frame < 0:
        raise ValueError(f"{spec.scene_id}: start frame must be non-negative.")
    if end_frame < start_frame:
        raise ValueError(f"{spec.scene_id}: end frame must be >= start frame.")
    return start_frame, end_frame


def _write_scene_frames(
    *,
    spec: SceneSpec,
    video_path: Path,
    output_root: Path,
    overwrite: bool,
    dry_run: bool,
    jpeg_quality: int,
) -> int:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0:
            raise RuntimeError(f"{spec.scene_id}: could not read FPS from {video_path.name}")

        start_frame, end_frame = _resolve_frame_window(spec, fps)
        expected_count = end_frame - start_frame + 1
        if total_frames and end_frame >= total_frames:
            raise RuntimeError(
                f"{spec.scene_id}: requested frame {end_frame}, but "
                f"{video_path.name} reports only {total_frames} frames"
            )

        print(
            f"{spec.scene_id}: {video_path.name} "
            f"time={_format_time(spec.start_time)}-{_format_time(spec.end_time)} "
            f"fps={fps:.6g} frames={start_frame}-{end_frame} "
            f"count={expected_count}"
        )
        if dry_run:
            return expected_count

        images_dir = output_root / spec.scene_id / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        current_frame = start_frame
        written = 0
        while current_frame <= end_frame:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError(
                    f"{spec.scene_id}: failed reading frame {current_frame} "
                    f"from {video_path.name}"
                )

            out_path = images_dir / f"frame_{current_frame:06d}.jpg"
            if out_path.exists() and not overwrite:
                raise FileExistsError(
                    f"Refusing to overwrite existing frame: {out_path}. "
                    "Pass --overwrite or choose a new --output-root."
                )
            cv2.imwrite(
                str(out_path),
                frame,
                [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
            )
            written += 1
            current_frame += 1

        if written != expected_count:
            raise RuntimeError(f"{spec.scene_id}: wrote {written}, expected {expected_count}")
        return written
    finally:
        cap.release()


def _copy_points(points_json: Path, output_root: Path, overwrite: bool, dry_run: bool) -> None:
    if not points_json.exists():
        raise FileNotFoundError(f"Points JSON not found: {points_json}")
    with points_json.open("r", encoding="utf-8") as handle:
        json.load(handle)

    out_path = output_root / "selected_points.json"
    if out_path.resolve() == points_json.resolve():
        return
    if dry_run:
        print(f"points: {points_json} -> {out_path}")
        return
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"Refusing to overwrite existing points file: {out_path}. "
            "Pass --overwrite or choose a new --output-root."
        )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(points_json, out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract scene frames from user-provided videos.")
    parser.add_argument("--video-dir", type=Path, default=_default_video_dir())
    parser.add_argument("--segments-json", type=Path, default=_default_segments_json())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument(
        "--points-json",
        type=Path,
        default=None,
        help="Optional prompt JSON to validate and copy to the output root.",
    )
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing frames/points.")
    parser.add_argument("--dry-run", action="store_true", help="Print frame ranges without writing files.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    output_root = args.output_root.resolve()
    specs = _load_scene_specs(args.segments_json.resolve())

    total = 0
    for spec in specs:
        video_path = (video_dir / spec.video_file).resolve()
        if not video_path.exists():
            raise FileNotFoundError(
                f"{spec.scene_id}: missing source video {video_path}\n"
                "Place videos under --video-dir or update video_file in the segment JSON."
            )
        total += _write_scene_frames(
            spec=spec,
            video_path=video_path,
            output_root=output_root,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            jpeg_quality=int(args.jpeg_quality),
        )

    if args.points_json is not None:
        _copy_points(
            points_json=args.points_json.resolve(),
            output_root=output_root,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )

    print(f"Done. Expected frames: {total}. Output root: {output_root}")


if __name__ == "__main__":
    main()
