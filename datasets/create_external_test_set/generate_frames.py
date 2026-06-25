#!/usr/bin/env python3
"""Rebuild the external-test catheter frames from source videos.

Place `yt1.mp4`, `yt2.mp4`, `yt3.mp4`, and `yt4.mp4` under `source_videos/`.
The extracted frame windows are fixed to the released prompt file.
"""

from __future__ import annotations

import argparse
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple


TimePoint = Tuple[int, int]  # minute, second


@dataclass(frozen=True)
class VideoSource:
    key: str
    filename: str
    legacy_prefix: Optional[str] = None


@dataclass(frozen=True)
class SceneSpec:
    scene_id: str
    video_key: str
    start_time: TimePoint
    end_time: TimePoint
    start_frame: int
    end_frame: int


VIDEO_SOURCES: dict[str, VideoSource] = {
    "yt1": VideoSource("yt1", "yt1.mp4", legacy_prefix="1"),
    "yt2": VideoSource("yt2", "yt2.mp4", legacy_prefix="2"),
    "yt3": VideoSource("yt3", "yt3.mp4", legacy_prefix="3"),
    "yt4": VideoSource("yt4", "yt4.mp4", legacy_prefix="4"),
}


SCENE_SPECS: tuple[SceneSpec, ...] = (
    SceneSpec("scene_yt1_001", "yt1", (3, 2), (3, 7), 5454, 5604),
    SceneSpec("scene_yt2_002", "yt2", (2, 25), (2, 53), 4359, 5184),
    SceneSpec("scene_yt3_003", "yt3", (6, 43), (7, 43), 10075, 11575),
    SceneSpec("scene_yt4_004", "yt4", (1, 38), (2, 19), 2940, 4170),
)


def _default_video_dir() -> Path:
    return Path(__file__).resolve().parent / "source_videos"


def _default_output_root() -> Path:
    return Path(__file__).resolve().parents[1] / "external_test_catheter"


def _default_points_json() -> Path:
    return Path(__file__).resolve().parent / "selected_points.json"


def _format_time(t: TimePoint) -> str:
    minute, second = t
    return f"{minute:02d}:{second:02d}"


def _find_video(video_dir: Path, source: VideoSource, allow_prefix_fallback: bool) -> Path:
    expected = video_dir / source.filename
    if expected.exists():
        return expected

    if allow_prefix_fallback and source.legacy_prefix is not None:
        matches = sorted(video_dir.glob(f"{source.legacy_prefix}_*.mp4"))
        if len(matches) == 1:
            print(f"{source.key}: using legacy local video name {matches[0].name}")
            return matches[0]
        if len(matches) > 1:
            names = ", ".join(p.name for p in matches)
            raise RuntimeError(
                f"{source.key}: expected one legacy video with prefix "
                f"'{source.legacy_prefix}_', found: {names}"
            )

    raise FileNotFoundError(
        f"Missing {source.key} video.\n"
        f"Expected source video path:\n  {expected}\n"
        "Use --allow-prefix-fallback only for older local copies named like '1_*.mp4'."
    )


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
        expected_count = spec.end_frame - spec.start_frame + 1
        if total_frames and spec.end_frame >= total_frames:
            raise RuntimeError(
                f"{spec.scene_id}: requested frame {spec.end_frame}, but "
                f"{video_path.name} reports only {total_frames} frames"
            )

        print(
            f"{spec.scene_id}: {video_path.name} "
            f"time={_format_time(spec.start_time)}-{_format_time(spec.end_time)} "
            f"fps={fps:.6g} frames={spec.start_frame}-{spec.end_frame} "
            f"count={expected_count}"
        )
        if dry_run:
            return expected_count

        images_dir = output_root / spec.scene_id / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        cap.set(cv2.CAP_PROP_POS_FRAMES, spec.start_frame)
        current_frame = spec.start_frame
        written = 0
        while current_frame <= spec.end_frame:
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
    parser = argparse.ArgumentParser(description="Extract external-test frames from public videos.")
    parser.add_argument("--video-dir", type=Path, default=_default_video_dir())
    parser.add_argument("--output-root", type=Path, default=_default_output_root())
    parser.add_argument("--points-json", type=Path, default=_default_points_json())
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--overwrite", action="store_true", help="Allow replacing existing frames/points.")
    parser.add_argument("--no-copy-points", action="store_true", help="Do not copy selected_points.json.")
    parser.add_argument("--dry-run", action="store_true", help="Print frame ranges without writing files.")
    parser.add_argument(
        "--allow-prefix-fallback",
        action="store_true",
        help="Accept older local names like 1_*.mp4 if yt1.mp4 is missing.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    video_dir = args.video_dir.resolve()
    output_root = args.output_root.resolve()

    total = 0
    for spec in SCENE_SPECS:
        source = VIDEO_SOURCES[spec.video_key]
        video_path = _find_video(
            video_dir=video_dir,
            source=source,
            allow_prefix_fallback=bool(args.allow_prefix_fallback),
        )
        total += _write_scene_frames(
            spec=spec,
            video_path=video_path,
            output_root=output_root,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
            jpeg_quality=int(args.jpeg_quality),
        )

    if not args.no_copy_points:
        _copy_points(
            points_json=args.points_json.resolve(),
            output_root=output_root,
            overwrite=bool(args.overwrite),
            dry_run=bool(args.dry_run),
        )

    print(f"Done. Expected frames: {total}. Output root: {output_root}")


if __name__ == "__main__":
    main()
