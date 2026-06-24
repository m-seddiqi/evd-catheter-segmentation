from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from copy import deepcopy
from dataclasses import asdict, is_dataclass, replace
from pathlib import Path
from typing import Any, Dict, List


SCENE_SEED_EPOCH_STRIDE = 1_000_000


def _load_generator_modules():
    try:
        from .config import default_config
        from .renderer import render_scene_arrays
        from .sampler import apply_incremental_motion, sample_scene_spec
    except ImportError:
        datasets_root = Path(__file__).resolve().parents[1]
        if str(datasets_root) not in sys.path:
            sys.path.insert(0, str(datasets_root))
        from create_synthetic_dataset.config import default_config
        from create_synthetic_dataset.renderer import render_scene_arrays
        from create_synthetic_dataset.sampler import apply_incremental_motion, sample_scene_spec
    return default_config, render_scene_arrays, apply_incremental_motion, sample_scene_spec


def _jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    return value


def _scene_spec_to_jsonable(spec: Any) -> Dict[str, Any]:
    return {
        "background": spec.background_name,
        "objects": [
            {
                "category": obj.category,
                "rel_path": obj.rel_path,
                "rotation_deg": obj.rotation_deg.tolist(),
                "translation": obj.translation.tolist(),
                "scale": obj.scale.tolist(),
                "color_override": obj.color_override,
            }
            for obj in spec.objects
        ],
    }


def _positive_int(name: str, value: int) -> int:
    value = int(value)
    if value <= 0:
        raise ValueError(f"{name} must be > 0, got {value}")
    return value


def _jpeg_quality(value: int) -> int:
    value = _positive_int("jpeg_quality", value)
    if value > 100:
        raise ValueError(f"jpeg_quality must be <= 100, got {value}")
    return value


def _normalized_ext(fmt: str) -> str:
    ext = str(fmt).lower().strip().lstrip(".")
    return "jpg" if ext == "jpeg" else ext


def _save_rgb_frame(image, path: Path, *, image_format: str, jpeg_quality: int) -> None:
    ext = _normalized_ext(image_format)
    pil_img = image.convert("RGB")
    if ext in {"jpg", "jpeg"}:
        pil_img.save(path, format="JPEG", quality=int(jpeg_quality), optimize=True)
    elif ext == "png":
        pil_img.save(path, format="PNG")
    else:
        pil_img.save(path)


def _default_output_root(generator_root: Path) -> Path:
    return generator_root.parent / "synthetic_dataset"


def _epoch_seed(seed_base: int, epoch: int) -> int:
    return int(seed_base) + int(epoch) * SCENE_SEED_EPOCH_STRIDE


def _scene_seed(seed: int, scene_id: int) -> int:
    return int(seed) + int(scene_id)


def _write_generation_config(
    *,
    cfg,
    output_dir: Path,
    seed: int,
    scene_count: int,
    start_index: int,
    frames_per_scene: int,
    snapshot_name: str,
) -> None:
    if not snapshot_name:
        return
    payload = {
        "seed": int(seed),
        "seed_formula": "scene_seed = seed + scene_id",
        "scene_count": int(scene_count),
        "start_index": int(start_index),
        "frames_per_scene": int(frames_per_scene),
        "config": _jsonable(cfg),
    }
    (output_dir / snapshot_name).write_text(json.dumps(payload, indent=2))


def generate_folder(
    *,
    generator_root: Path,
    output_dir: Path | None,
    scene_count: int,
    frames_per_scene: int | None,
    image_format: str | None,
    jpeg_quality: int | None,
    seed: int,
    start_index: int = 0,
    config_snapshot_name: str = "generation_config.json",
) -> None:
    import numpy as np
    from PIL import Image

    default_config, render_scene_arrays, apply_incremental_motion, sample_scene_spec = _load_generator_modules()

    cfg = default_config(generator_root)
    if output_dir is not None:
        cfg = replace(cfg, paths=replace(cfg.paths, output_dir=output_dir))
    if image_format is not None:
        cfg = replace(cfg, image_format=str(image_format))
    if jpeg_quality is not None:
        cfg = replace(cfg, jpeg_quality=_jpeg_quality(jpeg_quality))

    scene_count = _positive_int("scene_count", scene_count)
    frames = cfg.frames_per_scene if frames_per_scene is None else _positive_int("frames_per_scene", frames_per_scene)
    cfg.paths.output_dir.mkdir(parents=True, exist_ok=True)
    min_mask_pixels = max(1, int(cfg.min_catheter_mask_pixels))
    max_visibility_retries = max(1, int(cfg.max_visibility_retries))

    def is_visible(mask) -> bool:
        return int(np.count_nonzero(mask)) >= min_mask_pixels

    def sample_visible_initial(scene_rng):
        last_error: str | None = None
        for _ in range(max_visibility_retries):
            candidate = sample_scene_spec(cfg, scene_rng)
            rgb, mask = render_scene_arrays(cfg, candidate)
            if is_visible(mask):
                return candidate, rgb, mask
            last_error = f"mask_pixels={int(np.count_nonzero(mask))}"
        raise RuntimeError(
            "Unable to sample a visible initial catheter after "
            f"{max_visibility_retries} attempts ({last_error})."
        )

    def next_visible_motion(previous_spec, scene_rng):
        for _ in range(max_visibility_retries):
            candidate = deepcopy(previous_spec)
            apply_incremental_motion(candidate, cfg, scene_rng)
            rgb, mask = render_scene_arrays(cfg, candidate)
            if is_visible(mask):
                return candidate, rgb, mask

        # Keep the sequence valid instead of writing an empty supervision mask.
        fallback = deepcopy(previous_spec)
        rgb, mask = render_scene_arrays(cfg, fallback)
        return fallback, rgb, mask

    _write_generation_config(
        cfg=cfg,
        output_dir=cfg.paths.output_dir,
        seed=int(seed),
        scene_count=scene_count,
        start_index=int(start_index),
        frames_per_scene=frames,
        snapshot_name=config_snapshot_name,
    )

    image_ext = _normalized_ext(cfg.image_format)
    mask_ext = _normalized_ext(cfg.mask_format)
    for offset in range(scene_count):
        scene_id = int(start_index) + offset
        rng = np.random.default_rng(_scene_seed(seed, scene_id))
        initial_spec, first_rgb, first_mask = sample_visible_initial(rng)
        frame_spec = deepcopy(initial_spec)

        scene_dir = cfg.paths.output_dir / f"scene_{scene_id:05d}"
        images_dir = scene_dir / "images"
        masks_dir = scene_dir / "masks"
        images_dir.mkdir(parents=True, exist_ok=True)
        masks_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx in range(frames):
            if frame_idx == 0:
                rgb, mask = first_rgb, first_mask
            else:
                frame_spec, rgb, mask = next_visible_motion(frame_spec, rng)

            frame_stem = f"frame_{frame_idx:05d}"
            _save_rgb_frame(
                Image.fromarray(rgb),
                images_dir / f"{frame_stem}.{image_ext}",
                image_format=cfg.image_format,
                jpeg_quality=cfg.jpeg_quality,
            )
            Image.fromarray(mask).save(masks_dir / f"{frame_stem}.{mask_ext}")

        scene_meta = {
            "scene_id": scene_id,
            "frames_per_scene": frames,
            "background": initial_spec.background_name,
            "initial_objects": _scene_spec_to_jsonable(initial_spec)["objects"],
        }
        (scene_dir / "scene.json").write_text(json.dumps(scene_meta, indent=2))
        print(f"[{offset + 1}/{scene_count}] scene={scene_dir} frames={frames}")


def _clear_epoch_dir(epoch_dir: Path) -> None:
    for scene_dir in epoch_dir.glob("scene_*"):
        if scene_dir.is_dir():
            shutil.rmtree(scene_dir)
    for path in epoch_dir.glob("generation_config*.json"):
        path.unlink()
    for path in (epoch_dir / "generation_plan.json",):
        if path.exists():
            path.unlink()
    logs_dir = epoch_dir / "logs"
    if logs_dir.exists():
        shutil.rmtree(logs_dir)


def _worker_commands(
    *,
    epoch: int,
    epoch_dir: Path,
    scene_count: int,
    scenes_per_worker: int,
    seed_base: int,
    generator_root: Path,
    frames_per_scene: int | None,
    image_format: str | None,
    jpeg_quality: int | None,
) -> list[tuple[list[str], Path]]:
    commands: list[tuple[list[str], Path]] = []
    generate_py = Path(__file__).resolve()
    logs_dir = epoch_dir / "logs"
    seed = _epoch_seed(seed_base, epoch)

    for worker_id, start_index in enumerate(range(0, scene_count, scenes_per_worker)):
        count = min(scenes_per_worker, scene_count - start_index)
        end_index = start_index + count - 1
        log_path = logs_dir / f"worker_{worker_id}_{start_index}_{end_index}.log"
        cmd = [
            sys.executable,
            str(generate_py),
            "--count",
            str(count),
            "--start-index",
            str(start_index),
            "--seed",
            str(seed),
            "--generator-root",
            str(generator_root),
            "--output-dir",
            str(epoch_dir),
            "--config-snapshot-name",
            f"generation_config_worker_{worker_id}_{start_index}_{end_index}.json",
        ]
        if frames_per_scene is not None:
            cmd.extend(["--frames", str(frames_per_scene)])
        if image_format is not None:
            cmd.extend(["--image-format", str(image_format)])
        if jpeg_quality is not None:
            cmd.extend(["--jpeg-quality", str(jpeg_quality)])
        commands.append((cmd, log_path))

    return commands


def generate_epoch(
    *,
    epoch: int,
    output_root: Path,
    scene_count: int,
    frames_per_scene: int | None,
    workers: int,
    scenes_per_worker: int,
    seed_base: int,
    generator_root: Path,
    clear_existing: bool,
    image_format: str | None,
    jpeg_quality: int | None,
) -> None:
    scene_count = _positive_int("scene_count", scene_count)
    workers = _positive_int("workers", workers)
    scenes_per_worker = _positive_int("scenes_per_worker", scenes_per_worker)

    epoch_dir = output_root / f"e{int(epoch)}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    if clear_existing:
        _clear_epoch_dir(epoch_dir)
    logs_dir = epoch_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)

    plan = {
        "epoch": int(epoch),
        "output_dir": str(epoch_dir),
        "scene_count": int(scene_count),
        "frames_per_scene": frames_per_scene,
        "workers": int(workers),
        "scenes_per_worker": int(scenes_per_worker),
        "seed_base": int(seed_base),
        "epoch_seed": _epoch_seed(seed_base, epoch),
        "seed_formula": f"scene_seed = seed_base + epoch * {SCENE_SEED_EPOCH_STRIDE} + scene_id",
    }
    (epoch_dir / "generation_plan.json").write_text(json.dumps(plan, indent=2))

    commands = _worker_commands(
        epoch=epoch,
        epoch_dir=epoch_dir,
        scene_count=scene_count,
        scenes_per_worker=scenes_per_worker,
        seed_base=seed_base,
        generator_root=generator_root,
        frames_per_scene=frames_per_scene,
        image_format=image_format,
        jpeg_quality=jpeg_quality,
    )

    active: List[tuple[subprocess.Popen, Path]] = []
    failures: List[Path] = []

    def wait_one() -> None:
        proc, log_path = active.pop(0)
        if proc.wait() != 0:
            failures.append(log_path)

    for cmd, log_path in commands:
        with log_path.open("w") as log_file:
            proc = subprocess.Popen(cmd, stdout=log_file, stderr=subprocess.STDOUT)
        active.append((proc, log_path))
        if len(active) >= workers:
            wait_one()

    while active:
        wait_one()

    if failures:
        failed_logs = "\n".join(f"- {p}" for p in failures[:10])
        raise RuntimeError(f"Generation failed for e{epoch}. Check logs:\n{failed_logs}")

    print(f"completed e{epoch}: {epoch_dir}")


def _resolve_epoch_range(args: argparse.Namespace) -> tuple[int, int] | None:
    if args.epoch is not None:
        if args.start_epoch is not None or args.end_epoch is not None:
            raise ValueError("Use either --epoch or --start-epoch/--end-epoch, not both.")
        return int(args.epoch), int(args.epoch)

    epoch_mode = args.epoch_folders or args.start_epoch is not None or args.end_epoch is not None
    if not epoch_mode:
        return None

    start_epoch = 1 if args.start_epoch is None else int(args.start_epoch)
    if args.end_epoch is not None:
        end_epoch = int(args.end_epoch)
    elif args.epoch_count is not None:
        end_epoch = start_epoch + _positive_int("epoch_count", args.epoch_count) - 1
    else:
        end_epoch = start_epoch

    if end_epoch < start_epoch:
        raise ValueError("--end-epoch must be >= --start-epoch")
    return start_epoch, end_epoch


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic catheter image/mask sequences.")
    parser.add_argument("--count", "--scenes", "--scenes-per-epoch", dest="scene_count", type=int, default=1)
    parser.add_argument("--frames", "--frames-per-scene", dest="frames_per_scene", type=int, default=None)
    parser.add_argument("--image-format", choices=["jpg", "jpeg", "png"], default=None)
    parser.add_argument("--jpeg-quality", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=None)
    parser.add_argument("--generator-root", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--repo-root", dest="generator_root", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--config-snapshot-name", default="generation_config.json")

    parser.add_argument("--epoch", type=int, default=None, help="Generate one folder, e.g. --epoch 0.")
    parser.add_argument("--start-epoch", type=int, default=None, help="First e* folder for range generation.")
    parser.add_argument("--end-epoch", type=int, default=None, help="Last e* folder for range generation.")
    parser.add_argument("--epoch-count", type=int, default=None, help="Range length when --end-epoch is omitted.")
    parser.add_argument("--epoch-folders", action="store_true", help="Compatibility alias for epoch range mode.")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scenes-per-worker", type=int, default=25)
    parser.add_argument("--seed-base", type=int, default=100000)
    parser.add_argument("--clear-existing", "--overwrite", dest="clear_existing", action="store_true")

    args = parser.parse_args()
    generator_root = args.generator_root.resolve()
    epoch_range = _resolve_epoch_range(args)

    if epoch_range is None:
        generate_folder(
            generator_root=generator_root,
            output_dir=args.output_dir,
            scene_count=args.scene_count,
            frames_per_scene=args.frames_per_scene,
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
            seed=int(args.seed),
            start_index=int(args.start_index),
            config_snapshot_name=str(args.config_snapshot_name),
        )
        return

    output_root = args.output_root or _default_output_root(generator_root)
    output_root.mkdir(parents=True, exist_ok=True)
    start_epoch, end_epoch = epoch_range
    print(
        "synthetic generation plan: "
        f"e{start_epoch}..e{end_epoch}, "
        f"{int(args.scene_count)} scenes/folder, "
        f"workers={int(args.workers)}, output_root={output_root}"
    )

    for epoch in range(start_epoch, end_epoch + 1):
        generate_epoch(
            epoch=epoch,
            output_root=output_root,
            scene_count=args.scene_count,
            frames_per_scene=args.frames_per_scene,
            workers=args.workers,
            scenes_per_worker=args.scenes_per_worker,
            seed_base=args.seed_base,
            generator_root=generator_root,
            clear_existing=bool(args.clear_existing),
            image_format=args.image_format,
            jpeg_quality=args.jpeg_quality,
        )

    print(f"done: generated e{start_epoch}..e{end_epoch} under {output_root}")


if __name__ == "__main__":
    main()
