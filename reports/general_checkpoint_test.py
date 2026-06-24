from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch

from fastvit_models.factory import build_model as build_fastvit_model
from other_models.factory import build_model as build_other_model
from reports.folder_inference import (
    FolderInferenceSettings,
    collect_samples,
    load_state_dict,
    run_folder_inference,
)
from utils.data.pipeline import parse_crop_sizes


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _find_weights(run_dir: Path, explicit_file: Path | None) -> Path:
    if explicit_file is not None:
        if not explicit_file.is_file():
            raise FileNotFoundError(f"Checkpoint weights not found: {explicit_file}")
        return explicit_file
    for filename in ("best_model.pth", "model.pth", "model.pt"):
        path = run_dir / filename
        if path.is_file():
            return path
    found = sorted(p for pattern in ("*.pth", "*.pt") for p in run_dir.glob(pattern) if p.name != "last_train.pt")
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"No model weights found in {run_dir}")
    raise RuntimeError(f"Multiple checkpoint files found in {run_dir}; pass a direct .pth/.pt checkpoint.")


def _resolve_checkpoint(checkpoint: str, config: str | None) -> tuple[Path, Path, Dict[str, Any], Path]:
    checkpoint_path = _resolve_path(checkpoint)
    explicit_file = checkpoint_path if checkpoint_path.is_file() else None
    run_dir = checkpoint_path.parent if checkpoint_path.is_file() else checkpoint_path
    if not run_dir.is_dir():
        raise FileNotFoundError(f"Checkpoint folder not found: {run_dir}")
    config_path = _resolve_path(config) if config else run_dir / "config.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"Config not found: {config_path}. Pass --config for direct checkpoint files.")
    return run_dir, _find_weights(run_dir, explicit_file), _load_json(config_path), config_path


def _build_model(cfg: Dict[str, Any]) -> torch.nn.Module:
    model_cfg = cfg["model"]
    variant = str(model_cfg["variant"])
    num_classes = int(model_cfg["num_classes"])
    if variant.startswith("fastvit_"):
        return build_fastvit_model(variant=variant, num_classes=num_classes)
    return build_other_model(
        variant=variant,
        num_classes=num_classes,
        fpn_dim=int(model_cfg.get("fpn_dim", 256)),
        pretrained=False,
    )


def _settings(cfg: Dict[str, Any]) -> FolderInferenceSettings:
    loss_cfg = cfg.get("loss", {})
    return FolderInferenceSettings(
        crop_sizes=tuple(tuple(int(x) for x in cs) for cs in parse_crop_sizes(cfg["data"])),
        num_classes=int(cfg.get("model", {}).get("num_classes", 2)),
        img_pad_value=int(cfg["data"].get("img_pad_value", 0)),
        mask_pad_value=int(cfg["data"].get("mask_pad_value", 0)),
        include_background=bool(loss_cfg.get("eval_include_background", False)),
        background_index=int(loss_cfg.get("background_index", 0)),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create masks from a checkpoint with config metadata; optionally compute metrics."
    )
    parser.add_argument("--checkpoint", required=True, help="Output run folder or direct .pth/.pt checkpoint.")
    parser.add_argument("--config", default=None, help="Config JSON if --checkpoint is a direct weights file.")
    parser.add_argument("--image-root", required=True, help="Folder containing images, images/, or scene*/images/.")
    parser.add_argument("--output-root", default="outputs/folder_predictions")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compute-metrics", "--eval-masks", action="store_true", dest="compute_metrics")
    parser.add_argument("--mask-dirname", default="masks", help="Mask folder name beside images/ when computing metrics.")
    parser.add_argument("--mask-root", default=None, help="Optional mask root override.")
    parser.add_argument("--copy-gt", action="store_true", help="Copy ground-truth masks beside predictions.")
    parser.add_argument("--mask-yt3-bottom", action="store_true", help="Mask bottom 15% for scene_yt3_003 before inference.")
    parser.add_argument(
        "--metric-space",
        choices=("eval", "original"),
        default="eval",
        help="Compute metrics in model eval space by default; use original for saved-mask geometry.",
    )
    parser.add_argument("--surface-stride", type=int, default=2)
    parser.add_argument("--nsd-tolerance-px", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir, weights, cfg, config_path = _resolve_checkpoint(args.checkpoint, args.config)
    image_root = _resolve_path(args.image_root)
    output_root = _resolve_path(args.output_root)
    mask_root = _resolve_path(args.mask_root) if args.mask_root else None
    device = torch.device(args.device)

    samples = collect_samples(
        image_root,
        compute_metrics=bool(args.compute_metrics),
        mask_dirname=str(args.mask_dirname),
        mask_root=mask_root,
    )
    if args.max_samples is not None:
        samples = samples[: max(0, int(args.max_samples))]

    model = _build_model(cfg).to(device)
    model.load_state_dict(load_state_dict(weights), strict=True)
    result = run_folder_inference(
        model=model,
        settings=_settings(cfg),
        samples=samples,
        output_dir=output_root / run_dir.name / image_root.name,
        device=device,
        batch_size=int(args.batch_size),
        compute_metrics=bool(args.compute_metrics),
        copy_gt=bool(args.copy_gt),
        surface_stride=int(args.surface_stride),
        nsd_tolerance_px=float(args.nsd_tolerance_px),
        mask_yt3_bottom=bool(args.mask_yt3_bottom),
        metric_space=str(args.metric_space),
        metadata={
            "checkpoint": str(weights.resolve()),
            "config": str(config_path.resolve()),
            "image_root": str(image_root.resolve()),
            "variant": str(cfg["model"]["variant"]),
        },
    )

    print(f"Saved masks: {result['output_dir']}")
    if args.compute_metrics:
        m = result["metrics"]
        print(f"macro mIoU={m['miou']:.6f}, Dice={m['dice']:.6f}, NSD={m['nsd']:.6f}, frames={m['n_frames']}")


if __name__ == "__main__":
    main()
