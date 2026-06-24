from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

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
    now_utc,
    run_folder_inference,
)


@dataclass(frozen=True)
class PaperModelSpec:
    name: str
    variant: str
    num_classes: int = 2
    fpn_dim: int = 256
    crop_sizes: tuple[tuple[int, int], ...] = ((512, 512), (384, 640))
    include_background: bool = False
    background_index: int = 0


PAPER_MODELS: Dict[str, PaperModelSpec] = {
    "fastvit_t8_both": PaperModelSpec(name="fastvit_t8_both", variant="fastvit_t8"),
    "fastvit_sa12_both": PaperModelSpec(name="fastvit_sa12_both", variant="fastvit_sa12"),
    "fastvit_sa36_both": PaperModelSpec(name="fastvit_sa36_both", variant="fastvit_sa36"),
    "convnext_tiny_both": PaperModelSpec(name="convnext_tiny_both", variant="convnext_tiny"),
    "mobilenetv3_large_100_both": PaperModelSpec(
        name="mobilenetv3_large_100_both",
        variant="mobilenetv3_large_100",
    ),
}


def _resolve_path(path: str | Path) -> Path:
    p = Path(path)
    return p if p.is_absolute() else REPO_ROOT / p


def _selected_models(tokens: Sequence[str]) -> List[str]:
    if any(token.lower() == "all" for token in tokens):
        return list(PAPER_MODELS)
    selected: List[str] = []
    for token in tokens:
        name = Path(token).name
        if name not in PAPER_MODELS:
            raise ValueError(f"Unknown paper model '{token}'. Available: {', '.join(PAPER_MODELS)}")
        if name not in selected:
            selected.append(name)
    return selected


def _weight_file(model_name: str, paper_root: Path) -> Path:
    spec = PAPER_MODELS[model_name]
    model_dir = paper_root / model_name
    if not model_dir.is_dir():
        raise FileNotFoundError(f"Paper checkpoint folder not found: {model_dir}")
    for filename in (
        "model.pth",
        "model.pt",
        f"{spec.variant}.pth",
        f"{spec.variant}.pt",
        f"{spec.name}.pth",
        f"{spec.name}.pt",
    ):
        path = model_dir / filename
        if path.is_file():
            return path
    found = sorted(p for pattern in ("*.pth", "*.pt") for p in model_dir.rglob(pattern) if p.is_file())
    if len(found) == 1:
        return found[0]
    if not found:
        raise FileNotFoundError(f"No .pth or .pt weights found under {model_dir}")
    raise RuntimeError(f"Multiple weight files found under {model_dir}; keep one or rename the intended file to model.pth.")


def _build_model(spec: PaperModelSpec) -> torch.nn.Module:
    if spec.variant.startswith("fastvit_"):
        return build_fastvit_model(variant=spec.variant, num_classes=spec.num_classes)
    return build_other_model(
        variant=spec.variant,
        num_classes=spec.num_classes,
        fpn_dim=spec.fpn_dim,
        pretrained=False,
    )


def _settings(spec: PaperModelSpec) -> FolderInferenceSettings:
    return FolderInferenceSettings(
        crop_sizes=spec.crop_sizes,
        num_classes=spec.num_classes,
        include_background=spec.include_background,
        background_index=spec.background_index,
    )


def _write_summary(output_root: Path, results: Iterable[dict]) -> None:
    rows = []
    for r in results:
        base = {
            "model": r["paper_model"],
            "variant": r["variant"],
            "image_root": r["image_root"],
            "output_dir": r["output_dir"],
        }
        if r.get("compute_metrics"):
            total = r.get("metrics", {})
            rows.append(
                {
                    **base,
                    "scene_id": "total",
                    "metric_average": r.get("metric_average", ""),
                    "miou": total.get("miou", ""),
                    "dice": total.get("dice", ""),
                    "nsd": total.get("nsd", ""),
                    "n_frames": total.get("n_frames", ""),
                }
            )
            for scene_row in r.get("per_scene", []):
                rows.append(
                    {
                        **base,
                        "scene_id": scene_row.get("scene_id", ""),
                        "metric_average": "scene",
                        "miou": scene_row.get("miou", ""),
                        "dice": scene_row.get("dice", ""),
                        "nsd": scene_row.get("nsd", ""),
                        "n_frames": scene_row.get("n_frames", ""),
                    }
                )
        else:
            rows.append(
                {
                    **base,
                    "scene_id": "",
                    "metric_average": "",
                    "miou": "",
                    "dice": "",
                    "nsd": "",
                    "n_frames": r.get("num_images", ""),
                }
            )

    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "summary.json").write_text(json.dumps({"generated_at_utc": now_utc(), "results": rows}, indent=2))
    with (output_root / "summary.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "variant",
                "scene_id",
                "image_root",
                "metric_average",
                "miou",
                "dice",
                "nsd",
                "n_frames",
                "output_dir",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create masks from released paper checkpoints; optionally compute metrics if masks are present."
    )
    parser.add_argument("--model", nargs="+", default=["all"], help="Paper model name(s), or all.")
    parser.add_argument("--image-root", required=True, help="Folder containing images, images/, or scene*/images/.")
    parser.add_argument("--paper-root", default="paper_checkpoint")
    parser.add_argument("--output-root", default="outputs/paper_test")
    parser.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--compute-metrics", action="store_true", help="Load masks and compute mIoU/Dice/NSD.")
    parser.add_argument("--mask-dirname", default="masks", help="Mask folder name beside images/ when computing metrics.")
    parser.add_argument("--mask-root", default=None, help="Optional mask root override.")
    parser.add_argument("--no-copy-gt", action="store_true", help="Do not copy ground-truth masks beside predictions.")
    parser.add_argument("--no-mask-yt3-bottom", action="store_true", help="Disable the yt3 bottom-region mask used for paper evaluation.")
    parser.add_argument("--surface-stride", type=int, default=2)
    parser.add_argument("--nsd-tolerance-px", type=float, default=2.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_root = _resolve_path(args.image_root)
    paper_root = _resolve_path(args.paper_root)
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

    results: List[dict] = []
    for model_name in _selected_models(args.model):
        spec = PAPER_MODELS[model_name]
        weights = _weight_file(model_name, paper_root)
        model = _build_model(spec).to(device)
        model.load_state_dict(load_state_dict(weights), strict=True)
        result = run_folder_inference(
            model=model,
            settings=_settings(spec),
            samples=samples,
            output_dir=output_root / model_name / image_root.name,
            device=device,
            batch_size=int(args.batch_size),
            compute_metrics=bool(args.compute_metrics),
            copy_gt=(bool(args.compute_metrics) and not bool(args.no_copy_gt)),
            surface_stride=int(args.surface_stride),
            nsd_tolerance_px=float(args.nsd_tolerance_px),
            mask_yt3_bottom=not bool(args.no_mask_yt3_bottom),
            metric_space="eval",
            metadata={
                "paper_model": model_name,
                "variant": spec.variant,
                "weights": str(weights.resolve()),
                "image_root": str(image_root.resolve()),
            },
        )
        results.append(result)
        if args.compute_metrics:
            m = result["metrics"]
            print(
                f"{model_name}: macro mIoU={m['miou']:.6f}, "
                f"Dice={m['dice']:.6f}, NSD={m['nsd']:.6f}, frames={m['n_frames']}"
            )
        else:
            print(f"{model_name}: saved {result['num_images']} masks -> {result['output_dir']}")

    _write_summary(output_root, results)
    print(f"Saved summary: {output_root / 'summary.csv'}")


if __name__ == "__main__":
    main()
