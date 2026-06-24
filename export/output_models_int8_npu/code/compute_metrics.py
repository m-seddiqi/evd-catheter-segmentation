from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage
from scipy.spatial import cKDTree

from export.output_models_int8_npu.code.model_loader import discover_model_specs
from export.output_models_int8_npu.code.utils.common import load_cfg_file, load_json, load_yaml, resolve_path, resolve_project_root, utc_now_iso
from export.output_models_int8_npu.code.utils.data_loader import FrameRef, build_refs_from_manifest


def _load_pred_labels(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path).convert("L"), dtype=np.uint8)
    return (arr > 0).astype(np.uint8)


def _align_pred_to_gt(pred_labels: np.ndarray, gt_labels: np.ndarray) -> np.ndarray:
    gh, gw = int(gt_labels.shape[0]), int(gt_labels.shape[1])
    ph, pw = int(pred_labels.shape[0]), int(pred_labels.shape[1])
    if ph == gh and pw == gw:
        return pred_labels
    out = pred_labels
    if ph < gh or pw < gw:
        out = np.pad(out, ((0, max(0, gh - ph)), (0, max(0, gw - pw))), mode="constant", constant_values=0)
    if out.shape[0] > gh or out.shape[1] > gw:
        out = out[:gh, :gw]
    return out.astype(np.uint8)


def _surface(mask_fg: np.ndarray) -> np.ndarray:
    if mask_fg.any():
        eroded = ndimage.binary_erosion(mask_fg, structure=np.ones((3, 3), dtype=bool), border_value=0)
        return mask_fg ^ eroded
    return np.zeros_like(mask_fg, dtype=bool)


def _surface_points(mask_fg: np.ndarray, stride: int = 1) -> np.ndarray:
    ys, xs = np.nonzero(_surface(mask_fg.astype(bool)))
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.column_stack([ys, xs]).astype(np.float32)
    if stride > 1:
        pts = pts[::stride]
    return pts


def _surface_distances(gt_pts: np.ndarray, pred_pts: np.ndarray):
    n_gt = int(gt_pts.shape[0])
    n_pr = int(pred_pts.shape[0])
    if n_gt == 0 and n_pr == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    if (n_gt == 0) != (n_pr == 0):
        return None
    tree_gt = cKDTree(gt_pts)
    d_pr_to_gt, _ = tree_gt.query(pred_pts, k=1, workers=-1)
    tree_pr = cKDTree(pred_pts)
    d_gt_to_pr, _ = tree_pr.query(gt_pts, k=1, workers=-1)
    return d_pr_to_gt.astype(np.float32), d_gt_to_pr.astype(np.float32)


def _nsd_from_points(gt_pts: np.ndarray, pred_pts: np.ndarray, tolerance_px: float) -> float:
    d = _surface_distances(gt_pts, pred_pts)
    if d is None:
        return 0.0
    d_pr_to_gt, d_gt_to_pr = d
    if d_pr_to_gt.size == 0 and d_gt_to_pr.size == 0:
        return 1.0
    in_pred = int(np.count_nonzero(d_pr_to_gt <= float(tolerance_px)))
    in_gt = int(np.count_nonzero(d_gt_to_pr <= float(tolerance_px)))
    denom = int(d_pr_to_gt.size + d_gt_to_pr.size)
    return float((in_pred + in_gt) / max(denom, 1))


def _miou_from_stats(
    intersection: np.ndarray,
    union: np.ndarray,
    include_background: bool,
    background_index: int,
    eps: float = 1e-6,
) -> float:
    valid = union > 0
    if not include_background and 0 <= background_index < valid.size:
        valid[background_index] = False
    if not np.any(valid):
        return 0.0
    return float(np.mean(intersection[valid] / (union[valid] + eps)))


def _dice(tp: int, fp: int, fn: int) -> float:
    return float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))


def _eval_spec_key(cfg: Dict[str, Any], test_root: Path, mask_subdir: str) -> str:
    from utils.data.pipeline import parse_crop_sizes

    data_cfg = dict(cfg.get("data", {}))
    spec = {
        "crop_sizes": [list(x) for x in parse_crop_sizes(data_cfg)],
        "img_pad_value": int(data_cfg.get("img_pad_value", 0)),
        "mask_pad_value": int(data_cfg.get("mask_pad_value", 0)),
        "test_root": str(test_root),
        "test_mask_subdir": str(mask_subdir),
    }
    return json.dumps(spec, sort_keys=True)


def _build_gt_cache(refs: List[FrameRef], cfg: Dict[str, Any]) -> Dict[Tuple[str, str], np.ndarray]:
    from utils.data.pipeline import MonoEvalPipeline, MonoPipelineConfig, parse_crop_sizes

    data_cfg = dict(cfg.get("data", {}))
    pipe_cfg = MonoPipelineConfig(
        crop_sizes=parse_crop_sizes(data_cfg),
        img_pad_value=int(data_cfg.get("img_pad_value", 0)),
        mask_pad_value=int(data_cfg.get("mask_pad_value", 0)),
    )
    eval_pipeline = MonoEvalPipeline(pipe_cfg)

    cache: Dict[Tuple[str, str], np.ndarray] = {}
    for r in refs:
        img = Image.open(r.img_path).convert("RGB")
        gt = Image.open(r.gt_path).convert("L")
        out = eval_pipeline(img=img, mask=gt, meta={"scene_id": r.scene_id, "frame_id": r.frame_id})
        cache[(r.scene_id, r.frame_id)] = (np.array(out["mask"], dtype=np.uint8) > 0).astype(np.uint8)
    return cache


def _resolve_pred_path(pred_root: Path, scene_id: str, frame_id: str, idx: int) -> Path | None:
    candidates = [
        pred_root / f"{scene_id}__{frame_id}.png",
        pred_root / f"{frame_id}.png",
        pred_root / f"idx_{idx:05d}.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def _evaluate_model(
    pred_root: Path,
    refs: List[FrameRef],
    gt_cache: Dict[Tuple[str, str], np.ndarray],
    num_classes: int,
    ignore_index: int,
    include_background: bool,
    background_index: int,
    surface_stride: int,
    nsd_tolerance_px: float,
):
    total_inter = np.zeros((num_classes,), dtype=np.float64)
    total_union = np.zeros((num_classes,), dtype=np.float64)
    all_tp = all_fp = all_fn = 0
    nsd_vals: List[float] = []
    per_scene: Dict[str, Dict[str, Any]] = {}
    missing: List[Dict[str, str]] = []
    used_frames = 0

    for idx, r in enumerate(refs):
        pred_path = _resolve_pred_path(pred_root, r.scene_id, r.frame_id, idx)
        if pred_path is None:
            missing.append({"scene_id": r.scene_id, "frame_id": r.frame_id})
            continue

        gt_labels = gt_cache.get((r.scene_id, r.frame_id))
        if gt_labels is None:
            continue

        pred_labels = _align_pred_to_gt(_load_pred_labels(pred_path), gt_labels)
        valid = gt_labels != ignore_index
        gt_v = gt_labels[valid].astype(np.int64)
        pr_v = pred_labels[valid].astype(np.int64)
        if gt_v.size == 0:
            continue

        for c in range(num_classes):
            pred_c = pr_v == c
            gt_c = gt_v == c
            total_inter[c] += float(np.count_nonzero(pred_c & gt_c))
            total_union[c] += float(np.count_nonzero(pred_c | gt_c))

        gt_fg = gt_v == 1
        pr_fg = pr_v == 1
        tp = int(np.count_nonzero(pr_fg & gt_fg))
        fp = int(np.count_nonzero(pr_fg & ~gt_fg))
        fn = int(np.count_nonzero(~pr_fg & gt_fg))
        nsd = _nsd_from_points(
            _surface_points(gt_labels == 1, stride=surface_stride),
            _surface_points(pred_labels == 1, stride=surface_stride),
            tolerance_px=nsd_tolerance_px,
        )

        if r.scene_id not in per_scene:
            per_scene[r.scene_id] = {
                "inter": np.zeros((num_classes,), dtype=np.float64),
                "union": np.zeros((num_classes,), dtype=np.float64),
                "tp": 0,
                "fp": 0,
                "fn": 0,
                "nsd_vals": [],
                "n_frames": 0,
            }
        s = per_scene[r.scene_id]
        for c in range(num_classes):
            s["inter"][c] += float(np.count_nonzero((pr_v == c) & (gt_v == c)))
            s["union"][c] += float(np.count_nonzero((pr_v == c) | (gt_v == c)))
        s["tp"] += tp
        s["fp"] += fp
        s["fn"] += fn
        s["nsd_vals"].append(float(nsd))
        s["n_frames"] += 1

        all_tp += tp
        all_fp += fp
        all_fn += fn
        nsd_vals.append(float(nsd))
        used_frames += 1

    overall = {
        "miou": _miou_from_stats(total_inter, total_union, include_background, background_index),
        "dice": _dice(all_tp, all_fp, all_fn),
        "nsd": float(np.mean(nsd_vals)) if nsd_vals else float("nan"),
        "n_frames": int(used_frames),
        "n_missing_preds": int(len(missing)),
    }

    per_scene_rows = []
    for scene_id in sorted(per_scene.keys()):
        s = per_scene[scene_id]
        per_scene_rows.append(
            {
                "scene_id": scene_id,
                "miou": _miou_from_stats(s["inter"], s["union"], include_background, background_index),
                "dice": _dice(int(s["tp"]), int(s["fp"]), int(s["fn"])),
                "nsd": float(np.mean(s["nsd_vals"])) if s["nsd_vals"] else float("nan"),
                "n_frames": int(s["n_frames"]),
            }
        )

    return overall, per_scene_rows, missing


def _discover_inference_dirs(outputs_root: Path) -> List[Path]:
    found = []
    for pred_dir in outputs_root.glob("*/*/*/pred_masks_png"):
        if pred_dir.is_dir():
            found.append(pred_dir)
    return sorted(found)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute mIoU, Dice, NSD for all available inference outputs.")
    parser.add_argument("--config", type=Path, default=Path("export/output_models_int8_npu/code/config.yaml"))
    parser.add_argument("--prefix", type=str, default=None, help="Optional output prefix filter")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    project_root = resolve_project_root(args.config, cfg)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    dataset_cfg = dict(cfg["dataset"])
    paths_cfg = dict(cfg["paths"])
    metrics_cfg = dict(cfg["metrics"])
    model_specs = {spec.model_key: spec for spec in discover_model_specs(project_root=project_root, cfg=cfg)}
    train_cfg_path = resolve_path(project_root, dataset_cfg["config_path"])
    train_cfg = load_cfg_file(train_cfg_path)

    manifest_path = resolve_path(project_root, paths_cfg["dataset_manifest_path"])
    outputs_root = resolve_path(project_root, paths_cfg["outputs_root"])
    metrics_json_path = resolve_path(project_root, paths_cfg["metrics_json_path"])
    metrics_json_path.parent.mkdir(parents=True, exist_ok=True)

    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}. Run upload_dataset.py first.")

    manifest = load_json(manifest_path)
    manifest_channel_first = bool(manifest.get("channel_first", True))
    refs = build_refs_from_manifest(manifest)

    pred_dirs = _discover_inference_dirs(outputs_root)
    if args.prefix:
        pred_dirs = [p for p in pred_dirs if p.parent.parent.name == args.prefix]
    if not pred_dirs:
        raise RuntimeError(f"No inference outputs found under: {outputs_root}")

    gt_cache_by_spec: Dict[str, Dict[Tuple[str, str], np.ndarray]] = {}
    model_results: List[Dict[str, Any]] = []

    t0 = time.time()
    for pred_dir in pred_dirs:
        model_dir = pred_dir.parent
        model_key = model_dir.name
        output_prefix = model_dir.parent.name
        source_group = model_dir.parent.parent.name
        model_spec = model_specs.get(model_key)

        summary_path = model_dir / "compile_infer_summary.json"
        summary: Dict[str, Any] = {}
        if summary_path.exists():
            summary = load_json(summary_path)
            summary_channel_first = summary.get("channel_first")
            if summary_channel_first is not None and bool(summary_channel_first) != manifest_channel_first:
                print(
                    f"[WARN] Layout mismatch for {source_group}/{model_key}: "
                    f"manifest channel_first={manifest_channel_first}, "
                    f"summary channel_first={bool(summary_channel_first)}"
                )

        if model_spec is None:
            print(f"[WARN] {model_key} is not listed in config models; using compile summary and train config defaults.")
            run_cfg = train_cfg
            variant = str(summary.get("variant", "unknown"))
            weights_path = str(summary.get("weights_path", ""))
            num_classes = int(summary.get("num_classes", train_cfg.get("model", {}).get("num_classes", 2)))
        else:
            run_cfg = model_spec.run_cfg if model_spec.run_cfg is not None else train_cfg
            variant = model_spec.variant
            weights_path = str(model_spec.weights_path)
            num_classes = int(model_spec.num_classes)

        loss_cfg = dict(run_cfg.get("loss", {}))
        spec_key = _eval_spec_key(
            run_cfg,
            test_root=Path(str(manifest["test_root"])),
            mask_subdir=str(manifest["mask_subdir"]),
        )
        if spec_key not in gt_cache_by_spec:
            gt_cache_by_spec[spec_key] = _build_gt_cache(refs=refs, cfg=run_cfg)

        overall, per_scene, missing = _evaluate_model(
            pred_root=pred_dir,
            refs=refs,
            gt_cache=gt_cache_by_spec[spec_key],
            num_classes=num_classes,
            ignore_index=int(loss_cfg.get("ignore_index", 255)),
            include_background=bool(loss_cfg.get("eval_include_background", False)),
            background_index=int(loss_cfg.get("background_index", 0)),
            surface_stride=int(metrics_cfg.get("surface_stride", 2)),
            nsd_tolerance_px=float(metrics_cfg.get("nsd_tolerance_px", 2.0)),
        )

        model_results.append(
            {
                "source_group": source_group,
                "output_prefix": output_prefix,
                "model_key": model_key,
                "variant": variant,
                "pred_dir": str(pred_dir),
                "weights_path": weights_path,
                "manifest_channel_first": manifest_channel_first,
                "miou": float(overall["miou"]),
                "dice": float(overall["dice"]),
                "nsd": float(overall["nsd"]),
                "n_frames": int(overall["n_frames"]),
                "n_missing_preds": int(overall["n_missing_preds"]),
                "per_scene": per_scene,
                "missing_predictions": missing,
            }
        )

    model_results.sort(key=lambda x: (x["source_group"], x["output_prefix"], x["model_key"]))

    payload = {
        "generated_at_utc": utc_now_iso(),
        "manifest_path": str(manifest_path),
        "outputs_root": str(outputs_root),
        "manifest_channel_first": manifest_channel_first,
        "surface_stride": int(metrics_cfg.get("surface_stride", 2)),
        "nsd_tolerance_px": float(metrics_cfg.get("nsd_tolerance_px", 2.0)),
        "num_eval_specs": int(len(gt_cache_by_spec)),
        "elapsed_sec": float(time.time() - t0),
        "results": model_results,
    }
    metrics_json_path.write_text(json.dumps(payload, indent=2))

    print(f"Saved metrics JSON: {metrics_json_path}")
    print(f"Models evaluated: {len(model_results)}")


if __name__ == "__main__":
    main()
