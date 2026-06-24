from __future__ import annotations

import os
import random
import re
import gc
import argparse
from pathlib import Path
import torch
import yaml
from typing import Dict, Any, List, Tuple

from fastvit_models.factory import build_model as build_fastvit_model
from other_models.factory import build_model as build_other_model

from utils.env import require_conda_env
from utils.run_dir import prepare_run_dir, save_json
from utils.checkpointing import (
    save_train_state,
    load_train_state,
    save_best_weights,
    EFFECTIVE_CFG_FNAME,
)
from utils.schedulers import build_scheduler
from utils.losses import build_segmentation_loss
from utils.line_distance_loss import maybe_wrap_with_line_prior
from utils.data.indexing import MonoSample, build_splits_custom, build_test_samples
from utils.data.pipeline import (
    MonoTrainPipeline,
    MonoEvalPipeline,
    MonoPipelineConfig,
    parse_crop_sizes,
)
from utils.data.loaders import (
    build_train_val_test_loaders,
    make_mono_dataset,
    make_mono_loader,
    LoaderConfig,
)

from utils.training.amp import set_seed, get_amp_setup
from utils.training.engine import SegmentationTrainer
from utils.training.logging import (
    setup_logger,
    log_run_header,
    format_epoch_line,
    format_test_scene_line,
    get_lr,
)


def _shutdown_loader_workers(loader: Any) -> None:
    it = getattr(loader, "_iterator", None)
    if it is not None and hasattr(it, "_shutdown_workers"):
        it._shutdown_workers()


def _is_fastvit_variant(variant: str) -> bool:
    return str(variant).lower().startswith("fastvit_")


def compute_effective_cfg(cfg: Dict[str, Any], resume_enabled: bool):
    model_cfg = dict(cfg["model"])
    variant = str(model_cfg["variant"])

    if _is_fastvit_variant(variant):
        init_requested = bool(model_cfg.get("backbone_init", False))
        init_effective = bool(init_requested and (not resume_enabled))
        model_cfg["backbone_init_effective"] = init_effective
        model_cfg["pretrained_effective"] = False
    else:
        init_requested = bool(model_cfg.get("pretrained", True))
        init_effective = bool(init_requested and (not resume_enabled))
        model_cfg["pretrained_effective"] = init_effective
        model_cfg["backbone_init_effective"] = False

    effective_cfg = dict(cfg)
    effective_cfg["model"] = model_cfg
    return effective_cfg, init_requested, init_effective


def _as_range2(x, default):
    if x is None:
        return default
    vals = tuple(float(v) for v in x)
    if len(vals) != 2:
        raise ValueError(f"Expected 2 values for range, got: {x}")
    return vals


_EPOCH_DIR_RE = re.compile(r"^e(\d+)$")


def _get_real_val_scene_names(data_cfg: Dict[str, Any]) -> List[str]:
    names = data_cfg.get("real_val_scene_names", [])
    if names is None:
        return []
    if not isinstance(names, list):
        raise ValueError("data.real_val_scene_names must be a list of scene folder names.")
    return sorted(str(name).strip() for name in names if str(name).strip())


def _get_synthetic_base_root(data_cfg: Dict[str, Any]) -> Path:
    catheter_root = str(data_cfg.get("catheter_root", data_cfg.get("root", ""))).strip()
    default_root = str(Path(catheter_root).parent / "synthetic_dataset") if catheter_root else "datasets/synthetic_dataset"
    return Path(str(data_cfg.get("synthetic_root", default_root)).strip())


def _discover_synthetic_epoch_dirs(base_root: Path) -> List[Tuple[int, Path]]:
    if not base_root.is_dir():
        return []
    found: List[Tuple[int, Path]] = []
    for p in sorted(base_root.iterdir()):
        if not p.is_dir():
            continue
        m = _EPOCH_DIR_RE.match(p.name)
        if m is None:
            continue
        found.append((int(m.group(1)), p))
    return sorted(found, key=lambda t: t[0])


def _resolve_synthetic_train_root_for_epoch(
    data_cfg: Dict[str, Any],
    *,
    epoch: int,
) -> Path:
    base_root = _get_synthetic_base_root(data_cfg)
    epoch_num = max(1, int(epoch))
    train_start = int(data_cfg.get("synthetic_train_start_epoch", 1))
    epoch_dirs = [(n, p) for n, p in _discover_synthetic_epoch_dirs(base_root) if n >= train_start]

    if not epoch_dirs:
        raise ValueError(
            f"No synthetic training epoch dirs found under {base_root}. "
            f"Expected folders like e{train_start}, e{train_start + 1}, ...; e0 is reserved for validation."
        )

    idx = (epoch_num - 1) % len(epoch_dirs)
    return epoch_dirs[idx][1]


def _resolve_synthetic_val_root(data_cfg: Dict[str, Any]) -> Path:
    base_root = _get_synthetic_base_root(data_cfg)
    val_folder = str(data_cfg.get("synthetic_val_folder", "e0")).strip()
    val_path = Path(val_folder)
    synthetic_val_root = val_path if val_path.is_absolute() else (base_root / val_folder)
    if not synthetic_val_root.is_dir():
        raise FileNotFoundError(
            f"Configured synthetic validation folder not found: {synthetic_val_root}. "
            "Generate e0 or set data.synthetic_val_folder."
        )
    return synthetic_val_root


def _select_rotating_fraction(
    samples: List[MonoSample],
    *,
    fraction: float,
    epoch: int,
    seed: int,
) -> List[MonoSample]:
    fraction = max(0.0, min(1.0, float(fraction)))
    if fraction <= 0.0 or not samples:
        return []
    if fraction >= 1.0:
        return samples

    cycle_len = max(1, int(round(1.0 / fraction)))
    shuffled = list(samples)
    random.Random(int(seed)).shuffle(shuffled)

    chunk_idx = (max(1, int(epoch)) - 1) % cycle_len
    start = int(round(len(shuffled) * chunk_idx / cycle_len))
    end = int(round(len(shuffled) * (chunk_idx + 1) / cycle_len))
    return shuffled[start:end]


def _resolve_train_val_samples(
    data_cfg: Dict[str, Any],
    *,
    epoch: int,
    fallback_seed: int,
) -> tuple[List[MonoSample], List[MonoSample], List[MonoSample], List[MonoSample]]:
    source = str(data_cfg.get("dataset_source", "catheter_dataset")).strip().lower()
    valid = {"catheter_dataset", "synthetic_data", "both"}
    if source not in valid:
        raise ValueError(f"Invalid data.dataset_source='{source}'. Expected one of {sorted(valid)}")

    train_all: List[MonoSample] = []
    val_all: List[MonoSample] = []

    catheter_train: List[MonoSample] = []
    catheter_val: List[MonoSample] = []
    synthetic_val: List[MonoSample] = []

    if source in {"catheter_dataset", "both"}:
        catheter_root = str(data_cfg.get("catheter_root", data_cfg.get("root", ""))).strip()
        if not catheter_root:
            raise ValueError("Missing data.catheter_root.")
        tr, va = build_splits_custom(
            root_dir=catheter_root,
            mask_subdir="masks",
            val_scene_names=_get_real_val_scene_names(data_cfg),
            val_fraction=None,
            split_seed=42,
        )
        catheter_train = tr
        catheter_val = va

    if source == "both":
        frac = float(data_cfg.get("catheter_fraction_when_both", 1.0))
        catheter_train = _select_rotating_fraction(
            catheter_train,
            fraction=frac,
            epoch=epoch,
            seed=int(data_cfg.get("catheter_fraction_seed", fallback_seed)),
        )

    train_all.extend(catheter_train)
    val_all.extend(catheter_val)

    if source in {"synthetic_data", "both"}:
        synthetic_train_root = _resolve_synthetic_train_root_for_epoch(
            data_cfg,
            epoch=epoch,
        )
        synthetic_train, _ = build_splits_custom(
            root_dir=str(synthetic_train_root),
            mask_subdir=str(data_cfg.get("synthetic_mask_subdir", "masks")),
            val_scene_names=None,
            val_fraction=None,
            split_seed=42,
        )
        synthetic_val_root = _resolve_synthetic_val_root(data_cfg)
        synthetic_val, _ = build_splits_custom(
            root_dir=str(synthetic_val_root),
            mask_subdir=str(data_cfg.get("synthetic_mask_subdir", "masks")),
            val_scene_names=None,
            val_fraction=None,
            split_seed=42,
        )
        train_all.extend(synthetic_train)
        val_all.extend(synthetic_val)

    if len(train_all) == 0:
        raise ValueError(f"No training samples found for data.dataset_source='{source}'.")

    return train_all, val_all, catheter_val, synthetic_val


def _is_mean_scene_selector(selector: str) -> bool:
    s = selector.strip().lower()
    return s in {"mean_test", "mean_scene", "mean_scene_miou", "avg_scene"}


def _is_val_selector(selector: str) -> bool:
    return selector.strip().lower() in {"val_miou", "validation_miou", "val"}


def _mean_scene_miou_equal_weight(scene_miou: Dict[str, float]) -> float:
    vals = [float(v) for v in scene_miou.values()]
    return float(sum(vals) / len(vals)) if vals else 0.0


def _best_score_from_eval(
    *,
    selector: str,
    val_miou: float,
    test_miou_total: float,
    test_scene_miou: Dict[str, float],
) -> float:
    if _is_val_selector(selector):
        return float(val_miou)
    if _is_mean_scene_selector(selector):
        return _mean_scene_miou_equal_weight(test_scene_miou)
    return float(test_scene_miou.get(selector, test_miou_total))


def _normalize_data_source(source: str) -> str:
    mapping = {
        "real": "catheter_dataset",
        "catheter": "catheter_dataset",
        "catheter_dataset": "catheter_dataset",
        "synthetic": "synthetic_data",
        "synthetic_data": "synthetic_data",
        "both": "both",
    }
    key = str(source).strip().lower()
    if key not in mapping:
        raise ValueError("data source must be one of: real, synthetic, both")
    return mapping[key]


def _default_run_name_for_source(cfg: Dict[str, Any], source: str) -> str:
    variant = str(cfg.get("model", {}).get("variant", "model"))
    suffix = {
        "catheter_dataset": "real",
        "synthetic_data": "synthetic",
        "both": "both",
    }[source]
    return f"{variant}_{suffix}"


def _normalize_model_cfg(cfg: Dict[str, Any]) -> None:
    model_cfg = cfg.setdefault("model", {})
    variant = str(model_cfg.get("variant", "")).strip()
    if not variant:
        raise ValueError("Missing model.variant.")

    model_cfg.setdefault("num_classes", 2)
    model_cfg.setdefault("fpn_dim", 256)
    if _is_fastvit_variant(variant):
        model_cfg.setdefault("backbone_init", True)
        model_cfg.setdefault("backbone_init_dir", "third_party/fvit/unfused_checkpoints")
        model_cfg.pop("pretrained", None)
    else:
        model_cfg.setdefault("pretrained", True)
        model_cfg.pop("backbone_init", None)
        model_cfg.pop("backbone_init_dir", None)


def _apply_model_runtime_defaults(cfg: Dict[str, Any], *, amp_override: bool | None) -> None:
    if amp_override is not None:
        cfg["device"]["amp"] = bool(amp_override)
        return

    variant = str(cfg["model"]["variant"]).strip().lower()
    if variant == "convnext_tiny":
        cfg["device"]["amp"] = False


def _build_model_from_cfg(
    cfg: Dict[str, Any],
    *,
    init_effective: bool,
) -> torch.nn.Module:
    model_cfg = cfg["model"]
    variant = str(model_cfg["variant"])
    num_classes = int(model_cfg["num_classes"])
    fpn_dim = int(model_cfg.get("fpn_dim", 256))

    if _is_fastvit_variant(variant):
        model = build_fastvit_model(variant=variant, num_classes=num_classes)
        if init_effective:
            ckpt_name = f"{variant}.pth.tar"
            ckpt_path = os.path.join(str(model_cfg["backbone_init_dir"]), ckpt_name)
            model.initialize_backbone(ckpt_path)
        return model

    return build_other_model(
        variant=variant,
        num_classes=num_classes,
        fpn_dim=fpn_dim,
        pretrained=init_effective,
    )


def _build_optional_eval_loader(
    samples: List[MonoSample],
    *,
    eval_pipeline: MonoEvalPipeline,
    loader_cfg: LoaderConfig,
    compute_line_prior_eval: bool,
) -> Any:
    if not samples:
        return None
    dataset = make_mono_dataset(samples, pipeline=eval_pipeline)
    return make_mono_loader(
        dataset,
        cfg=loader_cfg,
        shuffle=False,
        drop_last=False,
        compute_line_prior=compute_line_prior_eval,
    )


def main(
    cfg_path: str = "configs/train.yaml",
    *,
    model: str | None = None,
    data_source: str | None = None,
    run_name: str | None = None,
    epochs: int | None = None,
    device: str | None = None,
    amp: bool | None = None,
):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    if not isinstance(cfg, dict):
        raise ValueError("Config YAML must be a dict")

    if model is not None:
        cfg.setdefault("model", {})["variant"] = str(model).strip()
    if data_source is not None:
        normalized_source = _normalize_data_source(data_source)
        cfg["data"]["dataset_source"] = normalized_source
    else:
        normalized_source = _normalize_data_source(cfg["data"].get("dataset_source", "catheter_dataset"))
        cfg["data"]["dataset_source"] = normalized_source
    _normalize_model_cfg(cfg)
    if run_name is not None:
        cfg["run"]["run_name"] = run_name
    elif model is not None or data_source is not None:
        cfg["run"]["run_name"] = _default_run_name_for_source(cfg, normalized_source)
    if epochs is not None:
        cfg["train"]["epochs"] = int(epochs)
    if device is not None:
        cfg["device"]["device"] = str(device)
    _apply_model_runtime_defaults(cfg, amp_override=amp)

    require_conda_env(cfg.get("required_conda_env"))

    device = torch.device(cfg["device"]["device"])
    set_seed(int(cfg["device"]["seed"]))

    resume_enabled = bool(cfg["resume"]["enabled"])
    resume_from_dir = cfg["resume"].get("from_dir", None)
    start_epoch, global_step, best_test_miou = 1, 0, -1.0

    if resume_enabled:
        if not resume_from_dir:
            raise ValueError("resume.enabled=true but resume.from_dir is null")
        run_dir = str(resume_from_dir)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = prepare_run_dir(cfg)

    logger = setup_logger(os.path.join(run_dir, "train.log"), verbose=True)
    _log = logger.info

    # configs
    effective_cfg, init_requested, init_effective = compute_effective_cfg(cfg, resume_enabled)
    if not resume_enabled:
        save_json(os.path.join(run_dir, "config.json"), cfg)
    save_json(os.path.join(run_dir, EFFECTIVE_CFG_FNAME), effective_cfg)

    if resume_enabled and init_requested:
        _log("NOTE: resume.enabled=true -> ignoring model initialization/pretraining (resume wins).")

    use_amp = bool(cfg["device"]["amp"]) and (device.type == "cuda")

    log_run_header(
        _log,
        run_dir=run_dir,
        device=device,
        amp=use_amp,
        resume_enabled=resume_enabled,
        resume_from_dir=resume_from_dir,
        cfg=cfg,
        backbone_init_requested=init_requested,
        backbone_init_effective=init_effective,
    )

    # -----------------------
    # data (MONO)
    # -----------------------
    catheter_root = str(cfg["data"].get("catheter_root", cfg["data"].get("root")))
    test_root = cfg["data"].get("test_root", str(Path(catheter_root).parent / "external_test_catheter"))
    test_mask_subdir = str(cfg["data"].get("test_mask_subdir", "masks"))
    test_samples = build_test_samples(test_root, mask_subdir=test_mask_subdir)
    train_samples, val_samples, real_val_samples, synthetic_val_samples = _resolve_train_val_samples(
        cfg["data"],
        epoch=start_epoch,
        fallback_seed=int(cfg["device"]["seed"]),
    )


    crop_sizes = parse_crop_sizes(cfg["data"])
    photo_cfg = cfg["data"].get("photo_aug", {})
    pipe_cfg = MonoPipelineConfig(
        crop_sizes=crop_sizes,
        img_pad_value=int(cfg["data"]["img_pad_value"]),
        mask_pad_value=int(cfg["data"]["mask_pad_value"]),
        photo_blur_p=float(photo_cfg.get("blur_p", 0.30)),
        photo_blur_radius_range=_as_range2(photo_cfg.get("blur_radius_range"), (0.1, 1.6)),
        photo_catheter_motion_blur_p=float(photo_cfg.get("catheter_motion_blur_p", 0.0)),
        photo_catheter_motion_blur_kernel_range=_as_range2(
            photo_cfg.get("catheter_motion_blur_kernel_range"),
            (7.0, 15.0),
        ),
        photo_catheter_motion_blur_angle_range=_as_range2(
            photo_cfg.get("catheter_motion_blur_angle_range"),
            (0.0, 180.0),
        ),
        photo_catheter_specular_p=float(photo_cfg.get("catheter_specular_p", 0.0)),
        photo_catheter_specular_count_range=_as_range2(
            photo_cfg.get("catheter_specular_count_range"),
            (1.0, 3.0),
        ),
        photo_catheter_specular_radius_frac_range=_as_range2(
            photo_cfg.get("catheter_specular_radius_frac_range"),
            (0.01, 0.05),
        ),
        photo_catheter_specular_strength_range=_as_range2(
            photo_cfg.get("catheter_specular_strength_range"),
            (0.20, 0.55),
        ),
        photo_exposure_p=float(photo_cfg.get("exposure_p", 0.35)),
        photo_exposure_ev_range=_as_range2(photo_cfg.get("exposure_ev_range"), (-0.7, 0.7)),
        photo_jitter_p=float(photo_cfg.get("jitter_p", 0.90)),
        photo_brightness_range=_as_range2(photo_cfg.get("brightness_range"), (0.75, 1.25)),
        photo_contrast_range=_as_range2(photo_cfg.get("contrast_range"), (0.70, 1.30)),
        photo_saturation_range=_as_range2(photo_cfg.get("saturation_range"), (0.70, 1.30)),
        photo_hue_range=_as_range2(photo_cfg.get("hue_range"), (-0.08, 0.08)),
    )
    for ch, cw in pipe_cfg.crop_sizes:
        if (ch % 32 != 0) or (cw % 32 != 0):
            _log(
                f"WARNING: data crop bucket {(ch, cw)} is not divisible by 32. "
                "Backbone has a 1/32 stage; use multiples of 32 for cleaner feature alignment."
            )

    loader_cfg_dict = dict(cfg["loader"])
    loader_cfg = LoaderConfig(**loader_cfg_dict)
    line_prior_enabled = bool(((cfg.get("loss", {}) or {}).get("line_prior", {}) or {}).get("enabled", False))
    eval_cfg = cfg.get("eval", {}) or {}
    compute_line_prior_eval = bool(eval_cfg.get("compute_line_prior", False))
    train_pipeline = MonoTrainPipeline(pipe_cfg)
    eval_pipeline = MonoEvalPipeline(pipe_cfg)

    loaders = build_train_val_test_loaders(
        train_samples=train_samples,
        val_samples=val_samples,
        test_samples=test_samples,
        train_pipeline=train_pipeline,
        eval_pipeline=eval_pipeline,
        cfg=loader_cfg,
        compute_line_prior_train=line_prior_enabled,
        compute_line_prior_eval=compute_line_prior_eval,
    )
    train_loader = loaders["train"]
    val_loader = loaders["val"]
    test_loader = loaders["test"]
    real_val_loader = _build_optional_eval_loader(
        real_val_samples,
        eval_pipeline=eval_pipeline,
        loader_cfg=loader_cfg,
        compute_line_prior_eval=compute_line_prior_eval,
    )
    synthetic_val_loader = _build_optional_eval_loader(
        synthetic_val_samples,
        eval_pipeline=eval_pipeline,
        loader_cfg=loader_cfg,
        compute_line_prior_eval=compute_line_prior_eval,
    )

    steps_per_epoch = len(train_loader)
    _log(
        f"Data splits: train={len(train_samples)} val={len(val_samples)} "
        f"(real={len(real_val_samples)} synthetic={len(synthetic_val_samples)}) test={len(test_samples)}"
    )
    _log(
        f"Data loaders: train={len(train_loader)} val={len(val_loader)} "
        f"real_val={len(real_val_loader) if real_val_loader is not None else 0} "
        f"synthetic_val={len(synthetic_val_loader) if synthetic_val_loader is not None else 0} "
        f"test={len(test_loader)} batches"
    )

    # -----------------------
    # model (MONO)
    # -----------------------
    model_obj = _build_model_from_cfg(cfg, init_effective=init_effective).to(device)

    # -----------------------
    # optim/loss/sched
    # -----------------------
    criterion = build_segmentation_loss(cfg["loss"]["name"])
    criterion = maybe_wrap_with_line_prior(seg_loss=criterion, loss_cfg=cfg["loss"])
    criterion = criterion.to(device)
    lr = float(cfg["train"]["lr"])
    wd = float(cfg["train"]["weight_decay"])
    backbone_scale = float(cfg["train"]["backbone_lr_scale"])

    optimizer = torch.optim.AdamW(
        [
            {"params": model_obj.backbone.parameters(), "lr": lr * backbone_scale},
            {"params": model_obj.head.parameters(), "lr": lr},
        ],
        weight_decay=wd,
        betas=(0.9, 0.999),
    )

    warmup_override = 0 if resume_enabled else None
    scheduler = build_scheduler(
        optimizer,
        cfg,
        steps_per_epoch=steps_per_epoch,
        warmup_steps_override=warmup_override,
    )

    amp_setup = get_amp_setup(device, bool(cfg["device"]["amp"]))

    # -----------------------
    # resume
    # -----------------------
    best_scene_name = str(cfg["data"].get("best_scene_name", "mean_scene_miou")).strip()
    last_eval_val_miou = -1.0
    last_eval_per_class_iou = None
    last_eval_test_miou = -1.0
    last_eval_test_scene_miou: Dict[str, float] = {}
    if resume_enabled:
        start_epoch, global_step, ckpt = load_train_state(
            from_dir=str(resume_from_dir),
            model=model_obj,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=(amp_setup[1] if amp_setup[0] else None),
            map_location="cpu",
            steps_per_epoch=steps_per_epoch,
        )
        last_eval_val_miou = float(ckpt.get("val_miou", -1.0))
        last_eval_per_class_iou = ckpt.get("per_class_iou", None)
        last_eval_test_miou = float(ckpt.get("test_miou_total", ckpt.get("val_miou", -1.0)))
        last_eval_test_scene_miou = dict(ckpt.get("test_scene_miou", {}))
        best_test_miou = _best_score_from_eval(
            selector=best_scene_name,
            val_miou=last_eval_val_miou,
            test_miou_total=last_eval_test_miou,
            test_scene_miou=last_eval_test_scene_miou,
        )
        _log(
            f"Resumed from {resume_from_dir} at start_epoch={start_epoch}, "
            f"global_step={global_step} (prev best selector='{best_scene_name}' score={best_test_miou:.4f})."
        )

    epochs = int(cfg["train"]["epochs"])
    source = str(cfg["data"].get("dataset_source", "catheter_dataset")).strip().lower()
    catheter_frac = float(cfg["data"].get("catheter_fraction_when_both", 1.0))
    data_changes_each_epoch = (source in {"synthetic_data", "both"}) or ((source == "both") and catheter_frac < 1.0)
    eval_enabled = bool(eval_cfg.get("enabled", True))
    eval_interval_epochs = max(1, int(eval_cfg.get("interval_epochs", 1)))
    always_eval_final_epoch = bool(eval_cfg.get("always_eval_final_epoch", True))
    val_real_weight = max(0.0, min(1.0, float(cfg["data"].get("val_real_weight", 0.5))))

    if start_epoch != 1 and data_changes_each_epoch:
        train_samples, val_samples, real_val_samples, synthetic_val_samples = _resolve_train_val_samples(
            cfg["data"],
            epoch=start_epoch,
            fallback_seed=int(cfg["device"]["seed"]),
        )
        _shutdown_loader_workers(train_loader)
        _shutdown_loader_workers(val_loader)
        _shutdown_loader_workers(real_val_loader)
        _shutdown_loader_workers(synthetic_val_loader)
        loaders = build_train_val_test_loaders(
            train_samples=train_samples,
            val_samples=val_samples,
            test_samples=test_samples,
            train_pipeline=train_pipeline,
            eval_pipeline=eval_pipeline,
            cfg=loader_cfg,
            compute_line_prior_train=line_prior_enabled,
            compute_line_prior_eval=compute_line_prior_eval,
        )
        train_loader = loaders["train"]
        val_loader = loaders["val"]
        real_val_loader = _build_optional_eval_loader(
            real_val_samples,
            eval_pipeline=eval_pipeline,
            loader_cfg=loader_cfg,
            compute_line_prior_eval=compute_line_prior_eval,
        )
        synthetic_val_loader = _build_optional_eval_loader(
            synthetic_val_samples,
            eval_pipeline=eval_pipeline,
            loader_cfg=loader_cfg,
            compute_line_prior_eval=compute_line_prior_eval,
        )
        _log(
            f"[data_refresh] resume start_epoch={start_epoch}: "
            f"train={len(train_loader)} val={len(val_loader)} "
            f"real_val={len(real_val_loader) if real_val_loader is not None else 0} "
            f"synthetic_val={len(synthetic_val_loader) if synthetic_val_loader is not None else 0}"
        )

    # -----------------------
    # trainer
    # -----------------------
    trainer = SegmentationTrainer(
        cfg=cfg,
        model=model_obj,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        criterion=criterion,
        scheduler=scheduler,
        device=device,
        amp_setup=amp_setup,
        real_val_loader=real_val_loader,
        synthetic_val_loader=synthetic_val_loader,
        val_real_weight=val_real_weight,
    )

    for epoch in range(start_epoch, epochs + 1):
        if epoch > start_epoch and data_changes_each_epoch:
            train_samples, val_samples, real_val_samples, synthetic_val_samples = _resolve_train_val_samples(
                cfg["data"],
                epoch=epoch,
                fallback_seed=int(cfg["device"]["seed"]),
            )
            _shutdown_loader_workers(trainer.train_loader)
            _shutdown_loader_workers(trainer.val_loader)
            _shutdown_loader_workers(trainer.real_val_loader)
            _shutdown_loader_workers(trainer.synthetic_val_loader)
            loaders = build_train_val_test_loaders(
                train_samples=train_samples,
                val_samples=val_samples,
                test_samples=test_samples,
                train_pipeline=train_pipeline,
                eval_pipeline=eval_pipeline,
                cfg=loader_cfg,
                compute_line_prior_train=line_prior_enabled,
                compute_line_prior_eval=compute_line_prior_eval,
            )
            trainer.train_loader = loaders["train"]
            trainer.val_loader = loaders["val"]
            trainer.real_val_loader = _build_optional_eval_loader(
                real_val_samples,
                eval_pipeline=eval_pipeline,
                loader_cfg=loader_cfg,
                compute_line_prior_eval=compute_line_prior_eval,
            )
            trainer.synthetic_val_loader = _build_optional_eval_loader(
                synthetic_val_samples,
                eval_pipeline=eval_pipeline,
                loader_cfg=loader_cfg,
                compute_line_prior_eval=compute_line_prior_eval,
            )
            gc.collect()
            _log(
                f"[data_refresh] epoch={epoch} refreshed loaders: "
                f"train={len(trainer.train_loader)} val={len(trainer.val_loader)} "
                f"real_val={len(trainer.real_val_loader) if trainer.real_val_loader is not None else 0} "
                f"synthetic_val={len(trainer.synthetic_val_loader) if trainer.synthetic_val_loader is not None else 0}"
            )

        should_eval = (
            (eval_enabled and ((epoch - 1) % eval_interval_epochs == 0))
            or (always_eval_final_epoch and epoch == epochs)
        )
        metrics, global_step = trainer.run_epoch(
            epoch=epoch,
            global_step=global_step,
            evaluate=should_eval,
        )

        _log(
            format_epoch_line(
                epoch=epoch,
                epochs=epochs,
                global_step=global_step,
                lr=get_lr(optimizer),
                train_loss=metrics.train_loss,
                val_loss=metrics.val_loss,
                val_miou=metrics.val_miou,
                test_miou=metrics.test_miou,
                dt_sec=metrics.dt_sec,
                evaluated=metrics.evaluated,
            )
        )
        if metrics.evaluated:
            last_eval_val_miou = metrics.val_miou
            last_eval_per_class_iou = metrics.val_per_class_iou
            last_eval_test_miou = metrics.test_miou
            last_eval_test_scene_miou = dict(metrics.test_scene_miou)
            if metrics.val_component_miou:
                comps = metrics.val_component_miou
                _log(
                    "val_component_mIoU: "
                    f"real={comps['real']:.4f} (w={comps['real_weight']:.2f}) | "
                    f"synthetic={comps['synthetic']:.4f} (w={comps['synthetic_weight']:.2f})"
                )
            _log(format_test_scene_line(metrics.test_scene_miou))

        save_train_state(
            run_dir=run_dir,
            epoch=epoch,
            global_step=global_step,
            model=model_obj,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=(trainer.scaler if trainer.amp_enabled else None),
            cfg=cfg,
            val_miou=last_eval_val_miou,
            per_class_iou=last_eval_per_class_iou,
            test_miou_total=last_eval_test_miou,
            test_scene_miou=last_eval_test_scene_miou,
        )

        if not metrics.evaluated:
            continue

        current_best_score = _best_score_from_eval(
            selector=best_scene_name,
            val_miou=metrics.val_miou,
            test_miou_total=metrics.test_miou,
            test_scene_miou=metrics.test_scene_miou,
        )
        if current_best_score > best_test_miou:
            best_test_miou = current_best_score
            save_best_weights(run_dir, model_obj)
            if _is_val_selector(best_scene_name):
                _log(f"  -> NEW BEST (by val mIoU): {best_test_miou:.4f} @ epoch={epoch}")
            elif _is_mean_scene_selector(best_scene_name):
                _log(f"  -> NEW BEST (by equal-weight mean test scene mIoU): {best_test_miou:.4f} @ epoch={epoch}")
            else:
                _log(f"  -> NEW BEST (by test scene mIoU: {best_scene_name}): {best_test_miou:.4f} @ epoch={epoch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train an EVD catheter segmentation model.")
    parser.add_argument("--config", default="configs/train.yaml", help="Path to the YAML training config.")
    parser.add_argument(
        "--model",
        default=None,
        help="Model variant, e.g. fastvit_t8, fastvit_sa12, convnext_tiny, mobilenetv3_large_100.",
    )
    parser.add_argument(
        "--data-source",
        choices=["real", "synthetic", "both", "catheter_dataset", "synthetic_data"],
        default=None,
        help="Training data source. 'real' maps to catheter_dataset.",
    )
    parser.add_argument("--epochs", type=int, default=None, help="Override train.epochs.")
    parser.add_argument("--device", default=None, help="Torch device, e.g. cuda:0, cuda:1, or cpu.")
    amp_group = parser.add_mutually_exclusive_group()
    amp_group.add_argument("--amp", dest="amp", action="store_true", default=None, help="Enable AMP.")
    amp_group.add_argument("--no-amp", dest="amp", action="store_false", help="Disable AMP.")
    parser.add_argument("--run-name", default=None, help="Optional output run folder name under run.out_dir.")
    args = parser.parse_args()
    main(
        args.config,
        model=args.model,
        data_source=args.data_source,
        run_name=args.run_name,
        epochs=args.epochs,
        device=args.device,
        amp=args.amp,
    )
