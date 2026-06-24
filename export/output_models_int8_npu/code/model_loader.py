from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from fastvit_models.factory import build_model as build_fastvit_model
from other_models.factory import build_model as build_other_model


@dataclass(frozen=True)
class ModelSpec:
    source_group: str
    model_key: str
    builder_kind: str
    variant: str
    weights_path: Path
    run_dir: Path | None = None
    num_classes: int = 2
    fpn_dim: int = 256
    run_cfg: Dict[str, Any] | None = None


class HubWrapper(torch.nn.Module):
    """QAI Hub wrapper: return a tensor, not a dict."""

    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        out = self.model(image)
        if isinstance(out, dict):
            if "out" in out:
                return out["out"]
            if len(out) == 1:
                return next(iter(out.values()))
            raise KeyError(f"Unexpected dict output keys: {list(out.keys())}")
        return out


class HubWrapperInputNHWC(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, image):
        image_nchw = image.permute(0, 3, 1, 2).contiguous()
        out = self.model(image_nchw)
        if isinstance(out, dict):
            if "out" in out:
                return out["out"]
            if len(out) == 1:
                return next(iter(out.values()))
            raise KeyError(f"Unexpected dict output keys: {list(out.keys())}")
        return out


def _resolve_path(project_root: Path, p: str | Path) -> Path:
    path = Path(p)
    return path if path.is_absolute() else (project_root / path).resolve()


def _weights_path_for_run(run_dir: Path) -> Path | None:
    for name in ("best_model.pth", "model.pth", "model.pt"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _config_path_for_run(run_dir: Path) -> Path | None:
    for name in ("effective_config.json", "config.json"):
        path = run_dir / name
        if path.is_file():
            return path
    return None


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def _infer_builder_kind(variant: str) -> str:
    return "fastvit" if variant.startswith("fastvit_") else "other"


def _model_cfg_from_run_cfg(run_cfg: Dict[str, Any]) -> Dict[str, Any]:
    model_cfg = dict(run_cfg["model"])
    if "variant" not in model_cfg:
        raise KeyError("Model config is missing required key: model.variant")
    return model_cfg


def direct_model_spec(
    project_root: Path,
    weights_path: str | Path,
    variant: str | None = None,
    model_key: str | None = None,
    builder_kind: str | None = None,
    num_classes: int = 2,
    fpn_dim: int = 256,
    source_group: str = "direct_model",
) -> ModelSpec:
    """Build a spec from a checkpoint path, using adjacent run config when available."""

    resolved_weights = _resolve_path(project_root, weights_path)
    if not resolved_weights.is_file():
        raise FileNotFoundError(f"Model weights not found: {resolved_weights}")

    run_dir = resolved_weights.parent
    run_cfg = None
    cfg_path = _config_path_for_run(run_dir)
    if cfg_path is not None:
        run_cfg = _load_json(cfg_path)
        model_cfg = _model_cfg_from_run_cfg(run_cfg)
        variant = variant or str(model_cfg["variant"])
        num_classes = int(model_cfg.get("num_classes", num_classes))
        fpn_dim = int(model_cfg.get("fpn_dim", fpn_dim))

    if variant is None:
        raise ValueError(
            "A direct .pth/.pt checkpoint needs --variant unless a sibling "
            "effective_config.json or config.json is available."
        )

    key = model_key or (run_dir.name if run_cfg is not None else resolved_weights.stem)
    return ModelSpec(
        source_group=source_group,
        model_key=str(key),
        builder_kind=str(builder_kind or _infer_builder_kind(str(variant))),
        variant=str(variant),
        weights_path=resolved_weights,
        run_dir=run_dir if run_cfg is not None else None,
        num_classes=int(num_classes),
        fpn_dim=int(fpn_dim),
        run_cfg=run_cfg,
    )


def _discover_output_model_specs(project_root: Path, model_sources: List[Dict[str, Any]]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for src in model_sources:
        source_dir = _resolve_path(project_root, src["source_dir"])
        if not source_dir.exists():
            continue

        source_group = str(src.get("source_group", source_dir.name))
        requested = [str(x) for x in src.get("select_models", []) if str(x).strip()]
        run_dirs = [source_dir / name for name in requested] if requested else [p for p in sorted(source_dir.iterdir()) if p.is_dir()]

        for run_dir in run_dirs:
            cfg_path = _config_path_for_run(run_dir)
            weights_path = _weights_path_for_run(run_dir)
            if cfg_path is None or weights_path is None:
                continue

            run_cfg = _load_json(cfg_path)
            model_cfg = _model_cfg_from_run_cfg(run_cfg)
            variant = str(model_cfg["variant"])
            specs.append(
                ModelSpec(
                    source_group=source_group,
                    model_key=str(src.get("model_key_prefix", "")) + run_dir.name,
                    builder_kind=str(src.get("builder_kind", _infer_builder_kind(variant))),
                    variant=variant,
                    weights_path=weights_path,
                    run_dir=run_dir,
                    num_classes=int(model_cfg.get("num_classes", 2)),
                    fpn_dim=int(model_cfg.get("fpn_dim", 256)),
                    run_cfg=run_cfg,
                )
            )
    return specs


def _discover_explicit_model_specs(project_root: Path, model_defs: List[Dict[str, Any]]) -> List[ModelSpec]:
    specs: List[ModelSpec] = []
    for item in model_defs:
        weights_path = _resolve_path(project_root, item["weights_path"])
        if not weights_path.is_file():
            raise FileNotFoundError(f"Model weights not found: {weights_path}")
        specs.append(
            ModelSpec(
                source_group=str(item.get("source_group", "explicit_models")),
                model_key=str(item["model_key"]),
                builder_kind=str(item["builder_kind"]),
                variant=str(item["variant"]),
                weights_path=weights_path,
                run_dir=None,
                num_classes=int(item.get("num_classes", 2)),
                fpn_dim=int(item.get("fpn_dim", 256)),
                run_cfg=None,
            )
        )
    return specs


def discover_model_specs(project_root: Path, cfg: Dict[str, Any]) -> List[ModelSpec]:
    if "model_sources" in cfg:
        specs = _discover_output_model_specs(project_root, list(cfg["model_sources"]))
    else:
        specs = _discover_explicit_model_specs(project_root, list(cfg.get("models", [])))
    return sorted(specs, key=lambda s: (s.source_group, s.model_key))


def _load_state_dict(weights_path: Path) -> Dict[str, torch.Tensor]:
    state = torch.load(weights_path, map_location="cpu")
    if isinstance(state, dict) and "model" in state:
        return state["model"]
    return state


def _build_model(spec: ModelSpec) -> torch.nn.Module:
    if spec.builder_kind == "fastvit":
        return build_fastvit_model(variant=spec.variant, num_classes=spec.num_classes)
    if spec.builder_kind == "other":
        return build_other_model(
            variant=spec.variant,
            num_classes=spec.num_classes,
            fpn_dim=spec.fpn_dim,
            pretrained=False,
        )
    raise ValueError(f"Unknown builder_kind: {spec.builder_kind}")


def build_hub_wrapper_for_spec(spec: ModelSpec, channel_first: bool = True):
    model = _build_model(spec)
    model.load_state_dict(_load_state_dict(spec.weights_path), strict=True)

    if spec.builder_kind == "fastvit":
        try:
            from third_party.fvit.models.modules.mobileone import reparameterize_model

            model.backbone.model = reparameterize_model(model.backbone.model)
        except Exception as exc:
            print(f"[WARN] FastViT reparameterization skipped: {exc}")

    model.eval()
    if channel_first:
        return HubWrapper(model).eval()
    return HubWrapperInputNHWC(model).eval()
