from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import SceneSynthConfig, TransformRanges, list_glb_files


BACKGROUND_PATTERNS = (
    "*.png",
    "*.jpg",
    "*.jpeg",
    "*.bmp",
    "*.tif",
    "*.tiff",
    "*.PNG",
    "*.JPG",
    "*.JPEG",
    "*.BMP",
    "*.TIF",
    "*.TIFF",
)


@dataclass
class ObjectSpec:
    category: str
    rel_path: str
    rotation_deg: np.ndarray
    translation: np.ndarray
    scale: np.ndarray
    color_override: Optional[str] = None


@dataclass
class SceneSpec:
    background_path: Path
    background_name: str
    objects: List[ObjectSpec]


def _sample_vec3(rng: np.random.Generator, ranges: Dict[str, tuple[float, float]]) -> np.ndarray:
    return np.array(
        [
            rng.uniform(*ranges["x"]),
            rng.uniform(*ranges["y"]),
            rng.uniform(*ranges["z"]),
        ],
        dtype=float,
    )


def _sample_uniform_scale(rng: np.random.Generator, ranges: Dict[str, tuple[float, float]]) -> np.ndarray:
    low = max(ranges["x"][0], ranges["y"][0], ranges["z"][0])
    high = min(ranges["x"][1], ranges["y"][1], ranges["z"][1])
    if low > high:
        raise ValueError(f"Scale ranges have no overlap for uniform sampling: {ranges}")
    scale = float(rng.uniform(low, high))
    return np.array([scale, scale, scale], dtype=float)


def sample_transform(rng: np.random.Generator, tr: TransformRanges) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        _sample_vec3(rng, tr.rotation_deg),
        _sample_vec3(rng, tr.translation),
        _sample_uniform_scale(rng, tr.scale),
    )


def _choose_random_file(rng: np.random.Generator, files: List[Path], label: str) -> Path:
    if not files:
        raise FileNotFoundError(f"No {label} files found.")
    return files[int(rng.integers(0, len(files)))]


def _background_files(cfg: SceneSynthConfig) -> List[Path]:
    files: List[Path] = []
    for pattern in BACKGROUND_PATTERNS:
        files.extend(cfg.paths.background_dir.glob(pattern))
    return sorted(set(files))


def sample_scene_spec(cfg: SceneSynthConfig, rng: np.random.Generator) -> SceneSpec:
    backgrounds = _background_files(cfg)
    background_path = _choose_random_file(rng, backgrounds, "background")
    background_name = background_path.name

    catheter_files = list_glb_files(cfg.paths.model_root, "catheter")
    catheter_path = _choose_random_file(rng, catheter_files, "catheter GLB")
    rotation, translation, scale = sample_transform(rng, cfg.catheter_ranges)

    color_override: Optional[str] = None
    if rng.random() < cfg.catheter_color_override_prob:
        palette = sorted(cfg.catheter_color_palette.keys())
        color_override = palette[int(rng.integers(0, len(palette)))]

    objects = [
        ObjectSpec(
            category="catheter",
            rel_path=str(catheter_path.relative_to(cfg.paths.model_root)),
            rotation_deg=rotation,
            translation=translation,
            scale=scale,
            color_override=color_override,
        )
    ]

    for category, rule in cfg.category_rules.items():
        if rng.random() >= rule.probability:
            continue

        tool_files = list_glb_files(cfg.paths.model_root, category)
        if not tool_files:
            continue
        tool_path = _choose_random_file(rng, tool_files, f"{category} GLB")
        tool_rel_path = str(tool_path.relative_to(cfg.paths.model_root))
        tool_rotation, tool_translation, tool_scale = sample_transform(rng, rule.transform_ranges)
        objects.append(
            ObjectSpec(
                category=category,
                rel_path=tool_rel_path,
                rotation_deg=tool_rotation,
                translation=tool_translation,
                scale=tool_scale,
            )
        )

    return SceneSpec(
        background_path=background_path,
        background_name=background_name,
        objects=objects,
    )


def apply_incremental_motion(spec: SceneSpec, cfg: SceneSynthConfig, rng: np.random.Generator) -> None:
    for obj in spec.objects:
        if obj.category == "catheter":
            motion_prob = cfg.catheter_motion_prob
            rotation_inc = np.array(cfg.catheter_rotation_inc_deg, dtype=float)
            translation_inc = np.array(cfg.catheter_translation_inc, dtype=float)
            scale_inc = float(cfg.catheter_scale_inc)
        else:
            motion_prob = cfg.other_motion_prob
            rotation_inc = np.array(cfg.other_rotation_inc_deg, dtype=float)
            translation_inc = np.array(cfg.other_translation_inc, dtype=float)
            scale_inc = float(cfg.other_scale_inc)

        if rng.random() > motion_prob:
            continue

        obj.rotation_deg = obj.rotation_deg + rng.uniform(-rotation_inc, rotation_inc)
        obj.translation = obj.translation + rng.uniform(-translation_inc, translation_inc)

        scale = float(obj.scale[0] + rng.uniform(-scale_inc, scale_inc))
        scale = max(scale, 0.05)
        obj.scale = np.array([scale, scale, scale], dtype=float)
