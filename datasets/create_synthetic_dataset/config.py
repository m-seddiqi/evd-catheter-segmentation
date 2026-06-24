from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


Vec3Range = Dict[str, Tuple[float, float]]
GLB_PATTERNS = ("*.glb", "*.GLB")


def _v3(x: Tuple[float, float], y: Tuple[float, float], z: Tuple[float, float]) -> Vec3Range:
    return {"x": x, "y": y, "z": z}


def _uniform_scale(r: Tuple[float, float]) -> Vec3Range:
    return {"x": r, "y": r, "z": r}


@dataclass(frozen=True)
class CameraConfig:
    position: Tuple[float, float, float]
    target: Tuple[float, float, float]
    up: Tuple[float, float, float]
    fov_y_deg: float
    near: float
    far: float
    width: int
    height: int


@dataclass(frozen=True)
class TransformRanges:
    rotation_deg: Vec3Range
    translation: Vec3Range
    scale: Vec3Range


@dataclass(frozen=True)
class CategoryRule:
    probability: float
    transform_ranges: TransformRanges


@dataclass(frozen=True)
class PathsConfig:
    root: Path
    model_root: Path
    background_dir: Path
    output_dir: Path


@dataclass(frozen=True)
class SceneSynthConfig:
    paths: PathsConfig
    camera: CameraConfig
    catheter_ranges: TransformRanges
    category_rules: Dict[str, CategoryRule]
    image_format: str
    mask_format: str
    jpeg_quality: int
    render_supersample: int
    smooth_shading: bool
    enable_directional_shadows: bool
    catheter_color_override_prob: float
    catheter_color_palette: Dict[str, Tuple[float, float, float]]
    frames_per_scene: int
    catheter_motion_prob: float
    other_motion_prob: float
    catheter_rotation_inc_deg: Tuple[float, float, float]
    other_rotation_inc_deg: Tuple[float, float, float]
    catheter_translation_inc: Tuple[float, float, float]
    other_translation_inc: Tuple[float, float, float]
    catheter_scale_inc: float
    other_scale_inc: float
    background_crop_ratio: float
    min_catheter_mask_pixels: int
    max_visibility_retries: int


def default_config(generator_root: Path) -> SceneSynthConfig:
    model_root = generator_root / "3d_tools"
    background_dir = model_root / "background_img"
    output_dir = generator_root.parent / "synthetic_dataset" / "e1"

    return SceneSynthConfig(
        paths=PathsConfig(
            root=generator_root,
            model_root=model_root,
            background_dir=background_dir,
            output_dir=output_dir,
        ),
        camera=CameraConfig(
            position=(0.0, -0.5, 0.0),
            target=(0.0, 0.0, 0.0),
            up=(0.0, 0.0, 0.05),
            fov_y_deg=40.0,
            near=0.05,
            far=1.2,
            width=1920,
            height=1080,
        ),
        catheter_ranges=TransformRanges(
            rotation_deg=_v3((0.0, 90.0), (0.0, 360.0), (-90.0, 90.0)),
            translation=_v3((-0.10, 0.10), (-0.10, 0.01), (-0.10, 0.10)),
            scale=_uniform_scale((1.0, 2.0)),
        ),
        category_rules={
            # Optional example tool category. To add more tools, create another
            # folder under 3d_tools/ and add one category-level rule here.
            "clippers": CategoryRule(
                probability=0.0,
                transform_ranges=TransformRanges(
                    rotation_deg=_v3((0.0, 20.0), (0.0, 360.0), (0.0, 360.0)),
                    translation=_v3((-0.25, 0.25), (-0.05, 0.20), (-0.05, 0.10)),
                    scale=_uniform_scale((0.80, 1.20)),
                ),
            ),
        },
        image_format="jpg",
        mask_format="png",
        jpeg_quality=95,
        render_supersample=2,
        smooth_shading=True,
        enable_directional_shadows=True,
        catheter_color_override_prob=0.8,
        catheter_color_palette={
            "metallic_creamy": (0.96, 0.94, 0.88),
            "white": (0.98, 0.98, 0.98),
            "orange": (0.95, 0.50, 0.12),
            "blue": (0.20, 0.45, 0.92),
        },
        frames_per_scene=10,
        catheter_motion_prob=1.0,
        other_motion_prob=0.8,
        catheter_rotation_inc_deg=(3.0, 3.0, 3.0),
        other_rotation_inc_deg=(3.0, 3.0, 3.0),
        catheter_translation_inc=(0.05, 0.05, 0.05),
        other_translation_inc=(0.05, 0.05, 0.05),
        catheter_scale_inc=0.02,
        other_scale_inc=0.02,
        background_crop_ratio=0.80,
        min_catheter_mask_pixels=100,
        max_visibility_retries=25,
    )


def list_glb_files(model_root: Path, category: str) -> List[Path]:
    files: List[Path] = []
    for pattern in GLB_PATTERNS:
        files.extend((model_root / category).glob(pattern))
    return sorted(set(files))
