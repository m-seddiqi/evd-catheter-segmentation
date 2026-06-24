# utils/data/pipeline.py
from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional, Tuple

from PIL import Image

from utils.data.transforms import (
    resize_keep_aspect,
    letterbox_to_size,
    crop_pair,
    CropParams,
)

from utils.data.augmentations import (
    augment_geometric,
    augment_photometric,
    GeomAugParams,
    PhotoAugParams,
)


CropSize = Tuple[int, int]  # (H, W)


def _as_crop_size(x: Iterable[int]) -> CropSize:
    vals = tuple(int(v) for v in x)
    if len(vals) != 2:
        raise ValueError(f"Crop size must be length-2 [H, W], got: {x}")
    h, w = vals
    if h <= 0 or w <= 0:
        raise ValueError(f"Crop size must be positive, got: {(h, w)}")
    return h, w


def parse_crop_sizes(data_cfg: Dict[str, Any]) -> Tuple[CropSize, ...]:
    """
    Supports either:
      - data.crop_size: [H, W]
      - data.crop_sizes: [[H, W], [H, W], ...]
    """
    if "crop_sizes" in data_cfg and data_cfg["crop_sizes"] is not None:
        raw = data_cfg["crop_sizes"]
        if not isinstance(raw, (list, tuple)) or len(raw) == 0:
            raise ValueError("data.crop_sizes must be a non-empty list of [H, W].")
        sizes = tuple(_as_crop_size(x) for x in raw)
    elif "crop_size" in data_cfg and data_cfg["crop_size"] is not None:
        sizes = (_as_crop_size(data_cfg["crop_size"]),)
    else:
        sizes = ((512, 512),)
    return sizes


def choose_closest_crop_size(crop_sizes: Tuple[CropSize, ...], *, img_w: int, img_h: int) -> CropSize:
    """
    Picks the bucket with closest aspect ratio to input image.
    """
    if len(crop_sizes) == 1:
        return crop_sizes[0]

    img_ar = float(img_w) / float(img_h)
    best = crop_sizes[0]
    best_dist = float("inf")

    for ch, cw in crop_sizes:
        bucket_ar = float(cw) / float(ch)
        # Log-distance is symmetric for wide/tall mismatches.
        dist = abs(math.log(bucket_ar) - math.log(img_ar))
        if dist < best_dist:
            best_dist = dist
            best = (ch, cw)

    return best


@dataclass(frozen=True)
class MonoPipelineConfig:
    crop_sizes: Tuple[CropSize, ...] = ((512, 512),)
    img_pad_value: int = 0
    mask_pad_value: int = 0
    photo_blur_p: float = 0.30
    photo_blur_radius_range: Tuple[float, float] = (0.1, 1.6)
    photo_catheter_motion_blur_p: float = 0.0
    photo_catheter_motion_blur_kernel_range: Tuple[float, float] = (7.0, 15.0)
    photo_catheter_motion_blur_angle_range: Tuple[float, float] = (0.0, 180.0)
    photo_catheter_specular_p: float = 0.0
    photo_catheter_specular_count_range: Tuple[float, float] = (1.0, 3.0)
    photo_catheter_specular_radius_frac_range: Tuple[float, float] = (0.01, 0.05)
    photo_catheter_specular_strength_range: Tuple[float, float] = (0.20, 0.55)
    photo_exposure_p: float = 0.35
    photo_exposure_ev_range: Tuple[float, float] = (-0.7, 0.7)
    photo_jitter_p: float = 0.90
    photo_brightness_range: Tuple[float, float] = (0.75, 1.25)
    photo_contrast_range: Tuple[float, float] = (0.70, 1.30)
    photo_saturation_range: Tuple[float, float] = (0.70, 1.30)
    photo_hue_range: Tuple[float, float] = (-0.08, 0.08)


class MonoTrainPipeline:
    """
    TRAIN pipeline (always-on transforms + augmentations).

    Always applies:
      - resize_keep_aspect
      - crop_pair (random_crop=True)
      - augment_geometric
      - augment_photometric

    Meta:
      - Consumes params from meta if present
      - Writes back params actually used (crop_params, geom_params, photo_params)
    """

    def __init__(self, cfg: MonoPipelineConfig):
        self.cfg = cfg

    def __call__(
        self,
        *,
        img: Image.Image,
        mask: Optional[Image.Image],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = meta if meta is not None else {}
        crop_size = random.choice(self.cfg.crop_sizes)
        meta["crop_size_used"] = crop_size

        # Keep aspect ratio and ensure both sides can support crop_size.
        # This enables true random crop (instead of no-op after exact letterbox).
        img, mask = resize_keep_aspect(
            img, mask,
            crop_size=crop_size,
        )

        crop_params: Optional[CropParams] = meta.get("crop_params")
        img, mask, crop_params = crop_pair(
            img,
            mask,
            crop_size=crop_size,
            random_crop=True,
            img_pad_value=self.cfg.img_pad_value,
            mask_pad_value=self.cfg.mask_pad_value,
            params=crop_params,
        )
        meta["crop_params"] = crop_params

        geom_params: Optional[GeomAugParams] = meta.get("geom_params")
        img, mask, geom_params = augment_geometric(
            img,
            mask,
            img_pad_value=self.cfg.img_pad_value,
            mask_pad_value=self.cfg.mask_pad_value,
            params=geom_params,
        )
        meta["geom_params"] = geom_params

        photo_params: Optional[PhotoAugParams] = meta.get("photo_params")
        img, photo_params = augment_photometric(
            img,
            mask,
            blur_p=self.cfg.photo_blur_p,
            blur_radius_range=self.cfg.photo_blur_radius_range,
            catheter_motion_blur_p=self.cfg.photo_catheter_motion_blur_p,
            catheter_motion_blur_kernel_range=self.cfg.photo_catheter_motion_blur_kernel_range,
            catheter_motion_blur_angle_range=self.cfg.photo_catheter_motion_blur_angle_range,
            catheter_specular_p=self.cfg.photo_catheter_specular_p,
            catheter_specular_count_range=self.cfg.photo_catheter_specular_count_range,
            catheter_specular_radius_frac_range=self.cfg.photo_catheter_specular_radius_frac_range,
            catheter_specular_strength_range=self.cfg.photo_catheter_specular_strength_range,
            exposure_p=self.cfg.photo_exposure_p,
            exposure_ev_range=self.cfg.photo_exposure_ev_range,
            jitter_p=self.cfg.photo_jitter_p,
            brightness_range=self.cfg.photo_brightness_range,
            contrast_range=self.cfg.photo_contrast_range,
            saturation_range=self.cfg.photo_saturation_range,
            hue_range=self.cfg.photo_hue_range,
            params=photo_params,
        )
        meta["photo_params"] = photo_params

        return {
            "img": img,
            "mask": mask,
            "meta": meta,
        }


class MonoEvalPipeline:
    """
    VAL / TEST / REALTIME:
      - letterbox_to_size
      - no augmentation
    """

    def __init__(self, cfg: MonoPipelineConfig):
        self.cfg = cfg

    def __call__(
        self,
        *,
        img: Image.Image,
        mask: Optional[Image.Image],
        meta: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        meta = meta if meta is not None else {}
        w, h = img.size
        crop_size = choose_closest_crop_size(self.cfg.crop_sizes, img_w=w, img_h=h)
        meta["crop_size_used"] = crop_size

        img, mask = letterbox_to_size(
            img, mask,
            crop_size=crop_size,
            img_pad_value=self.cfg.img_pad_value,
            mask_pad_value=self.cfg.mask_pad_value,
        )

        return {
            "img": img,
            "mask": mask,
            "meta": meta,
        }
