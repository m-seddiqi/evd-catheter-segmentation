from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image, ImageFilter
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision.transforms.functional import InterpolationMode



@dataclass(frozen=True)
class GeomAugParams:
    do_hflip: bool
    do_vflip: bool
    do_affine: bool
    angle_deg: float
    scale: float
    translate_xy: Tuple[int, int]          # (tx, ty) in pixels
    shear: float                           # keep scalar 0.0 for now


def augment_geometric(
    img: Image.Image,
    mask: Image.Image,
    *,
    # flips
    hflip_p: float = 0.5,
    vflip_p: float = 0.0,                  # recommended: 0 for most medical
    # affine
    affine_p: float = 0.7,
    rot_range: Tuple[float, float] = (-10.0, 10.0),   # degrees
    scale_range: Tuple[float, float] = (0.9, 1.25),   # avoid aggressive downscale for thin objects
    translate_pct: float = 0.05,           # smaller than 0.10 to reduce “empty crop” risk
    # fill values
    img_pad_value: int = 0,
    mask_pad_value: int = 0,
    params: Optional[GeomAugParams] = None,
) -> Tuple[Image.Image, Image.Image, GeomAugParams]:
    """
    Segmentation-safe geometric augmentations (img + mask)

    - Uses BILINEAR for image, NEAREST for mask.
    - Uses img_pad_value/mask_pad_value for fill (avoid hardcoded 0 corners).
    - vflip defaults to 0.0 (often not physically plausible in medical).
    - If `params` is provided, applies exactly those decisions/values
      Otherwise samples new params and returns them.

    Returns:
        (img_aug, mask_aug, params_used)
    """
    # -----------------------
    # sample / reuse params
    # -----------------------
    if params is None:
        do_hflip = (random.random() < hflip_p)
        do_vflip = (random.random() < vflip_p)
        do_affine = (random.random() < affine_p)

        angle = random.uniform(*rot_range) if do_affine else 0.0
        scale = random.uniform(*scale_range) if do_affine else 1.0
        shear = 0.0

        w, h = img.size
        if do_affine:
            max_dx = translate_pct * w
            max_dy = translate_pct * h
            tx = int(round(random.uniform(-max_dx, max_dx)))
            ty = int(round(random.uniform(-max_dy, max_dy)))
        else:
            tx, ty = 0, 0

        params = GeomAugParams(
            do_hflip=do_hflip,
            do_vflip=do_vflip,
            do_affine=do_affine,
            angle_deg=float(angle),
            scale=float(scale),
            translate_xy=(int(tx), int(ty)),
            shear=float(shear),
        )

    # -----------------------
    # apply transforms
    # -----------------------
    if params.do_hflip:
        img = TF.hflip(img)
        mask = TF.hflip(mask)

    if params.do_vflip:
        img = TF.vflip(img)
        mask = TF.vflip(mask)

    if params.do_affine:
        img = TF.affine(
            img,
            angle=params.angle_deg,
            translate=params.translate_xy,
            scale=params.scale,
            shear=params.shear,
            interpolation=InterpolationMode.BILINEAR,
            fill=img_pad_value,
        )
        mask = TF.affine(
            mask,
            angle=params.angle_deg,
            translate=params.translate_xy,
            scale=params.scale,
            shear=params.shear,
            interpolation=InterpolationMode.NEAREST,
            fill=mask_pad_value,
        )

    return img, mask, params





@dataclass(frozen=True)
class PhotoAugParams:
    do_blur: bool
    blur_radius: float
    do_catheter_motion_blur: bool
    catheter_motion_blur_kernel: int
    catheter_motion_blur_angle_deg: float
    do_catheter_specular: bool
    catheter_specular_centers_xy: Tuple[Tuple[float, float], ...]
    catheter_specular_radii_px: Tuple[float, ...]
    catheter_specular_strengths: Tuple[float, ...]
    do_exposure: bool
    exposure_ev: float
    do_jitter: bool
    # jitter factors are multiplicative: 1.0 = no change
    brightness: float
    contrast: float
    saturation: float
    hue: float  # additive in [-0.5, 0.5] per torchvision convention


def augment_photometric(
    img: Image.Image,
    mask: Optional[Image.Image] = None,
    *,
    # keep these gentle for medical imagery
    blur_p: float = 0.2,
    blur_radius_range: Tuple[float, float] = (0.1, 1.2),
    catheter_motion_blur_p: float = 0.0,
    catheter_motion_blur_kernel_range: Tuple[float, float] = (7.0, 15.0),
    catheter_motion_blur_angle_range: Tuple[float, float] = (0.0, 180.0),
    catheter_specular_p: float = 0.0,
    catheter_specular_count_range: Tuple[float, float] = (1.0, 3.0),
    catheter_specular_radius_frac_range: Tuple[float, float] = (0.01, 0.05),
    catheter_specular_strength_range: Tuple[float, float] = (0.20, 0.55),
    exposure_p: float = 0.35,
    exposure_ev_range: Tuple[float, float] = (-0.7, 0.7),
    jitter_p: float = 0.7,
    brightness_range: Tuple[float, float] = (0.90, 1.10),
    contrast_range: Tuple[float, float] = (0.85, 1.15),
    saturation_range: Tuple[float, float] = (0.95, 1.05),  # often keep tight for medical
    hue_range: Tuple[float, float] = (-0.02, 0.02),         # usually tiny / or set to (0,0)
    params: Optional[PhotoAugParams] = None,
) -> Tuple[Image.Image, PhotoAugParams]:
    """
    Photometric (appearance) augmentations for the IMAGE ONLY (mask unchanged).

    Design choices for catheter/medical:
      - Gentle jitter (avoid unrealistic colors).
      - Blur is occasional and mild.
      - Exposure simulation is sampled in EV stops (factor = 2 ** EV).
      - Returns params so you can reuse them for (share) or perturb slightly.

    usage:
      - If you want identical appearance: call on left, reuse params for right.
      - If you want mild camera mismatch: sample independently or tweak ranges slightly.

    Returns:
      (img_aug, params_used)
    """
    if params is None:
        do_blur = (random.random() < blur_p)
        blur_radius = random.uniform(*blur_radius_range) if do_blur else 0.0

        do_catheter_motion_blur = (mask is not None) and (random.random() < catheter_motion_blur_p)
        if do_catheter_motion_blur:
            k_lo = int(round(catheter_motion_blur_kernel_range[0]))
            k_hi = int(round(catheter_motion_blur_kernel_range[1]))
            kernel_size = random.randint(min(k_lo, k_hi), max(k_lo, k_hi))
            if kernel_size < 3:
                kernel_size = 3
            if kernel_size % 2 == 0:
                kernel_size += 1
            motion_angle_deg = random.uniform(*catheter_motion_blur_angle_range)
        else:
            kernel_size = 0
            motion_angle_deg = 0.0

        do_catheter_specular = False
        spec_centers: Tuple[Tuple[float, float], ...] = ()
        spec_radii: Tuple[float, ...] = ()
        spec_strengths: Tuple[float, ...] = ()
        if (mask is not None) and (random.random() < catheter_specular_p):
            mask_t = TF.pil_to_tensor(mask)
            fg_yx = torch.nonzero(mask_t[0] > 0, as_tuple=False)  # (N,2) as (y,x)
            if fg_yx.shape[0] > 0:
                do_catheter_specular = True
                n_lo = int(round(catheter_specular_count_range[0]))
                n_hi = int(round(catheter_specular_count_range[1]))
                n_spots = max(1, random.randint(min(n_lo, n_hi), max(n_lo, n_hi)))
                h = int(mask_t.shape[1])
                w = int(mask_t.shape[2])
                min_hw = float(min(h, w))
                centers = []
                radii = []
                strengths = []
                for _ in range(n_spots):
                    idx = random.randrange(int(fg_yx.shape[0]))
                    y = int(fg_yx[idx, 0].item())
                    x = int(fg_yx[idx, 1].item())
                    rad_frac = random.uniform(*catheter_specular_radius_frac_range)
                    radius_px = max(1.0, rad_frac * min_hw)
                    strength = random.uniform(*catheter_specular_strength_range)
                    centers.append((float(x), float(y)))
                    radii.append(float(radius_px))
                    strengths.append(float(strength))
                spec_centers = tuple(centers)
                spec_radii = tuple(radii)
                spec_strengths = tuple(strengths)

        do_exposure = (random.random() < exposure_p)
        exposure_ev = random.uniform(*exposure_ev_range) if do_exposure else 0.0

        do_jitter = (random.random() < jitter_p)
        if do_jitter:
            b = random.uniform(*brightness_range)
            c = random.uniform(*contrast_range)
            s = random.uniform(*saturation_range)
            h = random.uniform(*hue_range)
        else:
            b, c, s, h = 1.0, 1.0, 1.0, 0.0

        params = PhotoAugParams(
            do_blur=do_blur,
            blur_radius=float(blur_radius),
            do_catheter_motion_blur=do_catheter_motion_blur,
            catheter_motion_blur_kernel=int(kernel_size),
            catheter_motion_blur_angle_deg=float(motion_angle_deg),
            do_catheter_specular=do_catheter_specular,
            catheter_specular_centers_xy=spec_centers,
            catheter_specular_radii_px=spec_radii,
            catheter_specular_strengths=spec_strengths,
            do_exposure=do_exposure,
            exposure_ev=float(exposure_ev),
            do_jitter=do_jitter,
            brightness=float(b),
            contrast=float(c),
            saturation=float(s),
            hue=float(h),
        )

    # Apply blur
    if params.do_blur and params.blur_radius > 0.0:
        img = img.filter(ImageFilter.GaussianBlur(radius=params.blur_radius))

    # Motion blur catheter foreground only (mask == 1 / 255).
    if (
        params.do_catheter_motion_blur
        and mask is not None
        and params.catheter_motion_blur_kernel > 0
    ):
        img_t = TF.pil_to_tensor(img).to(torch.float32)  # (C,H,W)
        mask_t = TF.pil_to_tensor(mask)  # (1,H,W)
        fg = (mask_t > 0).to(dtype=img_t.dtype)
        if torch.count_nonzero(fg) > 0:
            k = int(params.catheter_motion_blur_kernel)
            theta = math.radians(float(params.catheter_motion_blur_angle_deg))
            coords = torch.arange(k, dtype=img_t.dtype) - ((k - 1) / 2.0)
            yy, xx = torch.meshgrid(coords, coords, indexing="ij")
            t = (xx * math.cos(theta)) + (yy * math.sin(theta))
            d = (-xx * math.sin(theta)) + (yy * math.cos(theta))
            half = (k - 1) / 2.0
            kernel_2d = ((torch.abs(d) <= 0.5) & (torch.abs(t) <= half)).to(img_t.dtype)
            if float(kernel_2d.sum()) <= 0.0:
                kernel_2d[k // 2, k // 2] = 1.0
            kernel_2d = kernel_2d / kernel_2d.sum()

            c, _, _ = img_t.shape
            weight = kernel_2d.view(1, 1, k, k).repeat(c, 1, 1, 1)
            padded = F.pad(img_t.unsqueeze(0), (k // 2, k // 2, k // 2, k // 2), mode="reflect")
            blurred = F.conv2d(padded, weight, groups=c).squeeze(0)
            img_t = (fg * blurred) + ((1.0 - fg) * img_t)
            img_t = torch.clamp(img_t, min=0.0, max=255.0).to(torch.uint8)
            img = TF.to_pil_image(img_t)

    # Synthetic specular highlights on catheter foreground only.
    if (
        params.do_catheter_specular
        and mask is not None
        and len(params.catheter_specular_centers_xy) > 0
    ):
        img_t = TF.pil_to_tensor(img).to(torch.float32)  # (C,H,W)
        mask_t = TF.pil_to_tensor(mask)  # (1,H,W)
        fg = (mask_t > 0).to(dtype=img_t.dtype)[0]  # (H,W)
        if torch.count_nonzero(fg) > 0:
            _, h, w = img_t.shape
            yy = torch.arange(h, dtype=img_t.dtype, device=img_t.device).view(h, 1)
            xx = torch.arange(w, dtype=img_t.dtype, device=img_t.device).view(1, w)
            highlight = torch.zeros((h, w), dtype=img_t.dtype, device=img_t.device)

            for (cx, cy), radius_px, strength in zip(
                params.catheter_specular_centers_xy,
                params.catheter_specular_radii_px,
                params.catheter_specular_strengths,
            ):
                sigma = max(1.0, float(radius_px) * 0.5)
                dist2 = ((xx - float(cx)) ** 2 + (yy - float(cy)) ** 2) / (2.0 * sigma * sigma)
                blob = torch.exp(-dist2) * float(strength)
                highlight = torch.maximum(highlight, blob)

            highlight = torch.clamp(highlight * fg, min=0.0, max=1.0)
            img_t = img_t + ((255.0 - img_t) * highlight.unsqueeze(0))
            img_t = torch.clamp(img_t, min=0.0, max=255.0).to(torch.uint8)
            img = TF.to_pil_image(img_t)

    # Simulate random camera exposure by scaling intensities and clipping.
    if params.do_exposure and params.exposure_ev != 0.0:
        exposure_factor = float(2.0 ** params.exposure_ev)
        img_t = TF.pil_to_tensor(img).to(torch.float32)
        img_t = torch.clamp(img_t * exposure_factor, min=0.0, max=255.0).to(torch.uint8)
        img = TF.to_pil_image(img_t)

    # Apply jitter (order is a design choice; this is a common, stable order)
    if params.do_jitter:
        if params.brightness != 1.0:
            img = TF.adjust_brightness(img, params.brightness)
        if params.contrast != 1.0:
            img = TF.adjust_contrast(img, params.contrast)
        if params.saturation != 1.0:
            img = TF.adjust_saturation(img, params.saturation)
        if params.hue != 0.0:
            img = TF.adjust_hue(img, params.hue)

    return img, params
