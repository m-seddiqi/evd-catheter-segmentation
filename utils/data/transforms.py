import random
from dataclasses import dataclass
from typing import Optional, Tuple

from PIL import Image
import torchvision.transforms.functional as TF


def resize_keep_aspect(
    img: Image.Image,
    mask: Optional[Image.Image],
    crop_size: Tuple[int, int],
) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    TRAIN helper:
    Resize so that BOTH sides are >= crop_size (H, W),
    keeping aspect ratio.

    Guarantees:
      new_h >= crop_h and new_w >= crop_w
    so a subsequent crop_pair(crop_size) never needs padding.

    This function resizes both up or down. Downscaling large frames is
    important so random crops are sampled from a meaningful field of view
    instead of mostly background on very high-resolution inputs.

    Uses bilinear for image, nearest for mask.
    """
    w, h = img.size
    ch, cw = crop_size

    # scale needed so that both dimensions can support the crop
    scale = max(ch / h, cw / w)

    if abs(scale - 1.0) > 1e-6:
        new_w = int(round(w * scale))
        new_h = int(round(h * scale))

        img = img.resize((new_w, new_h), Image.BILINEAR)
        if mask is not None:
            mask = mask.resize((new_w, new_h), Image.NEAREST)

    return img, mask




def letterbox_to_size(
    img: Image.Image,
    mask: Optional[Image.Image],
    crop_size: Tuple[int, int],
    img_pad_value: int = 0,
    mask_pad_value: int = 0,
) -> Tuple[Image.Image, Optional[Image.Image]]:
    """
    VAL/TEST/REAL-TIME helper:
    Fit inside crop_size (H,W) while keeping aspect ratio, then pad.
    No random cropping. This mimics real-time preprocessing.
    """
    w, h = img.size
    ch, cw = crop_size

    scale = min(cw / w, ch / h)
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))

    img = img.resize((new_w, new_h), Image.BILINEAR)
    if mask is not None:
        mask = mask.resize((new_w, new_h), Image.NEAREST)

    pad_left = (cw - new_w) // 2
    pad_top = (ch - new_h) // 2
    pad_right = cw - new_w - pad_left
    pad_bottom = ch - new_h - pad_top

    img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=img_pad_value)
    if mask is not None:
        mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=mask_pad_value)

    return img, mask



@dataclass(frozen=True)
class CropParams:
    pad: Tuple[int, int, int, int]   # (left, top, right, bottom)
    box: Tuple[int, int, int, int]   # (x1, y1, x2, y2)

def crop_pair(
    img: Image.Image,
    mask: Optional[Image.Image],
    crop_size: Tuple[int, int],
    random_crop: bool = True,
    img_pad_value: int = 0,
    mask_pad_value: int = 0,
    params: Optional[CropParams] = None,
) -> Tuple[Image.Image, Optional[Image.Image], CropParams]:
    """
    TRAIN helper:
    Pad to at least crop_size (centered), then crop both img & mask
    to exactly (crop_h, crop_w).

    If params is provided, applies the same pad+crop.
    Returns (img, mask, params) so you can reuse for the right image.
    """
    w, h = img.size
    ch, cw = crop_size

    if params is None:
        # compute padding
        pad_left = max(0, (cw - w) // 2)
        pad_right = max(0, cw - w - pad_left)
        pad_top = max(0, (ch - h) // 2)
        pad_bottom = max(0, ch - h - pad_top)

        # apply padding (and update size)
        if pad_left or pad_right or pad_top or pad_bottom:
            img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=img_pad_value)
            if mask is not None:
                mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=mask_pad_value)
            w, h = img.size

        # sample crop
        if random_crop:
            x1 = random.randint(0, w - cw)
            y1 = random.randint(0, h - ch)
        else:
            x1 = (w - cw) // 2
            y1 = (h - ch) // 2

        params = CropParams(
            pad=(pad_left, pad_top, pad_right, pad_bottom),
            box=(x1, y1, x1 + cw, y1 + ch),
        )
    else:
        # apply the same padding first
        pad_left, pad_top, pad_right, pad_bottom = params.pad
        if pad_left or pad_right or pad_top or pad_bottom:
            img = TF.pad(img, (pad_left, pad_top, pad_right, pad_bottom), fill=img_pad_value)
            if mask is not None:
                mask = TF.pad(mask, (pad_left, pad_top, pad_right, pad_bottom), fill=mask_pad_value)

    # apply crop box
    x1, y1, x2, y2 = params.box
    img = img.crop((x1, y1, x2, y2))
    if mask is not None:
        mask = mask.crop((x1, y1, x2, y2))

    return img, mask, params


