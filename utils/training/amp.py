from __future__ import annotations

import random
from typing import Any, Tuple

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    # (optional) keep benchmark False for determinism; set True only if you want speed.
    torch.backends.cudnn.benchmark = False


def get_amp_setup(device: torch.device, use_amp: bool) -> Tuple[bool, Any, Any, dict]:
    """
    Version-robust AMP:
      - Prefer torch.amp.autocast / torch.amp.GradScaler when available
      - Fall back to torch.cuda.amp
    Returns: (enabled, scaler, autocast_ctx, autocast_kwargs)
    """
    enabled = bool(use_amp) and (device.type == "cuda")

    # autocast
    if hasattr(torch, "amp") and hasattr(torch.amp, "autocast"):
        autocast_ctx = torch.amp.autocast
        autocast_kwargs = {"device_type": "cuda"}
    else:
        from torch.cuda.amp import autocast as autocast_ctx  # type: ignore
        autocast_kwargs = {}

    # GradScaler
    if hasattr(torch, "amp") and hasattr(torch.amp, "GradScaler"):
        scaler = torch.amp.GradScaler(enabled=enabled)
    else:
        from torch.cuda.amp import GradScaler  # type: ignore
        scaler = GradScaler(enabled=enabled)

    return enabled, scaler, autocast_ctx, autocast_kwargs
