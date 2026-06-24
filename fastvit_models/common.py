from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Literal, Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

try:
    # Registers Apple's FastViT implementations with timm. The segmentation
    # wrapper below relies on the Apple model internals (`patch_embed`,
    # `network`) used by the paper checkpoints.
    import third_party.fvit.models.fastvit  # type: ignore  # noqa: F401
except Exception:
    pass

FASTVIT_CHANNELS: Dict[str, List[int]] = {
    "fastvit_t8":  [48, 96, 192, 384],
    "fastvit_t12": [64, 128, 256, 512],
    "fastvit_s12": [64, 128, 256, 512],
    "fastvit_s24": [64, 128, 256, 512],
    "fastvit_m12": [76, 152, 304, 608],
    "fastvit_m24": [76, 152, 304, 608],
    "fastvit_sa12": [64, 128, 256, 512],
    "fastvit_sa24": [64, 128, 256, 512],
    "fastvit_sa36": [64, 128, 256, 512],
    "fastvit_ma36": [76, 152, 304, 608],
}

def _strip_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sd
    out = {}
    for k, v in sd.items():
        out[k[len(prefix):] if k.startswith(prefix) else k] = v
    return out

def _add_prefix(sd: Dict[str, torch.Tensor], prefix: str) -> Dict[str, torch.Tensor]:
    if not prefix:
        return sd
    return { (prefix + k if not k.startswith(prefix) else k): v for k, v in sd.items() }

def load_backbone_init_checkpoint(
    backbone_model: nn.Module,
    checkpoint_path: str,
    *,
    strict: bool = False,
    verbose: bool = True,
) -> Tuple[List[str], List[str]]:
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    state_dict = ckpt.get("state_dict", ckpt)

    # common cleanup
    state_dict = _strip_prefix(state_dict, "module.")

    model_keys = list(backbone_model.state_dict().keys())
    sd_keys = list(state_dict.keys())

    model_has_model_prefix = any(k.startswith("model.") for k in model_keys)
    sd_has_model_prefix = any(k.startswith("model.") for k in sd_keys)

    # If model expects "model." but checkpoint doesn't have it -> add it
    if model_has_model_prefix and not sd_has_model_prefix:
        state_dict = _add_prefix(state_dict, "model.")

    # If checkpoint has "model." but model doesn't -> strip it
    if (not model_has_model_prefix) and sd_has_model_prefix:
        state_dict = _strip_prefix(state_dict, "model.")

    missing, unexpected = backbone_model.load_state_dict(state_dict, strict=strict)

    if verbose:
        print(f"[load_backbone_init_checkpoint] loaded: {checkpoint_path}")
        if missing:
            print(f"  missing({len(missing)}): {missing[:20]}{' ...' if len(missing) > 20 else ''}")
        if unexpected:
            print(f"  unexpected({len(unexpected)}): {unexpected[:20]}{' ...' if len(unexpected) > 20 else ''}")

    return missing, unexpected



class FastViTBackbone(nn.Module):
    """
    Wraps a timm FastViT classifier and exposes 4 feature maps:
      [C1 (1/4), C2 (1/8), C3 (1/16), C4 (1/32)]
    """

    def __init__(self,  base_model: str):
        super().__init__()
        self.model = create_model(base_model, pretrained=False)
        if not hasattr(self.model, "patch_embed") or not hasattr(self.model, "network"):
            raise RuntimeError(
                "FastViT training requires Apple's ml-fastvit implementation under "
                "third_party/fvit. Run scripts/download_external_deps.sh first."
            )
        key = base_model.lower()
        if key not in FASTVIT_CHANNELS:
            raise ValueError(f"Unknown FastViT variant '{base_model}' for channel config.")
        self.out_channels = FASTVIT_CHANNELS[key]

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        m = self.model
        feats: List[torch.Tensor] = []

        x = m.patch_embed(x)
        x = m.network[0](x)
        feats.append(x)  # 1/4

        x = m.network[1](x)
        x = m.network[2](x)
        feats.append(x)  # 1/8

        x = m.network[3](x)
        x = m.network[4](x)
        feats.append(x)  # 1/16

        x = m.network[5](x)
        x = m.network[6](x)
        feats.append(x)  # 1/32

        return feats
