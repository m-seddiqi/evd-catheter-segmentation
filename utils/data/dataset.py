# utils/data/dataset.py
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

from PIL import Image
from torch.utils.data import Dataset

from utils.data.indexing import MonoSample
from utils.data.pipeline import MonoTrainPipeline, MonoEvalPipeline


def _pil_open_rgb(path: Path) -> Image.Image:
    # jpg frames: ensure RGB
    return Image.open(path).convert("RGB")


def _pil_open_mask(path: Path) -> Image.Image:
    # png masks: keep single-channel (L)
    return Image.open(path).convert("L")


@dataclass(frozen=True)
class DatasetItem:
    """
    Returned by __getitem__.
    meta contains at least video_dir, scene_id, frame_id (and pipeline params).
    """
    img: Any
    mask: Any
    meta: Dict[str, Any]


class CatheterMonoDataset(Dataset):
    """
    Dataset that:
      - consumes a list of MonoSample (from indexing.py)
      - loads PIL image/mask from disk
      - calls the provided pipeline (train/eval) which applies transforms + aug
      - returns whatever the pipeline returns + stable metadata

    No filesystem globbing here.
    No transforms/aug logic here.
    """

    def __init__(
        self,
        samples: Sequence[MonoSample],
        *,
        pipeline: Union[MonoTrainPipeline, MonoEvalPipeline],
        return_paths: bool = False,
    ):
        self.samples: List[MonoSample] = list(samples)
        self.pipeline = pipeline
        self.return_paths = return_paths

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        s = self.samples[idx]

        img = _pil_open_rgb(s.img)
        mask = _pil_open_mask(s.mask) if s.mask is not None else None

        meta: Dict[str, Any] = {
            "video_name": s.video_dir.name,
            "video_dir": str(s.video_dir),
            "scene_id": s.scene_id,
            "frame_id": s.frame_id,
        }
        if self.return_paths:
            meta.update(
                {
                    "img_path": str(s.img),
                    "mask_path": str(s.mask) if s.mask is not None else None,
                }
            )

        out = self.pipeline(img=img, mask=mask, meta=meta)

        # Ensure meta is always present and includes our stable fields
        if "meta" not in out or out["meta"] is None:
            out["meta"] = meta
        else:
            for k, v in meta.items():
                out["meta"].setdefault(k, v)

        return out