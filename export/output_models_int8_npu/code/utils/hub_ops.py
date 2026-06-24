from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import qai_hub as hub
import torch
from PIL import Image


def get_job_id(job: Any) -> str:
    jid = getattr(job, "job_id", None)
    if jid:
        return str(jid)
    url = getattr(job, "url", "")
    if url:
        return str(url).rstrip("/").split("/")[-1]
    return "unknown"


def wait_success(job: Any, kind: str) -> str:
    job.wait()
    status = job.get_status().code
    if status != "SUCCESS":
        raise RuntimeError(f"{kind} failed: {status} ({job.url})")
    return str(status)


def upload_or_get_dataset(hub_inputs: Dict[str, Any], dataset_name: str, dataset_id_path: Path, upload: bool):
    dataset_id_path.parent.mkdir(parents=True, exist_ok=True)
    if upload:
        ds = hub.upload_dataset(hub_inputs, name=dataset_name)
        dataset_id = str(ds.dataset_id)
        dataset_id_path.write_text(dataset_id + "\n")
        return ds, dataset_id

    if not dataset_id_path.exists():
        raise FileNotFoundError(f"Dataset id file not found: {dataset_id_path}")
    dataset_id = dataset_id_path.read_text().strip()
    if not dataset_id:
        raise ValueError(f"Dataset id file is empty: {dataset_id_path}")
    return hub.get_dataset(dataset_id), dataset_id


def require_dataset(dataset_id_path: Path):
    if not dataset_id_path.exists():
        raise FileNotFoundError(
            f"Inference enabled but dataset id file is missing: {dataset_id_path}. Run upload_dataset.py first."
        )
    dataset_id = dataset_id_path.read_text().strip()
    if not dataset_id:
        raise ValueError(f"Dataset id file is empty: {dataset_id_path}")
    return hub.get_dataset(dataset_id), dataset_id


def decode_masks_from_outputs(outputs: Dict[str, Any], num_classes: int) -> np.ndarray:
    key = "output_0" if "output_0" in outputs else next(iter(outputs))
    raw = outputs[key]

    if isinstance(raw, np.ndarray):
        raw_list = [raw[i] for i in range(raw.shape[0])]
    else:
        raw_list = list(raw)

    pred_masks = []
    for item in raw_list:
        t = torch.from_numpy(item).float() if isinstance(item, np.ndarray) else torch.tensor(item).float()

        if t.ndim == 4 and t.shape[0] == 1:
            t = t.squeeze(0)
        if t.ndim == 3 and t.shape[0] == 1:
            t = t.squeeze(0)

        if t.ndim == 3:
            if t.shape[0] == num_classes or t.shape[0] > 1:
                pred = torch.argmax(t, dim=0)
            else:
                pred = t.squeeze(0).round().long()
        elif t.ndim == 2:
            pred = t.round().long()
        else:
            raise ValueError(f"Unexpected output tensor shape: {tuple(t.shape)}")

        pred_masks.append(pred.cpu().numpy().astype(np.uint8))

    return np.stack(pred_masks, axis=0)


def save_mask_artifacts(pred_masks: np.ndarray, frame_keys: list[tuple[str, str]], model_out_dir: Path) -> Tuple[Path, Path]:
    masks_npy_path = model_out_dir / "pred_masks.npy"
    np.save(masks_npy_path, pred_masks)

    png_dir = model_out_dir / "pred_masks_png"
    png_dir.mkdir(parents=True, exist_ok=True)

    for i, m in enumerate(pred_masks):
        if i < len(frame_keys):
            scene_id, frame_id = frame_keys[i]
            fname = f"{scene_id}__{frame_id}.png"
        else:
            fname = f"idx_{i:05d}.png"
        binary = (m > 0).astype(np.uint8) * 255
        Image.fromarray(binary, mode="L").save(png_dir / fname)

    return masks_npy_path, png_dir
