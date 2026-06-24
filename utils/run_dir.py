# utils/run_dir.py
from __future__ import annotations

import os
import json
from dataclasses import asdict
from typing import Any, Dict, Optional


def make_run_name(cfg: Dict[str, Any]) -> str:
    base = cfg["model"]["base_model"]
    loss = cfg["loss"]["name"]
    return f"{base}_{loss}_mono"


def prepare_run_dir(cfg: Dict[str, Any]) -> str:
    out_dir = cfg["run"]["out_dir"]
    run_name = cfg["run"]["run_name"] or make_run_name(cfg)
    overwrite = bool(cfg["run"].get("overwrite", False))

    os.makedirs(out_dir, exist_ok=True)
    run_dir = os.path.join(out_dir, run_name)

    if os.path.exists(run_dir) and not overwrite:
        raise FileExistsError(f"Run directory already exists: {run_dir} (set run.overwrite=true to overwrite)")

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_json(path: str, obj: Any) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=2)
