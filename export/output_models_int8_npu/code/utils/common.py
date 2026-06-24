from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

import yaml


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_yaml(path: Path) -> Dict[str, Any]:
    return yaml.safe_load(path.read_text())


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def load_cfg_file(path: Path) -> Dict[str, Any]:
    if path.suffix.lower() == ".json":
        return load_json(path)
    return yaml.safe_load(path.read_text())


def resolve_path(project_root: Path, p: str | Path) -> Path:
    x = Path(p)
    return x if x.is_absolute() else (project_root / x).resolve()


def resolve_project_root(config_path: Path, cfg: Dict[str, Any]) -> Path:
    project_root = resolve_path(config_path.parent, cfg.get("project_root", "."))
    if (project_root / "configs").exists():
        return project_root
    for parent in project_root.parents:
        if (parent / "configs").exists():
            return parent.resolve()
    return project_root
