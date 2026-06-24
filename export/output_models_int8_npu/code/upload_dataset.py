from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from export.output_models_int8_npu.code.utils.common import load_cfg_file, load_yaml, resolve_path, resolve_project_root
from export.output_models_int8_npu.code.utils.data_loader import build_manifest, build_upload_payload
from export.output_models_int8_npu.code.utils.hub_ops import upload_or_get_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload preprocessed dataset to QAI Hub and save dataset id.")
    parser.add_argument("--config", type=Path, default=Path("export/output_models_int8_npu/code/config.yaml"))
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    project_root = resolve_project_root(args.config, cfg)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    dataset_cfg = dict(cfg["dataset"])
    qai_cfg = dict(cfg["qai"])
    paths_cfg = dict(cfg["paths"])
    channel_first = bool(qai_cfg.get("channel_first", True))

    train_cfg_path = resolve_path(project_root, dataset_cfg["config_path"])
    train_cfg = load_cfg_file(train_cfg_path)
    test_root = resolve_path(project_root, dataset_cfg["test_root"])
    manifest_path = resolve_path(project_root, paths_cfg["dataset_manifest_path"])
    dataset_id_path = resolve_path(project_root, paths_cfg["dataset_id_path"])

    hub_inputs, frame_keys = build_upload_payload(
        dataset_cfg=dataset_cfg,
        train_cfg=train_cfg,
        test_root=test_root,
        channel_first=channel_first,
    )
    manifest = build_manifest(
        dataset_cfg=dataset_cfg,
        train_cfg_path=train_cfg_path,
        test_root=test_root,
        frame_keys=frame_keys,
        channel_first=channel_first,
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2))

    _, dataset_id = upload_or_get_dataset(
        hub_inputs=hub_inputs,
        dataset_name=str(dataset_cfg["dataset_name"]),
        dataset_id_path=dataset_id_path,
        upload=bool(dataset_cfg.get("upload", True)),
    )

    print(f"Dataset id: {dataset_id}")
    print(f"Saved manifest: {manifest_path}")
    print(f"Saved dataset id path: {dataset_id_path}")
    print(f"Prepared samples: {len(frame_keys)}")


if __name__ == "__main__":
    main()
