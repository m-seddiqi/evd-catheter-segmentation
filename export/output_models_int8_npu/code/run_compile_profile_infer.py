from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import qai_hub as hub
import torch

from export.output_models_int8_npu.code.model_loader import build_hub_wrapper_for_spec, direct_model_spec, discover_model_specs
from export.output_models_int8_npu.code.utils.common import load_yaml, resolve_path, resolve_project_root, utc_now_iso
from export.output_models_int8_npu.code.utils.hub_ops import (
    decode_masks_from_outputs,
    get_job_id,
    require_dataset,
    save_mask_artifacts,
    wait_success,
)


def _load_manifest(manifest_path: Path) -> Dict[str, Any]:
    return json.loads(manifest_path.read_text())


def _load_frame_keys_from_manifest(payload: Dict[str, Any]) -> List[tuple[str, str]]:
    out = []
    for item in payload.get("frames", []):
        out.append((str(item["scene_id"]), str(item["frame_id"])))
    return out


def _parse_quant_dtype(name: str):
    key = str(name).strip().upper()
    mapping = {
        "INT8": hub.QuantizeDtype.INT8,
        "INT16": hub.QuantizeDtype.INT16,
    }
    if key not in mapping:
        raise ValueError(f"Unsupported quant dtype: {name}. Supported: {sorted(mapping.keys())}")
    return mapping[key]


def main() -> None:
    parser = argparse.ArgumentParser(description="Export output checkpoints to INT8/NPU TFLite with QAI Hub.")
    parser.add_argument("--config", type=Path, default=Path("export/output_models_int8_npu/code/config.yaml"))
    parser.add_argument("--models", nargs="*", default=None, help="Optional subset of outputs/<run_name> model keys")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=None,
        help="Optional direct .pth/.pt checkpoint path. If set, model_sources scanning is skipped.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Model variant for --model-path, for example fastvit_t8 or convnext_tiny.",
    )
    parser.add_argument("--model-key", type=str, default=None, help="Output name for --model-path exports.")
    parser.add_argument(
        "--builder-kind",
        choices=("fastvit", "other"),
        default=None,
        help="Builder family for --model-path. Defaults to fastvit for fastvit_* variants, otherwise other.",
    )
    parser.add_argument("--num-classes", type=int, default=2, help="Number of segmentation classes for --model-path.")
    parser.add_argument("--fpn-dim", type=int, default=256, help="FPN width used by non-FastViT models.")
    args = parser.parse_args()
    if args.model_path is not None and args.models:
        raise ValueError("Use either --model-path for one direct checkpoint or --models for output-run scanning, not both.")

    cfg = load_yaml(args.config)
    project_root = resolve_project_root(args.config, cfg)
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    qai_cfg = dict(cfg["qai"])
    paths_cfg = dict(cfg["paths"])

    profile_name = str(qai_cfg["active_compile_profile"])
    profile_cfg = dict(qai_cfg["compile_profiles"][profile_name])
    output_prefix = str(profile_cfg["output_prefix"])

    do_compile = bool(profile_cfg.get("do_compile", True))
    do_profile = bool(profile_cfg.get("do_profile", False))
    do_inference = bool(profile_cfg.get("do_inference", True))
    if (do_profile or do_inference) and not do_compile:
        raise ValueError("Invalid config: do_compile must be true when do_profile or do_inference is enabled.")
    quant_cfg = dict(profile_cfg.get("quantization", {}))
    use_quant = bool(quant_cfg.get("enabled", False))

    channel_first = bool(qai_cfg.get("channel_first", True))
    input_shape_nchw = tuple(int(v) for v in qai_cfg["input_shape_nchw"])
    if len(input_shape_nchw) != 4:
        raise ValueError(f"input_shape_nchw must be rank-4, got: {input_shape_nchw}")
    n, c, h, w = input_shape_nchw
    input_shape = input_shape_nchw if channel_first else (n, h, w, c)
    device_name = str(qai_cfg["device_name"])
    compile_options = str(profile_cfg["compile_options"])

    dataset_id_path = resolve_path(project_root, paths_cfg["dataset_id_path"])
    manifest_path = resolve_path(project_root, paths_cfg["dataset_manifest_path"])
    outputs_root = resolve_path(project_root, paths_cfg["outputs_root"])
    run_registry_path = resolve_path(project_root, paths_cfg["run_registry_path"])
    outputs_root.mkdir(parents=True, exist_ok=True)
    run_registry_path.parent.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, Any] | None = None
    frame_keys: List[tuple[str, str]] = []
    if manifest_path.exists():
        manifest = _load_manifest(manifest_path)
        frame_keys = _load_frame_keys_from_manifest(manifest)

    dataset_id = ""
    hub_dataset = None
    if do_inference or use_quant:
        if manifest is None:
            raise FileNotFoundError(
                f"Dataset manifest not found: {manifest_path}. Run upload_dataset.py first for the current layout."
            )
        manifest_channel_first = manifest.get("channel_first")
        if manifest_channel_first is not None and bool(manifest_channel_first) != channel_first:
            expected_layout = "NCHW (channel_first=true)" if channel_first else "NHWC (channel_first=false)"
            manifest_layout = (
                "NCHW (channel_first=true)" if bool(manifest_channel_first) else "NHWC (channel_first=false)"
            )
            raise ValueError(
                "Dataset layout mismatch between config and manifest. "
                f"Config expects {expected_layout}, but manifest has {manifest_layout}. "
                "Use layout-specific dataset files and rerun upload_dataset.py."
            )
        hub_dataset, dataset_id = require_dataset(dataset_id_path)

    if args.model_path is not None:
        model_specs = [
            direct_model_spec(
                project_root=project_root,
                weights_path=args.model_path,
                variant=args.variant,
                model_key=args.model_key,
                builder_kind=args.builder_kind,
                num_classes=args.num_classes,
                fpn_dim=args.fpn_dim,
            )
        ]
    else:
        model_specs = discover_model_specs(project_root=project_root, cfg=cfg)
        if args.models:
            requested = set(args.models)
            model_specs = [s for s in model_specs if s.model_key in requested]
    if not model_specs:
        raise RuntimeError("No model checkpoints were selected for export.")

    target_device = hub.Device(device_name)

    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    for spec in model_specs:
        try:
            num_classes = int(spec.num_classes)
            wrapper = build_hub_wrapper_for_spec(
                spec=spec,
                channel_first=channel_first,
            )
            traced_model = torch.jit.trace(wrapper, (torch.randn(input_shape),))

            model_out_dir = outputs_root / spec.source_group / output_prefix / spec.model_key
            model_out_dir.mkdir(parents=True, exist_ok=True)

            compile_id_path = model_out_dir / "compile_job_id.txt"
            onnx_compile_id_path = model_out_dir / "onnx_compile_job_id.txt"
            quantize_id_path = model_out_dir / "quantize_job_id.txt"
            profile_id_path = model_out_dir / "profile_job_id.txt"
            infer_id_path = model_out_dir / "inference_job_id.txt"
            tflite_path = model_out_dir / f"{spec.model_key}_{output_prefix}.tflite"
            raw_out_path = model_out_dir / f"{spec.model_key}_{output_prefix}_outputs.pt"
            summary_path = model_out_dir / "compile_infer_summary.json"

            summary: Dict[str, Any] = {
                "model_key": spec.model_key,
                "source_group": spec.source_group,
                "builder_kind": spec.builder_kind,
                "variant": spec.variant,
                "num_classes": spec.num_classes,
                "fpn_dim": spec.fpn_dim,
                "run_dir": str(spec.run_dir) if spec.run_dir else None,
                "weights_path": str(spec.weights_path),
                "profile_name": profile_name,
                "output_prefix": output_prefix,
                "dataset_id": dataset_id if do_inference else None,
                "device_name": device_name,
                "channel_first": channel_first,
                "input_shape": list(input_shape),
                "input_shape_nchw": list(input_shape_nchw),
                "compile_options": compile_options,
                "use_quantization": use_quant,
                "updated_at_utc": utc_now_iso(),
            }

            target_model = None
            if do_compile:
                if use_quant:
                    # Step 1: make a static ONNX model before QAI Hub quantization.
                    onnx_options = str(quant_cfg.get("precompile_options", "--target_runtime onnx"))
                    onnx_job = hub.submit_compile_job(
                        model=traced_model,
                        device=target_device,
                        input_specs={"image": input_shape},
                        options=onnx_options,
                        name=f"{spec.model_key}_{output_prefix}_onnx_compile",
                    )
                    onnx_job_id = get_job_id(onnx_job)
                    onnx_compile_id_path.write_text(onnx_job_id + "\n")
                    onnx_status = wait_success(onnx_job, "ONNX pre-compile")
                    static_onnx_model = onnx_job.get_target_model()

                    # Step 2: INT8 quantization uses the uploaded calibration dataset.
                    weights_dtype = _parse_quant_dtype(str(quant_cfg.get("weights_dtype", "int8")))
                    activations_dtype = _parse_quant_dtype(str(quant_cfg.get("activations_dtype", "int8")))
                    quant_options = str(quant_cfg.get("quant_options", "")).strip()
                    quant_kwargs: Dict[str, Any] = {
                        "model": static_onnx_model,
                        "calibration_data": hub_dataset,
                        "weights_dtype": weights_dtype,
                        "activations_dtype": activations_dtype,
                        "name": f"{spec.model_key}_{output_prefix}_quantize",
                    }
                    if quant_options:
                        quant_kwargs["options"] = quant_options
                    quant_job = hub.submit_quantize_job(**quant_kwargs)
                    quant_job_id = get_job_id(quant_job)
                    quantize_id_path.write_text(quant_job_id + "\n")
                    quant_status = wait_success(quant_job, "Quantization")
                    quantized_model = quant_job.get_target_model()

                    # Step 3: compile the quantized model to the selected device/runtime.
                    compile_job = hub.submit_compile_job(
                        model=quantized_model,
                        device=target_device,
                        input_specs={"image": input_shape},
                        options=compile_options,
                        name=f"{spec.model_key}_{output_prefix}_compile",
                    )
                    compile_id = get_job_id(compile_job)
                    compile_id_path.write_text(compile_id + "\n")
                    compile_status = wait_success(compile_job, "Quantized compile")
                    compile_job.download_target_model(str(tflite_path))
                    target_model = compile_job.get_target_model()
                    summary.update(
                        {
                            "onnx_compile_job_id": onnx_job_id,
                            "onnx_compile_url": onnx_job.url,
                            "onnx_compile_status": onnx_status,
                            "onnx_compile_options": onnx_options,
                            "quantize_job_id": quant_job_id,
                            "quantize_url": quant_job.url,
                            "quantize_status": quant_status,
                            "quantize_weights_dtype": str(quant_cfg.get("weights_dtype", "int8")).upper(),
                            "quantize_activations_dtype": str(quant_cfg.get("activations_dtype", "int8")).upper(),
                            "quantize_options": quant_options,
                            "compile_job_id": compile_id,
                            "compile_url": compile_job.url,
                            "compile_status": compile_status,
                            "tflite_path": str(tflite_path),
                        }
                    )
                else:
                    compile_job = hub.submit_compile_job(
                        model=traced_model,
                        device=target_device,
                        input_specs={"image": input_shape},
                        options=compile_options,
                        name=f"{spec.model_key}_{output_prefix}_compile",
                    )
                    compile_id = get_job_id(compile_job)
                    compile_id_path.write_text(compile_id + "\n")
                    compile_status = wait_success(compile_job, "Compile")
                    compile_job.download_target_model(str(tflite_path))
                    target_model = compile_job.get_target_model()
                    summary.update(
                        {
                            "compile_job_id": compile_id,
                            "compile_url": compile_job.url,
                            "compile_status": compile_status,
                            "tflite_path": str(tflite_path),
                        }
                    )

            if do_profile:
                if target_model is None:
                    raise RuntimeError("Profile requested but no compiled target model is available.")
                profile_job = hub.submit_profile_job(
                    model=target_model,
                    device=target_device,
                    name=f"{spec.model_key}_{output_prefix}_profile",
                )
                profile_id = get_job_id(profile_job)
                profile_id_path.write_text(profile_id + "\n")
                profile_status = wait_success(profile_job, "Profile")
                summary.update(
                    {
                        "profile_job_id": profile_id,
                        "profile_url": profile_job.url,
                        "profile_status": profile_status,
                    }
                )

            if do_inference:
                if target_model is None:
                    raise RuntimeError("Inference requested but no compiled target model is available.")
                # Step 4: optional device inference. Disable do_inference to export only the TFLite file.
                inference_job = hub.submit_inference_job(
                    model=target_model,
                    device=target_device,
                    inputs=hub_dataset,
                    name=f"{spec.model_key}_{output_prefix}_inference",
                )
                infer_id = get_job_id(inference_job)
                infer_id_path.write_text(infer_id + "\n")
                infer_status = wait_success(inference_job, "Inference")

                outputs = inference_job.download_output_data()
                torch.save(outputs, raw_out_path)
                pred_masks = decode_masks_from_outputs(outputs, num_classes=num_classes)
                masks_npy_path, masks_png_dir = save_mask_artifacts(
                    pred_masks=pred_masks,
                    frame_keys=frame_keys,
                    model_out_dir=model_out_dir,
                )
                summary.update(
                    {
                        "inference_job_id": infer_id,
                        "inference_url": inference_job.url,
                        "inference_status": infer_status,
                        "raw_outputs_path": str(raw_out_path),
                        "pred_masks_npy_path": str(masks_npy_path),
                        "pred_masks_png_dir": str(masks_png_dir),
                        "num_output_masks": int(pred_masks.shape[0]),
                    }
                )

            summary_path.write_text(json.dumps(summary, indent=2))
            results.append(summary)
            print(f"[OK] {spec.source_group}/{spec.model_key}")
        except Exception as exc:
            errors.append({"model_key": spec.model_key, "source_group": spec.source_group, "error": str(exc)})
            print(f"[ERROR] {spec.source_group}/{spec.model_key}: {exc}")

    registry = {
        "profile_name": profile_name,
        "output_prefix": output_prefix,
        "do_compile": do_compile,
        "do_profile": do_profile,
        "do_inference": do_inference,
        "use_quantization": use_quant,
        "success_count": len(results),
        "error_count": len(errors),
        "results": results,
        "errors": errors,
        "updated_at_utc": utc_now_iso(),
    }
    run_registry_path.write_text(json.dumps(registry, indent=2))

    print(f"Success: {len(results)}")
    print(f"Errors: {len(errors)}")
    print(f"Run registry: {run_registry_path}")


if __name__ == "__main__":
    main()
