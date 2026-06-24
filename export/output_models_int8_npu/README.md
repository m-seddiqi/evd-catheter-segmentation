# Output Model INT8/NPU Export

This folder contains the QAI Hub export recipe for newly trained models. It is
for checkpoints in `outputs/` or for a direct `.pth` path. It is separate from
`paper_checkpoints_int8/`, which already contains the paper TFLite files.

## Configure Shape And Dataset

Edit `code/config.yaml` before launching jobs:

```yaml
dataset:
  test_root: datasets/external_test_catheter
  scene_id: scene_yt1_001
  input_hw: [384, 640]

qai:
  device_name: Samsung Galaxy S23
  channel_first: true
  input_shape_nchw: [1, 3, 384, 640]
```

`input_hw` is `[height, width]` after repository preprocessing. The last two
values in `input_shape_nchw` must match it. If these dimensions change, upload a
new dataset so QAI Hub receives tensors with the same shape used for tracing.

## Upload The Calibration Dataset

Run these commands from a Python environment that has the repository
dependencies plus `qai_hub` installed and configured.

```bash
python -m export.output_models_int8_npu.code.upload_dataset \
  --config export/output_models_int8_npu/code/config.yaml
```

This creates:

```text
export/output_models_int8_npu/generated/datasets/
  qai_manifest_*.json
  qai_dataset_id_*.txt
```

The manifest records the scene/frame order used for optional QAI inference and
later metrics. Set `dataset.upload: false` only when `qai_dataset_id_*.txt`
already contains a valid QAI Hub dataset id.

## Export From outputs/

Expected run layout:

```text
outputs/<run_name>/
  effective_config.json
  best_model.pth
```

Run one model:

```bash
python -m export.output_models_int8_npu.code.run_compile_profile_infer \
  --config export/output_models_int8_npu/code/config.yaml \
  --models t8_both
```

Leave `--models` out, or fill `model_sources[0].select_models`, only when you
intentionally want to launch jobs for several output runs.

## Export A Direct Checkpoint

If the checkpoint folder also has `config.json` or `effective_config.json`, the
architecture is inferred:

```bash
python -m export.output_models_int8_npu.code.run_compile_profile_infer \
  --config export/output_models_int8_npu/code/config.yaml \
  --model-path outputs/t8_both/best_model.pth
```

For a standalone `.pth`, pass the variant:

```bash
python -m export.output_models_int8_npu.code.run_compile_profile_infer \
  --config export/output_models_int8_npu/code/config.yaml \
  --model-path /path/to/model.pth \
  --variant fastvit_t8 \
  --model-key fastvit_t8_direct
```

Use `--builder-kind other` only if the variant name is unusual and does not
start with `fastvit_`.

## QAI Hub Steps

`run_compile_profile_infer.py` performs the following jobs:

1. Trace the PyTorch model with `qai.input_shape_nchw`.
2. Compile to static ONNX.
3. Quantize to INT8 using the uploaded calibration dataset.
4. Compile the quantized model to TFLite for the selected NPU device.
5. Optionally run QAI inference and save masks.

The optional parts are controlled in `code/config.yaml`:

```yaml
do_profile: false
do_inference: true
```

When `do_inference` is `true`, masks are written to:

```text
export/output_models_int8_npu/generated/inference/<source>/<prefix>/<model>/pred_masks_png/
```

Then compute metrics:

```bash
python -m export.output_models_int8_npu.code.compute_metrics \
  --config export/output_models_int8_npu/code/config.yaml
```

When `do_inference` is `false`, the output is just the compiled `.tflite`; masks
can be generated later with another TFLite runtime and evaluated separately.

Generated files stay under `export/output_models_int8_npu/generated/`, which is
gitignored.
