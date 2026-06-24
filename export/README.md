# Export

This folder contains the reproducible export code for newly trained models.

The already exported INT8/NPU TFLite artifacts for the released paper
checkpoints live at the repository root under `paper_checkpoints_int8/`. Those
files are final release artifacts, not inputs to the export code here.

`output_models_int8_npu/` contains the code used to export newly trained models
from `outputs/<run_name>` or from a directly provided `.pth` checkpoint path.
That code is the reproducible recipe for making Qualcomm AI Hub INT8/NPU TFLite
models.

## What The Export Code Does

The output-model export flow is:

1. Prepare a calibration/evaluation dataset with the same preprocessing used by
   the PyTorch test code.
2. Upload that tensor dataset to QAI Hub and save the dataset id plus a local
   manifest that maps tensors back to frame names.
3. Load a PyTorch checkpoint.
4. Trace the model with a fixed input shape.
5. Ask QAI Hub to create a static ONNX model.
6. Ask QAI Hub to quantize the ONNX model to INT8 using the uploaded dataset.
7. Ask QAI Hub to compile the quantized model to TFLite for the selected NPU
   device.
8. Optionally ask QAI Hub to run inference on the uploaded dataset, download the
   raw outputs, save PNG masks, and compute mIoU, Dice, and NSD.

The main shape settings live in
`export/output_models_int8_npu/code/config.yaml`:

```yaml
dataset:
  input_hw: [384, 640]

qai:
  channel_first: true
  input_shape_nchw: [1, 3, 384, 640]
```

`dataset.input_hw` is `[height, width]` after repository preprocessing.
`qai.input_shape_nchw` is `[batch, channels, height, width]`. These values must
agree. If you change one, change the other and upload a new dataset manifest.

## Export A Model From outputs/

Run these commands from a Python environment that has the repository
dependencies plus `qai_hub` installed and configured.

```bash
python -m export.output_models_int8_npu.code.upload_dataset \
  --config export/output_models_int8_npu/code/config.yaml

python -m export.output_models_int8_npu.code.run_compile_profile_infer \
  --config export/output_models_int8_npu/code/config.yaml \
  --models t8_both
```

This expects:

```text
outputs/t8_both/
  effective_config.json
  best_model.pth
```

The exporter reads the model variant and class count from the output config.

## Export A Direct .pth File

If the checkpoint is in an output run folder with `config.json` or
`effective_config.json`, only the checkpoint path is needed:

```bash
python -m export.output_models_int8_npu.code.run_compile_profile_infer \
  --config export/output_models_int8_npu/code/config.yaml \
  --model-path outputs/t8_both/best_model.pth
```

If the checkpoint is standalone, provide the architecture variant:

```bash
python -m export.output_models_int8_npu.code.run_compile_profile_infer \
  --config export/output_models_int8_npu/code/config.yaml \
  --model-path /path/to/model.pth \
  --variant fastvit_t8 \
  --model-key fastvit_t8_direct
```

For non-FastViT models, use variants such as `convnext_tiny` or
`mobilenetv3_large_100`. The exporter infers `builder_kind=fastvit` only for
variants starting with `fastvit_`; all other variants use the repository's
`other_models` builder.

## Masks And Metrics

`do_inference: true` in `config.yaml` asks QAI Hub to run the uploaded dataset on
the compiled model. The script then downloads outputs and writes masks under:

```text
export/output_models_int8_npu/generated/inference/<source>/<prefix>/<model>/pred_masks_png/
```

Then run:

```bash
python -m export.output_models_int8_npu.code.compute_metrics \
  --config export/output_models_int8_npu/code/config.yaml
```

If `do_inference: false`, the exporter only downloads the `.tflite` file. In
that case, masks and metrics can be produced later by running the TFLite model
with another runtime and evaluating the generated masks separately.

All generated QAI files are written under
`export/output_models_int8_npu/generated/`, which is gitignored.
