# Paper Checkpoints INT8 Exports

This folder contains only the provided Qualcomm AI Hub INT8/NPU TFLite exports
for the released paper checkpoints. These files are deployable artifacts, not
the source recipe for exporting new training runs.

Included artifacts:

```text
convnext_tiny_both/convnext_tiny_both_int8_npu.tflite
fastvit_sa12_both/fastvit_sa12_both_int8_npu.tflite
fastvit_sa36_both/fastvit_sa36_both_int8_npu.tflite
fastvit_t8_both/fastvit_t8_both_int8_npu.tflite
mobilenetv3_large_100_both/mobilenetv3_large_100_both_int8_npu.tflite
```

The matching export family was identified from the FastViT T8 yt1 signature:

```text
mIoU = 0.370617
Dice = 0.540803
NSD  = 0.504432
```

These round to `37.1 / 54.1 / 50.4`.

Use `export/output_models_int8_npu/` when exporting a newly trained model from
`outputs/` or from a direct `.pth` checkpoint path. The export code does not
read this folder.
