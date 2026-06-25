# Datasets

This repository expects three dataset roots under `datasets/`:

```text
datasets/
  real_dataset/
  synthetic_dataset/
  external_test_catheter/
```

The full research datasets are distributed separately because some source
material is not licensed for direct redistribution through git. Access to the
Hugging Face dataset archives may require approval for research use.

## Real Dataset

`datasets/real_dataset/` contains the real mobile/HMD catheter frames and
manually corrected catheter masks used for training and validation.

Download:

```text
https://huggingface.co/datasets/mussed/evd_catheter_segmentation_datasets/resolve/main/real_dataset.zip
```

Extract the archive so the final layout is:

```text
datasets/real_dataset/
  scene_xxx/
    images/
    masks/
```

The default validation scene names in `configs/train.yaml` are aligned with the
paper experiments. Some real-data scenes were split or reorganized during later
curation; the released layout and default config preserve the paper-compatible
validation selection.

## External Test Dataset

`datasets/external_test_catheter/` contains the external catheter test scenes
used for paper evaluation.

Download:

```text
https://huggingface.co/datasets/mussed/evd_catheter_segmentation_datasets/resolve/main/external_test_catheter.zip
```

Extract the archive so the final layout is:

```text
datasets/external_test_catheter/
  scene_*/images/
  scene_*/masks/
```

If the archive is not available, the evaluation frames and prompted masks can be
reconstructed from the public video references and annotations in:

```text
datasets/create_external_test_set/
```

That folder documents the source video URLs, fixed frame ranges, point prompts,
and SAM3 mask propagation workflow used to rebuild the external test set.

## Synthetic Dataset

`datasets/synthetic_dataset/` contains rendered synthetic image/mask scenes used
for training. The paper layout uses `e0` plus `e1` through `e10` folders:

```text
datasets/synthetic_dataset/
  e0/scene_00000/images/frame_00000.jpg
  e0/scene_00000/masks/frame_00000.png
  e1/scene_00000/images/frame_00000.jpg
  e1/scene_00000/masks/frame_00000.png
```

Generate these folders with:

```text
datasets/create_synthetic_dataset/
```

The generator includes the catheter GLB assets and sample backgrounds. The full
paper background-image archive is available separately:

```text
https://huggingface.co/datasets/mussed/evd_catheter_segmentation_datasets/resolve/main/background_img.zip
```

Replace or augment `datasets/create_synthetic_dataset/3d_tools/background_img/`
with the extracted `background_img/` folder before generating the full paper
synthetic set.

Regenerating synthetic data with the released settings should produce comparable
training behavior, but exact results can vary with renderer/library versions,
hardware, random seeds, and whether the full background archive is available.
Use the released checkpoints for exact paper-model evaluation, and regenerated
synthetic data for reproducible training experiments and adaptation studies.
