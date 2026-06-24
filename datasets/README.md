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

If you want to create masks for new videos rather than use the paper external
test set, see:

```text
datasets/create_external_test_set/
```

That folder contains a generic segment extraction and prompted SAM3 propagation
workflow.

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
