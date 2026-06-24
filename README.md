# EVD Catheter Segmentation

Reproducibility code for the MICCAI paper
**Device-Constrained Real-Time EVD Catheter Segmentation**.

This repository contains the training, evaluation, export, and dataset-generation
code used for the paper experiments. Due to licensing restrictions, not all
source data can be redistributed directly through git. The released code keeps
the experiment configurations, model definitions, rendering pipeline, evaluation
scripts, and deployment-export workflow needed to reproduce the reported setup
or adapt it to related thin-structure segmentation tasks with limited data.

## Contents

- `main_train.py`, `configs/train.yaml`: model-agnostic training entrypoint and paper-style configuration.
- `reports/paper_checkpoint_test.py`: evaluate released paper checkpoints on an image folder.
- `reports/general_checkpoint_test.py`: evaluate a newly trained checkpoint on an image folder.
- `datasets/`: dataset layout notes, access links, and dataset creation utilities.
- `datasets/create_synthetic_dataset/`: synthetic rendering pipeline for catheter image/mask pairs.
- `datasets/create_external_test_set/`: optional utilities for creating prompted masks on new videos.
- `paper_checkpoint/`: expected location for released FP32 paper checkpoints.
- `paper_checkpoints_int8/`: expected location for released INT8/NPU TFLite exports.
- `export/`: export code and export provenance.
- `outputs/`: destination for new training runs and prediction summaries.

## Data And Checkpoints

Dataset archives may require access approval for research use. See
`datasets/README.md` for the expected extraction layout.

The paper data resources are available from:

```text
https://huggingface.co/datasets/mussed/evd_catheter_segmentation_datasets/resolve/main/real_dataset.zip
https://huggingface.co/datasets/mussed/evd_catheter_segmentation_datasets/resolve/main/external_test_catheter.zip
https://huggingface.co/datasets/mussed/evd_catheter_segmentation_datasets/resolve/main/background_img.zip
```

Extract them so the final paths are:

```text
datasets/real_dataset/
datasets/external_test_catheter/
datasets/create_synthetic_dataset/3d_tools/background_img/
```

The released model checkpoints are public:

```text
https://huggingface.co/mussed/evd_catheter_segmentation/resolve/main/paper_checkpoint.zip
https://huggingface.co/mussed/evd_catheter_segmentation/resolve/main/int8_paper_checkpoints.zip
```

Extract `paper_checkpoint.zip` into `paper_checkpoint/`. Extract
`int8_paper_checkpoints.zip` into `paper_checkpoints_int8/` if the INT8/NPU
exports are needed.

## Setup

Use a Python or conda environment with the CUDA/PyTorch stack that matches your
machine, then install the project dependencies:

```bash
cd evd_catheter
pip install -r requirements.txt
pip install -e .
```

FastViT models also need Apple's `ml-fastvit` source and the unfused FastViT
initialization checkpoints:

```bash
bash scripts/download_external_deps.sh
```

The script prepares `third_party/fvit/`,
`third_party/fvit/unfused_checkpoints/`, and the small import patch needed to
use Apple's code from this repository.

## Test Released Checkpoints

Run all released paper checkpoints on the full external test set:

```bash
python reports/paper_checkpoint_test.py \
  --model all \
  --image-root datasets/external_test_catheter \
  --compute-metrics \
  --device cuda:0
```

Run one paper checkpoint on one scene:

```bash
python reports/paper_checkpoint_test.py \
  --model fastvit_t8_both \
  --image-root datasets/external_test_catheter/scene_yt1_001 \
  --compute-metrics \
  --device cuda:0
```

Run a newly trained output checkpoint on an arbitrary image folder:

```bash
python reports/general_checkpoint_test.py \
  --checkpoint outputs/fastvit_t8_both \
  --image-root datasets/external_test_catheter/scene_yt1_001 \
  --compute-metrics \
  --device cuda:0
```

Predicted masks and metric files are written under `outputs/paper_test/`.
Folder-prediction outputs are written under `outputs/folder_predictions/`.
See `reports/README.md` for the exact output structures.

## Train Models

All training runs save under `outputs/<run_name>`.

```bash
python main_train.py --model fastvit_t8 --data-source real
python main_train.py --model fastvit_t8 --data-source synthetic
python main_train.py --model fastvit_t8 --data-source both
```

The accepted training sources are:

- `real`: corrected catheter data only.
- `synthetic`: generated synthetic data only.
- `both`: combined real and synthetic supervision.

Supported paper model names are `fastvit_t8`, `fastvit_sa12`, `fastvit_sa36`,
`convnext_tiny`, and `mobilenetv3_large_100`. Use `--device cuda:1` or
`--device cpu` to override the config device. Use `--epochs N` for a short smoke
run. `convnext_tiny` defaults to `--no-amp`, matching the released checkpoint
training setup; pass `--amp` to override.

Example named runs:

```bash
python main_train.py --model convnext_tiny --data-source both
python main_train.py --model mobilenetv3_large_100 --data-source both
python main_train.py --model fastvit_sa36 --data-source both --run-name fastvit_sa36_both
```

## Synthetic Data

The synthetic pipeline is under:

```text
datasets/create_synthetic_dataset/
```

It contains the rendering code, paper configuration, catheter GLB assets, and
sample backgrounds. Some paper background images are distributed separately in
`background_img.zip`; replace or augment
`datasets/create_synthetic_dataset/3d_tools/background_img/` with the extracted
archive before regenerating the full paper synthetic set. The provided sample
backgrounds are enough for smoke tests and for similar-style experiments. For a
different device or tool, use background images from the target domain.

The catheter GLB files used in the paper are already scaled and oriented under
`datasets/create_synthetic_dataset/3d_tools/catheter/`. To adapt the method to a
new thin instrument, create a GLB from 2D tool imagery with a SAM3D-style
workflow, then scale, orient, and configure it for the renderer. Follow
`datasets/create_synthetic_dataset/README.md` to generate the `e0` through `e10`
synthetic folders used by the training code.

## Real Data

Place the extracted real dataset at:

```text
datasets/real_dataset/
```

This folder should contain scene folders with paired `images/` and `masks/`
subdirectories. Download information and the access note are in
`datasets/README.md`.

## Export Artifacts

The released Qualcomm AI Hub INT8/NPU TFLite exports for paper checkpoints use:

```text
paper_checkpoints_int8/
```

QAI export code for newly trained `outputs/<run_name>` models is under:

```text
export/output_models_int8_npu/
```
