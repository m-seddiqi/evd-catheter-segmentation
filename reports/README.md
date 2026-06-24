# Reports

This folder contains evaluation and prediction scripts.

## Paper Checkpoints

`paper_checkpoint_test.py` loads released checkpoints from `paper_checkpoint`,
runs them on an image folder, saves predicted masks, and optionally writes
mIoU/Dice/NSD metrics.

The paper checkpoint tester does not read `config.json`, `effective_config.json`,
or training logs from `paper_checkpoint`. The required model metadata is listed
in `paper_checkpoint/README.md` and encoded in the tester.

It is **not** used by training. `main_train.py` has its own validation/test pass
inside the training loop.

When `--compute-metrics` is used, paper checkpoint metrics are computed in the
same letterboxed evaluation space used by the original paper testing script.
Predicted PNG masks are still saved back at the original image size for visual
inspection and downstream use.

## Test Scenes And Masks

By default, the script reads test images and corrected masks from:

```text
datasets/external_test_catheter/
  scene_yt1_001/
    images/
    masks/
  scene_yt2_002/
    images/
    masks/
  scene_yt3_003/
    images/
    masks/
  scene_yt4_004/
    images/
    masks/
```

Download and extraction instructions for `datasets/external_test_catheter/` are
in `datasets/README.md`.

The `masks` folders are used as ground-truth masks when `--compute-metrics` is
passed. Paper testing also copies these masks into `gt_masks/` beside the
predictions unless `--no-copy-gt` is passed.

Use `--image-root` to choose the image folder. Use `--mask-root` or
`--mask-dirname` only if masks live somewhere other than `masks/` beside images.

## Common Commands

Evaluate all paper checkpoints on all external test scenes:

```bash
python reports/paper_checkpoint_test.py \
  --model all \
  --image-root datasets/external_test_catheter \
  --compute-metrics \
  --device cuda:0
```

Evaluate one paper checkpoint on one scene:

```bash
python reports/paper_checkpoint_test.py \
  --model fastvit_t8_both \
  --image-root datasets/external_test_catheter/scene_yt1_001 \
  --compute-metrics \
  --device cuda:0
```

## Output Checkpoints On Image Folders

Use `general_checkpoint_test.py` for checkpoints produced by `main_train.py`
under `outputs/`. This script reads the output run `config.json`, predicts masks
for an arbitrary image folder, and optionally computes metrics if masks are
available.

For output checkpoints, metrics are computed in the same letterboxed eval space
by default. Pass `--metric-space original` only when you intentionally want
metrics in the saved original-resolution mask geometry.

```bash
python reports/general_checkpoint_test.py \
  --checkpoint outputs/fastvit_t8_both \
  --image-root datasets/external_test_catheter/scene_yt1_001 \
  --compute-metrics \
  --device cuda:0
```

Without `--compute-metrics`, either tester only writes predictions. With
`--compute-metrics`, it
looks for `masks/` beside the image folder, or you can pass `--mask-root`.

Use `--max-samples 5 --device cpu --batch-size 1` for a quick smoke test.

## Outputs

Default output root:

```text
outputs/paper_test/
  summary.csv
  summary.json
  <model>/<image-folder>/
    pred_masks/<scene_id>/*.png
    gt_masks/<scene_id>/*.png
    predictions.csv
    metadata.json
    metrics.csv
    metrics.json
```

For `scene_yt3_003`, the script masks the bottom 15% of the input image before
inference to match the prior paper-evaluation script.

Output-checkpoint folder predictions are written under:

```text
outputs/folder_predictions/
  <checkpoint>/<image-folder>/
    pred_masks/
    gt_masks/        # only with --copy-gt
    predictions.csv
    metadata.json
    metrics.csv      # only with --compute-metrics
    metrics.json     # only with --compute-metrics
```
