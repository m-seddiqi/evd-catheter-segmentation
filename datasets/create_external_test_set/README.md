# Create External Test Set

Use this folder only when `datasets/external_test_catheter/` is not already
available. The scripts recreate the external-test images and prompted
`masks` used by the evaluation code.

Expected generated layout:

```text
datasets/external_test_catheter/
  selected_points.json
  scene_yt1_001/images/
  scene_yt1_001/masks/
  scene_yt2_002/images/
  scene_yt2_002/masks/
  scene_yt3_003/images/
  scene_yt3_003/masks/
  scene_yt4_004/images/
  scene_yt4_004/masks/
```

## Inputs

Place the four source videos here:

```text
datasets/create_external_test_set/source_videos/
  yt1.mp4
  yt2.mp4
  yt3.mp4
  yt4.mp4
```

Source references:

| File | Source |
| --- | --- |
| `yt1.mp4` | https://youtu.be/wNDx8kJsMKA |
| `yt2.mp4` | https://youtu.be/XaL8AElw79o |
| `yt3.mp4` | https://youtu.be/2-OR7BYb3K0 |
| `yt4.mp4` | https://youtu.be/3ugktyjfsk0 |

Clone the official SAM3 repository into `third_party/sam3/` and place the
checkpoint under its `checkpoints/` directory:

```text
datasets/create_external_test_set/third_party/
  sam3/
    checkpoints/sam3.pt
```

SAM3 references:

- GitHub: https://github.com/facebookresearch/sam3
- Project page: https://ai.meta.com/sam3
- Checkpoints: https://huggingface.co/facebook/sam3

Confirm the expected files are present:

```bash
test -f datasets/create_external_test_set/source_videos/yt1.mp4
test -f datasets/create_external_test_set/source_videos/yt2.mp4
test -f datasets/create_external_test_set/source_videos/yt3.mp4
test -f datasets/create_external_test_set/source_videos/yt4.mp4
test -f datasets/create_external_test_set/third_party/sam3/checkpoints/sam3.pt
```

Use a Python environment compatible with SAM3, PyTorch, OpenCV, Pillow, and your
CUDA runtime. No environment name is assumed by these scripts.

## 1. Extract Frames

Run from the `evd_catheter` repository root:

```bash
python datasets/create_external_test_set/generate_frames.py --overwrite
```

The extraction uses fixed frame ranges and copies `selected_points.json` into
the generated dataset:

| Scene | Source | Time range | Frame range | Count |
| --- | --- | --- | --- | --- |
| `scene_yt1_001` | `yt1.mp4` | `03:02-03:07` | `005454-005604` | 151 |
| `scene_yt2_002` | `yt2.mp4` | `02:25-02:53` | `004359-005184` | 826 |
| `scene_yt3_003` | `yt3.mp4` | `06:43-07:43` | `010075-011575` | 1501 |
| `scene_yt4_004` | `yt4.mp4` | `01:38-02:19` | `002940-004170` | 1231 |

Total: 3709 frames.

## 2. Generate SAM3 Masks

`selected_points.json` stores absolute pixel prompts as `[x, y, label]`, where
`label=1` is a positive catheter point and `label=0` is a
negative/background point. The mask script initializes SAM3 at the prompted
frame and propagates both forward and backward through each scene.

```bash
python datasets/create_external_test_set/generate_prompted_sam_masks.py \
  --device cuda
```

This writes `masks/` under each generated scene.
