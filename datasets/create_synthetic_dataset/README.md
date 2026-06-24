# Create Synthetic Dataset

This folder generates synthetic catheter image/mask sequences from the assets in
`3d_tools/`. The generator samples one background image, one catheter GLB, and
small per-frame motion for each scene. File names are not hardcoded: every
supported image in `3d_tools/background_img/` is eligible as a background, and
every GLB in `3d_tools/catheter/` is eligible as a catheter.

The repository includes a small set of background images so the generator can be
tested immediately. Some background images used for the paper release are not
stored directly in git. If you have dataset access, download the background
archive from:

```text
https://huggingface.co/datasets/mussed/evd_catheter_segmentation_datasets/resolve/main/background_img.zip
```

Extract it so that it replaces or augments:

```text
datasets/create_synthetic_dataset/3d_tools/background_img/
```

The generator also works with the currently provided sample backgrounds. For a
different instrument or imaging setup, replace this folder with background
images from the target domain.

Optional non-catheter tools are configured by category folder. The default config
keeps one disabled example category, `clippers` with `probability=0.0`, as a
template for adding distractor tools later. If enabled, distractors are rendered
into RGB frames only; masks remain catheter-only.

Generated data is written as:

```text
datasets/synthetic_dataset/
  e0/
    scene_00000/
      images/frame_00000.jpg
      masks/frame_00000.png
      scene.json
    generation_plan.json
  e1/
  ...
```

Each `scene_*` folder contains JPEG RGB frames under `images/`, PNG binary
catheter masks under `masks/`, and the sampled initial transform in
`scene.json`. Masks stay PNG so segmentation labels remain lossless.

## Setup

Install the generator dependencies:

```bash
python -m pip install -r datasets/create_synthetic_dataset/requirements.txt
```

For headless rendering, set an OpenGL backend if your machine needs it:

```bash
export PYOPENGL_PLATFORM=egl
```

## Paper Dataset Commands

Create `e0` with 1,600 scenes and 10 frames per scene:

```bash
python datasets/create_synthetic_dataset/generate.py \
  --epoch 0 \
  --scenes 1600 \
  --frames 10 \
  --workers 16 \
  --scenes-per-worker 50 \
  --clear-existing
```

Create `e1` through `e10` with 1,440 scenes per folder and 10 frames per scene:

```bash
python datasets/create_synthetic_dataset/generate.py \
  --start-epoch 1 \
  --end-epoch 10 \
  --scenes 1440 \
  --frames 10 \
  --workers 24 \
  --scenes-per-worker 60 \
  --clear-existing
```

Both commands write to `datasets/synthetic_dataset/` by default. Use
`--output-root /path/to/root` to write the `e*` folders somewhere else.

## 3D Tool Assets

The catheter GLB assets used by the paper experiments are provided under:

```text
datasets/create_synthetic_dataset/3d_tools/catheter/
```

For the paper, the catheter meshes were prepared from 2D tool imagery with a
SAM3D-style reconstruction workflow, then manually scaled and oriented so the
renderer produced anatomically plausible projections. If you adapt this pipeline
to another thin instrument, create or obtain a GLB for that tool, scale and
orient it in the same coordinate convention, and add it to a category folder
under `3d_tools/`. Rendering behavior is controlled in `config.py`.

## Smoke Test

Run a tiny local render before launching the full paper dataset:

```bash
python datasets/create_synthetic_dataset/generate.py \
  --count 1 \
  --frames 2 \
  --seed 42 \
  --output-dir /tmp/evd_synthetic_smoke
```

Expected output:

```text
/tmp/evd_synthetic_smoke/
  scene_00000/images/frame_00000.jpg
  scene_00000/images/frame_00001.jpg
  scene_00000/masks/frame_00000.png
  scene_00000/masks/frame_00001.png
  scene_00000/scene.json
```

## Useful Options

```text
--epoch N              generate one folder named eN
--start-epoch A        first folder for a range
--end-epoch B          last folder for a range
--scenes N             scenes per generated folder
--frames N             frames per scene
--seed N               base seed for direct generation
--seed-base N          base seed for epoch folders
--workers N            parallel render worker processes
--scenes-per-worker N  scenes assigned to each worker job
--clear-existing       remove old scene_* folders before generating
--output-root PATH     root that receives e0, e1, ...
--image-format FORMAT  jpg, jpeg, or png for RGB frames; default jpg
--jpeg-quality N       JPEG quality for RGB frames; default 95
```

The defaults live in `config.py`: `1920 x 1080` RGB frames saved as JPEG
quality 95, PNG masks, 10 frames per scene, catheter color variation, all
available backgrounds from `3d_tools/background_img/`, all available catheter
GLBs from `3d_tools/catheter/`, the disabled `clippers` distractor example, and
mild per-frame object motion. Frames with empty/tiny catheter masks are retried
during generation (`min_catheter_mask_pixels=100` by default). On the current
generated dataset, JPEG quality 95 reduces RGB-frame storage by roughly 70%
compared with PNG while keeping masks lossless.

To add a non-catheter tool category, put GLB files under
`3d_tools/<category_name>/` and add one `CategoryRule` entry in `config.py`. The
sampler will choose from every GLB in that folder, so the rule stays independent
of individual file names.

Seeds are scene-index based, not worker based. Direct generation uses
`scene_seed = seed + scene_id`; epoch-folder generation uses
`scene_seed = seed_base + epoch * 1000000 + scene_id`, so changing
`--workers` or `--scenes-per-worker` does not change sampled scenes.
