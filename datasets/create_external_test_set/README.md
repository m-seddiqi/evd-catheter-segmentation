# Create External Test Masks

The paper external-test images and masks are distributed as a dataset archive;
see `datasets/README.md` for the download link and expected
`datasets/external_test_catheter/` layout.

Use this folder only if you want to build a new video-based test set or create
prompted masks for your own videos. The workflow is:

1. place source videos locally,
2. define the temporal segment for each scene,
3. select point prompts on one or more extracted frames,
4. propagate masks through the segment with SAM3.

The same idea can be implemented with SAM2 video propagation if that is the
available tool in your environment.

## Expected Output

The scripts write the standard image/mask layout used by the evaluation code:

```text
datasets/external_test_catheter/
  scene_example_001/
    images/frame_000120.jpg
    masks/frame_000120.png
```

Use `--output-root` if you want to create a different dataset folder.

## 1. Add Videos

Create a local video directory and add your own files:

```text
datasets/create_external_test_set/source_videos/
  example_video.mp4
```

The videos are not part of the repository. Keep only local or redistributable
files in this directory.

## 2. Define Segments

Create `video_segments.json` from the provided template:

```bash
cp datasets/create_external_test_set/video_segments.example.json \
  datasets/create_external_test_set/video_segments.json
```

Each scene entry names the output scene, the local video file, and the segment
to extract. Exact frame indices are preferred for reproducibility:

```json
{
  "scenes": [
    {
      "scene_id": "scene_example_001",
      "video_file": "example_video.mp4",
      "start_frame": 120,
      "end_frame": 240
    }
  ]
}
```

You may also use `start_time` and `end_time` instead of frame indices. Time
values can be seconds or strings such as `"00:04"` and `"01:12.5"`.

Extract frames from the repository root:

```bash
python datasets/create_external_test_set/generate_frames.py \
  --segments-json datasets/create_external_test_set/video_segments.json \
  --overwrite
```

## 3. Define Point Prompts

Edit `selected_points.json` so that each scene contains at least one prompted
frame. Points are absolute pixel coordinates in `[x, y, label]` format:

- `label=1`: positive point on the catheter or target tool.
- `label=0`: negative point on background or nearby non-target anatomy/tooling.

Example:

```json
{
  "scene_example_001": {
    "obj_id": 1,
    "prompts": [
      {
        "frame_name": "frame_000120.jpg",
        "points": [
          [960, 540, 1],
          [1000, 560, 1],
          [900, 520, 0]
        ]
      }
    ]
  }
}
```

For thin instruments, use multiple positive points along the visible axis and
one or more negative points close to confusing background structures.

## 4. Generate SAM3 Masks

Clone SAM3 and place the checkpoint where the script can find it:

```text
datasets/create_external_test_set/third_party/
  sam3/
    checkpoints/sam3.pt
```

Then run:

```bash
python datasets/create_external_test_set/generate_prompted_sam_masks.py \
  --dataset-root datasets/external_test_catheter \
  --points-json datasets/create_external_test_set/selected_points.json \
  --device cuda
```

This writes one PNG mask under `masks/` for each extracted frame. Review the
generated masks before using them as ground truth; prompted video propagation
usually benefits from a quick manual quality check.
