# Paper Checkpoints

Each subfolder contains one released model-weight file. The paper tester does
not read training configs from this folder; model metadata is fixed in
`reports/paper_checkpoint_test.py` and summarized here.

Expected file layout:

```text
paper_checkpoint/
  fastvit_t8_both/model.pth
  fastvit_sa12_both/model.pth
  fastvit_sa36_both/model.pth
  convnext_tiny_both/model.pth
  mobilenetv3_large_100_both/model.pth
```

The tester also accepts a single `.pth` or `.pt` file under a checkpoint folder
if it is uniquely identifiable, but `model.pth` is the preferred release name.

## Model Metadata

| Checkpoint folder | Model variant | Classes | FPN dim | Eval crop sizes |
| --- | --- | ---: | ---: | --- |
| `fastvit_t8_both` | `fastvit_t8` | 2 | 256 | `512x512`, `384x640` |
| `fastvit_sa12_both` | `fastvit_sa12` | 2 | 256 | `512x512`, `384x640` |
| `fastvit_sa36_both` | `fastvit_sa36` | 2 | 256 | `512x512`, `384x640` |
| `convnext_tiny_both` | `convnext_tiny` | 2 | 256 | `512x512`, `384x640` |
| `mobilenetv3_large_100_both` | `mobilenetv3_large_100` | 2 | 256 | `512x512`, `384x640` |

Evaluation uses foreground-only mIoU (`background_index=0`,
`eval_include_background=false`) and `ignore_index=255`.

`reports/paper_checkpoint_test.py --compute-metrics` computes mIoU/Dice/NSD in
the letterboxed evaluation crop space (`512x512` or `384x640`) to match the
paper testing script. Predicted mask PNGs are saved at the original image size.
