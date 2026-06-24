from __future__ import annotations

from typing import Optional, Tuple

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


def surface(mask_fg: np.ndarray) -> np.ndarray:
    if mask_fg.any():
        eroded = ndimage.binary_erosion(mask_fg, structure=np.ones((3, 3), dtype=bool), border_value=0)
        return mask_fg ^ eroded
    return np.zeros_like(mask_fg, dtype=bool)


def surface_points(mask_fg: np.ndarray, stride: int = 1) -> np.ndarray:
    ys, xs = np.nonzero(surface(mask_fg.astype(bool)))
    if ys.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    pts = np.column_stack([ys, xs]).astype(np.float32)
    if stride > 1:
        pts = pts[::stride]
    return pts


def surface_distances(
    gt_pts: np.ndarray,
    pred_pts: np.ndarray,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    n_gt = int(gt_pts.shape[0])
    n_pr = int(pred_pts.shape[0])
    if n_gt == 0 and n_pr == 0:
        return np.array([], dtype=np.float32), np.array([], dtype=np.float32)
    if (n_gt == 0) != (n_pr == 0):
        return None

    tree_gt = cKDTree(gt_pts)
    d_pr_to_gt, _ = tree_gt.query(pred_pts, k=1, workers=-1)
    tree_pr = cKDTree(pred_pts)
    d_gt_to_pr, _ = tree_pr.query(gt_pts, k=1, workers=-1)
    return d_pr_to_gt.astype(np.float32), d_gt_to_pr.astype(np.float32)


def nsd_from_points(gt_pts: np.ndarray, pred_pts: np.ndarray, tolerance_px: float) -> float:
    d = surface_distances(gt_pts, pred_pts)
    if d is None:
        return 0.0
    d_pr_to_gt, d_gt_to_pr = d
    if d_pr_to_gt.size == 0 and d_gt_to_pr.size == 0:
        return 1.0
    in_pred = int(np.count_nonzero(d_pr_to_gt <= float(tolerance_px)))
    in_gt = int(np.count_nonzero(d_gt_to_pr <= float(tolerance_px)))
    denom = int(d_pr_to_gt.size + d_gt_to_pr.size)
    return float((in_pred + in_gt) / max(denom, 1))


def miou_from_stats(
    intersection: np.ndarray,
    union: np.ndarray,
    include_background: bool,
    background_index: int,
    eps: float = 1e-6,
) -> float:
    valid = union > 0
    if not include_background and 0 <= background_index < valid.size:
        valid[background_index] = False
    if not np.any(valid):
        return 0.0
    return float(np.mean(intersection[valid] / (union[valid] + eps)))


def dice(tp: int, fp: int, fn: int) -> float:
    return float((2.0 * tp) / (2.0 * tp + fp + fn + 1e-8))
