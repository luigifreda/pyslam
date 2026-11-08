"""
Match filtering for geomatch.

A filter decides which putative matches to keep. Filters are pure functions registered
by name in FILTERS, so a new method can be added without touching demo.py:

    @register("my_filter")
    def my_filter(ctx, **params) -> np.ndarray:
        ...  # return a boolean mask over the matches (True = keep)

and then used as:

    keep = filter_matches(ctx, method="my_filter", ransac="after")

filter_matches() composes a named filter with optional RANSAC outlier rejection
(fundamental matrix, USAC-MAGSAC) applied before or after it:

    method : name of a registered filter ("none", "depth_valid", "desc_threshold",
             "gt_epipolar", ...) -- see available_filters()
    ransac : "none" | "before" | "after" -- when to run RANSAC relative to the filter
    **params : forwarded to the filter (e.g. max_dist=..., err_px=...) and to RANSAC
               (ransac_err=...)

MatchContext carries everything a filter may need. Note on honesty: T1/T2 (ground-truth
poses) are included ONLY for oracle filters that serve as an upper-bound reference in
the demo evaluation. A production filter must not use them -- in the real pipeline no
GT pose exists at matching time.
"""

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MatchContext:
    """All data a filter may look at. Index arrays i1/i2 select the current putative
    matches: matched points are pts1[i1], pts2[i2]; per-match depths are z1[i1], z2[i2];
    matched descriptors are des1[i1], des2[i2]."""
    i1: np.ndarray      # indices of matched keypoints in image 1
    i2: np.ndarray      # indices of matched keypoints in image 2
    pts1: np.ndarray    # all keypoints image 1, N1x2
    pts2: np.ndarray    # all keypoints image 2, N2x2
    des1: np.ndarray    # all descriptors image 1 (concatenated appearance+geometry), N1xD
    des2: np.ndarray    # all descriptors image 2, N2xD
    depth1: np.ndarray  # dense depth map image 1, HxW
    depth2: np.ndarray  # dense depth map image 2, HxW
    z1: np.ndarray      # depth at keypoints image 1, N1
    z2: np.ndarray      # depth at keypoints image 2, N2
    K: np.ndarray       # 3x3 intrinsics
    T1: np.ndarray = None  # ORACLE ONLY: GT camera-to-world pose image 1
    T2: np.ndarray = None  # ORACLE ONLY: GT camera-to-world pose image 2


FILTERS = {}


def register(name):
    """Decorator: add a filter function to the registry under `name`."""
    def deco(fn):
        FILTERS[name] = fn
        return fn
    return deco


def available_filters():
    return sorted(FILTERS)


def _ransac_mask(p1, p2, err_thld=1.0):
    """Boolean mask over matches: inliers of a fundamental matrix estimated with
    USAC-MAGSAC (falls back to RANSAC). Degenerate inputs keep nothing."""
    if len(p1) < 8:
        return np.zeros(len(p1), dtype=bool)
    method = cv2.USAC_MAGSAC if hasattr(cv2, "USAC_MAGSAC") else cv2.RANSAC
    _, mask = cv2.findFundamentalMat(p1, p2, method, err_thld, confidence=0.999)
    if mask is None:
        return np.zeros(len(p1), dtype=bool)
    return mask.ravel().astype(bool)


def filter_matches(ctx, method="none", ransac="none", **params):
    """Apply a registered filter (and optional RANSAC) to the putative matches in ctx.

    Returns a boolean mask over the matches (True = keep). The named filter and RANSAC
    are applied in the order given by `ransac` ("before"/"after"); both are optional.
    """
    if method not in FILTERS:
        raise KeyError(f"unknown filter '{method}', available: {available_filters()}")
    if ransac not in ("none", "before", "after"):
        raise ValueError(f"ransac must be 'none', 'before' or 'after', got '{ransac}'")

    def ransac_step(mask):
        sub = _ransac_mask(ctx.pts1[ctx.i1[mask]], ctx.pts2[ctx.i2[mask]],
                           err_thld=params.get("ransac_err", 1.0))
        out = np.zeros_like(mask)
        out[mask] = sub
        return out

    keep = np.ones(len(ctx.i1), dtype=bool)
    if ransac == "before":
        keep = ransac_step(keep)
    keep &= FILTERS[method](ctx, **params)
    if ransac == "after":
        keep = ransac_step(keep)
    return keep


# ---------- built-in filters ----------

@register("none")
def _none(ctx, **params):
    """Keep everything (baseline)."""
    return np.ones(len(ctx.i1), dtype=bool)


@register("depth_valid")
def _depth_valid(ctx, **params):
    """Drop matches whose predicted depth is missing or non-positive in either image."""
    z1, z2 = ctx.z1[ctx.i1], ctx.z2[ctx.i2]
    return np.isfinite(z1) & np.isfinite(z2) & (z1 > 1e-6) & (z2 > 1e-6)


@register("desc_threshold")
def _desc_threshold(ctx, max_dist=1.0, **params):
    """Keep matches whose (concatenated) descriptor distance is below max_dist."""
    d = np.linalg.norm(ctx.des1[ctx.i1] - ctx.des2[ctx.i2], axis=1)
    return d < max_dist


@register("gt_epipolar")
def _gt_epipolar(ctx, err_px=4.0, **params):
    """ORACLE / upper bound only: keep matches within err_px of the GT epipolar line.
    Uses the ground-truth poses -- never use this as a real filter, only as a reference
    for what perfect geometric filtering could achieve in the demo."""
    if ctx.T1 is None or ctx.T2 is None:
        raise ValueError("gt_epipolar is an oracle filter and needs GT poses (T1, T2) in the context")
    T21 = np.linalg.inv(ctx.T2) @ ctx.T1
    R, t = T21[:3, :3], T21[:3, 3]
    tx = np.array([[0, -t[2], t[1]], [t[2], 0, -t[0]], [-t[1], t[0], 0]])
    Kinv = np.linalg.inv(ctx.K)
    F = Kinv.T @ tx @ R @ Kinv
    p1, p2 = ctx.pts1[ctx.i1], ctx.pts2[ctx.i2]
    x1 = np.c_[p1, np.ones(len(p1))]
    x2 = np.c_[p2, np.ones(len(p2))]
    lines = x1 @ F.T
    err = np.abs(np.sum(x2 * lines, 1)) / np.hypot(lines[:, 0], lines[:, 1])
    return err < err_px


@register("relative_depth_threshold")
def _relative_depth_threshold(ctx, rel_depth_thresh=0.5, **params):
    """Keep matches whose absolute difference in depth divided by 
    the sum of depths is below rel_depth_thresh."""
    rel_depth = np.abs(ctx.z1[ctx.i1] - ctx.z2[ctx.i2]) / (ctx.z1[ctx.i1] + ctx.z2[ctx.i2])
    return rel_depth < rel_depth_thresh