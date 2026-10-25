"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
*
* NumPy-only translation ATE used by replays and external SLAM comparisons.
"""

import numpy as np


def translation_ate(timestamps, positions, gt_timestamps, gt_positions,
                    find_scale=False, max_gap=0.1):
    """Interpolate within observed mocap intervals, then fit SE(3) or Sim(3).

    The gap bound applies to the entire bracketing interval. No extrapolation or
    interpolation across an unobserved section of the trajectory is permitted.
    """
    ts, xyz = np.asarray(timestamps, dtype=float), np.asarray(positions, dtype=float)
    gt_ts, gt_xyz = np.asarray(gt_timestamps, dtype=float), np.asarray(gt_positions, dtype=float)
    for times, points in ((ts, xyz), (gt_ts, gt_xyz)):
        if times.ndim != 1 or points.shape != (len(times), 3):
            raise ValueError("Expected timestamps (N,) and positions (N,3)")
        if not np.isfinite(times).all() or not np.isfinite(points).all():
            raise ValueError("Trajectory contains nonfinite samples")
        if np.any(np.diff(times) <= 0):
            raise ValueError("Trajectory timestamps must be strictly increasing")
    if len(gt_ts) < 2 or not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("Need at least two mocap samples and a positive gap bound")
    right = np.searchsorted(gt_ts, ts, side="left")
    bounded = np.minimum(right, len(gt_ts) - 1)
    exact = (right < len(gt_ts)) & (gt_ts[bounded] == ts)
    left = np.maximum(right - 1, 0)
    span = gt_ts[bounded] - gt_ts[left]
    keep = exact | ((right > 0) & (right < len(gt_ts)) & (span > 0) & (span <= max_gap))
    if np.count_nonzero(keep) < 3:
        raise ValueError("Fewer than three trajectory/mocap associations")
    alpha = np.divide(ts - gt_ts[left], span, out=np.zeros_like(ts), where=span > 0)
    reference = (gt_xyz[left] + alpha[:, None] * (gt_xyz[bounded] - gt_xyz[left]))[keep]
    reference[exact[keep]] = gt_xyz[bounded[keep][exact[keep]]]
    estimated = xyz[keep]
    x, y = estimated - estimated.mean(axis=0), reference - reference.mean(axis=0)
    variance = np.sum(x * x)
    if variance <= np.finfo(float).eps or np.sum(y * y) <= np.finfo(float).eps:
        raise ValueError("Trajectory alignment needs translation in both trajectories")
    U, singular, Vt = np.linalg.svd(y.T @ x)
    sign = np.ones(3)
    sign[-1] = np.linalg.det(U @ Vt)
    rotation = (U * sign) @ Vt
    scale = float(singular @ sign / variance) if find_scale else 1.0
    aligned = scale * (x @ rotation.T) + reference.mean(axis=0)
    errors = np.linalg.norm(aligned - reference, axis=1)
    # Do not count a jump between disconnected mocap intervals as traveled distance.
    adjacent = np.diff(ts[keep]) <= max_gap
    return dict(ate_rmse_m=float(np.sqrt(np.mean(errors**2))),
                ate_median_m=float(np.median(errors)), ate_max_m=float(errors.max()),
                gt_associations=int(keep.sum()), trajectory_poses=len(ts),
                association_fraction=float(keep.mean()), alignment="sim3" if find_scale else "se3",
                alignment_scale=scale, associated_start_s=float(ts[keep][0]),
                associated_end_s=float(ts[keep][-1]), max_mocap_gap_s=max_gap,
                gt_path_length_m=float(np.linalg.norm(np.diff(reference, axis=0), axis=1)[adjacent].sum()))
