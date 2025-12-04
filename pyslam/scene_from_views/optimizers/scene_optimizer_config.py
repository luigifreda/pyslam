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
"""

from enum import Enum
from typing import Dict, Any


class SceneOptimizerType(str, Enum):
    """Enumeration of available scene optimizer types."""

    DENSE = "dense_scene_optimizer"
    SPARSE = "sparse_scene_optimizer"

    def __str__(self) -> str:
        return self.value


# ============================================================================
# DENSE OPTIMIZER CONFIGURATION GUIDE
# ============================================================================
# Single-stage: Optimizes poses, depths (per-pixel), intrinsics jointly. Loss: geometric consistency.
# TROUBLESHOOTING: Weird maps → reduce lr; Misalignment → increase niter; Divergence → reduce lr significantly.
# ============================================================================

# Default configurations for each optimizer type
DEFAULT_DENSE_OPTIMIZER_CONFIG: Dict[str, Any] = {
    "type": SceneOptimizerType.DENSE.value,
    # Optimizes: poses, depths (per-pixel), intrinsics jointly. Loss: geometric consistency (aligned pairwise 3D predictions).
    # niter: Number of iterations (range: 200-500). More = better alignment, slower.
    "niter": 300,
    # schedule: Learning rate schedule ("linear" or "cosine"). "linear" is more stable.
    "schedule": "linear",
    # lr: Learning rate (range: 0.005-0.02). Lower than sparse due to all-pixel gradients.
    "lr": 0.01,
}


# ============================================================================
# SPARSE OPTIMIZER CONFIGURATION GUIDE
# ============================================================================
# Two-stage: (1) COARSE: optimizes poses/scales (freezes intrinsics/depths), loss: 3D matching (gamma_loss 1.5).
#            (2) FINE: optimizes poses/intrinsics/depths, loss: 2D reprojection (gamma_loss 0.5).
# KEY: subsample affects anchor points QUADRATICALLY. Lower subsample = more points = need lower lr, more iterations.
# TROUBLESHOOTING: Weird maps → reduce lr1/lr2; Clusters (subsample=4) → lr1=0.05, niter1=900, lr2=0.012, niter2=450, matching_conf_thr=4.5.
#                  Depth noise at discontinuities → reduce lr2 to 0.01-0.015, increase niter2 to 400-500, lower matching_conf_thr to 5.0-6.0.
# ============================================================================
DEFAULT_SPARSE_OPTIMIZER_CONFIG: Dict[str, Any] = {
    "type": SceneOptimizerType.SPARSE.value,
    # subsample: Anchor point spacing (default: 8). Lower = more points (quadratically).
    # 2 and 4 are not working well, 8 is the best so far.
    # subsample=4: 4x more points, needs lr1=0.05, niter1=900, lr2=0.012, niter2=450, matching_conf_thr=4.5.
    # subsample=2: 16x more points, needs lr1=0.03, niter1=1000, lr2=0.01, niter2=500.
    "subsample": 8,
    # COARSE STAGE (Stage 1): Optimizes poses/scales (freezes intrinsics/depths). Loss: 3D point matching (gamma_loss 1.5).
    # lr1: Coarse learning rate (range: 0.05-0.15 for subsample=8). subsample=4: 0.04-0.06, subsample=2: 0.03-0.04.
    "lr1": 0.08,
    # niter1: Coarse iterations (range: 400-800 for subsample=8). subsample=4: 800-1000, subsample=2: 1000-1200.
    "niter1": 600,
    # FINE STAGE (Stage 2): Optimizes poses/intrinsics/depths. Loss: 2D reprojection (gamma_loss 0.5).
    # lr2: Fine learning rate (range: 0.01-0.05 for subsample=8). subsample=4: 0.01-0.015, subsample=2: 0.008-0.01.
    #      Depth noise at discontinuities: reduce to 0.01-0.015 for smoother depth optimization.
    "lr2": 0.01,
    # niter2: Fine iterations (range: 200-500). Critical for depth quality. subsample=4: 400-500.
    #         Depth noise at discontinuities: increase to 400-500 for better convergence.
    "niter2": 500,
    # matching_conf_thr: Confidence threshold for correspondences (range: 3.0-7.0). subsample=4 clustering: try 4.0-4.5.
    #                    Depth noise at discontinuities: lower to 5.0-6.0 to get more edge correspondences (current: 7.0 may be too high).
    "matching_conf_thr": 7.0,
    # shared_intrinsics: Share intrinsics across cameras (False = per-camera intrinsics).
    "shared_intrinsics": False,
    # optim_level: "coarse" (niter2=0), "refine" (no depth), "refine+depth" (full optimization).
    # "refine" seems to work the best so far. "refine+depth" is adding noise to the depth maps.
    "optim_level": "refine",
    # kinematic_mode: "mst" (faster) or "hclust-ward" (more robust, recommended for subsample=4).
    "kinematic_mode": "hclust-ward",
    # adaptive_scaling: Auto-adjust lr/iterations for different subsample (experimental, manual tuning may be needed).
    "adaptive_scaling": True,
}


def adapt_config_for_subsample(config: Dict[str, Any], subsample: int) -> Dict[str, Any]:
    """
    Adapt optimization parameters based on subsample value.

    When subsample decreases, the number of anchor points increases quadratically.
    This function scales learning rates and iterations to maintain optimization stability.

    NOTE: This is an experimental heuristic. The relationship between subsample and
    optimal learning rates/iterations is complex and dataset-dependent. Manual tuning
    may still be required for best results.

    Args:
        config: Configuration dictionary with lr1, lr2, niter1, niter2
        subsample: Subsampling factor (typically 2, 4, 8, etc.)

    Returns:
        Adapted configuration dictionary with scaled learning rates and iterations

    Scaling Strategy:
        - Learning rates scale DOWN with more points
          * subsample=4: square root scaling (more aggressive to prevent clustering)
          * subsample=2: cube root scaling (very aggressive)
        - Iterations scale UP with more points (more aggressive for subsample=4)
          * subsample=4: ~1.5-1.7x more iterations (ensures global consistency)
          * subsample=2: ~1.7-2.0x more iterations (capped at 2x)
        - Minimum learning rate scale: 0.25x for subsample=4, 0.2x for subsample=2
    """
    config = config.copy()
    reference_subsample = 8  # Default reference value (configs are tuned for this)

    if subsample == reference_subsample:
        return config  # No adaptation needed

    # Calculate how many more/fewer points we have compared to reference
    # Example: subsample=4 → point_ratio = (8/4)^2 = 4 (4x more points)
    #          subsample=2 → point_ratio = (8/2)^2 = 16 (16x more points)
    point_ratio = (reference_subsample / subsample) ** 2

    # Learning rate scaling: more points = need smaller learning rates
    # For subsample=4 (point_ratio=4), use more aggressive scaling to prevent clustering
    # For subsample=2 (point_ratio=16), use even more aggressive scaling
    if point_ratio <= 4:
        # For subsample=4: use square root scaling (more aggressive than cube root)
        # This helps prevent independent cluster convergence
        lr_scale = (1.0 / point_ratio) ** 0.5
    else:
        # For subsample=2 or lower: use cube root for very aggressive scaling
        lr_scale = (1.0 / point_ratio) ** (1.0 / 3.0)

    # Safety: don't reduce learning rates too much (minimum 25% of original for subsample=4)
    # Lower minimum for subsample=2 to allow more aggressive scaling
    min_lr_scale = 0.25 if point_ratio <= 4 else 0.2
    lr_scale = max(lr_scale, min_lr_scale)

    # Iteration scaling: more points need significantly more iterations for global consistency
    # More aggressive scaling for subsample=4 to prevent clustering issues
    if point_ratio <= 4:
        # For subsample=4: scale iterations more aggressively (1.5-1.7x)
        # This ensures coarse stage has enough time to establish global consistency
        iter_scale = 1.0 + 0.6 * (point_ratio - 1.0) / point_ratio
    else:
        # For subsample=2: even more aggressive (but cap at reasonable maximum)
        iter_scale = 1.0 + 0.7 * (point_ratio - 1.0) / point_ratio
        iter_scale = min(iter_scale, 2.0)  # Cap at 2x to avoid excessive runtime

    # Apply scaling to learning rates and iterations
    if "lr1" in config:
        config["lr1"] = config["lr1"] * lr_scale
    if "lr2" in config:
        config["lr2"] = config["lr2"] * lr_scale
    if "niter1" in config:
        config["niter1"] = int(config["niter1"] * iter_scale)
    if "niter2" in config:
        config["niter2"] = int(config["niter2"] * iter_scale)

    return config


def get_default_config(optimizer_type: SceneOptimizerType) -> Dict[str, Any]:
    """
    Get default configuration for a given optimizer type.

    Args:
        optimizer_type: Type of optimizer

    Returns:
        Dictionary with default configuration parameters

    Raises:
        ValueError: If optimizer_type is not recognized
    """
    if optimizer_type == SceneOptimizerType.DENSE:
        return DEFAULT_DENSE_OPTIMIZER_CONFIG.copy()
    elif optimizer_type == SceneOptimizerType.SPARSE:
        return DEFAULT_SPARSE_OPTIMIZER_CONFIG.copy()
    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")
