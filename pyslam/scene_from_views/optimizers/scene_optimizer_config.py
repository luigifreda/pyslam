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


# Default configurations for each optimizer type
DEFAULT_DENSE_OPTIMIZER_CONFIG: Dict[str, Any] = {
    "type": SceneOptimizerType.DENSE.value,
    "niter": 300,
    "schedule": "linear",
    "lr": 0.01,
}

DEFAULT_SPARSE_OPTIMIZER_CONFIG: Dict[str, Any] = {
    "type": SceneOptimizerType.SPARSE.value,
    "subsample": 8,
    "lr1": 0.08,  # Increased from 0.07 for better initial convergence
    "niter1": 600,  # Increased from 500 for more thorough coarse alignment
    "lr2": 0.02,  # Increased from 0.014 for better fine-tuning convergence
    "niter2": 300,  # Increased from 200 for more thorough refinement (critical for quality)
    "matching_conf_thr": 3.0,  # Lowered from 5.0 to allow more matches while filtering noise
    "shared_intrinsics": False,
    "optim_level": "refine+depth",
    "kinematic_mode": "hclust-ward",
}


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
