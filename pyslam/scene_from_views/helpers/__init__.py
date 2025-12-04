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

"""
Scene optimizer helper utilities.

This module provides utility classes and functions for scene optimization.
All classes are re-exported here for backward compatibility.
"""

# Re-export all classes for backward compatibility
from .geometry import GeometryUtils, ProjectionUtils
from .camera import CameraUtils, PoseUtils, PoseInitialization
from .pointcloud import PointCloudUtils
from .matching import MatchingUtils, NearestNeighborMatcher
from .loss import LossFunctions, ConfidenceUtils
from .canonical_views import CanonicalViewUtils
from .sparse_ga import (
    SparseGA,
    GraphUtils,
    EdgeUtils,
    sparse_scene_optimizer,
    _extract_initial_poses_from_pairwise,
    PairOfSlices,
)
from .spectral import SpectralUtils
from .parameters import ParameterUtils
from .image import ImageUtils

__all__ = [
    # Geometry
    "GeometryUtils",
    "ProjectionUtils",
    # Camera
    "CameraUtils",
    "PoseUtils",
    "PoseInitialization",
    # Point Cloud
    "PointCloudUtils",
    # Matching
    "MatchingUtils",
    "NearestNeighborMatcher",
    # Loss
    "LossFunctions",
    "ConfidenceUtils",
    # Canonical Views
    "CanonicalViewUtils",
    # Sparse GA
    "SparseGA",
    "GraphUtils",
    "EdgeUtils",
    "sparse_scene_optimizer",
    "_extract_initial_poses_from_pairwise",
    "PairOfSlices",
    # Spectral
    "SpectralUtils",
    # Parameters
    "ParameterUtils",
    # Image
    "ImageUtils",
]
