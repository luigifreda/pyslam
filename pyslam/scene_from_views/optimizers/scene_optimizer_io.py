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

from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass
import torch


@dataclass
class PairPrediction:
    """
    Prediction data for a single image pair.

    This represents the 3D reconstruction results for a pair of images,
    used by both dense and sparse optimizers.
    """

    # Image indices for this pair
    image_idx_i: int  # Index of first image
    image_idx_j: int  # Index of second image

    # 3D point predictions
    pts3d_i: torch.Tensor  # [H, W, 3] 3D points for image i
    pts3d_j: (
        torch.Tensor
    )  # [H, W, 3] 3D points for image j (in image i's coordinate frame for dense, or canonical for sparse)

    # Confidence maps
    conf_i: torch.Tensor  # [H, W] confidence map for image i
    conf_j: torch.Tensor  # [H, W] confidence map for image j

    # Images (optional, can be extracted from images list using indices)
    image_i: Optional[torch.Tensor] = None  # [H, W, 3] RGB image i
    image_j: Optional[torch.Tensor] = None  # [H, W, 3] RGB image j


@dataclass
class SceneOptimizerInput:
    """
    Unified input structure for scene optimizers.

    This structure provides a single standard format that both dense and sparse
    optimizers can use, eliminating the need for format-specific fields.
    """

    # Image data
    images: List[torch.Tensor]  # List of [H, W, 3] RGB images (all images in the scene)

    # Pair predictions - unified format for both optimizers
    pair_predictions: List[PairPrediction]  # List of predictions, one per image pair

    # Pairs information (metadata about pairs, e.g., from make_pairs)
    pairs: List[Any]  # List of image pair metadata (kept for compatibility with existing code)

    # Metadata
    filelist: Optional[List[str]] = None  # List of image file names/identifiers
    cache_dir: Optional[str] = (
        None  # Cache directory for intermediate results (used by sparse optimizer)
    )

    # Canonical view data (for sparse optimizer)
    #
    # Canonical views are averaged 3D pointmaps computed per image by combining multiple
    # pairwise predictions. Instead of using raw pairwise predictions directly, the sparse
    # optimizer builds a canonical representation for each image by averaging all pairwise
    # predictions that include that image. This provides a more stable and consistent 3D
    # representation per image, which is then used for optimization.
    #
    # The pairs_output dictionary maps image pairs to cached file paths containing the
    # pairwise prediction data needed to compute canonical views:
    #   {(img1, img2): ((path1, path2), path_corres)}
    #
    # Where:
    #   - (img1, img2): Tuple of image identifiers (e.g., filenames)
    #   - path1: Path to cached file containing (X1, C1, X2, C2) tensors for view1
    #            X1: [H, W, 3] 3D points for image 1
    #            C1: [H, W] confidence map for image 1
    #            X2: [H, W, 3] 3D points for image 2 (in image 1's coordinate frame)
    #            C2: [H, W] confidence map for image 2
    #   - path2: Path to cached file containing (X1, C1, X2, C2) tensors for view2
    #   - path_corres: Path to cached correspondence data between the two images
    #
    # The sparse optimizer uses this data to:
    #   1. Load pairwise predictions from cache
    #   2. Compute canonical views by averaging multiple pairwise predictions per image
    #   3. Use canonical views for more stable optimization of camera poses and depths
    pairs_output: Optional[Dict[Tuple[str, str], Tuple[Tuple[str, str], str]]] = None


@dataclass
class SceneOptimizerOutput:
    """
    Unified output structure from scene optimizers.

    This structure provides a consistent format for optimizer results
    regardless of the underlying optimizer implementation.
    """

    # Scene object (optimizer-specific, but all have common interface)
    scene: Any

    # Optimizer type identifier
    optimizer_type: str

    # Common extracted results
    rgb_imgs: Optional[List[torch.Tensor]] = None  # List of [H, W, 3] RGB images
    focals: Optional[torch.Tensor] = None  # [n] focal lengths
    cams2world: Optional[torch.Tensor] = None  # [n, 4, 4] camera poses
    pts3d: Optional[List[torch.Tensor]] = None  # List of [H, W, 3] point clouds
    confs: Optional[List[torch.Tensor]] = None  # List of [H, W] confidence maps

    # Additional optimizer-specific data
    additional_data: Optional[Dict[str, Any]] = None
