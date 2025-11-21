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

import numpy as np
from typing import List, Optional, Tuple
import trimesh


class SceneFromViewsResult:
    """
    Result structure for scene reconstruction from multiple views.

    Attributes:
        global_point_cloud: trimesh.PointCloud or None - Merged point cloud in world coordinates
        global_mesh: trimesh.Trimesh or None - Merged mesh in world coordinates
        camera_poses: List[np.ndarray] - List of camera-to-world transformation matrices (4x4)
        processed_images: List[np.ndarray] - List of processed images (as used by the model)
        depth_predictions: List[np.ndarray] or None - List of depth maps (H, W) if available
        point_clouds: List[PointCloud] or None - List of per-view point clouds (camera or world frame)
        intrinsics: List[np.ndarray] or None - List of camera intrinsic matrices (3x3) if available
        confidences: List[np.ndarray] or None - List of confidence maps (H, W) if available
    """

    def __init__(
        self,
        global_point_cloud: Optional[trimesh.PointCloud] = None,
        global_mesh: Optional[trimesh.Trimesh] = None,
        camera_poses: Optional[List[np.ndarray]] = None,
        processed_images: Optional[List[np.ndarray]] = None,
        depth_predictions: Optional[List[np.ndarray]] = None,
        point_clouds: Optional[List] = None,
        intrinsics: Optional[List[np.ndarray]] = None,
        confidences: Optional[List[np.ndarray]] = None,
    ):
        self.global_point_cloud = global_point_cloud
        self.global_mesh = global_mesh
        self.camera_poses = camera_poses if camera_poses is not None else []
        self.processed_images = processed_images if processed_images is not None else []
        self.depth_predictions = depth_predictions
        self.point_clouds = point_clouds
        self.intrinsics = intrinsics
        self.confidences = confidences


class SceneFromViewsBase:
    """
    Base class for scene reconstruction from multiple views.

    This class defines the interface for different 3D reconstruction models
    that can recover a 3D scene from a set of images.
    """

    def __init__(self, device=None, **kwargs):
        """
        Initialize the scene reconstruction model.

        Args:
            device: Device to run inference on (e.g., 'cuda', 'cpu')
            **kwargs: Additional model-specific parameters
        """
        self.device = device
        self.model = None

    def reconstruct(
        self, images: List[np.ndarray], as_pointcloud: bool = True, **kwargs
    ) -> SceneFromViewsResult:
        """
        Reconstruct a 3D scene from multiple views.

        This method implements the shared reconstruction pipeline:
        1. Preprocess images
        2. Run inference
        3. Postprocess results

        Args:
            images: List of input images (BGR or RGB format, numpy arrays)
            as_pointcloud: If True, return point cloud; if False, return mesh
            **kwargs: Additional reconstruction parameters (model-specific)

        Returns:
            SceneFromViewsResult: Result containing point cloud/mesh, poses, images, etc.
        """
        # Step 1: Preprocess images
        processed_images = self.preprocess_images(images, **kwargs)

        # Step 2: Run inference
        raw_output = self.infer(processed_images, **kwargs)

        # Step 3: Postprocess results
        result = self.postprocess_results(
            raw_output=raw_output,
            images=images,
            processed_images=processed_images,
            as_pointcloud=as_pointcloud,
            **kwargs,
        )

        return result

    def preprocess_images(self, images: List[np.ndarray], **kwargs) -> List:
        """
        Preprocess images for the model.

        Args:
            images: List of input images
            **kwargs: Additional preprocessing parameters (model-specific)

        Returns:
            Preprocessed images in the format expected by the model
        """
        raise NotImplementedError("Subclasses must implement preprocess_images()")

    def infer(self, processed_images: List, **kwargs):
        """
        Run inference on preprocessed images.

        Args:
            processed_images: Preprocessed images from preprocess_images()
            **kwargs: Additional inference parameters (model-specific)

        Returns:
            Raw output from the model (format depends on the specific model)
        """
        raise NotImplementedError("Subclasses must implement infer()")

    def postprocess_results(
        self,
        raw_output,
        images: List[np.ndarray],
        processed_images: List,
        as_pointcloud: bool = True,
        **kwargs,
    ) -> SceneFromViewsResult:
        """
        Postprocess raw model output into SceneFromViewsResult.

        Args:
            raw_output: Raw output from infer()
            images: Original input images (for reference)
            processed_images: Preprocessed images used for inference
            as_pointcloud: Whether to return point cloud or mesh
            **kwargs: Additional postprocessing parameters

        Returns:
            SceneFromViewsResult: Processed results
        """
        raise NotImplementedError("Subclasses must implement postprocess_results()")
