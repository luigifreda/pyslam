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
        4. Optionally apply scene optimizer as post-processing

        Args:
            images: List of input images (BGR or RGB format, numpy arrays)
            as_pointcloud: If True, return point cloud; if False, return mesh
            **kwargs: Additional reconstruction parameters (model-specific):
                - optimizer: Optional SceneOptimizerBase instance for post-processing
                - optimizer_kwargs: Optional kwargs to pass to optimizer

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

        # Step 4: Optionally apply scene optimizer as post-processing
        optimizer = kwargs.get("optimizer", None)
        if optimizer is not None:
            from pyslam.scene_from_views.optimizers import SceneOptimizerBase

            if isinstance(optimizer, SceneOptimizerBase):
                if optimizer.can_optimize_from_result(result):
                    optimizer_kwargs = kwargs.get("optimizer_kwargs", {})
                    result = self.apply_optimizer_postprocessing(
                        result, optimizer, processed_images, **optimizer_kwargs
                    )
                else:
                    import warnings

                    warnings.warn(
                        f"Optimizer {optimizer.optimizer_type} cannot process this result. "
                        "Skipping optimizer post-processing."
                    )

        return result

    def apply_optimizer_postprocessing(
        self,
        result: SceneFromViewsResult,
        optimizer,
        processed_images: List = None,
        **optimizer_kwargs,
    ) -> SceneFromViewsResult:
        """
        Apply a scene optimizer as post-processing step on a SceneFromViewsResult.

        Args:
            result: SceneFromViewsResult from initial reconstruction
            optimizer: SceneOptimizerBase instance to apply
            processed_images: Preprocessed images used for initial reconstruction
            **optimizer_kwargs: Additional parameters for the optimizer

        Returns:
            SceneFromViewsResult: Updated result with optimized poses/depths/intrinsics
        """
        from pyslam.scene_from_views.optimizers import SceneOptimizerBase

        if not isinstance(optimizer, SceneOptimizerBase):
            raise TypeError(
                f"optimizer must be an instance of SceneOptimizerBase, got {type(optimizer)}"
            )

        # Try to optimize from result
        try:
            optimizer_result = optimizer.optimize_from_result(
                result, processed_images=processed_images, **optimizer_kwargs
            )
            scene = optimizer_result["scene"]

            # Create SceneOptimizerOutput from result
            from pyslam.scene_from_views.optimizers import SceneOptimizerOutput

            optimizer_output = SceneOptimizerOutput(
                scene=scene,
                optimizer_type=optimizer.optimizer_type,
            )

            # Extract optimized results
            extracted_output = optimizer.extract_results(
                optimizer_output, processed_images, **optimizer_kwargs
            )
            extracted = {
                "rgb_imgs": extracted_output.rgb_imgs,
                "focals": extracted_output.focals,
                "cams2world": extracted_output.cams2world,
                "pts3d": extracted_output.pts3d,
                "confs": extracted_output.confs,
            }

            # Update result with optimized data
            result.camera_poses = [
                extracted["cams2world"][i] for i in range(len(extracted["cams2world"]))
            ]
            result.intrinsics = [
                self._build_intrinsics_matrix(
                    extracted["focals"][i], extracted["rgb_imgs"][i].shape[:2]
                )
                for i in range(len(extracted["focals"]))
            ]
            result.depth_predictions = [
                self._extract_depth_from_pts3d(extracted["pts3d"][i], extracted["confs"][i])
                for i in range(len(extracted["pts3d"]))
            ]
            result.confidences = extracted["confs"]

            # Rebuild point clouds with optimized data
            import trimesh

            point_clouds = []
            for i, (pts, msk, img) in enumerate(
                zip(extracted["pts3d"], extracted["confs"], extracted["rgb_imgs"])
            ):
                h, w = img.shape[:2]
                pts_reshaped = pts.reshape(h, w, 3)
                mask = msk > 0  # Simple threshold
                pts_valid = pts_reshaped[mask].reshape(-1, 3)
                img_valid = img[mask].reshape(-1, 3)
                if len(pts_valid) > 0:
                    pc = trimesh.PointCloud(vertices=pts_valid, colors=img_valid)
                    point_clouds.append(pc)
                else:
                    point_clouds.append(None)
            result.point_clouds = point_clouds

        except NotImplementedError as e:
            import warnings

            warnings.warn(
                f"Optimizer {optimizer.optimizer_type} does not support post-processing from "
                f"SceneFromViewsResult: {e}. Skipping optimizer post-processing."
            )

        return result

    def _build_intrinsics_matrix(self, focal: float, image_shape: tuple) -> np.ndarray:
        """Build camera intrinsics matrix from focal length and image shape."""
        K = np.eye(3)
        K[0, 0] = focal
        K[1, 1] = focal
        K[0, 2] = image_shape[1] / 2
        K[1, 2] = image_shape[0] / 2
        return K

    def _extract_depth_from_pts3d(self, pts3d: np.ndarray, conf: np.ndarray) -> np.ndarray:
        """Extract depth map from 3D points (z-component)."""
        h, w = conf.shape
        depth = np.zeros((h, w), dtype=pts3d.dtype)
        pts_reshaped = pts3d.reshape(h, w, 3)
        mask = conf > 0
        depth[mask] = pts_reshaped[mask, 2]  # z-component
        return depth

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

    def create_optimizer_input(
        self,
        raw_output,
        pairs: List,
        processed_images: List,
        **kwargs,
    ):
        """
        Create SceneOptimizerInput from model output.

        This method converts model-specific output into a unified SceneOptimizerInput
        structure that can be used by any scene optimizer.

        Args:
            raw_output: Raw output from infer() (format depends on the specific model)
            pairs: List of image pairs used for inference
            processed_images: Preprocessed images used for inference
            **kwargs: Additional parameters (e.g., filelist, cache_dir)

        Returns:
            SceneOptimizerInput: Unified input representation for optimizers
        """
        raise NotImplementedError("Subclasses must implement create_optimizer_input()")
