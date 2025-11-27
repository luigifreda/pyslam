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

import cv2
import numpy as np
import torch
from typing import List, Optional
import trimesh
import os

from .scene_from_views_base import SceneFromViewsBase, SceneFromViewsResult
from pyslam.utilities.depth import depth2pointcloud, PointCloud, filter_shadow_points
from pyslam.utilities.dust3r import convert_mv_output_to_geometry
from pyslam.utilities.geometry import inv_poseRt


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class SceneFromViewsDepthAnythingV3(SceneFromViewsBase):
    """
    Scene reconstruction using Depth Anything V3.

    This model predicts monocular depth and can also output camera poses/intrinsics
    when using the multi-view/any-view variants; metric presets provide scale-aware
    depth together with the poses returned by the model.
    """

    def __init__(
        self, device=None, model_type="depth-anything/DA3-LARGE", filter_depth=True, **kwargs
    ):
        """
        Initialize Depth Anything V3 model.

        Args:
            device: Device to run inference on (e.g., 'cuda', 'cpu')
            model_type: Model type to use. Options:
                - "depth-anything/DA3-LARGE": Any-view model with pose/intrinsic prediction
                - "depth-anything/DA3METRIC-LARGE": Metric depth estimation with pose/intrinsic prediction
            filter_depth: Whether to filter shadow points from depth maps
        """
        super().__init__(device=device, **kwargs)

        if device is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        from depth_anything_3.api import DepthAnything3

        self.model = DepthAnything3.from_pretrained(model_type)
        self.model = self.model.to(device=self.device)
        self.model_type = model_type
        self.filter_depth = filter_depth

    def preprocess_images(self, images: List[np.ndarray], **kwargs) -> List[np.ndarray]:
        """
        Preprocess images for Depth Anything V3.
        The model handles preprocessing internally, so we just return the images.

        Args:
            images: List of input images
            **kwargs: Additional preprocessing parameters (not used for this model)
        """
        return images

    def infer(self, processed_images: List[np.ndarray], **kwargs):
        """
        Run inference on preprocessed images using Depth Anything V3.

        Args:
            processed_images: Preprocessed images (same as input for this model)
            **kwargs: Additional inference parameters (not used for this model)

        Returns:
            Raw depth prediction output from the model
        """
        # Run inference
        depth_prediction = self.model.inference(processed_images)
        return depth_prediction

    def postprocess_results(
        self,
        raw_output,
        images: List[np.ndarray],
        processed_images: List[np.ndarray],
        as_pointcloud: bool = True,
        **kwargs,
    ) -> SceneFromViewsResult:
        """
        Postprocess raw Depth Anything V3 output into SceneFromViewsResult.

        Args:
            raw_output: Raw depth prediction output from infer()
            images: Original input images (for reference)
            processed_images: Preprocessed images used for inference
            as_pointcloud: Whether to return point cloud or mesh
            **kwargs: Additional postprocessing parameters (not used for this model)

        Returns:
            SceneFromViewsResult: Reconstruction results
        """
        depth_prediction = raw_output

        # Extract results
        depth_prediction_processed_images = (
            depth_prediction.processed_images
        )  # [N, H, W, 3] array (format may vary)
        depth_maps = depth_prediction.depth  # [N, H, W] float32
        extrinsics = depth_prediction.extrinsics  # [N, 3, 4] or None - w2c (world-to-camera) format
        intrinsics = depth_prediction.intrinsics  # [N, 3, 3] or None
        confidences = depth_prediction.conf  # [N, H, W] or None

        # Convert to numpy if needed
        if isinstance(depth_maps, torch.Tensor):
            depth_maps = depth_maps.cpu().numpy()
        if isinstance(depth_prediction_processed_images, torch.Tensor):
            depth_prediction_processed_images = depth_prediction_processed_images.cpu().numpy()
        if extrinsics is not None and isinstance(extrinsics, torch.Tensor):
            extrinsics = extrinsics.cpu().numpy()
        if intrinsics is not None and isinstance(intrinsics, torch.Tensor):
            intrinsics = intrinsics.cpu().numpy()
        if confidences is not None and isinstance(confidences, torch.Tensor):
            confidences = confidences.cpu().numpy()

        # Process each image
        processed_images_list = []
        depth_predictions_list = []
        camera_poses_list = []
        intrinsics_list = []
        point_clouds_list = []
        confidences_list = []

        poses_are_available = extrinsics is not None and intrinsics is not None

        for i in range(len(images)):
            # Processed image - make a copy to avoid all images referencing the same data
            processed_images_list.append(np.copy(depth_prediction_processed_images[i]))

            # Filter depth if requested
            if self.filter_depth:
                depth_filtered = filter_shadow_points(depth_maps[i], delta_depth=None)
            else:
                depth_filtered = depth_maps[i]

            depth_predictions_list.append(depth_filtered)

            # Camera poses (convert from [3, 4] w2c to [4, 4] c2w)
            # SceneFromViewsBase expects camera_poses to be c2w (camera-to-world)
            if poses_are_available:
                # Depth Anything 3 extrinsics are in w2c (world-to-camera) format
                # Convert to c2w (camera-to-world) for camera_poses
                Rcw = extrinsics[i][:3, :3]
                tcw = extrinsics[i][:3, 3]
                Twc = inv_poseRt(Rcw, tcw)  # Convert w2c to c2w
                camera_poses_list.append(Twc)
                intrinsics_list.append(intrinsics[i])

                # Create point cloud in camera frame
                fx = intrinsics[i][0, 0]
                fy = intrinsics[i][1, 1]
                cx = intrinsics[i][0, 2]
                cy = intrinsics[i][1, 2]

                # Convert processed image for point cloud
                # depth2pointcloud expects image in [0, 255] range
                processed_img_for_pc = depth_prediction_processed_images[i]

                point_cloud_cam = depth2pointcloud(
                    depth_filtered,
                    processed_img_for_pc,
                    fx,
                    fy,
                    cx,
                    cy,
                )

                # Transform to world coordinates
                # Depth Anything 3 extrinsics are in w2c (world-to-camera) format
                # We need to convert to c2w (camera-to-world) to transform points
                Rcw = extrinsics[i][:3, :3]
                tcw = extrinsics[i][:3, 3]
                Twc = inv_poseRt(Rcw, tcw)  # Convert w2c to c2w
                Rwc = Twc[:3, :3]
                twc = Twc[:3, 3].reshape(3, 1)
                point_cloud_cam.points = (Rwc @ point_cloud_cam.points.T + twc).T

                point_clouds_list.append(point_cloud_cam)
            else:
                camera_poses_list.append(None)
                intrinsics_list.append(None)
                point_clouds_list.append(None)

            # Confidences
            if confidences is not None:
                confidences_list.append(confidences[i])
            else:
                confidences_list.append(None)

        # Merge point clouds if poses are available
        global_point_cloud = None
        global_mesh = None

        if poses_are_available and len(point_clouds_list) > 0:
            # Filter out None point clouds
            valid_point_clouds = [pc for pc in point_clouds_list if pc is not None]
            if len(valid_point_clouds) > 0:
                if as_pointcloud:
                    # Merge all point clouds
                    all_points = np.concatenate([pc.points for pc in valid_point_clouds], axis=0)
                    all_colors = np.concatenate([pc.colors for pc in valid_point_clouds], axis=0)
                    global_point_cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)
                else:
                    # Create mesh from point clouds (simplified - could be improved)
                    # For now, we'll create a point cloud and let trimesh handle it
                    all_points = np.concatenate([pc.points for pc in valid_point_clouds], axis=0)
                    all_colors = np.concatenate([pc.colors for pc in valid_point_clouds], axis=0)
                    global_point_cloud = trimesh.PointCloud(vertices=all_points, colors=all_colors)

        return SceneFromViewsResult(
            global_point_cloud=global_point_cloud,
            global_mesh=global_mesh,
            camera_poses=camera_poses_list if poses_are_available else None,
            processed_images=processed_images_list,
            depth_predictions=depth_predictions_list,
            point_clouds=point_clouds_list if poses_are_available else None,
            intrinsics=intrinsics_list if poses_are_available else None,
            confidences=confidences_list if confidences is not None else None,
        )
