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
import os
import sys
import platform

# import pyslam.config as config

# config.cfg.set_lib("depth_anything_v3")


from pyslam.slam import Camera
from pyslam.io.dataset_types import DatasetEnvironmentType
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.depth import img_from_depth
from pyslam.utilities.logging import Printer
from pyslam.utilities.system import set_rlimit

from .depth_estimator_base import DepthEstimator


import torch
from depth_anything_3.api import DepthAnything3


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


# Monocular depth estimator using the DepthAnythingV3 model.
class DepthEstimatorDepthAnythingV3(DepthEstimator):
    def __init__(
        self,
        device=None,
        camera: Camera = None,
        min_depth=0,
        max_depth=50,
        dataset_env_type=DatasetEnvironmentType.OUTDOOR,
        model_type="depth-anything/DA3METRIC-LARGE",
        precision=None,
        use_linear_interpolation=True,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        if device.type == "cuda":
            print("DepthEstimatorDepthAnythingV3: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise Exception("DepthEstimatorDepthAnythingV3: MPS is not available")
            print("DepthEstimatorDepthAnythingV3: Using MPS")
        else:
            print("DepthEstimatorDepthAnythingV3: Using CPU")

        transform = None
        model = DepthAnything3.from_pretrained(model_type)
        model = model.to(device=device).eval()

        self.use_linear_interpolation = use_linear_interpolation

        super().__init__(
            model,
            transform,
            device,
            camera=camera,
            min_depth=min_depth,
            max_depth=max_depth,
            dataset_env_type=dataset_env_type,
            precision=precision,
        )

    # Return the predicted depth map and the point cloud (if any)
    def infer(self, image, image_right=None):
        # DepthAnything3.inference expects a list of images
        # Wrap single image in a list
        images = [image]

        # Run inference
        depth_prediction = self.model.inference(images)

        # Extract depth map from prediction (first image in batch)
        # depth_prediction.depth is [N, H, W] float32 array
        depth_map = depth_prediction.depth[0]  # Extract first image depth

        # Convert to numpy if it's a tensor
        if isinstance(depth_map, torch.Tensor):
            depth_map = depth_map.cpu().numpy()

        # Rescale depth prediction to match original image dimensions if needed
        if depth_map.shape[:2] != image.shape[:2]:
            # Use INTER_NEAREST for depth to preserve discrete depth values
            # Use INTER_LINEAR if smooth depth maps are preferred
            interpolation_method = (
                cv2.INTER_LINEAR if self.use_linear_interpolation else cv2.INTER_NEAREST
            )
            rescaled_depth_map = cv2.resize(
                depth_map,
                (image.shape[1], image.shape[0]),
                interpolation=interpolation_method,
            )
        else:
            rescaled_depth_map = depth_map

        self.depth_map = rescaled_depth_map
        return rescaled_depth_map, None
