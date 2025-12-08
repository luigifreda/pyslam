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

import pyslam.config as config

config.cfg.set_lib("depth_pro")

import torch
import depth_pro

from pyslam.slam import Camera
from pyslam.io.dataset_types import DatasetEnvironmentType
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.depth import img_from_depth
from pyslam.utilities.system import Printer, set_rlimit

from .depth_estimator_base import DepthEstimator


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


# Moncocular depth estimator using the DepthPro model.
class DepthEstimatorDepthPro(DepthEstimator):
    def __init__(
        self,
        device=None,
        camera: Camera = None,
        min_depth=0,
        max_depth=50,
        dataset_env_type=DatasetEnvironmentType.OUTDOOR,
        precision=torch.float16,
    ):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print("DepthEstimatorDepthPro: Using CUDA")
        else:
            print("DepthEstimatorDepthPro: Using CPU")
        # Load model and preprocessing transform
        # NOTE: precision=torch.float16 makes the inference much faster
        if device.type == "cpu":
            print(
                f"DepthEstimatorDepthPro: Forcing precision {precision} on CPU since float16 may not be supported"
            )
            precision = torch.float32
        model, transform = depth_pro.create_model_and_transforms(device=device, precision=precision)
        model = model.to(device).eval()
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
        image = self.transform(image)
        f_px = torch.tensor(self.camera.fx) if self.camera is not None else None
        prediction = self.model.infer(image, f_px=f_px)
        # Extract depth and focal length.
        depth_prediction = prediction["depth"]  # Depth in [m].
        # focallength_px = prediction["focallength_px"]  # Focal length in pixels.
        depth_prediction = depth_prediction.squeeze().cpu().numpy()
        self.depth_map = depth_prediction
        return depth_prediction, None
