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

import config


from camera import Camera
from dataset import DatasetEnvironmentType


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'


# Base class for depth estimators via inference.
class DepthEstimator:
    def __init__(self, model, transform, device, camera=None,
                 min_depth=0, max_depth=50, dataset_env_type=DatasetEnvironmentType.OUTDOOR, precision=None):
        self.model = model
        self.transform = transform
        self.device = device
        self.camera = camera
        self.dataset_env_type = dataset_env_type
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.precision = precision
        
        self.depth_map = None
        self.disparity_map = None 

    def infer(self, image, image_right=None):
        raise NotImplementedError


# Stereo depth estimator using the Stereo SGBM algorithm. 
class DepthEstimatorSgbm(DepthEstimator):
    def __init__(self, device=None, camera:Camera=None,
                 min_depth=0, max_depth=50, dataset_env_type=DatasetEnvironmentType.OUTDOOR):
        super().__init__(None, None, None, camera=camera, 
                         min_depth=min_depth, max_depth=max_depth, 
                         dataset_env_type=dataset_env_type, precision=None)        
        # Stereo SGBM Parameters
        min_z = self.camera.b
        self.min_disparity = 0 # camera.bf / max_depth
        self.max_disparity = camera.bf / min_z 
        self.num_disparities = 16 * 8  # Must be divisible by 16
        self.block_size = 5  # Typically an odd number, >= 5

        # Create Stereo SGBM matcher
        self.stereo = cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=self.num_disparities,
            blockSize=self.block_size,
            P1=8 * 3 * self.block_size ** 2,  # Smoothness parameter (smaller = less smooth)
            P2=32 * 3 * self.block_size ** 2,  # Smoothness parameter (larger = more smooth)
            disp12MaxDiff=1,
            uniquenessRatio=15,
            speckleWindowSize=50,
            speckleRange=1,
            preFilterCap=63,
        )

    def infer(self, image, image_right=None):
        if image_right is None:
            message = 'Image right is None. Are you using a stereo dataset?'
            Printer.red(message)
            raise ValueError(message)
        # Compute disparity map
        disparity_map = self.stereo.compute(image, image_right).astype(np.float32) / 16.0
        self.disparity_map = disparity_map
        
        bf = self.camera.bf if self.camera is not None else 1.0
        if self.camera is None:
            Printer.red('Camera is None!')
                
        # Compute depth map
        # valid_mask = disparity_map > 0
        # depth_map = np.zeros_like(disparity_map, dtype=disparity_map.dtype)
        # depth_map[valid_mask] = bf / disparity_map[valid_mask] 
        abs_disparity_map = np.abs(disparity_map, dtype=float)
        depth_map = np.where(abs_disparity_map > self.min_disparity, bf / abs_disparity_map, 0.0)        
        self.depth_map = depth_map
        return depth_map
    