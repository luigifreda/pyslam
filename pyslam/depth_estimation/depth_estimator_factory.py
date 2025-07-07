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
import torch
import platform

from pyslam.slam.camera import Camera
from pyslam.io.dataset_types import DatasetEnvironmentType
from pyslam.utilities.utils_sys import import_from

from pyslam.utilities.utils_serialization import SerializableEnum, register_class

from .depth_estimator_base import DepthEstimator, DepthEstimatorSgbm
from .depth_estimator_depth_pro import DepthEstimatorDepthPro
from .depth_estimator_depth_anything_v2 import DepthEstimatorDepthAnythingV2
from .depth_estimator_raft_stereo import DepthEstimatorRaftStereo
from .depth_estimator_crestereo_megengine import DepthEstimatorCrestereoMegengine
from .depth_estimator_crestereo_pytorch import DepthEstimatorCrestereoPytorch


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'


@register_class
class DepthEstimatorType(SerializableEnum):
    DEPTH_SGBM                 = 0    # Depth SGBM from OpenCV [Stereo, it requires a stereo dataset]    
    DEPTH_ANYTHING_V2          = 1    # "Depth Anything V2" [Monocular]
    DEPTH_PRO                  = 2    # "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second" [Monocular]
    DEPTH_RAFT_STEREO          = 3    # "RAFT-Stereo: Multilevel Recurrent Field Transforms for Stereo Matching" [Stereo, it requires a stereo dataset]
    DEPTH_CRESTEREO_MEGENGINE  = 4    # "Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation" [Stereo, it requires a stereo dataset]
                                      # Note: Linking problems under Linux, not supported under mac [WIP] => use DepthEstimatorType.DEPTH_CRESTEREO_PYTORCH
    DEPTH_CRESTEREO_PYTORCH    = 5    # "Practical Stereo Matching via Cascaded Recurrent Network with Adaptive Correlation", pytorch implementation [Stereo, it requires a stereo dataset]
    DEPTH_MAST3R               = 6    # "Grounding Image Matching in 3D with MASt3R"
    DEPTH_MVDUST3R             = 7    # "MV-DUSt3R+: Single-Stage Scene Reconstruction from Sparse Views In 2 Seconds"
 

    @staticmethod
    def from_string(name: str):
        try:
            return DepthEstimatorType[name]
        except KeyError:
            raise ValueError(f"Invalid DepthEstimatorType: {name}")


def depth_estimator_factory(depth_estimator_type=DepthEstimatorType.DEPTH_ANYTHING_V2,
                            device=None,
                            camera:Camera=None,
                            min_depth=0,      # not used at the moment
                            max_depth=50,
                            dataset_env_type=DatasetEnvironmentType.OUTDOOR,
                            precision=torch.float16):
    if depth_estimator_type == DepthEstimatorType.DEPTH_ANYTHING_V2:
        return DepthEstimatorDepthAnythingV2(device=device, camera=camera, 
                                             min_depth=min_depth, max_depth=max_depth, 
                                             dataset_env_type=dataset_env_type, precision=precision)
    elif depth_estimator_type == DepthEstimatorType.DEPTH_PRO:
        return DepthEstimatorDepthPro(device=device, camera=camera, 
                                      max_depth=max_depth, 
                                      dataset_env_type=dataset_env_type, precision=precision)
    elif depth_estimator_type == DepthEstimatorType.DEPTH_SGBM:
        return DepthEstimatorSgbm(device=device, camera=camera, 
                                  min_depth=min_depth, max_depth=max_depth, 
                                  dataset_env_type=dataset_env_type)
    elif depth_estimator_type == DepthEstimatorType.DEPTH_RAFT_STEREO:
        return DepthEstimatorRaftStereo(device=device, camera=camera, 
                                  min_depth=min_depth, max_depth=max_depth, 
                                  dataset_env_type=dataset_env_type)
    elif depth_estimator_type == DepthEstimatorType.DEPTH_CRESTEREO_MEGENGINE:
        if platform.system() == 'Darwin':
            raise ValueError('DepthEstimatorType.DEPTH_CRESTEREO_MEGENGINE is not supported on macOS. Use DepthEstimatorType.DEPTH_CRESTEREO_PYTORCH instead!')
        return DepthEstimatorCrestereoMegengine(device=device, camera=camera, 
                                       min_depth=min_depth, max_depth=max_depth, 
                                       dataset_env_type=dataset_env_type)
    elif depth_estimator_type == DepthEstimatorType.DEPTH_CRESTEREO_PYTORCH:
        return DepthEstimatorCrestereoPytorch(device=device, camera=camera, 
                                              min_depth=min_depth, max_depth=max_depth, 
                                              dataset_env_type=dataset_env_type)
    elif depth_estimator_type == DepthEstimatorType.DEPTH_MAST3R:
        from pyslam.depth_estimation.depth_estimator_mast3r import DepthEstimatorMast3r # available only with CUDA       
        return DepthEstimatorMast3r(device=device, camera=camera, 
                                    min_depth=min_depth, max_depth=max_depth, 
                                    dataset_env_type=dataset_env_type)
    elif depth_estimator_type == DepthEstimatorType.DEPTH_MVDUST3R:
        from pyslam.depth_estimation.depth_estimator_mvdust3r import DepthEstimatorMvdust3r # available only with CUDA      
        return DepthEstimatorMvdust3r(device=device, camera=camera, 
                                    min_depth=min_depth, max_depth=max_depth, 
                                    dataset_env_type=dataset_env_type)
    else:
        raise ValueError(f'Invalid depth estimator type: {depth_estimator_type}')

