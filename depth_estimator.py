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

from camera import Camera

import platform

import config
config.cfg.set_lib('depth_anything_v2')
config.cfg.set_lib('depth_pro')


from dataset import DatasetEnvironmentType
from utils_serialization import SerializableEnum, register_class
from utils_sys import Printer 

import torch
import depth_pro
from depth_anything_v2.dpt import DepthAnythingV2


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder



@register_class
class DepthEstimatorType(SerializableEnum):
    DEPTH_ANYTHING_V2 = 0   # Depth Anything V2
    DEPTH_PRO = 1           # "Depth Pro: Sharp Monocular Metric Depth in Less Than a Second"


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
    else:
        raise ValueError(f'Invalid depth estimator type: {depth_estimator_type}')


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

    def infer(self, image):
        raise NotImplementedError


# Depth estimator using the DepthPro model.
class DepthEstimatorDepthPro(DepthEstimator):
    def __init__(self, device=None, camera:Camera=None, 
                 min_depth=0, max_depth=50, dataset_env_type=DatasetEnvironmentType.OUTDOOR, precision=torch.float16):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print('DepthEstimatorDepthPro: Using CUDA')
        else:
            print('DepthEstimatorDepthPro: Using CPU')    
        # Load model and preprocessing transform
        # NOTE: precision=torch.float16 makes the inference much faster 
        if device.type == 'cpu':
            print(f'DepthEstimatorDepthPro: Forcing precision {precision} on CPU since float16 may not be supported')
            precision=torch.float32
        model, transform = depth_pro.create_model_and_transforms(device=device, precision=precision)
        model = model.to(device).eval()
        super().__init__(model, transform, device, camera=camera, 
                         min_depth=min_depth, max_depth=max_depth, 
                         dataset_env_type=dataset_env_type, precision=precision)

    def infer(self, image):
        image = self.transform(image)
        f_px = torch.tensor(self.camera.fx) if self.camera is not None else None
        prediction = self.model.infer(image, f_px=f_px)
        # Extract depth and focal length.
        depth_prediction = prediction["depth"]  # Depth in [m].
        #focallength_px = prediction["focallength_px"]  # Focal length in pixels.            
        depth_prediction = depth_prediction.squeeze().cpu().numpy()        
        return depth_prediction


# Depth estimator using the DepthAnythingV2 model.
class DepthEstimatorDepthAnythingV2(DepthEstimator):
    depth_anything_v2_path = kRootFolder + '/thirdparty/depth_anything_v2/metric_depth/checkpoints'
    model_configs = {
        'vits': {'encoder': 'vits', 'features': 64, 'out_channels': [48, 96, 192, 384]},
        'vitb': {'encoder': 'vitb', 'features': 128, 'out_channels': [96, 192, 384, 768]},
        'vitl': {'encoder': 'vitl', 'features': 256, 'out_channels': [256, 512, 1024, 1024]}
    } 
    def __init__(self, device=None, camera:Camera=None, 
                 min_depth=0, max_depth=50, dataset_env_type=DatasetEnvironmentType.OUTDOOR,
                 encoder_name='vitl', precision=None):  # or 'vits', 'vitb'   we use the largest by default
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == 'cuda':
            print('DepthEstimatorDepthPro: Using CUDA')
        else:
            print('DepthEstimatorDepthPro: Using CPU')   
        transform = None
        model = DepthAnythingV2(**{**DepthEstimatorDepthAnythingV2.model_configs[encoder_name], 'max_depth': max_depth})
        dataset_name = 'vkitti' if dataset_env_type == DatasetEnvironmentType.OUTDOOR else 'hypersim'
        model_path = f'{DepthEstimatorDepthAnythingV2.depth_anything_v2_path}/depth_anything_v2_metric_{dataset_name}_{encoder_name}.pth'
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model = model.to(device).eval()
        super().__init__(model, transform, device, camera=camera, 
                         min_depth=min_depth, max_depth=max_depth, 
                         dataset_env_type=dataset_env_type, precision=None)

    def infer(self, image):
        depth_prediction = self.model.infer_image(image)
        return depth_prediction