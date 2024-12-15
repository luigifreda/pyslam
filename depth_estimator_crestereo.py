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
config.cfg.set_lib('crestereo')


from camera import Camera
from dataset import DatasetEnvironmentType
from utils_depth import img_from_depth 
from utils_sys import Printer, set_rlimit

from depth_estimator_base import DepthEstimator


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder


def enforce_megengine_linking():
    import ctypes
    # Path to the directory containing your libraries
    site_packages_path = next(p for p in sys.path if 'site-packages' in p)
    lib_path = os.path.join(site_packages_path, 'megengine', 'core', 'lib')        
    print(f'DepthEstimatorCrestereo: lib_path = {lib_path}')
    if not os.path.exists(lib_path):
        Printer.red(f'DepthEstimatorCrestereo: lib_path does not exist: {lib_path}')
        return

    # Add the library path to the environment variable
    os.environ["LD_LIBRARY_PATH"] = lib_path + ":" + os.environ.get("LD_LIBRARY_PATH", "")

    # Explicitly load the libraries in the correct order
    ctypes.CDLL(os.path.join(lib_path, "libnvinfer.so.8"))
    ctypes.CDLL(os.path.join(lib_path, "libnvinfer_plugin.so.8"))
    ctypes.CDLL(os.path.join(lib_path, "libcudnn.so.8"))
    ctypes.CDLL(os.path.join(lib_path, "libcublas.so.11"))
    ctypes.CDLL(os.path.join(lib_path, "libcublasLt.so.11"))
    ctypes.CDLL(os.path.join(lib_path, "libnvrtc.so.11.2"))
    ctypes.CDLL(os.path.join(lib_path, "libmegengine_shared.so"))  # Load the main library last
    
    
#enforce_megengine_linking()
import megengine as mge
import crestereo.nets as crestereo_nets


# Stereo depth prediction using the Crestereo model.
class DepthEstimatorCrestereo(DepthEstimator):
    kCrestereoBasePath=kRootFolder +'/thirdparty/crestereo'
    kCrestereoModelPath=kCrestereoBasePath +'/crestereo_eth3d.mge'
    def __init__(self, device=None, camera:Camera=None,
                 min_depth=0, max_depth=50, dataset_env_type=DatasetEnvironmentType.OUTDOOR,
                 n_iter = 20):
        
        min_z = camera.b
        self.min_disparity = camera.bf / max_depth
        self.max_disparity = camera.bf / min_z          
                
        self.n_iter = n_iter
        model = self.load_model()
        super().__init__(model=model, transform=None, device=None, camera=camera, 
                         min_depth=min_depth, max_depth=max_depth, 
                         dataset_env_type=dataset_env_type, precision=None)        
        
    def load_model(self, model_path=kCrestereoModelPath):
        print("DepthEstimatorCrestereo: Loading model:", os.path.abspath(model_path))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")        
        pretrained_dict = mge.load(model_path)
        model = crestereo_nets.Model(max_disp=self.max_disparity, mixed_precision=False, test_mode=True)
        model.load_state_dict(pretrained_dict["state_dict"], strict=True)
        model.eval()
        return model

    def infer(self, image, image_right=None):     
        if image_right is None:
            message = 'Image right is None. Are you using a stereo dataset?'
            Printer.red(message)
            raise ValueError(message)
        
        in_shape = image.shape 
        print(f'DepthEstimatorCrestereo: Running inference: {image.shape} {image_right.shape}')

        # Compute disparity map
        imgL = image.transpose(2, 0, 1)
        imgR = image_right.transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL[None, :, :, :])
        imgR = np.ascontiguousarray(imgR[None, :, :, :])

        imgL = mge.tensor(imgL).astype("float32")
        imgR = mge.tensor(imgR).astype("float32")

        imgL_dw2 = mge.functional.nn.interpolate(
            imgL,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        imgR_dw2 = mge.functional.nn.interpolate(
            imgR,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        pred_flow_dw2 = self.model(imgL_dw2, imgR_dw2, iters=self.n_iter, flow_init=None)

        pred_flow = self.model(imgL, imgR, iters=self.n_iter, flow_init=pred_flow_dw2)
        disparity_map = mge.functional.squeeze(pred_flow[:, 0, :, :]).numpy()
            
        out_shape = disparity_map.shape
        if out_shape[0] != in_shape[0] or out_shape[1] != in_shape[1]:
            in_w = in_shape[1]
            out_w = out_shape[1]
            # got this formula from the original testing code
            t = float(in_w) / float(out_w)
            disparity_map = cv2.resize(disparity_map, (in_shape[1], in_shape[0]), interpolation=cv2.INTER_AREA)*t

        self.disparity_map = disparity_map

        bf = self.camera.bf if self.camera is not None else 1.0
        if self.camera is None:
            Printer.red('Camera is None!')
                
        # Compute depth map
        abs_disparity_map = np.abs(disparity_map, dtype=float)
        depth_map = np.where(abs_disparity_map > self.min_disparity, bf / abs_disparity_map, 0.0)
        self.depth_map = depth_map
        
        print(f'DepthEstimatorCrestereo: Depth map shape: {depth_map.shape}')
        return depth_map
    
    