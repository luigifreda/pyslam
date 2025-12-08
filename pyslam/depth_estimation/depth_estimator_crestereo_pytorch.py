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

config.cfg.set_lib("crestereo_pytorch")


from pyslam.slam import Camera
from pyslam.io.dataset_types import DatasetEnvironmentType
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.depth import img_from_depth
from pyslam.utilities.system import Printer, set_rlimit

from .depth_estimator_base import DepthEstimator

import torch
import crestereo_pytorch.nets as crestereo_pytorch_nets


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


# Stereo depth prediction using the Crestereo model with pytorch.
class DepthEstimatorCrestereoPytorch(DepthEstimator):
    kCrestereoBasePath = kRootFolder + "/thirdparty/crestereo_pytorch"
    kCrestereoModelPath = kCrestereoBasePath + "/models/crestereo_eth3d.pth"

    def __init__(
        self,
        device=None,
        camera: Camera = None,
        min_depth=0,
        max_depth=50,
        dataset_env_type=DatasetEnvironmentType.OUTDOOR,
        n_iter=10,
    ):
        min_z = camera.b
        self.min_disparity = camera.bf / max_depth
        self.max_disparity = camera.bf / min_z
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.n_iter = n_iter
        model = self.load_model()
        super().__init__(
            model=model,
            transform=None,
            device=device,
            camera=camera,
            min_depth=min_depth,
            max_depth=max_depth,
            dataset_env_type=dataset_env_type,
            precision=None,
        )

    def load_model(self, model_path=kCrestereoModelPath):
        print("DepthEstimatorCrestereoPytorch: Loading model:", os.path.abspath(model_path))
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        model = crestereo_pytorch_nets.Model(
            max_disp=self.max_disparity, mixed_precision=False, test_mode=True
        )
        model.load_state_dict(torch.load(model_path), strict=True)
        model.to(self.device)
        model.eval()
        return model

    # Return the predicted depth map and the point cloud (if any)
    def infer(self, image, image_right=None):
        if image_right is None:
            message = "Image right is None. Are you using a stereo dataset? If not, you cant use a stereo depth estimator here."
            Printer.red(message)
            raise ValueError(message)

        print(f"DepthEstimatorCrestereoPytorch: Input image: {image.shape} {image_right.shape}")
        in_h, in_w = image.shape[0:2]

        eval_h, eval_w = ((in_h // 8) * 8, (in_w // 8) * 8)
        # eval_h, eval_w = image.shape[0:2]
        imgL = cv2.resize(image, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        imgR = cv2.resize(image_right, (eval_w, eval_h), interpolation=cv2.INTER_LINEAR)
        print(f"DepthEstimatorCrestereoPytorch: Running inference: {imgL.shape}")

        # Compute disparity map
        imgL = imgL.transpose(2, 0, 1)
        imgR = imgR.transpose(2, 0, 1)
        imgL = np.ascontiguousarray(imgL[None, :, :, :])
        imgR = np.ascontiguousarray(imgR[None, :, :, :])

        # imgL = torch.tensor(imgL.astype("float32")).to(self.device)
        # imgR = torch.tensor(imgR.astype("float32")).to(self.device)
        imgL = torch.tensor(imgL.astype("float32"), device=self.device)
        imgR = torch.tensor(imgR.astype("float32"), device=self.device)

        import torch.nn.functional as F

        imgL_dw2 = F.interpolate(
            imgL,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        imgR_dw2 = F.interpolate(
            imgR,
            size=(imgL.shape[2] // 2, imgL.shape[3] // 2),
            mode="bilinear",
            align_corners=True,
        )
        # print(imgR_dw2.shape)
        with torch.inference_mode():
            pred_flow_dw2 = self.model(imgL_dw2, imgR_dw2, iters=self.n_iter, flow_init=None)
            pred_flow = self.model(imgL, imgR, iters=self.n_iter, flow_init=pred_flow_dw2)
        disparity_map = torch.squeeze(pred_flow[:, 0, :, :]).cpu().detach().numpy()

        out_shape = disparity_map.shape
        out_h, out_w = out_shape[0:2]

        print(f"DepthEstimatorCrestereoPytorch: Disparity map shape: {disparity_map.shape}")
        if in_h != out_h or in_w != out_w:
            # got this formula from the original testing code
            t = float(in_w) / float(out_w)
            disparity_map = (
                cv2.resize(disparity_map, (in_w, in_h), interpolation=cv2.INTER_AREA) * t
            )
        self.disparity_map = disparity_map

        if False:
            disp_min = disparity_map.min()
            disp_max = disparity_map.max()
            print(f"DepthEstimatorCrestereoPytorch: disp_min: {disp_min}, disp_max: {disp_max}")

        if False:
            disp_vis = img_from_depth(disparity_map)
            cv2.imshow("disparity_map", disp_vis)

        bf = self.camera.bf if self.camera is not None else 1.0
        if self.camera is None:
            Printer.red("Camera is None!")

        # Compute depth map
        abs_disparity_map = np.abs(disparity_map, dtype=float)
        depth_map = np.where(abs_disparity_map > self.min_disparity, bf / abs_disparity_map, 0.0)
        self.depth_map = depth_map

        print(f"DepthEstimatorCrestereoPytorch: Depth map shape: {depth_map.shape}")
        return depth_map, None
