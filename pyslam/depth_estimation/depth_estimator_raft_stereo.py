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

config.cfg.set_lib("raft_stereo")


from pyslam.slam import Camera
from pyslam.io.dataset_types import DatasetEnvironmentType
from pyslam.utilities.serialization import SerializableEnum, register_class
from pyslam.utilities.depth import img_from_depth
from pyslam.utilities.system import Printer, set_rlimit

from .depth_estimator_base import DepthEstimator

import torch

from raft_stereo import RAFTStereo
from core.utils.utils import InputPadder


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class DepthEstimatorRaftStereoConfiguration:
    def __init__(self):
        # Architecture choices
        # parser.add_argument('--hidden_dims', nargs='+', type=int, default=[128]*3, help="hidden state and context dimensions")
        # parser.add_argument('--corr_implementation', choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg", help="correlation volume implementation")
        # parser.add_argument('--shared_backbone', action='store_true', help="use a single backbone for the context and feature encoders")
        # parser.add_argument('--corr_levels', type=int, default=4, help="number of levels in the correlation pyramid")
        # parser.add_argument('--corr_radius', type=int, default=4, help="width of the correlation pyramid")
        # parser.add_argument('--n_downsample', type=int, default=2, help="resolution of the disparity field (1/2^K)")
        # parser.add_argument('--context_norm', type=str, default="batch", choices=['group', 'batch', 'instance', 'none'], help="normalization of context encoder")
        # parser.add_argument('--slow_fast_gru', action='store_true', help="iterate the low-res GRUs more frequently")
        # parser.add_argument('--n_gru_layers', type=int, default=3, help="number of hidden GRU levels")
        # parser.add_argument('--mixed_precision', action='store_true', help='use mixed precision')
        # parser.add_argument('--valid_iters', type=int, default=32, help='number of flow-field updates during forward pass')

        self.hidden_dims = [128] * 3  # hidden state and context dimensions
        self.corr_implementation = "reg"  # correlation volume implementation, choices=["reg", "alt", "reg_cuda", "alt_cuda"], default="reg",
        self.shared_backbone = False  # use a single backbone for the context and feature encoders
        self.corr_levels = 4  # number of levels in the correlation pyramid
        self.corr_radius = 4  # width of the correlation pyramid
        self.n_downsample = 2  # resolution of the disparity field (1/2^K)
        self.context_norm = "batch"  # normalization of context encoder, choices=['group', 'batch', 'instance', 'none'],
        self.slow_fast_gru = False  # iterate the low-res GRUs more frequently
        self.n_gru_layers = 3  # number of hidden GRU levels
        self.mixed_precision = False  # use mixed precision
        self.valid_iters = 32  # number of flow-field updates during forward pass


# Stereo depth prediction using the Stereo Raft model.
class DepthEstimatorRaftStereo(DepthEstimator):
    kStereoRaftBasePath = kRootFolder + "/thirdparty/raft_stereo"
    kEth3dModelPath = kStereoRaftBasePath + "/models/raftstereo-eth3d.pth"
    kRealtimeModelPath = kStereoRaftBasePath + "/models/raftstereo-realtime.pth"
    kMiddleburyModelPath = kStereoRaftBasePath + "/models/raftstereo-middlebury.pth"

    def __init__(
        self,
        device=None,
        camera: Camera = None,
        min_depth=0,
        max_depth=50,
        dataset_env_type=DatasetEnvironmentType.OUTDOOR,
        model_name="realtime",
    ):
        self.model_config = DepthEstimatorRaftStereoConfiguration()
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type == "cuda":
            print("DepthEstimatorRaftStereo: Using CUDA")
        else:
            print("DepthEstimatorRaftStereo: Using CPU")
        transform = None

        if model_name == "eth3d":
            restore_ckpt = DepthEstimatorRaftStereo.kEth3dModelPath
        elif model_name == "realtime":
            restore_ckpt = (
                DepthEstimatorRaftStereo.kRealtimeModelPath
            )  # --shared_backbone --n_downsample 3 --n_gru_layers 2 --slow_fast_gru --valid_iters 7 --corr_implementation reg_cuda --mixed_precision
            self.model_config.shared_backbone = True
            self.model_config.n_downsample = 3
            self.model_config.n_gru_layers = 2
            self.model_config.slow_fast_gru = True
            self.model_config.valid_iters = 7
            # self.model_config.corr_implementation = 'reg_cuda'
            self.model_config.mixed_precision = True
        elif model_name == "middlebury":
            restore_ckpt = DepthEstimatorRaftStereo.kMiddleburyModelPath
        else:
            raise ValueError(f"Unknown model name: {model_name}")

        if os.path.exists(restore_ckpt):
            print(f"Loading model from {restore_ckpt}")
        else:
            raise ValueError(f"Model {restore_ckpt} does not exist")

        self.model = torch.nn.DataParallel(RAFTStereo(self.model_config), device_ids=[0])
        self.model.load_state_dict(torch.load(restore_ckpt, map_location="cpu"))
        self.module = self.model.module
        self.module.to(device)
        self.module.eval()

        super().__init__(
            model=self.model,
            transform=transform,
            device=device,
            camera=camera,
            min_depth=min_depth,
            max_depth=max_depth,
            dataset_env_type=dataset_env_type,
        )

    # Return the predicted depth map and the point cloud (if any)
    def infer(self, image, image_right=None):
        if image_right is None:
            message = "Image right is None. Are you using a stereo dataset? If not, you cant use a stereo depth estimator here."
            Printer.red(message)
            raise ValueError(message)
        with torch.no_grad():
            # Compute disparity map
            image1 = torch.from_numpy(image).permute(2, 0, 1).float()
            image1 = image1[None].to(self.device)
            image2 = torch.from_numpy(image_right).permute(2, 0, 1).float()
            image2 = image2[None].to(self.device)

            padder = InputPadder(image1.shape, divis_by=32)
            image1, image2 = padder.pad(image1, image2)

            _, flow_up = self.module(
                image1, image2, iters=self.model_config.valid_iters, test_mode=True
            )
            flow_up = padder.unpad(flow_up).squeeze()
            disparity_map = flow_up.cpu().numpy().squeeze()
            self.disparity_map = disparity_map

            if False:
                disparity_map_img = img_from_depth(disparity_map)
                cv2.imshow("disparity_map", disparity_map_img)

            bf = self.camera.bf if self.camera is not None else 1.0
            if self.camera is None:
                Printer.red("Camera is None!")

            # Compute depth map
            abs_disparity_map = np.abs(disparity_map)
            valid_mask = abs_disparity_map > 0
            depth_map = np.zeros_like(disparity_map, dtype=disparity_map.dtype)
            depth_map[valid_mask] = bf / abs_disparity_map[valid_mask]
            self.depth_map = depth_map
            return depth_map, None
