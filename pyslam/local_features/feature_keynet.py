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

import pyslam.config as config
import cv2
import numpy as np
import math

from pyslam.utilities.logging import Printer

import kornia as K
import kornia.feature as KF
import numpy as np
import torch
from kornia_moons.feature import *
from kornia_moons.viz import *

from typing_extensions import TypedDict
import matplotlib.pyplot as plt

from .feature_base import BaseFeature2D


kVerbose = True


class Detector_config(TypedDict):
    nms_size: int
    pyramid_levels: int
    up_levels: int
    scale_factor_levels: float
    s_mult: float


class KeyNet_conf(TypedDict):
    num_filters: int
    num_levels: int
    kernel_size: int
    Detector_conf: Detector_config


def get_default_detector_config() -> Detector_config:
    """Return default config."""
    return {
        # Extraction Parameters
        "nms_size": 15,
        "pyramid_levels": 4,
        "up_levels": 1,
        "scale_factor_levels": math.sqrt(2),
        "s_mult": 22.0,
    }


keynet_default_config: KeyNet_conf = {
    # Key.Net Model
    "num_filters": 8,
    "num_levels": 3,
    "kernel_size": 5,
    # Extraction Parameters
    "Detector_conf": get_default_detector_config(),
}


class KeyNetHardNet(KF.LocalFeature):
    """Convenience module, which implements KeyNet detector + HardNet descriptor."""

    def __init__(
        self,
        num_features: int = 8000,
        upright: bool = False,
        device=None,
        scale_laf: float = 1.0,
        patch_size=32,
        keynet_conf: KeyNet_conf = keynet_default_config,
    ) -> None:
        if device is None:
            device = torch.device("cpu")
        ori_module = KF.PassLAF() if upright else KF.LAFOrienter(angle_detector=KF.OriNet(True))
        detector = KF.KeyNetDetector(
            True, num_features=num_features, ori_module=ori_module, keynet_conf=keynet_conf
        ).to(device)
        descriptor = KF.LAFDescriptor(None, patch_size=32, grayscale_descriptor=True).to(device)
        super().__init__(detector, descriptor, scale_laf)


# Interface for pySLAM
class KeyNetDescFeature2D(BaseFeature2D):
    def __init__(self, num_features=2000, device=K.utils.get_cuda_or_mps_device_if_available()):
        print("Using KeyNetDescFeature2D")
        self.device = device
        self.num_features = num_features
        self.keynet_conf = keynet_default_config
        self.feature = (
            KeyNetHardNet(num_features, True, keynet_conf=keynet_default_config)
            .eval()
            .to(self.device)
        )

    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.num_features = num_features
        try:
            self.feature.detector.num_features = num_features
        except:
            Printer.red("[KeyNetDescFeature2D] Error setting num_features")

    @property
    def num_levels(self):
        try:
            return self.keynet_config["Detector_conf"]["pyramid_levels"]
        except:
            return 1

    @property
    def scale_factor(self):
        try:
            return self.keynet_config["Detector_conf"]["scale_factor_levels"]
        except:
            return 1

    def convert_to_keypoints_array(self, lafs):
        mkpts = KF.get_laf_center(lafs).squeeze().detach().cpu().numpy()
        return mkpts

    def convert_to_keypoints(self, lafs, resps, size=32):
        mkpts = self.convert_to_keypoints_array(lafs)
        # convert matrix [Nx2] of pts into list of keypoints
        kps = [
            cv2.KeyPoint(int(p[0]), int(p[1]), size=size, response=resps[i])
            for i, p in enumerate(mkpts)
        ]
        return kps

    def draw_matches(self, img1, img2, lafs1, lafs2, idxs, inliers):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        draw_LAF_matches(
            lafs1.cpu(),
            lafs2.cpu(),
            idxs.cpu(),
            K.tensor_to_image(img1.cpu()),
            K.tensor_to_image(img2.cpu()),
            inliers,
            draw_dict={
                "inlier_color": (0.2, 1, 0.2),
                "tentative_color": (1, 1, 0.2, 0.3),
                "feature_color": None,
                "vertical": False,
            },
            ax=ax,
        )

    # extract keypoints
    def detect(self, img, mask=None):  # mask is fake: it is not considered by the implementation
        kps = None
        with torch.inference_mode():
            if img.ndim > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = K.image_to_tensor(img, False).to(self.device).float() / 255.0
            lafs, resps = self.feature.detector(img)
            print(f"resps: {resps}")
            kps = self.convert_to_keypoints(lafs, resps.squeeze().detach().cpu().numpy())
        return kps

    def compute(self, img, kps, mask=None):
        Printer.orange(
            "WARNING: you are supposed to call detectAndCompute() for KeyNetDescFeature2D instead of compute()"
        )
        Printer.orange(
            "WARNING: KeyNetDescFeature2D is recomputing both kps and des on input frame", img.shape
        )
        return self.detectAndCompute(img)

    # compute both keypoints and descriptors
    def detectAndCompute(
        self, img, mask=None
    ):  # mask is fake: it is not considered by the c++ implementation
        # detect and compute
        kps = None
        des = None
        with torch.inference_mode():
            if img.ndim > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            img = K.image_to_tensor(img, False).to(self.device).float() / 255.0
            lafs, resps, des = self.feature(img)
            kps = self.convert_to_keypoints(lafs, resps.squeeze().detach().cpu().numpy())
            des = des.cpu().numpy()
            des = np.squeeze(des, axis=0)
            # print(f'des shape: {des.shape}, des type: {des.dtype}')
        return kps, des

    # compute both keypoints and descriptors
    def detectAndComputeWithTensors(
        self, img, mask=None
    ):  # mask is fake: it is not considered by the c++ implementation
        # detect and compute
        kps = None
        # a starting 't' means the variable is a tensor
        tdes = None
        tresps = None
        lafs = None
        with torch.inference_mode():
            if img.ndim > 2:
                img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
            timg = K.image_to_tensor(img, False).to(self.device).float() / 255.0
            lafs, tresps, tdes = self.feature(timg)
            kps = self.convert_to_keypoints(lafs)
        return kps, lafs, tresps, tdes
