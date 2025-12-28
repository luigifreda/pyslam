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
import torch
from threading import RLock

from pyslam.utilities.logging import Printer
from pyslam.utilities.system import import_from, is_opencv_version_greater_equal
from .feature_base import BaseFeature2D

import pyslam.config as config

config.cfg.set_lib("lightglue")

SIFT = import_from("lightglue", "SIFT")

kVerbose = True


def numpy_image_to_torch(image: np.ndarray) -> torch.Tensor:
    """Normalize the image tensor and reorder the dimensions."""
    if image.ndim == 3:
        image = image.transpose((2, 0, 1))  # HxWxC to CxHxW
    elif image.ndim == 2:
        image = image[None]  # add channel axis
    else:
        raise ValueError(f"Not an image: {image.shape}")
    return torch.tensor(image / 255.0, dtype=torch.float)


def convert_pts_to_keypoints(pts, scales, oris):
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        kps = [
            cv2.KeyPoint(p[0], p[1], size=s, angle=o, response=1.0, octave=0)
            for p, s, o in zip(pts, scales, oris)
        ]
    return kps


# Interface for pySLAM
class LightGlueSIFTFeature2D(BaseFeature2D):
    def __init__(self, num_features=2000):
        print("Using LightGlueSIFTFeature2D")
        self.num_features = num_features
        self.config = SIFT.default_conf.copy()
        self.config["max_num_keypoints"] = self.num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        self.SIFT = SIFT(conf=self.config)

    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.num_features = num_features
        self.config["max_num_keypoints"] = self.num_features
        self.SIFT = SIFT(conf=self.config)

    def extract(self, image):
        tensor = numpy_image_to_torch(image)
        feats = self.SIFT.extract(tensor.to(self.device))
        # print(f'feats: {feats}')
        kps = feats["keypoints"].cpu().numpy()[0]
        des = feats["descriptors"].cpu().numpy()[0]
        scales = feats["scales"].cpu().numpy()[0]
        oris = feats["oris"].cpu().numpy()[0]
        # print(f'kps: {kps}')
        # print(f'des: {des}')
        # print(f'scales: {scales}')
        # print(f'oris: {oris}')
        return kps, des, scales, oris

    # extract keypoints
    def detect(
        self, img, mask=None
    ):  # mask is fake: it is not considered by the c++ implementation
        # detect and compute
        kps, des, scales, oris = self.extract(img)
        return kps

    def compute(self, img, kps, mask=None):
        Printer.orange(
            "WARNING: you are supposed to call detectAndCompute() for LIGHTGLUESIFT instead of compute()"
        )
        Printer.orange(
            "WARNING: LIGHTGLUESIFT is recomputing both kps and des on input frame", img.shape
        )
        kps, des, scales, oris = self.extract(img)
        return des

    # compute both keypoints and descriptors
    def detectAndCompute(
        self, img, mask=None
    ):  # mask is fake: it is not considered by the c++ implementation
        # detect and compute
        kps, des, scales, oris = self.extract(img)
        kps = convert_pts_to_keypoints(kps, scales, oris)
        return kps, des
