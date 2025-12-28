"""
* This file is part of PYSLAM
* Adapted from https://github.com/cvlab-epfl/disk/blob/master/detect.py, see licence therein.
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

import sys
import pyslam.config as config

config.cfg.set_lib("lightglue")

import numpy as np
import torch

import cv2
from threading import RLock

from pyslam.utilities.logging import Printer
from pyslam.utilities.system import import_from, is_opencv_version_greater_equal

from .feature_base import BaseFeature2D


ALIKED = import_from("lightglue", "ALIKED")

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


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, size):
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        if is_opencv_version_greater_equal(4, 5, 3):
            kps = [cv2.KeyPoint(p[0], p[1], size=size, response=1.0, octave=0) for p in pts]
        else:
            kps = [cv2.KeyPoint(p[0], p[1], _size=size, _response=1.0, _octave=0) for p in pts]
    return kps


# Interface for pySLAM
# NOTE: from Fig. 3 in the paper "ALIKED: Learning local features with policy gradient"
# "Our approach can match many more points and produce more accurate poses. It can deal with large changes in scale (4th and 5th columns) but not in rotation..."
class AlikedFeature2D(BaseFeature2D):
    def __init__(self, num_features=2000):
        self.lock = RLock()
        self.num_features = num_features
        config = ALIKED.default_conf.copy()
        config["max_num_keypoints"] = self.num_features
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # 'mps', 'cpu'
        self.ALIKED = ALIKED(**config).eval().to(self.device)

    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.num_features = num_features

    def extract(self, image):
        tensor = numpy_image_to_torch(image)
        feats = self.ALIKED.extract(tensor.to(self.device))
        # print(f'feats: {feats}')
        kps = feats["keypoints"].cpu().numpy()[0]
        des = feats["descriptors"].cpu().numpy()[0]
        return kps, des

    def compute_kps_des(self, im):
        with self.lock:
            keypoints, descriptors = self.extract(im)
            # print('keypoints: ', keypoints)
            self.kps = convert_pts_to_keypoints(keypoints, size=1)
            return self.kps, descriptors

    def detectAndCompute(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            self.frame = frame
            self.kps, self.des = self.compute_kps_des(frame)
            if kVerbose:
                print(
                    "detector: ALIKED, descriptor: ALIKED, #features: ",
                    len(self.kps),
                    ", frame res: ",
                    frame.shape[0:2],
                )
            return self.kps, self.des

    # return keypoints if available otherwise call detectAndCompute()
    def detect(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            # if self.frame is not frame:
            self.detectAndCompute(frame)
            return self.kps

    # return descriptors if available otherwise call detectAndCompute()
    def compute(self, frame, kps=None, mask=None):  # kps is a fake input, mask is a fake input
        with self.lock:
            if self.frame is not frame:
                # Printer.orange('WARNING: ALIKED is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, self.des
