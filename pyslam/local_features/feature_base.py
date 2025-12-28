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

import numpy as np
import torch

import cv2
from threading import RLock

from pyslam.utilities.logging import Printer
from pyslam.utilities.system import import_from


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, sizes, levels):
    assert len(pts) == len(scores)
    assert len(scores) == len(sizes)
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        kps = [
            cv2.KeyPoint(p[0], p[1], size=sizes[i], response=scores[i], octave=levels[i])
            for i, p in enumerate(pts)
        ]
    return kps


# Base class for a Feature2D interface in pySLAM
class BaseFeature2D:
    def __init__(self, num_features=None, device=None):
        self.num_features = num_features
        self.device = device

    # Set the maximum number of features
    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.num_features = num_features

    # Detect keypoints and compute their descriptors
    def detectAndCompute(self, frame, mask=None):
        kps = None
        des = None
        raise NotImplementedError
        return kps, des

    # Detect keypoints
    def detect(self, frame, mask=None):
        kps = None
        raise NotImplementedError
        return kps

    # Compute descriptors for the given input keypoints
    def compute(self, frame, kps=None, mask=None):
        des = None
        raise NotImplementedError
        return kps, des
