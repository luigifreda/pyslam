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
import os
import cv2
import numpy as np

from pyslam.utilities.logging import Printer
from .feature_base import BaseFeature2D
from orbslam2_features import ORBextractor, ORBextractorDeterministic


kVerbose = True


# Interface for pySLAM
class Orbslam2Feature2D(BaseFeature2D):
    def __init__(self, num_features=2000, scale_factor=1.2, num_levels=8, deterministic=False):
        print("Using Orbslam2Feature2D")
        if deterministic:
            self.orb_extractor = ORBextractorDeterministic(num_features, scale_factor, num_levels)
        else:
            self.orb_extractor = ORBextractor(num_features, scale_factor, num_levels)

    # extract keypoints
    def detect(
        self, img, mask=None
    ):  # mask is fake: it is not considered by the c++ implementation
        # detect and compute
        kps_tuples = self.orb_extractor.detect(img)
        # convert keypoints
        kps = [cv2.KeyPoint(*kp) for kp in kps_tuples]
        return kps

    def compute(self, img, kps, mask=None):
        Printer.orange(
            "WARNING: you are supposed to call detectAndCompute() for ORB2 instead of compute()"
        )
        Printer.orange("WARNING: ORB2 is recomputing both kps and des on input frame", img.shape)
        return self.detectAndCompute(img)
        # return kps, np.array([])

    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.orb_extractor.SetNumFeatures(
            num_features
        )  # custom method name for OrbSlam2 python wrapper

    # compute both keypoints and descriptors
    def detectAndCompute(
        self, img, mask=None
    ):  # mask is fake: it is not considered by the c++ implementation
        # detect and compute
        kps_tuples, des = self.orb_extractor.detectAndCompute(img)
        # convert keypoints
        kps = [cv2.KeyPoint(*kp) for kp in kps_tuples]
        return kps, des
