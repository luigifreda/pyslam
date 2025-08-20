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
from enum import Enum
import numpy as np
import cv2

from .feature_base import BaseFeature2D


# https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
# adapated from https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
class RootSIFTFeature2D(BaseFeature2D):
    def __init__(self, feature):
        # initialize the SIFT feature detector
        self.feature = feature

    def detect(self, frame, mask=None):
        return self.feature.detect(frame, mask)

    def transform_descriptors(self, des, eps=1e-7):
        # apply the Hellinger kernel by first L1-normalizing and
        # taking the square-root
        des /= des.sum(axis=1, keepdims=True) + eps
        des = np.sqrt(des)
        return des

    def compute(self, frame, kps, eps=1e-7):
        # compute SIFT descriptors
        (kps, des) = self.feature.compute(frame, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and
        # taking the square-root
        des = self.transform_descriptors(des)

        # return a tuple of the keypoints and descriptors
        return (kps, des)

    # detect keypoints and their descriptors
    # out: kps, des
    def detectAndCompute(self, frame, mask=None):
        # compute SIFT keypoints and descriptors
        (kps, des) = self.feature.detectAndCompute(frame, mask)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and
        # taking the square-root
        des = self.transform_descriptors(des)

        # return a tuple of the keypoints and descriptors
        return (kps, des)
