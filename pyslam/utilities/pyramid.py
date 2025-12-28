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

import math
from enum import Enum

import numpy as np
import cv2

from .img_management import img_blocks
from .logging import Printer


kVerbose = True

kNumLevelsInitSigma = 20


# pyramid types
class PyramidType(Enum):
    RESIZE = 0  # only resize, do NOT filter (N.B.: filters are typically applied for obtaining a useful antialiasing effect)
    # both Pyramid.imgs and Pyramid.imgs_filtered contain unfiltered resized images
    RESIZE_AND_FILTER = 1  # compute separated resized images and filtered images: first resize then filter (typically used by ORB)
    # Pyramid.imgs contains (unfiltered) resized images, and Pyramid.imgs_filtered contain filtered resized images
    GAUSS_PYRAMID = 2  # compute images in the scale-space: first filter (with appropriate sigmas) than resize, see  https://www.vlfeat.org/api/sift.html#sift-tech-ss  (used by SIFT, SURF, etc...)
    # both Pyramid.imgs and Pyramid.imgs_filtered contain filtered images in the scale space


# PyramidAdaptor generate a pyramid of num_levels images and extracts features in each of these images
class Pyramid:
    def __init__(
        self,
        num_levels=4,
        scale_factor=1.2,
        sigma0=1.0,  # N.B.: SIFT use 1.6 for this value
        first_level=0,  # 0: start from input image; -1: start from up-scaled image*scale_factor
        pyramid_type=PyramidType.RESIZE,
    ):
        self.num_levels = num_levels
        self.scale_factor = scale_factor
        self.sigma0 = sigma0
        self.first_level = first_level
        self.pyramid_type = pyramid_type

        self.imgs = []
        self.imgs_filtered = []
        self.base_img = None

        self.scale_factors = None
        self.inv_scale_factors = None
        self.initSigmaLevels()

    def initSigmaLevels(self):
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        self.scale_factors = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.scale_factors[0] = 1.0
        self.inv_scale_factors[0] = 1.0 / self.scale_factors[0]
        for i in range(1, num_levels):
            self.scale_factors[i] = self.scale_factors[i - 1] * self.scale_factor
            self.inv_scale_factors[i] = 1.0 / self.scale_factors[i]
        # print('self.inv_scale_factors: ', self.inv_scale_factors)

    def compute(self, frame):
        # print(f'Pyramid: compute, first_level: {self.first_level}, pyramid_type: {self.pyramid_type}')
        if self.first_level == -1:
            frame = self.createBaseImg(
                frame
            )  # replace the image with the new level -1 (up-resized image)
        if self.pyramid_type == PyramidType.RESIZE:
            self.computeResize(frame)
        elif self.pyramid_type == PyramidType.RESIZE_AND_FILTER:
            self.computeResizeAndFilter(frame)
        elif self.pyramid_type == PyramidType.GAUSS_PYRAMID:
            self.computeGauss(frame)
        else:
            Printer.orange("Pyramid - unknown type")
            self.computeResize(frame)
        # print(f'Pyramid: compute done')

    def createBaseImg(self, frame):
        sigma_init = (
            0.5  # 0.5 is the base sigma from https://www.vlfeat.org/api/sift.html#sift-tech-ss
        )
        delta_sigma = math.sqrt(
            max(
                self.sigma0 * self.sigma0
                - (sigma_init * sigma_init * self.scale_factor * self.scale_factor),
                0.01,
            )
        )  # see https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L197
        frame_upscaled = cv2.resize(frame, (0, 0), fx=self.scale_factor, fy=self.scale_factor)
        if self.pyramid_type == PyramidType.GAUSS_PYRAMID:
            return cv2.GaussianBlur(frame_upscaled, ksize=(0, 0), sigmaX=delta_sigma)
        else:
            return frame_upscaled

    # only resize, do not filter
    def computeResize(self, frame):
        # print(f'Pyramid: computeResize, num_levels: {self.num_levels}, scale_factor: {self.scale_factor},frame shape: {frame.shape}')
        inv_scale = 1.0 / self.scale_factor
        self.imgs = []
        self.imgs_filtered = []
        pyr_cur = frame
        for i in range(0, self.num_levels):
            self.imgs.append(pyr_cur)
            self.imgs_filtered.append(pyr_cur)
            if i < self.num_levels - 1:
                pyr_down = cv2.resize(
                    pyr_cur, (0, 0), fx=inv_scale, fy=inv_scale
                )  # resize the unfiltered frame
                pyr_cur = pyr_down
                # print(f'Pyramid: computeResize, level: {i}, pyr_cur shape: {pyr_cur.shape}')
        # print(f'Pyramid: computeResize done')

    # keep separated resized images and filtered images: first resize than filter with constant sigma
    def computeResizeAndFilter(self, frame):
        inv_scale = 1.0 / self.scale_factor
        filter_sigmaX = 2  # setting used for computing ORB descriptors
        ksize = (5, 5)
        self.imgs = []
        self.imgs_filtered = []
        pyr_cur = frame
        for i in range(0, self.num_levels):
            filtered = cv2.GaussianBlur(pyr_cur, ksize, sigmaX=filter_sigmaX)
            self.imgs.append(pyr_cur)  # self.imgs contain resized image
            self.imgs_filtered.append(filtered)  # self.imgs_filtered contain filtered images
            if i < self.num_levels - 1:
                pyr_down = cv2.resize(
                    pyr_cur, (0, 0), fx=inv_scale, fy=inv_scale
                )  # resize the unfiltered frame
                pyr_cur = pyr_down

    # compute images in the scale space: first filter (with appropriate sigmas) than resize
    def computeGauss(self, frame):
        inv_scale = 1.0 / self.scale_factor

        # from https://github.com/opencv/opencv/blob/173442bb2ecd527f1884d96d7327bff293f0c65a/modules/nonfree/src/sift.cpp#L212
        # \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
        sigma_nominal = 0.5  # original image with nominal sigma=0.5  <=from https://www.vlfeat.org/api/sift.html#sift-tech-ss
        sigma0 = self.sigma0  # N.B.: SIFT use 1.6 for this value
        sigma_prev = sigma_nominal

        self.imgs = []
        self.imgs_filtered = []

        pyr_cur = frame

        for i in range(0, self.num_levels):
            if i == 0 and self.first_level == -1:
                sigma_prev = sigma0
                filtered = frame
            else:
                sigma_total = self.scale_factors[i] * sigma0
                sigma_cur = math.sqrt(
                    sigma_total * sigma_total - sigma_prev * sigma_prev
                )  # this the DELTA-SIGMA according to \sigma_{total}^2 = \sigma_{i}^2 + \sigma_{i-1}^2
                sigma_prev = sigma_cur
                filtered = cv2.GaussianBlur(pyr_cur, ksize=(0, 0), sigmaX=sigma_cur)

            # both self.imgs and self.imgs_filtered contain filtered images in the scale space
            self.imgs.append(filtered)
            self.imgs_filtered.append(filtered)

            if i < self.num_levels - 1:
                pyr_down = cv2.resize(
                    filtered, (0, 0), fx=inv_scale, fy=inv_scale
                )  # resize the filtered frame
                pyr_cur = pyr_down
