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

import os
import numpy as np
import cv2
import math


# A class for computing a moving average (mean value) with a given window size.
# Optionally, you can also compute standard devitation of the signal and standard deviation of the average
class MovingAverage:
    def __init__(self, average_width=10, compute_sigma=False):
        self._average_width = average_width
        self._idx_ring = 0
        self._average = 0
        self._sigma2 = 0
        self._is_init = False
        self._is_compute_sigma = compute_sigma
        self._one_over_average_width_min_one = 1.0 / (average_width - 1)
        self._ring_buffer = np.zeros(average_width)

    def init(self, initVal=None):
        if initVal is None:
            initVal = 0.0
        self._ring_buffer = np.full(self._average_width, initVal, dtype=float)
        self._average = initVal
        self._sigma2 = 0
        self._is_init = True

    def get_average(self, new_val=None):
        if not self._is_init:
            self.init(new_val)
        if new_val is None:
            return self._average
        averageOld = self._average
        oldVal = self._ring_buffer[self._idx_ring]
        self._average += (new_val - oldVal) / self._average_width
        if self._is_compute_sigma:
            self._sigma2 = self._sigma2 + self._one_over_average_width_min_one * (
                self._average_width * (averageOld * averageOld - self._average * self._average)
                - oldVal * oldVal
                + new_val * new_val
            )
        self._ring_buffer[self._idx_ring] = new_val
        self._idx_ring = (self._idx_ring + 1) % self._average_width
        return self._average

    def get_sigma(self):
        return math.sqrt(max(self._sigma2, 0.0))

    def get_sigma2(self):
        return self._sigma2
