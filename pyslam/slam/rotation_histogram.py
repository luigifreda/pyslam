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
import numpy as np


class RotationHistogram(object):
    def __init__(
        self, histogram_length=12
    ):  # NOTE: with 12 bins => new factor = 12/360 equals to old factor = 1/30
        self.histogram_length = histogram_length
        self.factor = float(histogram_length) / 360.0  # 1.0/histogram_length
        self.histo = [[] for i in range(self.histogram_length)]

    def push(self, rot, idx):
        rot = rot % 360.0  # get a 0-360 range
        bin = int(round(rot * self.factor))
        if bin == self.histogram_length:
            bin = 0
        assert bin >= 0 and bin < self.histogram_length
        self.histo[bin].append(idx)

    def push_entries(self, rots, idxs):
        rot_array = np.mod(rots, 360.0)
        bins = np.round(rot_array * self.factor).astype(int)
        bins[bins == self.histogram_length] = 0
        # assert np.all((bins >= 0) & (bins < self.histogram_length)), "Invalid bin index"
        if not np.all((bins >= 0) & (bins < self.histogram_length)):
            raise ValueError(f"RotationHistogram: Invalid bin index in push_entries()")
        for bin_idx, idx in zip(bins, idxs):
            self.histo[bin_idx].append(idx)

    def compute_3_max(self):
        counts = np.array([len(bin) for bin in self.histo])
        indices = np.argsort(counts)[::-1]
        max1, max2, max3 = indices[:3]
        if counts[max2] < 0.1 * counts[max1]:
            max2 = -1
        if counts[max3] < 0.1 * counts[max1]:
            max3 = -1
        return max1, max2, max3

    def get_invalid_idxs(self):
        ind1, ind2, ind3 = self.compute_3_max()
        invalid_idxs = []
        for i in range(self.histogram_length):
            if i != ind1 and i != ind2 and i != ind3:
                invalid_idxs.extend(self.histo[i])
        return invalid_idxs

    def get_valid_idxs(self):
        ind1, ind2, ind3 = self.compute_3_max()
        valid_idxs = []
        if ind1 != -1:
            valid_idxs.extend(self.histo[ind1])
        if ind2 != -1:
            valid_idxs.extend(self.histo[ind2])
        if ind3 != -1:
            valid_idxs.extend(self.histo[ind3])
        return valid_idxs

    def __str__(self):
        # return str(self.__class__) + ": " + str(self.__dict__)
        return "RotationHistogram " + str(self.histo)

    @staticmethod
    def filter_matches_with_histogram_orientation(idxs1, idxs2, angles1, angles2):
        if len(idxs1) == 0 or len(idxs2) == 0:
            return []
        assert len(idxs1) == len(idxs2)
        num_matches = len(idxs1)
        rot_histo = RotationHistogram()
        rots = angles1[idxs1] - angles2[idxs2]  # assuming input degs
        rot_histo.push_entries(rots, [ii for ii in range(num_matches)])
        valid_match_idxs = rot_histo.get_valid_idxs()
        return valid_match_idxs
