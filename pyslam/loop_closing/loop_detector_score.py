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
import time
import math
import numpy as np
import cv2
import sys
from enum import Enum

from pyslam.utilities.logging import Printer

from typing import List

from pyslam.config_parameters import Parameters
import torch


kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = kRootFolder + "/data"


class SCoreType(Enum):
    COSINE = 0
    SAD = 1


# Base class
class ScoreBase:
    def __init__(self, type, worst_score, best_score):
        self.type = type
        self.worst_score = worst_score
        self.best_score = best_score

    # g_des1 is [1, D], g_des2 is [M, D]
    def __call__(self, g_des1, g_des2):
        pass


class ScoreSad(ScoreBase):
    def __init__(self):
        super().__init__(SCoreType.SAD, worst_score=-sys.float_info.max, best_score=0.0)

    @staticmethod
    def score(g_des1, g_des2):
        diff = g_des1 - g_des2
        is_nan_diff = np.isnan(diff)
        nan_count_per_row = np.count_nonzero(is_nan_diff, axis=1)
        dim = diff.shape[1] - nan_count_per_row
        # print(f'dim: {dim}, diff.shape: {diff.shape}')
        diff[is_nan_diff] = 0
        return -np.sum(np.abs(diff), axis=1) / dim  # invert the sign of the standard SAD score

    # g_des1 is [1, D], g_des2 is [M, D]
    def __call__(self, g_des1, g_des2):
        return self.score(g_des1, g_des2)


class ScoreCosine(ScoreBase):
    def __init__(self):
        super().__init__(SCoreType.COSINE, worst_score=-1.0, best_score=1.0)

    @staticmethod
    def score(g_des1, g_des2):
        norm_g_des1 = np.linalg.norm(
            g_des1, axis=1, keepdims=True
        )  # g_des1 is [1, D], so norm is scalar
        norm_g_des2 = np.linalg.norm(g_des2, axis=1, keepdims=True)  # g_des2 is [M, D]
        dot_product = np.dot(g_des2, g_des1.T).ravel()
        cosine_similarity = dot_product / (norm_g_des1 * norm_g_des2.ravel())
        return cosine_similarity.ravel()

    # g_des1 is [1, D], g_des2 is [M, D]
    def __call__(self, g_des1, g_des2):
        return self.score(g_des1, g_des2)


class ScoreTorchCosine(ScoreBase):
    def __init__(self):
        super().__init__(SCoreType.COSINE, worst_score=-1.0, best_score=1.0)

    @staticmethod
    def score(g_des1, g_des2):
        # Ensure g_des1 is a 2D tensor of shape [1, D]
        if g_des1.dim() == 1:
            g_des1 = g_des1.unsqueeze(0)

        # Compute the norms
        norm_g_des1 = g_des1.norm(dim=1, keepdim=True)  # Shape [1, 1]
        norm_g_des2 = g_des2.norm(dim=1, keepdim=True)  # Shape [M, 1]

        # Dot product between g_des1 and each row of g_des2
        dot_product = torch.mm(g_des2, g_des1.T).squeeze()  # Shape [M]

        # Compute cosine similarity
        cosine_similarity = (dot_product / (norm_g_des1 * norm_g_des2).squeeze()).ravel()
        return cosine_similarity

    # g_des1 is [1, D], g_des2 is [M, D]
    def __call__(self, g_des1, g_des2):
        return self.score(g_des1, g_des2)
