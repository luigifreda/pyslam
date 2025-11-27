"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* The Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.

Learning rate schedule functions for scene optimization.
"""

import numpy as np


class LearningRateSchedules:

    @staticmethod
    def adjust_learning_rate_by_lr(optimizer, lr):
        for param_group in optimizer.param_groups:
            if "lr_scale" in param_group:
                param_group["lr"] = lr * param_group["lr_scale"]
            else:
                param_group["lr"] = lr

    @staticmethod
    def cosine_schedule(alpha_or_t, lr_base_or_start, lr_end=0):
        """
        Cosine learning rate schedule.

        Args:
            alpha_or_t: Progress value in [0, 1] (0 = start, 1 = end)
            lr_base_or_start: Starting/base learning rate
            lr_end: Ending learning rate (default: 0)

        Returns:
            Learning rate at the given progress value
        """
        return lr_end + (lr_base_or_start - lr_end) * (1 + np.cos(alpha_or_t * np.pi)) / 2

    @staticmethod
    def linear_schedule(alpha_or_t, lr_base_or_start, lr_end=0):
        """
        Linear learning rate schedule.

        Args:
            alpha_or_t: Progress value in [0, 1] (0 = start, 1 = end)
            lr_base_or_start: Starting/base learning rate
            lr_end: Ending learning rate (default: 0)

        Returns:
            Learning rate at the given progress value
        """
        return (1 - alpha_or_t) * lr_base_or_start + alpha_or_t * lr_end
