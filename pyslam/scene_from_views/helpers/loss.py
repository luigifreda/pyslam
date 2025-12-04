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

Part of the code is adapted from the original code by Naver Corporation.
Original code Copyright (C) 2024-present Naver Corporation.
Licensed under CC BY-NC-SA 4.0 (non-commercial use only).

"""

"""
Loss computation functions and confidence transformations.
"""

import torch
import numpy as np


class ConfidenceUtils:
    """Confidence transformation utilities."""

    @staticmethod
    def get_conf_trf(mode):
        """Get confidence transformation function.

        Args:
            mode: Transformation mode ('log', 'sqrt', 'm1', 'id', 'none')

        Returns:
            Transformation function
        """
        if mode == "log":

            def conf_trf(x):
                return x.log()

        elif mode == "sqrt":

            def conf_trf(x):
                return x.sqrt()

        elif mode == "m1":

            def conf_trf(x):
                return x - 1

        elif mode in ("id", "none"):

            def conf_trf(x):
                return x

        else:
            raise ValueError(f"bad mode for {mode=}")
        return conf_trf


class LossFunctions:
    """Loss computation functions."""

    @staticmethod
    def l05_loss(x, y):
        """Compute L0.5 loss (square root of L2 norm).

        Args:
            x: First tensor (..., D)
            y: Second tensor (..., D)

        Returns:
            L0.5 loss per element (...,)
        """
        return torch.linalg.norm(x - y, dim=-1).sqrt()

    @staticmethod
    def l1_loss(x, y):
        """Compute L1 loss (L2 norm).

        Args:
            x: First tensor (..., D)
            y: Second tensor (..., D)

        Returns:
            L1 loss per element (...,)
        """
        return torch.linalg.norm(x - y, dim=-1)

    @staticmethod
    def gamma_loss(gamma, mul=1, offset=None, clip=np.inf):
        """Create a gamma-parameterized loss function.

        This creates a loss function that applies a gamma power to the L1 distance,
        with an optional offset to ensure smooth gradients. When gamma=1, it reduces
        to standard L1 loss.

        Args:
            gamma: Power parameter (typically 0.5-2.0)
            mul: Multiplicative factor (default: 1)
            offset: Offset value (default: None, auto-computed)
            clip: Maximum value to clip distances (default: np.inf)

        Returns:
            Loss function that takes (x, y) and returns loss values
        """
        if offset is None:
            if gamma == 1:
                return LossFunctions.l1_loss
            # d(x**p)/dx = 1 ==> p * x**(p-1) == 1 ==> x = (1/p)**(1/(p-1))
            offset = (1 / gamma) ** (1 / (gamma - 1))

        def loss_func(x, y):
            return (
                mul * LossFunctions.l1_loss(x, y).clip(max=clip) + offset
            ) ** gamma - offset**gamma

        return loss_func

    @staticmethod
    def meta_gamma_loss():
        """Create a meta loss function that takes alpha as a parameter.

        Returns:
            Function that takes alpha and returns a gamma_loss function
        """
        return lambda alpha: LossFunctions.gamma_loss(alpha)

    @staticmethod
    def l2_dist(a, b, weight):
        """Compute L2 distance between two point sets with weights.

        Args:
            a: First point set (..., D)
            b: Second point set (..., D)
            weight: Weight tensor (...,)

        Returns:
            Weighted L2 distance (...,)
        """
        return (a - b).square().sum(dim=-1) * weight

    @staticmethod
    def l1_dist(a, b, weight):
        """Compute L1 distance between two point sets with weights.

        Args:
            a: First point set (..., D)
            b: Second point set (..., D)
            weight: Weight tensor (...,)

        Returns:
            Weighted L1 distance (...,)
        """
        return (a - b).norm(dim=-1) * weight

    @staticmethod
    def get_all_dists():
        """Get dictionary of all distance functions.

        Returns:
            Dictionary mapping distance names to distance functions
        """
        return dict(l1=LossFunctions.l1_dist, l2=LossFunctions.l2_dist)
