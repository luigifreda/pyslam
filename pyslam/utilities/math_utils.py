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

Mathematical utility functions for signed log/exp operations.
"""

import torch


def signed_log1p(x):
    """
    Compute signed log(1 + |x|) preserving the sign.

    Args:
        x: Input tensor

    Returns:
        Tensor with sign(x) * log(1 + |x|)
    """
    sign = torch.sign(x)
    return sign * torch.log1p(torch.abs(x))


def signed_expm1(x):
    """
    Compute signed (exp(|x|) - 1) preserving the sign.

    Args:
        x: Input tensor

    Returns:
        Tensor with sign(x) * (exp(|x|) - 1)
    """
    sign = torch.sign(x)
    return sign * torch.expm1(torch.abs(x))

