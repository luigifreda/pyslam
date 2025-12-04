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
Parameter manipulation and stacking utilities.
"""

import torch
import torch.nn as nn


class ParameterUtils:
    """Parameter manipulation and stacking utilities."""

    @staticmethod
    def NoGradParamDict(x):
        """Create a ParameterDict with requires_grad=False.

        Args:
            x: Dictionary of parameters

        Returns:
            nn.ParameterDict with requires_grad=False
        """
        assert isinstance(x, dict)
        return nn.ParameterDict(x).requires_grad_(False)

    @staticmethod
    def ParameterStack(params, keys=None, is_param=None, fill=0):
        """Stack parameters into a single tensor.

        Args:
            params: List or dict of parameters
            keys: Optional list of keys to select from dict
            is_param: Whether to create nn.Parameter
            fill: Padding size for raveled tensors

        Returns:
            Stacked parameter tensor
        """
        if keys is not None:
            params = [params[k] for k in keys]

        if fill > 0:
            params = [ParameterUtils.ravel_hw(p, fill) for p in params]

        requires_grad = params[0].requires_grad
        assert all(p.requires_grad == requires_grad for p in params)

        params = torch.stack(list(params)).float().detach()
        if is_param or requires_grad:
            params = nn.Parameter(params)
            params.requires_grad_(requires_grad)
        return params

    @staticmethod
    def ravel_hw(tensor, fill=0):
        """Ravel H,W dimensions and optionally pad.

        Args:
            tensor: Input tensor with H, W dimensions
            fill: Padding size (default: 0)

        Returns:
            Ravelled tensor, optionally padded
        """
        # ravel H,W
        tensor = tensor.view((tensor.shape[0] * tensor.shape[1],) + tensor.shape[2:])

        if len(tensor) < fill:
            tensor = torch.cat((tensor, tensor.new_zeros((fill - len(tensor),) + tensor.shape[1:])))
        return tensor
