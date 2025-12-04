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
Image shape and dimension utilities.
"""

import torch
import numpy as np


class ImageUtils:
    """Image shape and dimension utilities."""

    @staticmethod
    def rgb(ftensor, true_shape=None):
        if isinstance(ftensor, list):
            return [ImageUtils.rgb(x, true_shape=true_shape) for x in ftensor]
        if isinstance(ftensor, torch.Tensor):
            ftensor = ftensor.detach().cpu().numpy()  # H,W,3
        if ftensor.ndim == 3 and ftensor.shape[0] == 3:
            ftensor = ftensor.transpose(1, 2, 0)
        elif ftensor.ndim == 4 and ftensor.shape[1] == 3:
            ftensor = ftensor.transpose(0, 2, 3, 1)
        if true_shape is not None:
            H, W = true_shape
            ftensor = ftensor[:H, :W]
        if ftensor.dtype == np.uint8:
            img = np.float32(ftensor) / 255
        else:
            img = (ftensor * 0.5) + 0.5
        return img.clip(min=0, max=1)

    @staticmethod
    def get_imshapes(edges, pred_i, pred_j):
        """Extract image shapes from edge predictions.

        Args:
            edges: List of edge tuples (i, j)
            pred_i: Dictionary of predictions for view i
            pred_j: Dictionary of predictions for view j

        Returns:
            List of image shapes (H, W) for each image
        """
        n_imgs = max(max(e) for e in edges) + 1
        imshapes = [None] * n_imgs
        for e, (i, j) in enumerate(edges):
            shape_i = tuple(pred_i[e].shape[0:2])
            shape_j = tuple(pred_j[e].shape[0:2])
            if imshapes[i]:
                assert imshapes[i] == shape_i, f"incorrect shape for image {i}"
            if imshapes[j]:
                assert imshapes[j] == shape_j, f"incorrect shape for image {j}"
            imshapes[i] = shape_i
            imshapes[j] = shape_j
        return imshapes
