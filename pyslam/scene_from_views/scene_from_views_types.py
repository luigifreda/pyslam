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
from pyslam.utilities.serialization import SerializableEnum, register_class


@register_class
class SceneFromViewsType(SerializableEnum):
    DEPTH_ANYTHING_V3 = 0  # Depth Anything 3 - monocular depth with optional pose/intrinsics
    MAST3R = 1  # MASt3R - Grounding Image Matching in 3D with MASt3R
    MVDUST3R = 2  # MVDust3r - multi-view DUSt3R variant
    VGGT = 3  # VGGT - Visual Geometry Grounded Transformer
    VGGT_ROBUST = (
        4  # VGGT Robust - Emergent Outlier View Rejection in Visual Geometry Grounded Transformers
    )
    DUST3R = 5  # DUSt3R - Geometric 3D Vision Made Easy

    @staticmethod
    def from_string(name: str):
        """
        Create SceneFromViewsType from string name.

        Args:
            name: String name of the type (e.g., "DEPTH_ANYTHING_V3", "MAST3R")

        Returns:
            SceneFromViewsType: Corresponding enum value

        Raises:
            ValueError: If name is not a valid SceneFromViewsType
        """
        try:
            return SceneFromViewsType[name]
        except KeyError:
            raise ValueError(f"Invalid SceneFromViewsType: {name}")
