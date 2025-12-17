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

from .scene_from_views_types import SceneFromViewsType
from .scene_from_views_base import SceneFromViewsBase

from pyslam.utilities.system import import_from

from pyslam.scene_from_views.optimizers import (
    DEFAULT_SPARSE_OPTIMIZER_CONFIG,
    DEFAULT_DENSE_OPTIMIZER_CONFIG,
)


def scene_from_views_factory(
    scene_from_views_type: SceneFromViewsType = SceneFromViewsType.DEPTH_ANYTHING_V3,
    device=None,
    **kwargs,
) -> SceneFromViewsBase:
    """
    Factory function to create a SceneFromViews reconstructor based on the specified type.

    Args:
        scene_from_views_type: Type of scene reconstruction method to use
        device: Device to run inference on (e.g., 'cuda', 'cpu')
        **kwargs: Additional model-specific parameters passed to the reconstructor

    Returns:
        SceneFromViewsBase: Instance of the requested reconstructor class

    Raises:
        ValueError: If scene_from_views_type is not recognized
        ImportError: If required dependencies are not available
    """
    if scene_from_views_type == SceneFromViewsType.DEPTH_ANYTHING_V3:
        try:
            from .scene_from_views_depth_anything_v3 import SceneFromViewsDepthAnythingV3
        except ImportError:
            SceneFromViewsDepthAnythingV3 = import_from(
                "pyslam.scene_from_views.scene_from_views_depth_anything_v3",
                "SceneFromViewsDepthAnythingV3",
            )
        return SceneFromViewsDepthAnythingV3(device=device, **kwargs)

    elif scene_from_views_type == SceneFromViewsType.MAST3R:
        try:
            from .scene_from_views_mast3r import SceneFromViewsMast3r
        except ImportError:
            SceneFromViewsMast3r = import_from(
                "pyslam.scene_from_views.scene_from_views_mast3r",
                "SceneFromViewsMast3r",
            )
        return SceneFromViewsMast3r(
            device=device,
            optimizer_config=DEFAULT_SPARSE_OPTIMIZER_CONFIG.copy(),
            # optimizer_config=DEFAULT_DENSE_OPTIMIZER_CONFIG.copy(),
            **kwargs,
        )

    elif scene_from_views_type == SceneFromViewsType.MVDUST3R:
        try:
            from .scene_from_views_mvdust3r import SceneFromViewsMvdust3r
        except ImportError:
            SceneFromViewsMvdust3r = import_from(
                "pyslam.scene_from_views.scene_from_views_mvdust3r",
                "SceneFromViewsMvdust3r",
            )
        return SceneFromViewsMvdust3r(device=device, **kwargs)

    elif scene_from_views_type == SceneFromViewsType.VGGT:
        try:
            from .scene_from_views_vggt import SceneFromViewsVggt
        except ImportError:
            SceneFromViewsVggt = import_from(
                "pyslam.scene_from_views.scene_from_views_vggt",
                "SceneFromViewsVggt",
            )
        return SceneFromViewsVggt(device=device, **kwargs)

    elif scene_from_views_type == SceneFromViewsType.VGGT_ROBUST:
        try:
            from .scene_from_views_vggt_robust import SceneFromViewsVggtRobust
        except ImportError:
            SceneFromViewsVggtRobust = import_from(
                "pyslam.scene_from_views.scene_from_views_vggt_robust",
                "SceneFromViewsVggtRobust",
            )
        return SceneFromViewsVggtRobust(device=device, **kwargs)

    elif scene_from_views_type == SceneFromViewsType.DUST3R:
        try:
            from .scene_from_views_dust3r import SceneFromViewsDust3r
        except ImportError:
            SceneFromViewsDust3r = import_from(
                "pyslam.scene_from_views.scene_from_views_dust3r",
                "SceneFromViewsDust3r",
            )
        return SceneFromViewsDust3r(
            device=device,
            # optimizer_config=DEFAULT_SPARSE_OPTIMIZER_CONFIG.copy(),
            optimizer_config=DEFAULT_DENSE_OPTIMIZER_CONFIG.copy(),
            **kwargs,
        )

    else:
        raise ValueError(f"Invalid scene_from_views_type: {scene_from_views_type}")
