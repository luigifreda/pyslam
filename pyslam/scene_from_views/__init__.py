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

from .scene_from_views_base import SceneFromViewsBase, SceneFromViewsResult
from .scene_from_views_types import SceneFromViewsType
from .scene_from_views_factory import scene_from_views_factory

# Direct imports (optional, can use factory instead)
try:
    from .scene_from_views_depth_anything_v3 import SceneFromViewsDepthAnythingV3
except ImportError:
    SceneFromViewsDepthAnythingV3 = None

try:
    from .scene_from_views_mast3r import SceneFromViewsMast3r
except ImportError:
    SceneFromViewsMast3r = None

try:
    from .scene_from_views_mvdust3r import SceneFromViewsMvdust3r
except ImportError:
    SceneFromViewsMvdust3r = None

try:
    from .scene_from_views_vggt import SceneFromViewsVggt
except ImportError:
    SceneFromViewsVggt = None

try:
    from .scene_from_views_dust3r import SceneFromViewsDust3r
except ImportError:
    SceneFromViewsDust3r = None

__all__ = [
    "SceneFromViewsBase",
    "SceneFromViewsResult",
    "SceneFromViewsType",
    "scene_from_views_factory",
    "SceneFromViewsDepthAnythingV3",
    "SceneFromViewsMast3r",
    "SceneFromViewsMvdust3r",
    "SceneFromViewsVggt",
    "SceneFromViewsDust3r",
]

