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

import sys
import os

from pyslam.utilities.utils_sys import Printer

USE_CPP = False
try:
    # Try to get from environment variable and fallback to default declared above
    USE_CPP_VAR = os.environ.get("PYSLAM_USE_CPP", str(USE_CPP).lower())
    if USE_CPP_VAR == "true":
        print(f"[pyslam.slam module] USE_CPP_VAR: {USE_CPP_VAR}")
    USE_CPP = USE_CPP_VAR.lower() in (
        "true",
        "1",
        "yes",
        "on",
    )
except:
    pass

if USE_CPP:
    try:
        from .cpp import cpp_module

        if not cpp_module.CPP_AVAILABLE:
            Printer.orange("❌ cpp_module imported successfully but C++ core is not available")
            sys.exit(1)
        print("✅ cpp_module imported successfully, C++ core is available")
        from .cpp import (
            Frame,
            KeyFrame,
            MapPoint,
            Map,
            Camera,
            PinholeCamera,
            CameraType,
            CameraUtils,
            CameraPose,
            Sim3Pose,
            optimizer_g2o,
            CKDTree2d,
            CKDTree3d,
            CKDTreeDyn,
        )

    except ImportError as e:
        Printer.red(f"❌ Failed to import C++ module: {e}")
        sys.exit(1)
else:

    # Fallback to Python implementations
    from .frame import Frame
    from .keyframe import KeyFrame
    from .map_point import MapPoint
    from .map import Map
    from .camera import Camera, PinholeCamera, CameraType, CameraUtils
    from .camera_pose import CameraPose
    from .sim3_pose import Sim3Pose
    from . import optimizer_g2o
    from .ckdtree import CKDTree2d, CKDTree3d, CKDTreeDyn


from .feature_tracker_shared import FeatureTrackerShared

__all__ = [
    "MapPoint",
    "Frame",
    "KeyFrame",
    "Map",
    "CameraPose",
    "Camera",
    "PinholeCamera",
    "CameraType",
    "CameraUtils",
    "Sim3Pose",
    "FeatureTrackerShared",
    "optimizer_g2o",
    "CKDTree2d",
    "CKDTree3d",
    "CKDTreeDyn",
    "USE_CPP",
]
