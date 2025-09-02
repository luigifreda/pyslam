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

"""
C++ Core Module for PYSLAM

This module contains the C++ implementations of core SLAM classes
with Python bindings via pybind11. The C++ implementations provide
significant performance improvements while maintaining the same
Python interface.

Classes:
    MapPoint: C++ implementation of MapPoint with zero-copy data access
    Frame: C++ implementation of Frame
    KeyFrame: C++ implementation of KeyFrame
    Map: C++ implementation of Map
    CameraPose: C++ implementation of CameraPose
    Camera: C++ implementation of Camera

The C++ classes own all their data and provide zero-copy access
to Python through pybind11's automatic numpy array conversion.
"""

import sys
import os

# Add the C++ module lib directory to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_LIB_DIR = os.path.join(THIS_DIR, "lib")
if os.path.exists(CPP_LIB_DIR):
    sys.path.insert(0, CPP_LIB_DIR)


# Try to import the C++ core module directly
try:
    import cpp_core

    # Import all classes from the pybind11 module
    MapPoint = cpp_core.MapPoint
    Frame = cpp_core.Frame
    KeyFrame = cpp_core.KeyFrame
    Map = cpp_core.Map
    CameraPose = cpp_core.CameraPose
    Camera = cpp_core.Camera
    PinholeCamera = getattr(cpp_core, "PinholeCamera", None)

    # Import enums
    CameraType = cpp_core.CameraType
    SensorType = cpp_core.SensorType

    # Import any other classes that might be available
    ReloadedSessionMapInfo = getattr(cpp_core, "ReloadedSessionMapInfo", None)
    LocalCovisibilityMap = getattr(cpp_core, "LocalCovisibilityMap", None)

    CPP_AVAILABLE = True

except ImportError as e:
    print(f"Warning: C++ core module not available: {e}")
    print("Falling back to Python implementations")

    # Fallback to Python implementations
    from pyslam.slam.map_point import MapPoint
    from pyslam.slam.frame import Frame
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.map import Map
    from pyslam.slam.camera_pose import CameraPose
    from pyslam.slam.camera import Camera, PinholeCamera, CameraType
    from pyslam.io.dataset_types import SensorType

    # Create dummy enums for compatibility

    ReloadedSessionMapInfo = None
    LocalCovisibilityMap = None

    CPP_AVAILABLE = False

    __all__ = [
        "MapPoint",
        "Frame",
        "KeyFrame",
        "Map",
        "CameraPose",
        "Camera",
        "PinholeCamera",
        "CameraType",
        "SensorType",
        "ReloadedSessionMapInfo",
        "LocalCovisibilityMap",
        "CPP_AVAILABLE",
    ]
