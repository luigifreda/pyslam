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
import atexit
import gc

# Add the C++ module lib directory to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_LIB_DIR = os.path.join(THIS_DIR, "lib")
if os.path.exists(CPP_LIB_DIR):
    sys.path.insert(0, CPP_LIB_DIR)


def _cleanup_cpp_resources():
    """Clean up C++ resources and force garbage collection at exit
    NOTE: This is not strictly required. Python will free memory on process exit,
    and pybind11 objects tied to Python references will be destroyed as
    the interpreter shuts down. However, it can help ensure Python objects are finalized
    before module teardown, which can avoid rare shutdown-time issues
    (e.g., lingering reference cycles, ordering-sensitive destructors,
    or sporadic segfaults on exit).
    """
    try:
        gc.collect()
    except Exception:
        pass  # Ignore errors during shutdown


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
    CameraUtils = cpp_core.CameraUtils
    Sim3Pose = cpp_core.Sim3Pose
    optimizer_g2o = cpp_core.OptimizerG2o

    # Import enums
    CameraType = cpp_core.CameraType
    SensorType = cpp_core.SensorType

    ReloadedSessionMapInfo = cpp_core.ReloadedSessionMapInfo
    LocalCovisibilityMap = cpp_core.LocalCovisibilityMap

    CKDTree2d = cpp_core.CKDTree2d_d
    CKDTree3d = cpp_core.CKDTree3d_d
    CKDTreeDyn = cpp_core.CKDTreeDyn_d

    # Register cleanup function to run at exit
    atexit.register(_cleanup_cpp_resources)

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
    from pyslam.slam.camera import Camera, PinholeCamera, CameraType, CameraUtils
    from pyslam.slam.sim3_pose import Sim3Pose
    from pyslam.io.dataset_types import SensorType
    import pyslam.slam.optimizer_g2o as optimizer_g2o

    from pyslam.slam.map import ReloadedSessionMapInfo, LocalCovisibilityMap

    from pyslam.slam.ckdtree import CKDTree2d, CKDTree3d, CKDTreeDyn

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
    "CameraUtils",
    "SensorType",
    "Sim3Pose",
    "OptimizerG2o",
    "ReloadedSessionMapInfo",
    "LocalCovisibilityMap",
    "optimizer_g2o",
    "CKDTree2d",
    "CKDTree3d",
    "CKDTreeDynd",
    "CPP_AVAILABLE",
]


# Create a cpp_module object for compatibility with the main SLAM module
class CppModule:
    """Wrapper class to provide cpp_module interface"""

    CPP_AVAILABLE = CPP_AVAILABLE


cpp_module = CppModule()
