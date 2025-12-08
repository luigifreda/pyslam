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
import warnings
import traceback

# Add the C++ module lib directory to path
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
CPP_LIB_DIR = os.path.join(THIS_DIR, "lib")
if os.path.exists(CPP_LIB_DIR):
    sys.path.insert(0, CPP_LIB_DIR)
else:
    print(f"❌ CPP_LIB_DIR not found: {CPP_LIB_DIR}")


USE_PYTHON_FALLBACK = False


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


# nominal core classes that should be available both in C++ and Python core
core_classes = [
    "MapPoint",
    "Frame",
    "FrameBase",
    "KeyFrame",
    "Map",
    "CameraPose",
    "Camera",
    "PinholeCamera",
    "CameraUtils",
    "Sim3Pose",
    "optimizer_g2o",
    "TrackingUtils",
    "CameraType",
    "SensorType",
    "ReloadedSessionMapInfo",
    "LocalCovisibilityMap",
    "CKDTree2d",
    "CKDTree3d",
    "CKDTreeDyn",
    "FeatureSharedResources",
    "Quaternion",
    "Rotation2d",
    "AngleAxis",
    "Isometry2d",
    "Isometry3d",
    "RotationHistogram",
    "Parameters",
    "ProjectionMatcher",
    "EpipolarMatcher",
]


def _import_cpp_core():
    """Import the C++ core module and return all classes in a dictionary"""
    try:
        import cpp_core

        cpp_classes = {
            "MapPoint": cpp_core.MapPoint,
            "Frame": cpp_core.Frame,
            "FrameBase": cpp_core.FrameBase,
            "KeyFrame": cpp_core.KeyFrame,
            "Map": cpp_core.Map,
            "CameraPose": cpp_core.CameraPose,
            "Camera": cpp_core.Camera,
            "PinholeCamera": cpp_core.PinholeCamera,
            "CameraUtils": cpp_core.CameraUtils,
            "Sim3Pose": cpp_core.Sim3Pose,
            "optimizer_g2o": cpp_core.OptimizerG2o,
            "TrackingUtils": cpp_core.TrackingUtils,
            "CameraType": cpp_core.CameraType,
            "SensorType": cpp_core.SensorType,
            "ReloadedSessionMapInfo": cpp_core.ReloadedSessionMapInfo,
            "LocalCovisibilityMap": cpp_core.LocalCovisibilityMap,
            "CKDTree2d": cpp_core.CKDTree2d_d,
            "CKDTree3d": cpp_core.CKDTree3d_d,
            "CKDTreeDyn": cpp_core.CKDTreeDyn_d,
            "FeatureSharedResources": cpp_core.FeatureSharedResources,
            "Quaternion": cpp_core.Quaternion,
            "Rotation2d": cpp_core.Rotation2d,
            "AngleAxis": cpp_core.AngleAxis,
            "Isometry2d": cpp_core.Isometry2d,
            "Isometry3d": cpp_core.Isometry3d,
            "RotationHistogram": cpp_core.RotationHistogram,
            "Parameters": cpp_core.Parameters,
            "ProjectionMatcher": cpp_core.ProjectionMatcher,
            "EpipolarMatcher": cpp_core.EpipolarMatcher,
        }

        out_classes = {}
        # we check the coverage here to ensure we have all the classes
        for cls in core_classes:
            out_classes[cls] = cpp_classes[cls]
        return out_classes, True

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Failed to import C++ core module: {e}")
        return None, False


def _import_python_core():
    """Import Python fallback implementations"""
    try:
        from pyslam.slam.map_point import MapPoint
        from pyslam.slam.frame import Frame, FrameBase
        from pyslam.slam.keyframe import KeyFrame
        from pyslam.slam.map import Map
        from pyslam.slam.camera_pose import CameraPose
        from pyslam.slam.camera import Camera, PinholeCamera, CameraType, CameraUtils
        from pyslam.slam.sim3_pose import Sim3Pose
        from pyslam.io.dataset_types import SensorType
        import pyslam.slam.optimizer_g2o as optimizer_g2o
        from pyslam.slam.tracking_utils import TrackingUtils
        from pyslam.slam.map import ReloadedSessionMapInfo, LocalCovisibilityMap
        from pyslam.slam.ckdtree import CKDTree2d, CKDTree3d, CKDTreeDyn
        from pyslam.slam.rotation_histogram import RotationHistogram
        from pyslam.slam.geometry_matchers import ProjectionMatcher, EpipolarMatcher

        python_classes = {
            "MapPoint": MapPoint,
            "Frame": Frame,
            "FrameBase": FrameBase,
            "KeyFrame": KeyFrame,
            "Map": Map,
            "CameraPose": CameraPose,
            "Camera": Camera,
            "PinholeCamera": PinholeCamera,
            "CameraType": CameraType,
            "CameraUtils": CameraUtils,
            "SensorType": SensorType,
            "Sim3Pose": Sim3Pose,
            "optimizer_g2o": optimizer_g2o,
            "TrackingUtils": TrackingUtils,
            "ReloadedSessionMapInfo": ReloadedSessionMapInfo,
            "LocalCovisibilityMap": LocalCovisibilityMap,
            "CKDTree2d": CKDTree2d,
            "CKDTree3d": CKDTree3d,
            "CKDTreeDyn": CKDTreeDyn,
            "FeatureSharedResources": None,
            "Quaternion": None,
            "Rotation2d": None,
            "AngleAxis": None,
            "Isometry2d": None,
            "Isometry3d": None,
            "RotationHistogram": RotationHistogram,
            "Parameters": None,
            "ProjectionMatcher": ProjectionMatcher,
            "EpipolarMatcher": EpipolarMatcher,
        }
        out_classes = {}
        # we check the coverage here to ensure we have all the classes
        for cls in core_classes:
            out_classes[cls] = python_classes[cls]
        return out_classes, True

    except Exception as e:
        traceback.print_exc()
        print(f"❌ Failed to import Python fallback implementations: {e}")
        return None, False


# Try to import the C++ core module
cpp_classes, CPP_AVAILABLE = _import_cpp_core()
python_classes, PYTHON_AVAILABLE = _import_python_core()

if CPP_AVAILABLE:
    # print("✅ cpp_module imported successfully, C++ core is available")

    # Assign all classes from C++ module to global namespace
    for name, cls in cpp_classes.items():
        globals()[name] = cls

    # Register cleanup function to run at exit
    atexit.register(_cleanup_cpp_resources)
else:
    if USE_PYTHON_FALLBACK:
        print("❌ C++ core module not available. Falling back to Python implementations.")

        # Assign all classes from fallback to global namespace
        for name, cls in python_classes.items():
            globals()[name] = cls
    else:
        print("❌ C++ core module not available. Falling back to Python implementations.")
        # sys.exit(1)


# Create a cpp_module object for compatibility with the main SLAM module
class CppModule:
    """Wrapper class to provide cpp_module interface"""

    CPP_AVAILABLE = CPP_AVAILABLE
    classes = cpp_classes

    def __init__(self):
        if CPP_AVAILABLE:
            # Use the same classes dictionary from C++ import
            for name, cls in cpp_classes.items():
                setattr(self, name, cls)
            # Register cleanup function to run at exit
            self._cleanup_cpp_resources = _cleanup_cpp_resources
        else:
            if USE_PYTHON_FALLBACK:
                # Use fallback implementations
                for name, cls in python_classes.items():
                    setattr(self, name, cls)


class PythonModule:
    """Wrapper class to provide Python module interface"""

    classes = python_classes

    def __init__(self):
        for name, cls in python_classes.items():
            setattr(self, name, cls)


cpp_module = CppModule()
python_module = PythonModule()

__all__ = [
    "core_classes",
    "CPP_AVAILABLE",
    "cpp_module",
    "python_module",
]
# Add all core classes to the __all__ list
for cls in core_classes:
    __all__.append(cls)
