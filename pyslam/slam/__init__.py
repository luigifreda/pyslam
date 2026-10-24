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

from pyslam.config_parameters import Parameters
from pyslam.utilities.logging import Printer

from .cpp import (
    python_module,
    CPP_AVAILABLE,
    core_classes,
)

# USE_CPP_CORE keeps the two implementations separate.
# True: classes from cpp_core. False: Python classes, with the original
# computations left in the Python modules.
USE_CPP = Parameters.USE_CPP_CORE

classes_dict = {}

if USE_CPP:
    if CPP_AVAILABLE:
        from .cpp import cpp_module

        print("✅ cpp_module imported successfully, C++ core is available")
        # Assign all classes from C++ module to global namespace
        for name, cls in cpp_module.classes.items():
            globals()[name] = cls
            classes_dict[name] = cls
    else:
        print("❌ C++ core module not available. Falling back to Python implementations.")
        sys.exit(1)
else:
    # Assign all classes from fallback to global namespace
    for name, cls in python_module.classes.items():
        if cls is not None:
            globals()[name] = cls
            classes_dict[name] = cls


globals()["USE_CPP"] = USE_CPP

# Load the Python orchestrators only when requested. Importing Map/Frame in a
# worker process should not import the entire feature tracker and viewer stack.
_PYTHON_FRONTENDS = {
    "MotionModel": "motion_model",
    "MotionModelDamping": "motion_model",
    "Initializer": "initializer",
    "SLAMDynamicConfig": "slam_dynamic_config",
    "Tracking": "tracking",
    "LocalMapping": "local_mapping",
    "Slam": "slam",
}


def __getattr__(name):
    if not USE_CPP and name in _PYTHON_FRONTENDS:
        from importlib import import_module

        cls = getattr(import_module(f".{_PYTHON_FRONTENDS[name]}", __name__), name)
        globals()[name] = cls
        classes_dict[name] = cls
        python_module.classes[name] = cls
        setattr(python_module, name, cls)
        return cls
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def camera_from_config(config):
    """Build the active camera implementation selected by Camera.type."""
    from pyslam.slam.camera import camera_from_config as build_camera

    constructors = {
        name: classes_dict[name]
        for name in (
            "PinholeCamera",
            "KannalaBrandt4Camera",
            "PinholeRadtan8Camera",
            "DoubleSphereCamera",
        )
    }
    return build_camera(config, constructors)


__all__ = [
    "CPP_AVAILABLE",
    "cpp_module",
    "python_module",
    "USE_CPP",
    "camera_from_config",
]
# Add all core classes to the __all__ list
for cls in core_classes:
    __all__.append(cls)
