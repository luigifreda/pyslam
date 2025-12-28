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

USE_CPP = Parameters.USE_CPP_CORE
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
        globals()[name] = cls
        classes_dict[name] = cls


globals()["USE_CPP"] = USE_CPP

__all__ = [
    "CPP_AVAILABLE",
    "cpp_module",
    "python_module",
    "USE_CPP",
]
# Add all core classes to the __all__ list
for cls in core_classes:
    __all__.append(cls)
