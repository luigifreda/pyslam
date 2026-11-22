#!/usr/bin/env python3
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
*
* Minimal test to isolate GIL issue
"""
import sys
import os

import pyslam.config as config
from pyslam.config_parameters import Parameters

Parameters.USE_CPP_CORE = True

from pyslam.slam.cpp import cpp_module, python_module, CPP_AVAILABLE

print(f"Python: {sys.executable}")
print(f"Python version: {sys.version}")

try:
    import cpp_core

    print("✅ SUCCESS: cpp_core imported successfully")
except Exception as e:
    print(f"❌ FAILED: {type(e).__name__}: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)
