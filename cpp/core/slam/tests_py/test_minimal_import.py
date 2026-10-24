#!/usr/bin/env python3
"""Minimal test to isolate GIL issue"""
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
