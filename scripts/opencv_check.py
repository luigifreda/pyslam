#!/usr/bin/env -S python3
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

import cv2
import os
import ctypes.util
import numpy as np


def print_red(text):
    print("\033[91m" + text + "\033[0m")


def print_green(text):
    print("\033[92m" + text + "\033[0m")


if __name__ == "__main__":

    try:
        # Print OpenCV version
        print_green(f"Opencv info:")
        print(f"\tversion: {cv2.__version__}")

        # Print the path of the loaded cv2 library
        print(f"\tloaded library: {cv2.__file__}")

        # If you need more details about linked libraries
        # loaded_libs = ctypes.util.find_library("opencv")
        # print_green(f"OK: Linked OpenCV library details: {loaded_libs}")
    except Exception as e:
        print(f"Exception: {e}")
        print_red(f"NOK: Could not find linked OpenCV library details")

    # check if we have non-free OpenCV support
    try:
        detector = cv2.xfeatures2d.SURF_create()
        print_green(f"OK: You have non-free OpenCV support!")
        print(f"\t Succeeded to create detector: {detector}")
    except Exception as e:
        print(f"Exception: {e}")
        print_red(f"NOK: Could not create SURF detector: You dont have non-free OpenCV support!")

    # check if we have GUI support
    try:
        cv2.namedWindow("Test")
        cv2.imshow("Test", np.zeros((100, 100), dtype=np.uint8))
        cv2.waitKey(1)
        cv2.destroyWindow("Test")
        print_green(f"OK: You have GUI support!")
    except Exception as e:
        print(f"Exception: {e}")
        print_red(f"NOK: Could not create window: You dont have GUI support!")
