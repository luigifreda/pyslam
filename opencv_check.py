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
    
if __name__ == "__main__":
    # Print OpenCV version
    print(f"OpenCV version: {cv2.__version__}")

    # Print the path of the loaded cv2 library
    print(f"Loaded OpenCV library: {cv2.__file__}")

    if True:
        try:
            # If you need more details about linked libraries
            loaded_libs = ctypes.util.find_library("opencv")
            print(f"Linked OpenCV library details: {loaded_libs}")
        except Exception as e:
            print(f"Exception: {e}")
            print(f'Could not find linked OpenCV library details')
            
            
    if True: 
        try: 
            detector = cv2.xfeatures2d.SURF_create()
            print(f"Detector: {detector} created")
            print(f'You have non-free OpenCV support!')
        except Exception as e:
            print(f"Exception: {e}")
            print(f'Could not create SURF detector: You dont have non-free OpenCV support!')