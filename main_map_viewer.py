#!/usr/bin/env -S python3 -O
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

import argparse
import numpy as np
import cv2
import math
import time 

import platform 

from config import Config

from slam import Slam, SlamState
from camera  import PinholeCamera
from dataset import dataset_factory, SensorType


from viewer3D import Viewer3D
from utils_sys import getchar, Printer 

from feature_tracker_configs import FeatureTrackerConfigs

from parameters import Parameters


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='data/slam_state', help='path where we have saved the system state')
    args = parser.parse_args()

    config = Config()

    cam = PinholeCamera(config)
    feature_tracker_config = FeatureTrackerConfigs.TEST
    
    # create SLAM object 
    slam = Slam(cam, feature_tracker_config)
    time.sleep(1) # to show initial messages 

    slam.load_system_state(args.path)
    viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
    print(f'viewer_scale: {viewer_scale}')
        
    viewer3D = Viewer3D(viewer_scale)
            
    while True:            
        # 3D display (map display)
        if viewer3D is not None:
            viewer3D.draw_map(slam)   
                
    slam.quit()
    