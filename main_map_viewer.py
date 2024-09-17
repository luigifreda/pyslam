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
from ground_truth import groundtruth_factory
from dataset import dataset_factory, SensorType
from trajectory_writer import TrajectoryWriter

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

if platform.system()  == 'Linux':     
    from display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import FeatureMatcherTypes

from feature_tracker_configs import FeatureTrackerConfigs

from parameters import Parameters  
import multiprocessing as mp 


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', type=str, default='map.json', help='saved map name')
    args = parser.parse_args()

    config = Config()

    cam = PinholeCamera(config)
    tracker_config = FeatureTrackerConfigs.TEST    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    # create SLAM object 
    slam = Slam(cam, feature_tracker)
    time.sleep(1) # to show initial messages 
    
    viewer3D = Viewer3D()

    slam.load_map(args.filename)
            
    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
    while True:            
        # 3D display (map display)
        if viewer3D is not None:
            viewer3D.draw_map(slam)   
            
                   
    slam.quit()
    