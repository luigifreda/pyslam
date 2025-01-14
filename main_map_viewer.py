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

from slam import Slam, SlamState, SlamMode
from camera  import PinholeCamera
from dataset import dataset_factory, SensorType
from ground_truth import GroundTruth

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 

from feature_tracker_configs import FeatureTrackerConfigs

from config_parameters import Parameters


if __name__ == "__main__":

    config = Config()
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=config.system_state_folder_path, help='path where we have saved the system state')
    args = parser.parse_args()

    camera = PinholeCamera()
    feature_tracker_config = FeatureTrackerConfigs.TEST
    
    # create SLAM object 
    slam = Slam(camera, feature_tracker_config, slam_mode=SlamMode.MAP_BROWSER)   
    # load the system state
    slam.load_system_state(args.path)
    camera = slam.camera # update the camera after having reloaded the state 
    groundtruth = GroundTruth.load(args.path) # load ground truth from saved state
    viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
    print(f'viewer_scale: {viewer_scale}')
        
    viewer3D = Viewer3D(viewer_scale)
    if groundtruth is not None:
        gt_traj3d, gt_timestamps = groundtruth.getFull3dTrajectory()
        viewer3D.set_gt_trajectory(gt_traj3d, gt_timestamps, align_with_scale=slam.sensor_type==SensorType.MONOCULAR)    
            
    while not viewer3D.is_closed():  
        time.sleep(0.1)
                  
        # 3D display (map display)
        viewer3D.draw_map(slam)   
                
    slam.quit()
    