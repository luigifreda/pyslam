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

import sys 
from config import Config

import argparse
import numpy as np
import cv2
import math
import time 

import platform 

from config import Config

from config_parameters import Parameters

from slam import Slam, SlamState, SlamMode
from camera  import PinholeCamera
from dataset import dataset_factory, SensorType
from ground_truth import groundtruth_factory, GroundTruth

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 
from utils_geom import inv_T

from feature_tracker_configs import FeatureTrackerConfigs

from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType

from volumetric_integrator_base import VolumetricIntegratorBase
from volumetric_integrator_factory import volumetric_integrator_factory, VolumetricIntegratorType


import signal   


# intercept the SIGINT signal
def signal_handler(signal, frame):
    print('You pressed Ctrl+C!')
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)


if __name__ == "__main__":

    config = Config()
        
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default=config.system_state_folder_path, help='path where we have saved the system state')
    parser.add_argument('-o','--output_path', required=False, type=str, default=config.system_state_folder_path + '_dense_reconstruction', help="Path to save the system state with the dense reconstruction")
    args = parser.parse_args()

    camera = PinholeCamera()
    feature_tracker_config = FeatureTrackerConfigs.TEST
    
    # Create SLAM object 
    slam = Slam(camera, feature_tracker_config, slam_mode=SlamMode.MAP_BROWSER)
    # load the system state    
    slam.load_system_state(args.path)
    camera = slam.camera # update the camera after having reloaded the state 
    groundtruth = GroundTruth.load(args.path) # load ground truth from saved state   
    viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
    print(f'Viewer_scale: {viewer_scale}')    
    print(f'Sensor_type: {slam.sensor_type}')
                
    # Select your volumetric integrator here (see the file volumetric_integrator_factory.py) 
    volumetric_integrator_type = VolumetricIntegratorType.TSDF     # TSDF, GAUSSIAN_SPLATTING
    Parameters.kVolumetricIntegrationUseDepthEstimator = (slam.sensor_type==SensorType.STEREO) # Use depth estimator for volumetric integration in the back-end in the case of stereo data. 
                                                                                    # Since the depth inference time may be above 1 second, the volumetric integrator may be very slow.
                                                                                    # NOTE: The depth estimator estimates a metric depth (with an absolute scale). You can't combine it with a MONOCULAR SLAM output
                                                                                    # since the SLAM sparse map scale will not be consistent.
    Parameters.kVolumetricIntegrationDepthEstimatorType = "DEPTH_CRESTEREO_PYTORCH"  # "DEPTH_PRO","DEPTH_ANYTHING_V2, "DEPTH_SGBM", "DEPTH_RAFT_STEREO", "DEPTH_CRESTEREO_PYTORCH"  (see depth_estimator_factory.py)    
    Parameters.kVolumetricIntegrationMinNumLBATimes = 0 # NOTE: This avoids the volumetric integrator integrates just keyframes with lba_count >= kVolumetricIntegrationMinNumLBATimes    
    if Parameters.kVolumetricIntegrationUseDepthEstimator:
        print(f'Using depth estimator: {Parameters.kVolumetricIntegrationDepthEstimatorType}')
    volumetric_integrator = volumetric_integrator_factory(volumetric_integrator_type, camera, 
                                                          slam.environment_type, slam.sensor_type)

    map = slam.map
    num_map_keyframes = map.num_keyframes()
    keyframes = map.get_keyframes()
            
    viewer3D = Viewer3D(viewer_scale)
    if groundtruth is not None:
        gt_traj3d, gt_timestamps = groundtruth.getFull3dTrajectory()
        viewer3D.set_gt_trajectory(gt_traj3d, gt_timestamps, align_with_scale=slam.sensor_type==SensorType.MONOCULAR)    
            
    # wait the viewer3D to be ready
    while not viewer3D.is_running():
        time.sleep(0.1)

    viewer3D.draw_map(slam)               
            
    print(f'inserting #keyframes: {num_map_keyframes} ...')
    
    is_map_save = False  # save map on GUI
    is_paused = False    # pause/resume on GUI     
    
    i = 0
    if num_map_keyframes>0:       
        
        for kf in keyframes:
            
            print('-----------------------------------')
            print(f'inserting keyframe: {kf.id}, img_id: {kf.img_id}. img shape: {kf.img.shape}, depth shape: {kf.depth_img.shape if kf.depth_img is not None else None} type: {kf.depth_img.dtype if kf.depth_img is not None else None}, lba_count: {kf.lba_count}')
                        
            volumetric_integrator.add_keyframe(kf, kf.img, kf.img_right, kf.depth_img)
            volumetric_integrator.add_update_output_task()
            time.sleep(0.1)
            
            dense_map_output = None
            if volumetric_integrator.q_out.qsize() > 0:            
                dense_map_output = volumetric_integrator.pop_output()
                if dense_map_output is not None:
                    viewer3D.draw_dense_geometry(dense_map_output.point_cloud, dense_map_output.mesh)
            i += 1
            if i > 30:
                break

    
    print(f'processing and visualizing dense map...')
    
    i = 0
    while not viewer3D.is_closed():   
        time.sleep(0.1)   
        
        if i % 10 == 0:
            volumetric_integrator.add_update_output_task()
        
        dense_map_output = None
        if volumetric_integrator.q_out.qsize() > 0:            
            dense_map_output = volumetric_integrator.pop_output()
            if dense_map_output is not None:
                viewer3D.draw_dense_geometry(dense_map_output.point_cloud, dense_map_output.mesh)
                
        is_map_save = viewer3D.is_map_save() and is_map_save == False                   
        if is_map_save:
            output_path = args.output_path
            slam.save_system_state(output_path)
            volumetric_integrator.save(output_path)
                            
        i += 1
                              
    slam.quit()
    volumetric_integrator.quit()
    