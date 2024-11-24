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

import cv2
import time 
import os
import sys

import platform 

from config import Config

from slam import Slam, SlamState
from slam_plot_drawer import SlamPlotDrawer
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory, SensorType
from trajectory_writer import TrajectoryWriter

from mplot_thread import Mplot2d
import matplotlib.colors as mcolors

if platform.system()  == 'Linux':     
    from display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 
from utils_img import ImgWriter

from feature_tracker_configs import FeatureTrackerConfigs

from loop_detector_configs import LoopDetectorConfigs

from parameters import Parameters  

from rerun_interface import Rerun

import traceback


if __name__ == "__main__":
                               
    config = Config()

    dataset = dataset_factory(config)

    trajectory_writer = None
    if config.trajectory_settings['save_trajectory']:
        trajectory_writer = TrajectoryWriter(format_type=config.trajectory_settings['format_type'], filename=config.trajectory_settings['filename'])
        trajectory_writer.open_file()
    
    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config)
    
    num_features=2000 
    if config.num_features_to_extract > 0:  # override the number of features to extract if we set something in the settings file
        num_features = config.num_features_to_extract

    # Select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, FAST_TFEAT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    # WARNING: At present, SLAM does not support LOFTR and other "pure" image matchers (further details in the commenting notes about LOFTR in feature_tracker_configs.py).
    feature_tracker_config = FeatureTrackerConfigs.ORB2
    feature_tracker_config['num_features'] = num_features
    Printer.green('feature_tracker_config: ',feature_tracker_config)    
    
    # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing. 
    # LoopDetectorConfigs: DBOW2, DBOW3, etc.
    loop_detection_config = LoopDetectorConfigs.DBOW3
    Printer.green('loop_detection_config: ',loop_detection_config)
        
    # create SLAM object 
    slam = Slam(cam, feature_tracker_config, loop_detection_config, dataset.sensorType(), groundtruth=None) # groundtruth not actually used by Slam class
    slam.set_viewer_scale(dataset.scale_viewer_3d)
    time.sleep(1) # to show initial messages 
    
    if config.system_state_load: 
        slam.load_system_state(config.system_state_folder_path)
        viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
        print(f'viewer_scale: {viewer_scale}')
        slam.set_tracking_state(SlamState.INIT_RELOCALIZE)

    viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
    if groundtruth is not None:
        gt_traj3d, gt_timestamps = groundtruth.getFull3dTrajectory()
        if viewer3D is not None:
            viewer3D.set_gt_trajectory(gt_traj3d, gt_timestamps, align_with_scale=dataset.sensor_type==SensorType.MONOCULAR)
    
    if platform.system()  == 'Linux':    
        display2d = Display2D(cam.width, cam.height)  # pygame interface 
    else: 
        display2d = None  # enable this if you want to use opencv window

    plot_drawer = SlamPlotDrawer(slam)
    
    img_writer = ImgWriter(font_scale=0.7)

    do_step = False      # proceed step by step on GUI 
    do_reset = False     # reset on GUI 
    is_paused = False    # pause/resume on GUI 
    is_map_save = False  # save map on GUI
    
    key_cv = None
            
    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
    while True:
        
        img, img_right, depth = None, None, None    
        
        if do_step:
            Printer.orange('do step: ', do_step)
            
        if do_reset: 
            Printer.yellow('do reset: ', do_reset)
            slam.reset()
               
        if not is_paused or do_step:
        
            if dataset.isOk():
                print('..................................')               
                img = dataset.getImageColor(img_id)
                depth = dataset.getDepth(img_id)
                img_right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
            
            if img is not None:
                timestamp = dataset.getTimestamp()          # get current timestamp 
                next_timestamp = dataset.getNextTimestamp() # get next timestamp 
                frame_duration = next_timestamp-timestamp if (timestamp is not None and next_timestamp is not None) else -1

                print(f'image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}') 
                
                time_start = None 
                if img is not None:
                    time_start = time.time()                  
                    slam.track(img, img_right, depth, img_id, timestamp)  # main SLAM function 
                                    
                    # 3D display (map display)
                    if viewer3D is not None:
                        viewer3D.draw_map(slam)

                    img_draw = slam.map.draw_feature_trails(img)
                    img_writer.write(img_draw, f'id: {img_id}', (30, 30))
                    
                    # 2D display (image display)
                    if display2d is not None:
                        display2d.draw(img_draw)
                    else: 
                        cv2.imshow('Camera', img_draw)
                    
                    # draw 2d plots
                    plot_drawer.draw(img_id)
                        
                if trajectory_writer is not None and slam.tracking.cur_R is not None and slam.tracking.cur_t is not None:
                    trajectory_writer.write_trajectory(slam.tracking.cur_R, slam.tracking.cur_t, timestamp)
                    
                if time_start is not None: 
                    duration = time.time()-time_start
                    if(frame_duration > duration):
                        time.sleep(frame_duration-duration) 
                    
                img_id += 1 
            else: 
                time.sleep(0.1) 
                
            # 3D display (map display)
            if viewer3D is not None:
                viewer3D.draw_dense_map(slam)  
                              
        else:
            time.sleep(0.1)                                 
        
        # get keys 
        key = plot_drawer.get_key()
        if display2d is None:
            key_cv = cv2.waitKey(1) & 0xFF   
            
        # if key != '' and key is not None:
        #     print(f'key pressed: {key}') 
        
        # manage interface infos  
        
        if slam.tracking.state==SlamState.LOST:
            if display2d is None:  
                key_cv = cv2.waitKey(0) & 0xFF   # useful when drawing stuff for debugging                                 
            else: 
                #getchar()
                time.sleep(0.5)
                 
        if is_map_save:
            slam.save_system_state(config.system_state_folder_path)
            dataset.save_info(config.system_state_folder_path)
            Printer.green('uncheck pause checkbox on GUI to continue...\n')        
        
        if viewer3D is not None:
            is_paused = viewer3D.is_paused()    
            is_map_save = viewer3D.is_map_save() and is_map_save == False 
            do_step = viewer3D.do_step() and do_step == False  
            do_reset = viewer3D.reset() and do_reset == False  
                                  
        if key == 'q' or (key_cv == ord('q')):
            slam.quit()
            plot_drawer.quit()         
            if display2d is not None:
                display2d.quit()
            if viewer3D is not None:
                viewer3D.quit()           
            break
            
    trajectory_writer.close_file()
    
    #cv2.waitKey(0)
    #cv2.destroyAllWindows()
