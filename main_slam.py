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
from feature_tracker_configs import FeatureTrackerConfigs

from parameters import Parameters  
import multiprocessing as mp 


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

    tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
    #tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, FAST_TFEAT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    # WARNING: At present, SLAM does not support LOFTR and other "pure" image matchers (further details in the commenting notes of LOFTR in feature_tracker_configs.py).
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = num_features
    tracker_config['tracker_type'] = tracker_type
    
    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    # create SLAM object 
    slam = Slam(cam, feature_tracker, dataset.sensorType(), groundtruth=None) # groundtruth not actually used by Slam class
    time.sleep(1) # to show initial messages 
    
    viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
    if groundtruth is not None:
        gt_traj3d, gt_timestamps = groundtruth.getFull3dTrajectory()
        viewer3D.set_gt_trajectory(gt_traj3d, gt_timestamps, align_with_scale=dataset.sensor_type==SensorType.MONOCULAR)
    
    if platform.system()  == 'Linux':    
        display2d = Display2D(cam.width, cam.height)  # pygame interface 
    else: 
        display2d = None  # enable this if you want to use opencv window

    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')  
    info_3dpoints_plt = None #Mplot2d(xlabel='img id', ylabel='# points',title='info 3d points')      
    reproj_error_plt = Mplot2d(xlabel='img id', ylabel='error',title='mean chi2 error')

    do_step = False      # proceed step by step on GUI  
    is_paused = False    # pause/resume on GUI 
    is_map_save = False  # save map on GUI
    
    key_cv = None
            
    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
    last_processed_kf_img_id = -1
    while dataset.isOk():
            
        if do_step:
            Printer.green('do step: ', do_step)
                        
        if not is_paused or do_step: 
            print('..................................')               
            img = dataset.getImageColor(img_id)
            depth = dataset.getDepth(img_id)
            img_right = dataset.getImageColorRight(img_id) if dataset.sensor_type == SensorType.STEREO else None
            
            if img is None:
                print('image is empty')
                getchar()
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
                # 2D display (image display)
                if display2d is not None:
                    display2d.draw(img_draw)
                else: 
                    cv2.imshow('Camera', img_draw)

                # draw matching info
                if matched_points_plt is not None: 
                    if slam.tracking.num_matched_kps is not None: 
                        matched_kps_signal = [img_id, slam.tracking.num_matched_kps]     
                        matched_points_plt.draw(matched_kps_signal,'# keypoint matches',color='r')                         
                    if slam.tracking.num_inliers is not None: 
                        inliers_signal = [img_id, slam.tracking.num_inliers]                    
                        matched_points_plt.draw(inliers_signal,'# inliers',color='g')
                    if slam.tracking.num_matched_map_points is not None: 
                        valid_matched_map_points_signal = [img_id, slam.tracking.num_matched_map_points]   # valid matched map points (in current pose optimization)                                       
                        matched_points_plt.draw(valid_matched_map_points_signal,'# matched map pts', color='b')  
                    if slam.tracking.num_kf_ref_tracked_points is not None: 
                        kf_ref_tracked_points_signal = [img_id, slam.tracking.num_kf_ref_tracked_points]                    
                        matched_points_plt.draw(kf_ref_tracked_points_signal,'# $KF_{ref}$  tracked pts',color='c')   
                    if slam.tracking.descriptor_distance_sigma is not None: 
                        descriptor_sigma_signal = [img_id, slam.tracking.descriptor_distance_sigma]                    
                        matched_points_plt.draw(descriptor_sigma_signal,'descriptor distance $\sigma_{th}$',color='k')                                                                 
                    matched_points_plt.refresh()    
                    
                # draw info about 3D points management by local mapping 
                if info_3dpoints_plt is not None:
                    print(f'last_processed_kf_img_id: {last_processed_kf_img_id}')
                    if slam.local_mapping.last_processed_kf_img_id is not None and last_processed_kf_img_id != slam.local_mapping.last_processed_kf_img_id:
                        if slam.local_mapping.last_num_triangulated_points is not None:
                            last_processed_kf_img_id = slam.local_mapping.last_processed_kf_img_id 
                            num_triangulated_points_signal = [img_id, slam.local_mapping.last_num_triangulated_points]     
                            info_3dpoints_plt.draw(num_triangulated_points_signal,'# temporal triangulated pts',color='r') 
                        if slam.local_mapping.last_num_fused_points is not None:
                            num_fused_points_signal = [img_id, slam.local_mapping.last_num_fused_points]
                            info_3dpoints_plt.draw(num_fused_points_signal,'# fused pts',color='g')
                        if slam.local_mapping.last_num_culled_keyframes is not None:
                            num_culled_keyframes_signal = [img_id, slam.local_mapping.last_num_culled_keyframes]
                            info_3dpoints_plt.draw(num_culled_keyframes_signal,'# culled keyframes',color='b')
                        if slam.local_mapping.last_num_culled_points is not None:
                            num_culled_points_signal = [img_id, slam.local_mapping.last_num_culled_points]
                            info_3dpoints_plt.draw(num_culled_points_signal,'# culled pts',color='c')
                        if slam.tracking.last_num_static_stereo_map_points is not None: 
                            num_static_stereo_map_points_signal = [img_id, slam.tracking.last_num_static_stereo_map_points]
                            info_3dpoints_plt.draw(num_static_stereo_map_points_signal,'# static triangulated pts',color='k') 
                    info_3dpoints_plt.refresh()   
                    
                if reproj_error_plt is not None:
                    if slam.tracking.mean_pose_opt_chi2_error is not None:
                        mean_pose_opt_chi2_error_signal = [img_id, slam.tracking.mean_pose_opt_chi2_error]
                        reproj_error_plt.draw(mean_pose_opt_chi2_error_signal,'pose opt chi2 error',color='r')
                    if slam.local_mapping.mean_ba_chi2_error is not None:
                        mean_squared_reproj_error_signal = [img_id, slam.local_mapping.mean_ba_chi2_error]
                        reproj_error_plt.draw(mean_squared_reproj_error_signal,'BA chi2 error',color='g')
                                            
                       
            if trajectory_writer is not None and slam.tracking.cur_R is not None and slam.tracking.cur_t is not None:
                trajectory_writer.write_trajectory(slam.tracking.cur_R, slam.tracking.cur_t, timestamp)
                
            if time_start is not None: 
                duration = time.time()-time_start
                if(frame_duration > duration):
                    time.sleep(frame_duration-duration) 
                
            img_id += 1  
        else:
            time.sleep(0.1)                                 
        
        # get keys 
        key = matched_points_plt.get_key() if matched_points_plt is not None else None
        if key == '' or key is None:
            key = info_3dpoints_plt.get_key() if info_3dpoints_plt is not None else None
        if key == '' or key is None:
            key = reproj_error_plt.get_key() if reproj_error_plt is not None else None
        if display2d is None:
            key_cv = cv2.waitKey(1) & 0xFF   
            
        # if key != '' and key is not None:
        #     print(f'key pressed: {key}') 
        
        # manage interface infos  
        
        if slam.tracking.state==SlamState.LOST:
            if display2d is None:  
                key_cv = cv2.waitKey(0) & 0xFF   # useful when drawing stuff for debugging                                 
            else: 
                getchar()
                 
        
        if is_map_save:
            slam.save_map('map.json') 
            Printer.green('uncheck pause checkbox on GUI to continue...\n')        
                      
        if key == 'q' or (key_cv == ord('q')):
            slam.quit()
            if matched_points_plt is not None:
                matched_points_plt.quit()
            if info_3dpoints_plt is not None:
                info_3dpoints_plt.quit()    
            if reproj_error_plt is not None:
                reproj_error_plt.quit()             
            if display2d is not None:
                display2d.quit()
            if viewer3D is not None:
                viewer3D.quit()           
            break
        
        if viewer3D is not None:
            is_paused = viewer3D.is_paused()    
            is_map_save = viewer3D.is_map_save() and is_map_save == False 
            do_step = viewer3D.do_step() and do_step == False     
    
    trajectory_writer.close_file()                   
    slam.quit()
    
    #cv2.waitKey(0)
    cv2.destroyAllWindows()
