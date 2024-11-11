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
import matplotlib.colors as mcolors

if platform.system()  == 'Linux':     
    from display2D import Display2D  #  !NOTE: pygame generate troubles under macOS!

from viewer3D import Viewer3D
from utils_sys import getchar, Printer 
from utils_img import ImgWriter

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

from loop_detector_configs import LoopDetectorConfigs, loop_detector_factory

from parameters import Parameters  

from rerun_interface import Rerun

import traceback


class SlamPlotDrawer:
    def __init__(self, slam):
        self.slam = slam
        
        self.matched_points_plt = None
        self.info_3dpoints_plt = None
        self.chi2_error_plt = None
        self.timing_plt = None
        
        # To disable one of them just comment it out
        self.matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')  
        #self.info_3dpoints_plt = Mplot2d(xlabel='img id', ylabel='# points',title='info 3d points')      
        self.chi2_error_plt = Mplot2d(xlabel='img id', ylabel='error',title='mean chi2 error')
        self.timing_plt = Mplot2d(xlabel='img id', ylabel='s',title='timing')        
        
        self.last_processed_kf_img_id = -1
        
    def quit(self):
        if self.matched_points_plt is not None:
            self.matched_points_plt.quit()
        if self.info_3dpoints_plt is not None:
            self.info_3dpoints_plt.quit()    
        if self.chi2_error_plt is not None:
            self.chi2_error_plt.quit()
        if self.timing_plt is not None:
            self.timing_plt.quit()            
            
    def get_key(self):
        key = self.matched_points_plt.get_key() if self.matched_points_plt is not None else None
        if key == '' or key is None:
            key = self.info_3dpoints_plt.get_key() if self.info_3dpoints_plt is not None else None
        if key == '' or key is None:
            key = self.chi2_error_plt.get_key() if self.chi2_error_plt is not None else None
        if key == '' or key is None:
            key = self.timing_plt.get_key() if self.timing_plt is not None else None
        return key      

    def draw(self, img_id):
        try:
            # draw matching info
            if self.matched_points_plt is not None: 
                if self.slam.tracking.num_matched_kps is not None: 
                    matched_kps_signal = [img_id, self.slam.tracking.num_matched_kps]     
                    self.matched_points_plt.draw(matched_kps_signal,'# keypoint matches',color='r')                         
                if self.slam.tracking.num_inliers is not None: 
                    inliers_signal = [img_id, self.slam.tracking.num_inliers]                    
                    self.matched_points_plt.draw(inliers_signal,'# inliers',color='g')
                if self.slam.tracking.num_matched_map_points is not None: 
                    valid_matched_map_points_signal = [img_id, self.slam.tracking.num_matched_map_points]   # valid matched map points (in current pose optimization)                                       
                    self.matched_points_plt.draw(valid_matched_map_points_signal,'# matched map pts', color='b')  
                if self.slam.tracking.num_kf_ref_tracked_points is not None: 
                    kf_ref_tracked_points_signal = [img_id, self.slam.tracking.num_kf_ref_tracked_points]                    
                    self.matched_points_plt.draw(kf_ref_tracked_points_signal,'# $KF_{ref}$  tracked pts',color='c')   
                if self.slam.tracking.descriptor_distance_sigma is not None: 
                    descriptor_sigma_signal = [img_id, self.slam.tracking.descriptor_distance_sigma]                    
                    self.matched_points_plt.draw(descriptor_sigma_signal,'descriptor distance $\sigma_{th}$',color='k')                                                                  
                
            # draw info about 3D points management by local mapping 
            if self.info_3dpoints_plt is not None:
                if self.slam.local_mapping.last_processed_kf_img_id is not None and self.last_processed_kf_img_id != self.slam.local_mapping.last_processed_kf_img_id:
                    self.last_processed_kf_img_id = self.slam.local_mapping.last_processed_kf_img_id
                    print(f'last_processed_kf_img_id: {self.last_processed_kf_img_id}')                             
                    if self.slam.local_mapping.last_num_triangulated_points is not None:
                        num_triangulated_points_signal = [img_id, self.slam.local_mapping.last_num_triangulated_points]     
                        self.info_3dpoints_plt.draw(num_triangulated_points_signal,'# temporal triangulated pts',color='r') 
                    if self.slam.local_mapping.last_num_fused_points is not None:
                        num_fused_points_signal = [img_id, self.slam.local_mapping.last_num_fused_points]
                        self.info_3dpoints_plt.draw(num_fused_points_signal,'# fused pts',color='g')
                    if self.slam.local_mapping.last_num_culled_keyframes is not None:
                        num_culled_keyframes_signal = [img_id, self.slam.local_mapping.last_num_culled_keyframes]
                        self.info_3dpoints_plt.draw(num_culled_keyframes_signal,'# culled keyframes',color='b')
                    if self.slam.local_mapping.last_num_culled_points is not None:
                        num_culled_points_signal = [img_id, self.slam.local_mapping.last_num_culled_points]
                        self.info_3dpoints_plt.draw(num_culled_points_signal,'# culled pts',color='c')
                    if self.slam.tracking.last_num_static_stereo_map_points is not None: 
                        num_static_stereo_map_points_signal = [img_id, self.slam.tracking.last_num_static_stereo_map_points]
                        self.info_3dpoints_plt.draw(num_static_stereo_map_points_signal,'# static triangulated pts',color='k') 
                
            if self.chi2_error_plt is not None:
                if self.slam.tracking.mean_pose_opt_chi2_error is not None:
                    mean_pose_opt_chi2_error_signal = [img_id, self.slam.tracking.mean_pose_opt_chi2_error]
                    self.chi2_error_plt.draw(mean_pose_opt_chi2_error_signal,'pose opt chi2 error',color='r')
                if self.slam.local_mapping.mean_ba_chi2_error is not None:
                    mean_squared_reproj_error_signal = [img_id, self.slam.local_mapping.mean_ba_chi2_error]
                    self.chi2_error_plt.draw(mean_squared_reproj_error_signal,'LBA chi2 error',color='g')
                if self.slam.loop_closing is not None: 
                    if self.slam.loop_closing.mean_graph_chi2_error is not None:
                        mean_graph_chi2_error_signal = [img_id, self.slam.loop_closing.mean_graph_chi2_error]
                        self.chi2_error_plt.draw(mean_graph_chi2_error_signal,'graph chi2 error',color='b')
                    if self.slam.GBA.mean_squared_error.value >0:
                        mean_BA_chi2_error_signal = [img_id, self.slam.GBA.mean_squared_error.value]
                        self.chi2_error_plt.draw(mean_BA_chi2_error_signal,'GBA chi2 error',color='k')
                                        
            if self.timing_plt is not None:
                if self.slam.tracking.time_track is not None:
                    time_track_signal = [img_id, self.slam.tracking.time_track]
                    self.timing_plt.draw(time_track_signal,'tracking',color='r')
                if self.slam.local_mapping.time_local_mapping is not None:
                    time_local_mapping_signal = [img_id, self.slam.local_mapping.time_local_mapping]
                    self.timing_plt.draw(time_local_mapping_signal,'local mapping',color='g')  
                if self.slam.local_mapping.time_local_opt.last_elapsed:
                    time_LBA_signal = [img_id, self.slam.local_mapping.time_local_opt.last_elapsed]
                    self.timing_plt.draw(time_LBA_signal,'LBA',color='b')
                if self.slam.local_mapping.timer_triangulation.last_elapsed:
                    time_local_mapping_triangulation_signal = [img_id, self.slam.local_mapping.timer_triangulation.last_elapsed]
                    self.timing_plt.draw(time_local_mapping_triangulation_signal,'local mapping triangulation',color='k')
                if self.slam.local_mapping.timer_pts_culling.last_elapsed:
                    time_local_mapping_pts_culling_signal = [img_id, self.slam.local_mapping.timer_pts_culling.last_elapsed]
                    self.timing_plt.draw(time_local_mapping_pts_culling_signal,'local mapping pts culling',color='c')
                if self.slam.local_mapping.timer_kf_culling.last_elapsed:
                    time_local_mapping_kf_culling_signal = [img_id, self.slam.local_mapping.timer_kf_culling.last_elapsed]
                    self.timing_plt.draw(time_local_mapping_kf_culling_signal,'local mapping kf culling',color='m')
                if self.slam.local_mapping.timer_pts_fusion.last_elapsed:
                    time_local_mapping_pts_fusion_signal = [img_id, self.slam.local_mapping.timer_pts_fusion.last_elapsed]
                    self.timing_plt.draw(time_local_mapping_pts_fusion_signal,'local mapping pts fusion',color='y')
                if self.slam.loop_closing is not None:
                    if self.slam.loop_closing.timer.last_elapsed:
                        time_loop_closing_signal = [img_id, self.slam.loop_closing.timer.last_elapsed]
                        self.timing_plt.draw(time_loop_closing_signal,'loop closing',color=mcolors.CSS4_COLORS['darkgoldenrod'], marker='+')
                    if self.slam.loop_closing.time_loop_detection.value:
                        time_loop_detection_signal = [img_id, self.slam.loop_closing.time_loop_detection.value]
                        self.timing_plt.draw(time_loop_detection_signal,'loop detection',color=mcolors.CSS4_COLORS['slategrey'], marker='+')
                
        except Exception as e:
            Printer.red(f'SlamPlotDrawer: draw: encountered exception: {e}')
            traceback_details = traceback.format_exc()
            print(f'\t traceback details: {traceback_details}')
    

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
    
    viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
    if groundtruth is not None:
        gt_traj3d, gt_timestamps = groundtruth.getFull3dTrajectory()
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
    while dataset.isOk():
            
        if do_step:
            Printer.green('do step: ', do_step)
            
        if do_reset: 
            Printer.green('do reset: ', do_reset)
            slam.reset()
                        
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
                getchar()
                 
        if is_map_save:
            slam.save_map('map.json') 
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
