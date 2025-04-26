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
import numpy as np
import json

import platform 

from config import Config

from slam import Slam, SlamState
from slam_plot_drawer import SlamPlotDrawer
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset_factory import dataset_factory
from dataset_types import DatasetType, SensorType
from trajectory_writer import TrajectoryWriter

from viewer3D import Viewer3D
from utils_sys import getchar, Printer, force_kill_all_and_exit
from utils_img import ImgWriter
from utils_eval import eval_ate
from utils_geom_trajectory import find_poses_associations
from utils_colors import GlColors
from utils_serialization import SerializableEnumEncoder

from feature_tracker_configs import FeatureTrackerConfigs

from loop_detector_configs import LoopDetectorConfigs

from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
from utils_depth import img_from_depth, filter_shadow_points

from semantic_estimator_factory import semantic_estimator_factory, SemanticEstimatorType
from utils_semantics import labels_to_image

from config_parameters import Parameters  

from rerun_interface import Rerun

from datetime import datetime
import traceback

import argparse


datetime_string = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    
def draw_associated_cameras(viewer3D, assoc_est_poses, assoc_gt_poses, T_gt_est):       
    T_est_gt = np.linalg.inv(T_gt_est)
    scale = np.mean([np.linalg.norm(T_est_gt[i, :3]) for i in range(3)])
    R_est_gt = T_est_gt[:3, :3]/scale # we need a pure rotation to avoid camera scale changes
    assoc_gt_poses_aligned = [np.eye(4) for i in range(len(assoc_gt_poses))]
    for i in range(len(assoc_gt_poses)):
        assoc_gt_poses_aligned[i][:3,3] = T_est_gt[:3, :3] @ assoc_gt_poses[i][:3, 3] + T_est_gt[:3, 3]
        assoc_gt_poses_aligned[i][:3,:3] = R_est_gt @ assoc_gt_poses[i][:3,:3]
    viewer3D.draw_cameras([assoc_est_poses, assoc_gt_poses_aligned], [GlColors.kCyan, GlColors.kMagenta])    


if __name__ == "__main__":   
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--config_path', type=str, default=None, help='Optional path for custom configuration file')
    parser.add_argument('--no_output_date', action='store_true', help='Do not append date to output directory')
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')    
    args = parser.parse_args()
    
    if args.config_path:
        config = Config(args.config_path) # use the custom configuration path file
    else:
        config = Config()
        
    if args.no_output_date:
        print('Not appending date to output directory')
        datetime_string = None

    dataset = dataset_factory(config)
    is_monocular=(dataset.sensor_type==SensorType.MONOCULAR)    
    num_total_frames = dataset.num_frames

    online_trajectory_writer = None
    final_trajectory_writer = None
    if config.trajectory_saving_settings['save_trajectory']:
        trajectory_online_file_path, trajectory_final_file_path, trajectory_saving_base_path = config.get_trajectory_saving_paths(datetime_string)
        online_trajectory_writer = TrajectoryWriter(format_type=config.trajectory_saving_settings['format_type'], filename=trajectory_online_file_path)
        final_trajectory_writer = TrajectoryWriter(format_type=config.trajectory_saving_settings['format_type'], filename=trajectory_final_file_path)
    metrics_save_dir = trajectory_saving_base_path
        
    groundtruth = groundtruth_factory(config.dataset_settings)

    camera = PinholeCamera(config)

    # Select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    # WARNING: At present, SLAM does not support LOFTR and other "pure" image matchers (further details in the commenting notes about LOFTR in feature_tracker_configs.py).
    feature_tracker_config = FeatureTrackerConfigs.ORB2
        
    # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing. 
    # LoopDetectorConfigs: DBOW2, DBOW2_INDEPENDENT, DBOW3, DBOW3_INDEPENDENT, IBOW, OBINDEX2, VLAD, HDC_DELF, SAD, ALEXNET, NETVLAD, COSPLACE, EIGENPLACES  etc.
    # NOTE: under mac, the boost/text deserialization used by DBOW2 and DBOW3 may be very slow.
    loop_detection_config = LoopDetectorConfigs.DBOW3

    # Override the feature tracker and loop detector configuration from the `settings` file
    if config.feature_tracker_config_name is not None:  # Check if we set `FeatureTrackerConfig.name` in the `settings` file 
        feature_tracker_config = FeatureTrackerConfigs.get_config_from_name(config.feature_tracker_config_name) # Override the feature tracker configuration from the `settings` file
    if config.num_features_to_extract > 0:             # Check if we set `FeatureTrackerConfig.nFeatures` in the `settings` file 
        Printer.yellow('Setting feature_tracker_config num_features from settings: ',config.num_features_to_extract)
        feature_tracker_config['num_features'] = config.num_features_to_extract  # Override the number of features from the `settings` file
    if config.loop_detection_config_name is not None:  # Check if we set `LoopDetectorConfig.name` in the `settings` file 
        loop_detection_config = LoopDetectorConfigs.get_config_from_name(config.loop_detection_config_name) # Override the loop detector configuration from the `settings` file
        
    Printer.green('feature_tracker_config: ',json.dumps(feature_tracker_config, indent=4, cls=SerializableEnumEncoder))          
    Printer.green('loop_detection_config: ',json.dumps(loop_detection_config, indent=4, cls=SerializableEnumEncoder))
        
    # Select your depth estimator in the front-end (EXPERIMENTAL, WIP)
    depth_estimator = None
    if Parameters.kUseDepthEstimatorInFrontEnd:
        Parameters.kVolumetricIntegrationUseDepthEstimator = False  # Just use this depth estimator in the front-end (This is not a choice, we are imposing it for avoiding computing the depth twice)
        # Select your depth estimator (see the file depth_estimator_factory.py)
        # DEPTH_ANYTHING_V2, DEPTH_PRO, DEPTH_RAFT_STEREO, DEPTH_SGBM, etc.
        depth_estimator_type = DepthEstimatorType.DEPTH_PRO
        max_depth = 20
        depth_estimator = depth_estimator_factory(depth_estimator_type=depth_estimator_type, max_depth=max_depth,
                                                  dataset_env_type=dataset.environmentType(), camera=camera) 
        Printer.green(f'Depth_estimator_type: {depth_estimator_type.name}, max_depth: {max_depth}')       

    semantic_estimator = None
    if Parameters.kUseSemanticEstimator:
        semantic_estimator_type = SemanticEstimatorType.SEGFORMER
        semantic_estimator = semantic_estimator_factory(semantic_estimator_type=semantic_estimator_type)
        Printer.green(f'Semantic_estimator_type: {semantic_estimator_type.name}')

    # create SLAM object
    slam = Slam(camera, feature_tracker_config, 
                loop_detection_config, dataset.sensorType(), 
                environment_type=dataset.environmentType(), 
                config=config,
                headless=args.headless)
    slam.set_viewer_scale(dataset.scale_viewer_3d)
    time.sleep(1) # to show initial messages 
    
    # load system state if requested         
    if config.system_state_load: 
        slam.load_system_state(config.system_state_folder_path)
        viewer_scale = slam.viewer_scale() if slam.viewer_scale()>0 else 0.1  # 0.1 is the default viewer scale
        print(f'viewer_scale: {viewer_scale}')
        slam.set_tracking_state(SlamState.INIT_RELOCALIZE)

    if args.headless:
        viewer3D = None
        plot_drawer = None        
    else:
        viewer3D = Viewer3D(scale=dataset.scale_viewer_3d)
        viewer3D.set_semantics_rgb_map(semantic_estimator.semantics_rgb_map)
        plot_drawer = SlamPlotDrawer(slam, viewer3D)
        img_writer = ImgWriter(font_scale=0.7)
        if False:
            cv2.namedWindow('Camera', cv2.WINDOW_NORMAL) # to make it resizable if needed        
    
    if groundtruth:
        gt_traj3d, gt_poses, gt_timestamps = groundtruth.getFull6dTrajectory()
        if viewer3D:
            viewer3D.set_gt_trajectory(gt_traj3d, gt_timestamps, align_with_scale=is_monocular)
            
    do_step = False          # proceed step by step on GUI 
    do_reset = False         # reset on GUI 
    is_paused = False        # pause/resume on GUI 
    is_map_save = False      # save map on GUI
    is_bundle_adjust = False # bundle adjust on GUI
    is_viewer_closed = False # viewer GUI was closed
    
    key = None
    key_cv = None
    
    num_tracking_lost = 0
    num_frames = 0
            
    img_id = 0  #210, 340, 400, 770   # you can start from a desired frame id if needed 
    while not is_viewer_closed:
        
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
                #TODO(@dvdmc): Get semantics if GT
                semantic_prediction = None

            if img is not None:
                timestamp = dataset.getTimestamp()          # get current timestamp 
                next_timestamp = dataset.getNextTimestamp() # get next timestamp 
                frame_duration = next_timestamp-timestamp if (timestamp is not None and next_timestamp is not None) else -1

                print(f'image: {img_id}, timestamp: {timestamp}, duration: {frame_duration}') 
                
                time_start = None 
                if img is not None:
                    time_start = time.time()    
                    
                    if depth is None and depth_estimator:
                        depth_prediction, pts3d_prediction = depth_estimator.infer(img, img_right)
                        if Parameters.kDepthEstimatorRemoveShadowPointsInFrontEnd:
                            depth = filter_shadow_points(depth_prediction)
                        else: 
                            depth = depth_prediction
                        
                        if not args.headless:
                            depth_img = img_from_depth(depth_prediction, img_min=0, img_max=50)
                            cv2.imshow("depth prediction", depth_img)
            
                    if semantic_prediction is None and semantic_estimator:
                        semantic_prediction = semantic_estimator.infer(img)           
                        if not args.headless:
                            semantic_color_img = labels_to_image(semantic_prediction, semantic_estimator.semantics_rgb_map, bgr=True)
                            cv2.imshow("semantic prediction", semantic_color_img)

                    #TODO(@dvdmc): Add semantics            
                    slam.track(img, img_right, depth, img_id, timestamp, semantic_prediction)  # main SLAM function 
                                    
                    # 3D display (map display)
                    if viewer3D:
                        #TODO(@dvdmc): add semantics
                        viewer3D.draw_slam_map(slam)

                    if not args.headless:
                        #TODO(@dvdmc): add semantics
                        img_draw = slam.map.draw_feature_trails(img)
                        img_writer.write(img_draw, f'id: {img_id}', (30, 30))
                        # 2D display (image display)
                        cv2.imshow('Camera', img_draw)
                    
                    # draw 2d plots
                    if plot_drawer:
                        plot_drawer.draw(img_id)
                        
                if online_trajectory_writer is not None and slam.tracking.cur_R is not None and slam.tracking.cur_t is not None:
                    online_trajectory_writer.write_trajectory(slam.tracking.cur_R, slam.tracking.cur_t, timestamp)
                    
                if time_start is not None: 
                    duration = time.time()-time_start
                    if(frame_duration > duration):
                        time.sleep(frame_duration-duration) 
                    
                img_id += 1 
                num_frames += 1
            else: 
                time.sleep(0.1)     # img is None
                if args.headless:
                    break # exit from the loop if headless
                
            # 3D display (map display)
            if viewer3D:
                #TODO(@dvdmc): add semantics
                viewer3D.draw_dense_map(slam)  
                              
        else:
            time.sleep(0.1)     # pause or do step on GUI                           
        
        if not args.headless:
            # get keys 
            key = plot_drawer.get_key() if plot_drawer else None
            key_cv = cv2.waitKey(1) & 0xFF   
            
            # manage SLAM states 
            if slam.tracking.state==SlamState.LOST:
                #key_cv = cv2.waitKey(0) & 0xFF   # wait key for debugging
                key_cv = cv2.waitKey(500) & 0xFF   
                
        if slam.tracking.state==SlamState.LOST:
            num_tracking_lost += 1                              
                
        # manage interface infos  
        if is_map_save:
            slam.save_system_state(config.system_state_folder_path)
            dataset.save_info(config.system_state_folder_path)
            groundtruth.save(config.system_state_folder_path)
            Printer.blue('\nuncheck pause checkbox on GUI to continue...\n')    
            
        if is_bundle_adjust:
            slam.bundle_adjust()    
            Printer.blue('\nuncheck pause checkbox on GUI to continue...\n')
                            
        if viewer3D:
            
            if not is_paused and viewer3D.is_paused():  # when a pause is triggered
                est_poses, timestamps, ids = slam.get_final_trajectory()
                assoc_timestamps, assoc_est_poses, assoc_gt_poses = find_poses_associations(timestamps, est_poses, gt_timestamps, gt_poses)
                ape_stats, T_gt_est = eval_ate(poses_est=assoc_est_poses, poses_gt=assoc_gt_poses, frame_ids=ids, 
                        curr_frame_id=img_id, is_final=False, is_monocular=is_monocular, save_dir=None)
                Printer.green(f"EVO stats: {json.dumps(ape_stats, indent=4)}")
                #draw_associated_cameras(viewer3D, assoc_est_poses, assoc_gt_poses, T_gt_est)
                        
            is_paused = viewer3D.is_paused()    
            is_map_save = viewer3D.is_map_save() and is_map_save == False 
            is_bundle_adjust = viewer3D.is_bundle_adjust() and is_bundle_adjust == False
            do_step = viewer3D.do_step() and do_step == False  
            do_reset = viewer3D.reset() and do_reset == False
            is_viewer_closed = viewer3D.is_closed()
                               
        if key == 'q' or (key_cv == ord('q') or key_cv == 27):    # press 'q' or ESC for quitting
            break
            
    # here we save the online estimated trajectory
    online_trajectory_writer.close_file()
    
    # compute metrics on the estimated final trajectory 
    try: 
        est_poses, timestamps, ids = slam.get_final_trajectory()
        is_final = not dataset.isOk()
        assoc_timestamps, assoc_est_poses, assoc_gt_poses = find_poses_associations(timestamps, est_poses, gt_timestamps, gt_poses)        
        ape_stats, T_gt_est = eval_ate(poses_est=assoc_est_poses, poses_gt=assoc_gt_poses, frame_ids=ids, 
                 curr_frame_id=img_id, is_final=is_final, is_monocular=is_monocular, save_dir=metrics_save_dir)
        Printer.green(f"EVO stats: {json.dumps(ape_stats, indent=4)}")
        
        #TODO(@dvdmc): add semantics evaluation

        if final_trajectory_writer:
            final_trajectory_writer.write_full_trajectory(est_poses, timestamps)
            final_trajectory_writer.close_file()
            
        other_metrics_file_path = os.path.join(metrics_save_dir, 'other_metrics_info.txt')
        with open(other_metrics_file_path, 'w') as f:
            f.write(f'num_total_frames: {num_total_frames}\n')
            f.write(f'num_processed_frames: {num_frames}\n')
            f.write(f'num_lost_frames: {num_tracking_lost}\n')
            f.write(f'percent_lost: {num_tracking_lost/num_total_frames*100:.2f}\n')
        
    except Exception as e:
        print('Exception while computing metrics: ', e)
        print(f'traceback: {traceback.format_exc()}')

    # close stuff 
    slam.quit()
    if plot_drawer:
        plot_drawer.quit()         
    if viewer3D:
        viewer3D.quit()   
    
    if not args.headless:
        cv2.destroyAllWindows()

    if args.headless:
        force_kill_all_and_exit(verbose=False) # just in case