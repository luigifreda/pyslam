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

from config import Config #, dump_config_to_json

from semantic_mapping import SemanticMappingType
from semantic_types import SemanticFeatureType
from semantic_mapping_configs import SemanticMappingConfigs
from semantic_mapping_shared import SemanticMappingShared
from semantic_utils import SemanticDatasetType
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

from config_parameters import Parameters  

from rerun_interface import Rerun

from datetime import datetime
import traceback

import argparse

from matplotlib import pyplot as plt 


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


def evaluate_semantic_mapping(slam, dataset):
    if Parameters.kDoSemanticMapping and slam.semantic_mapping.semantic_dataset_type != SemanticDatasetType.FEATURE_SIMILARITY and dataset.has_gt_semantics and slam.semantic_mapping.semantic_mapping_type == SemanticMappingType.DENSE:
            Printer.green("Evaluating semantic mapping...")
            # Get all the KFs
            keyframes = slam.map.get_keyframes()
            Printer.green(f"Number of keyframes: {len(keyframes)}")

            labels_2d = []
            labels_3d = []
            gt_labels = []
            total_mps = 0
            # Get all the final MPs that project on it
            for kf in keyframes:
                if kf.kps_sem is None:
                    Printer.yellow(f"Keyframe {kf.id} has no semantics!")
                    continue
                if kf.points is None:
                    Printer.yellow(f"Keyframe {kf.id} has no points!")
                    continue

                semantic_gt = dataset.getSemanticGroundTruth(kf.id)

                # Get the semantic_des of projected points
                points = kf.get_points()
                total_mps += len(points)
                # Get the per-frame gt semantic label for projected MPs
                for idx, kp in enumerate(kf.kps):
                    if points[idx] is not None and points[idx].semantic_des is not None and kf.kps_sem[idx] is not None:
                        gt_kf_label = semantic_gt[int(kp[1]), int(kp[0])]
                        # Filter out ignore-labels
                        if dataset.ignore_label != None and gt_kf_label == dataset.ignore_label:
                            continue
                        gt_labels.append(gt_kf_label)
                        if SemanticMappingShared.semantic_feature_type == SemanticFeatureType.LABEL:
                            labels_2d.append(kf.kps_sem[idx])
                            labels_3d.append(points[idx].semantic_des)
                        elif SemanticMappingShared.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
                            labels_2d.append(np.argmax(kf.kps_sem[idx]))
                            labels_3d.append(np.argmax(points[idx].semantic_des))

                
                # For debugging:
                # Recover image
                # rgb_img = dataset.getImageColor(kf.id)
                # cv2.imshow('rgb', rgb_img)
                # semantic_gt_color = SemanticMappingShared.sem_img_to_rgb(semantic_gt, bgr=True)
                # cv2.imshow('semantic_gt', semantic_gt_color)
                # Get the predicted semantic label for the MP projection (baseline)
                # predicted_semantics = slam.semantic_mapping.semantic_segmentation.infer(rgb_img)
                # print(f"Predicted labels: {np.unique(predicted_semantics)}")
                # predicted_semantics_color = SemanticMappingShared.sem_img_to_rgb(predicted_semantics, bgr=True)
                # cv2.imshow('predicted_semantics', predicted_semantics_color)
                # cv2.waitKey(0) 
            Printer.orange(f"Number of projected MPs: {len(labels_2d)}")
            Printer.orange(f"Number of projected MPs (3D): {len(labels_3d)}")
            Printer.orange(f"Number of GT MPs: {len(gt_labels)}")
            Printer.orange(f"Number of evaluated MPs: {total_mps}")
            Printer.orange(f"Number of evaluated KFs: {len(keyframes)}")
            from sklearn.metrics import (
                classification_report,
                accuracy_score,
                confusion_matrix,
                ConfusionMatrixDisplay,
                precision_recall_fscore_support
            )

            # Class labels and names
            num_classes = dataset.num_labels
            labels_range = range(num_classes)
            labels_names = [str(i) for i in labels_range]

            # Determine which labels are actually present in the GT
            present_labels = sorted(set(gt_labels))  # list of int
            Printer.blue(f"Evaluating only on present GT labels: {present_labels}")

            # --- Baseline (2D) ---
            confusion_matrix_base = confusion_matrix(gt_labels, labels_2d, labels=labels_range)
            overall_accuracy_2d = accuracy_score(gt_labels, labels_2d)
            Printer.green(f"Overall Accuracy 2D: {overall_accuracy_2d:.4f}")

            # Macro average (only on present labels)
            report_2d = classification_report(
                gt_labels,
                labels_2d,
                labels=present_labels,
                zero_division=0,
                output_dict=True
            )
            macro_avg_2d = report_2d["macro avg"]
            Printer.green(f"2D Macro Avg: precision={macro_avg_2d['precision']:.4f}, recall={macro_avg_2d['recall']:.4f}, f1-score={macro_avg_2d['f1-score']:.4f}")

            # Micro average
            precision_2d, recall_2d, f1_2d, _ = precision_recall_fscore_support(gt_labels, labels_2d, average='micro', zero_division=0)
            Printer.green(f"2D Micro Avg: precision={precision_2d:.4f}, recall={recall_2d:.4f}, f1-score={f1_2d:.4f}")

            # Confusion matrix - 2D
            cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_base, display_labels=labels_names)
            fig, ax = plt.subplots(figsize=(24, 18))
            cm_display.plot(ax=ax, xticks_rotation=90)
            plt.savefig(os.path.join(metrics_save_dir, 'confusion_matrix_est2d.png'), dpi=300)

            # --- 3D Projection ---
            confusion_matrix_proj = confusion_matrix(gt_labels, labels_3d, labels=labels_range)
            overall_accuracy_3d = accuracy_score(gt_labels, labels_3d)
            Printer.green(f"Overall Accuracy 3D: {overall_accuracy_3d:.4f}")

            # Macro average (only on present labels)
            report_3d = classification_report(
                gt_labels,
                labels_3d,
                labels=present_labels,
                zero_division=0,
                output_dict=True
            )
            macro_avg_3d = report_3d["macro avg"]
            Printer.green(f"3D Macro Avg: precision={macro_avg_3d['precision']:.4f}, recall={macro_avg_3d['recall']:.4f}, f1-score={macro_avg_3d['f1-score']:.4f}")

            # Micro average
            precision_3d, recall_3d, f1_3d, _ = precision_recall_fscore_support(gt_labels, labels_3d, average='micro', zero_division=0)
            Printer.green(f"3D Micro Avg: precision={precision_3d:.4f}, recall={recall_3d:.4f}, f1-score={f1_3d:.4f}")

            # Confusion matrix - 3D
            cm_display = ConfusionMatrixDisplay(confusion_matrix=confusion_matrix_proj, display_labels=labels_names)
            fig, ax = plt.subplots(figsize=(24, 18))
            cm_display.plot(ax=ax, xticks_rotation=90)
            plt.savefig(os.path.join(metrics_save_dir, 'confusion_matrix_est3d.png'), dpi=300)

            semantic_metrics_file_path = os.path.join(metrics_save_dir, 'semantic_metrics_info.txt')
            with open(semantic_metrics_file_path, 'w') as f:
                f.write("Evaluated labels: " + str(present_labels) + "\n")
                f.write(f"Feature type: {slam.semantic_mapping.semantic_feature_type}\n")
                f.write(f"Number of KFs: {len(keyframes)}\n")
                f.write(f"Number of MPs: {total_mps}\n")
                f.write(f"Number of GT labels {len(gt_labels)}\n")
                f.write(f"Number of estimated labels 2D: {len(labels_2d)}\n")
                f.write(f"Number of estimated labels 3D: {len(labels_3d)}\n")
                # --- 2D Metrics ---
                f.write("=== 2D Semantic Evaluation ===\n")
                f.write(f"Accuracy: {overall_accuracy_2d:.4f}\n")
                f.write(f"Micro Precision: {precision_2d:.4f}\n")
                f.write(f"Micro Recall:    {recall_2d:.4f}\n")
                f.write(f"Micro F1-score:  {f1_2d:.4f}\n")
                f.write(f"Macro Precision: {macro_avg_2d['precision']:.4f}\n")
                f.write(f"Macro Recall:    {macro_avg_2d['recall']:.4f}\n")
                f.write(f"Macro F1-score:  {macro_avg_2d['f1-score']:.4f}\n\n")

                # --- 3D Metrics ---
                f.write("=== 3D Semantic Evaluation ===\n")
                f.write(f"Accuracy: {overall_accuracy_3d:.4f}\n")
                f.write(f"Micro Precision: {precision_3d:.4f}\n")
                f.write(f"Micro Recall:    {recall_3d:.4f}\n")
                f.write(f"Micro F1-score:  {f1_3d:.4f}\n")
                f.write(f"Macro Precision: {macro_avg_3d['precision']:.4f}\n")
                f.write(f"Macro Recall:    {macro_avg_3d['recall']:.4f}\n")
                f.write(f"Macro F1-score:  {macro_avg_3d['f1-score']:.4f}\n")
                

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
    # LoopDetectorConfigs: DBOW2, DBOW2_INDEPENDENT, DBOW3, DBOW3_INDEPENDENT, IBOW, OBINDEX2, VLAD, HDC_DELF, SAD, ALEXNET, NETVLAD, COSPLACE, EIGENPLACES, MEGALOC  etc.
    # NOTE: under mac, the boost/text deserialization used by DBOW2 and DBOW3 may be very slow.
    loop_detection_config = LoopDetectorConfigs.DBOW3

    # Select your semantic mapping configuration (see the file semantic_mapping_configs.py). Set it to None to disable semantic mapping.
    semantic_mapping_config = SemanticMappingConfigs.get_config_from_slam_dataset(dataset.type)
    
    # Override the feature tracker and loop detector configuration from the `settings` file
    if config.feature_tracker_config_name is not None:  # Check if we set `FeatureTrackerConfig.name` in the `settings` file 
        feature_tracker_config = FeatureTrackerConfigs.get_config_from_name(config.feature_tracker_config_name) # Override the feature tracker configuration from the `settings` file
    if config.num_features_to_extract > 0:             # Check if we set `FeatureTrackerConfig.nFeatures` in the `settings` file 
        Printer.yellow('Setting feature_tracker_config num_features from settings: ',config.num_features_to_extract)
        feature_tracker_config['num_features'] = config.num_features_to_extract  # Override the number of features from the `settings` file
    if config.loop_detection_config_name is not None:  # Check if we set `LoopDetectorConfig.name` in the `settings` file 
        loop_detection_config = LoopDetectorConfigs.get_config_from_name(config.loop_detection_config_name) # Override the loop detector configuration from the `settings` file
    if config.semantic_mapping_config_name is not None:  # Check if we set `SemanticMappingConfig.name` in the `settings` file. It is recommended to load semantics from the slam dataset name instead
        semantic_mapping_config = SemanticMappingConfigs.get_config_from_name(config.semantic_mapping_config_name) # Override the semantic mapping configuration from the `settings` file
    
    Printer.green('feature_tracker_config: ',json.dumps(feature_tracker_config, indent=4, cls=SerializableEnumEncoder))          
    Printer.green('loop_detection_config: ',json.dumps(loop_detection_config, indent=4, cls=SerializableEnumEncoder))
    if Parameters.kDoSemanticMapping:
        Printer.green('semantic_mapping_config: ',json.dumps(semantic_mapping_config, indent=4, cls=SerializableEnumEncoder))
    config.feature_tracker_config = feature_tracker_config
    config.loop_detection_config = loop_detection_config
    config.semantic_mapping_config = semantic_mapping_config

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

    # TODO(dvdmc): I did not manage to serialize the SerializableEnums for some reason.
    # dump_config_to_json(config, os.path.join(metrics_save_dir, 'config.json'))

    # create SLAM object
    slam = Slam(camera, feature_tracker_config, 
                loop_detection_config, semantic_mapping_config, 
                dataset.sensorType(), 
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
            
                    slam.track(img, img_right, depth, img_id, timestamp)  # main SLAM function 
                                    
                    # 3D display (map display)
                    if viewer3D:
                        viewer3D.draw_slam_map(slam)

                    if not args.headless:
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
                #TODO(dvdmc): add semantics
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

        if final_trajectory_writer:
            final_trajectory_writer.write_full_trajectory(est_poses, timestamps)
            final_trajectory_writer.close_file()
            
        other_metrics_file_path = os.path.join(metrics_save_dir, 'other_metrics_info.txt')
        with open(other_metrics_file_path, 'w') as f:
            f.write(f'num_total_frames: {num_total_frames}\n')
            f.write(f'num_processed_frames: {num_frames}\n')
            f.write(f'num_lost_frames: {num_tracking_lost}\n')
            f.write(f'percent_lost: {num_tracking_lost/num_total_frames*100:.2f}\n')
        
        evaluate_semantic_mapping(slam, dataset)
        
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