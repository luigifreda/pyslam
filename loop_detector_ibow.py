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


import os
import time
import math 
import numpy as np
import cv2
from enum import Enum

from utils_sys import getchar, Printer 
from utils_features import transform_float_to_binary_descriptor

from parameters import Parameters
from feature_types import FeatureInfo

from timer import TimerFps

from keyframe import KeyFrame
from frame import Frame, FrameShared

import traceback
from loop_detector_base import LoopDetectorTaskType, LoopDetectKeyframeData, LoopDetectorTask, LoopDetectorOutput, LoopDetectorBase

import config
config.cfg.set_lib('pyibow')
import pyibow as ibow


kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kDataFolder = kRootFolder + '/data'


if Parameters.kLoopClosingDebugAndPrintToFile:
    from loop_detector_base import print


class LoopDetectorIBow(LoopDetectorBase): 
    def __init__(self, local_feature_manager=None):
        super().__init__()
        self.local_feature_manager = local_feature_manager        
        self.lc_detector_parameters = ibow.LCDetectorParams()
        self.lc_detector_parameters.p = 100 # default in ibow: 250
        print(f'LoopDetectorIBow: min number of images to start detecting loops: {self.lc_detector_parameters.p}')    
        self.lc_detector = ibow.LCDetector(self.lc_detector_parameters)
        
    def reset(self):
        LoopDetectorBase.reset(self)
        self.lc_detector.clear()
        
    def run_task(self, task: LoopDetectorTask):
        print(f'LoopDetectorIBow: running task {task.keyframe_data.id}, img_count = {self.img_count}, task_type = {task.task_type.name}')              
        keyframe = task.keyframe_data     
        img_id = keyframe.id
        
        if self.loop_detection_imgs is not None:       
            self.map_kf_img_id_to_img[keyframe.id] = keyframe.img
            self.loop_detection_imgs.reset()
            
        self.resize_similary_matrix_if_needed()            
        
        kps, des = keyframe.kps, keyframe.des 
        #print(f'LoopDetectorIBow: kps = {len(kps)}, des = {des.shape}')
        if len(kps)>0 and len(kps[0])>2:
            kps_ = [(kp[0], kp[1], kp[2], kp[3], kp[4], kp[5]) for kp in kps]  # tuple_x_y_size_angle_response_octave
        else:
            # kp.response is not actually used
            #kps_ = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps]  # tuple_x_y_size_angle_response_octave            
            kps_ = [(kp[0], kp[1], keyframe.sizes[i], keyframe.angles[i], 1, keyframe.octaves[i]) for i,kp in enumerate(kps)]  # tuple_x_y_size_angle_response_octave
        des_ = des
        
        # if we are not using a binary descriptr then we conver the float descriptors to binary        
        if FeatureInfo.norm_type[FrameShared.feature_manager.descriptor_type] != cv2.NORM_HAMMING:
            des_ = transform_float_to_binary_descriptor(des)
                                    
        candidate_idxs = []
        candidate_scores = []
        g_des = None
                     
        # the img_ids are mapped to img_counts (entry ids) inside the database management
        self.map_img_count_to_kf_img_id[self.img_count] = img_id      
                                     
        detection_output = LoopDetectorOutput(task_type=task.task_type, g_des_vec=g_des, img_id=img_id, img=keyframe.img)                                              
                          
        if task.task_type == LoopDetectorTaskType.RELOCALIZATION:
            result = self.lc_detector.process_without_pushing(self.img_count, kps_, des_)
            other_img_count = result.train_id
            other_img_id = self.map_img_count_to_kf_img_id[other_img_count]
                
            if result.isLoop():
                candidate_idxs.append(other_img_id)
                candidate_scores.append(result.score)                                       
                                  
            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores 
                                              
        else:                                                                         
            result = self.lc_detector.process(self.img_count, kps_, des_)
            other_img_count = result.train_id
            other_img_id = self.map_img_count_to_kf_img_id[other_img_count]
                
            self.update_similarity_matrix(score=result.score, img_count=self.img_count, other_img_count=other_img_count)

            if result.isLoop():
                if abs(other_img_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure and \
                    other_img_id not in task.connected_keyframes_ids: 
                    candidate_idxs.append(other_img_id)
                    candidate_scores.append(result.score)                                                        
                    self.update_loop_closure_imgs(score=result.score, other_img_id = other_img_id)                 

            self.draw_loop_detection_imgs(keyframe.img, img_id, detection_output)  
                
            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores 
            detection_output.covisible_ids = [cov_kf.id for cov_kf in task.covisible_keyframes_data]
            detection_output.covisible_gdes_vecs = [cov_kf.g_des.toVec() if cov_kf.g_des is not None else None for cov_kf in task.covisible_keyframes_data]
            
        
        if result.status == ibow.LCDetectorStatus.LC_DETECTED:
            # NOTE: it's normal to get zero inliers in some cases where the loop is detected, for instance: 
            #       consecutive_loops_ > min_consecutive_loops_ and island.overlaps(last_lc_island_)
            print(f'LoopDetectorIBow: Loop detected: {result.train_id}, #inliers: {result.inliers}, score: {result.score}')
        elif result.status == ibow.LCDetectorStatus.LC_NOT_DETECTED:
            print('LoopDetectorIBow: No loop found')
        elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_IMAGES:
            print(f'LoopDetectorIBow: Not enough images to found a loop, min number of processed images for loop: {self.lc_detector_parameters.p}, num_pushed_images: {self.lc_detector.num_pushed_images()}')
        elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_ISLANDS:
            print('LoopDetectorIBow: Not enough islands to found a loop')
        elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_INLIERS:
            print('LoopDetectorIBow: Not enough inliers')
        elif result.status == ibow.LCDetectorStatus.LC_TRANSITION:
            print('LoopDetectorIBow: Transitional loop closure')
        else:
            print('LoopDetectorIBow: No status information')
                         

        if task.task_type != LoopDetectorTaskType.RELOCALIZATION:
            # NOTE: with relocalization we don't need to increment the img_count since we don't add frames to database        
            self.img_count += 1       
              
        return detection_output   
            