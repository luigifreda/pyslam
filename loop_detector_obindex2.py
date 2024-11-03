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

from loop_detector_base import LoopDetectorTaskType, LoopDetectKeyframeData, LoopDetectorTask, LoopDetectorOutput, LoopDetectorBase

import config
config.cfg.set_lib('pyobindex2')
import pyobindex2 as obindex2


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


class LoopDetectorOBIndex2(LoopDetectorBase): 
    def __init__(self, local_feature_manager=None):
        super().__init__()
        self.local_feature_manager = local_feature_manager        
        # Creating a new index of images
        self.index = obindex2.ImageIndex(16, 150, 4, obindex2.MERGE_POLICY_AND, True)               
        
    def run_task(self, task: LoopDetectorTask):
        print(f'LoopDetectorOBIndex2: running task {task.keyframe_data.id}, img_count = {self.img_count}, task_type = {task.task_type.name}')              
        keyframe = task.keyframe_data     
        img_id = keyframe.id
        
        if self.loop_detection_imgs is not None:       
            self.map_kf_img_id_to_img[keyframe.id] = keyframe.img
            self.loop_detection_imgs.reset()
            
        self.resize_similary_matrix_if_needed()            
        
        kps, des = keyframe.kps, keyframe.des 
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
                                                                         
        if task.task_type == LoopDetectorTaskType.LOOP_CLOSURE and self.img_count >= 1:
            # Search the query descriptors against the features in the index
            matches_feats = self.index.searchDescriptors(des_,2, 64)
        
            # Filter matches according to the ratio test
            matches = []
            for m in matches_feats: # vector of pairs of tuples (queryIdx, trainIdx, imgIdx, distance)
                if m[0][3] < m[1][3] * 0.8:
                    matches.append(m[0])
                    
            if len(matches)>0:
                #  Look for similar images according to the good matches found
                image_matches = self.index.searchImages(des_, matches, True)
                count_valid_candidates = 0
                for i,m in enumerate(image_matches):
                    other_img_id = self.map_img_count_to_kf_img_id[m.image_id]
                    other_img_count=m.image_id   
                    self.update_similarity_matrix(score=m.score, img_count=self.img_count, other_img_count=other_img_count)                     
                    if abs(other_img_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure and \
                        other_img_id not in task.connected_keyframes_ids:
                        candidate_idxs.append(other_img_id)
                        candidate_scores.append(m.score)
                        count_valid_candidates += 1                                                             
                        self.update_loop_closure_imgs(score=m.score, other_img_id = other_img_id) 
                        if count_valid_candidates >= kMaxResultsForLoopClosure: 
                            break                             
                    
            self.draw_loop_detection_imgs(keyframe.img, img_id, detection_output)  
                
            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores 
            detection_output.covisible_ids = [cov_kf.id for cov_kf in task.covisible_keyframes_data]
            detection_output.covisible_gdes_vecs = [cov_kf.g_des.toVec() if cov_kf.g_des is not None else None for cov_kf in task.covisible_keyframes_data]                     
        
            # Add the image to the index. Matched descriptors are used 
            # to update the index and the remaining ones are added as new visual words
            self.index.addImage(self.img_count, kps_, des_, matches)
            
        else: 
            # if we just wanted to compute the global descriptor (LoopDetectorTaskType.COMPUTE_GLOBAL_DES), we don't have to do anything            
            self.index.addImage(self.img_count, kps_, des_)              
                                        
        # Reindex features every 500 images
        if self.img_count % 250 == 0 and self.img_count>0:
            print("------ Rebuilding indices ------")
            self.index.rebuild() 
        
        self.img_count += 1      
        return detection_output          
            
