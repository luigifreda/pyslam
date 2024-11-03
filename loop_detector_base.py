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
from utils_img import float_to_color, convert_float_to_colored_uint8_image, LoopCandidateImgs

from parameters import Parameters
from feature_types import FeatureInfo

from timer import TimerFps

from keyframe import KeyFrame
from frame import Frame

import traceback
import torch.multiprocessing as mp
import logging
import sys
from utils_sys import Printer, getchar, Logging



kVerbose = True
kPrintTrackebackDetails = True 

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kDataFolder = kRootFolder + '/data'
kOrbVocabFile = kDataFolder + '/ORBvoc.txt'


if kVerbose:
    if Parameters.kLoopClosingDebugAndPrintToFile:
        # redirect the prints of local mapping to the file local_mapping.log 
        # you can watch the output in separate shell by running:
        # $ tail -f loop_detecting.log 
        import builtins as __builtin__
        logging_file=open('loop_detecting.log','w')
        def print(*args, **kwargs):
            return __builtin__.print(*args,**kwargs,file=logging_file,flush=True)
else:
    def print(*args, **kwargs):
        return

        
class LoopDetectorTaskType(Enum):
    NONE = 0
    COMPUTE_GLOBAL_DES = 2
    LOOP_CLOSURE = 3
    RELOCALIZATION = 4


# keyframe (pickable) data that are needed for loop detection
class LoopDetectKeyframeData:
    def __init__(self, keyframe: KeyFrame =None, img=None):
        # keyframe data
        self.id = keyframe.id if keyframe is not None else -1
        self.kps = keyframe.kps if keyframe is not None else []
        self.angles = keyframe.angles if keyframe is not None else []
        self.sizes = keyframe.sizes if keyframe is not None else []
        self.octaves = keyframe.octaves if keyframe is not None else []
        self.des = keyframe.des if keyframe is not None else []
        # NOTE: The kid is not actually used for the processing in this whole file
        self.kid = keyframe.kid if keyframe is not None else -1
        self.img = img if img is not None else (keyframe.img if keyframe is not None else None)
        self.g_des = keyframe.g_des if keyframe is not None else None


class LoopDetectorTask: 
    def __init__(self, keyframe: KeyFrame, img, task_type=LoopDetectorTaskType.NONE, covisible_keyframes = [], connected_keyframes = []):
        self.task_type = task_type
        self.keyframe_data = LoopDetectKeyframeData(keyframe, img)
        self.covisible_keyframes_data = [LoopDetectKeyframeData(kf) for kf in covisible_keyframes if not kf.is_bad] 
        self.connected_keyframes_ids =  [kf.id for kf in connected_keyframes]
        # # for loop closing 
        # self.loop_query_id = None
        # self.num_loop_words = 0
        # self.loop_score = None
        # # for relocalization 
        # self.reloc_query_id = None
        # self.num_reloc_words = 0
        # self.reloc_score = None             
        
    def __str__(self) -> str:
        return f'LoopDetectorTask: img id = {self.keyframe_data.id}, kid = {self.keyframe_data.kid}, task_type = {self.task_type.name}'
        
        
class LoopDetectorOutput: 
    def __init__(self, task_type, candidate_idxs=[], candidate_scores=[], g_des_vec=None, img_id=None, img=None, covisible_ids=[], covisible_gdes_vecs=[]):
        self.task_type = task_type
        # candidates information + input keyframe data
        self.candidate_idxs = candidate_idxs
        self.candidate_scores = candidate_scores
        self.g_des_vec = g_des_vec  # we use g_des_vec instead of g_des since the first can be used with multiprocessing and transported in the queue
        self.img_id = img_id
        self.img = img   # for debugging 
        # potential g_des updates/computations for covisible keyframes
        self.covisible_ids = covisible_ids
        self.covisible_gdes_vecs = covisible_gdes_vecs
        # potential img output 
        self.similarity_matrix = None
        self.loop_detection_img_candidates = None
        
         
    def __str__(self) -> str:
        return f'LoopDetectorOutput: task_type = {self.task_type.name}, candidate_idxs = {self.candidate_idxs}, candidate_scores = {self.candidate_scores}, img_id = {self.img_id}'
        
        
# Base class for loop detectors
class LoopDetectorBase:
    def __init__(self):
        self.img_count = 0 # this corresponds to the internal detector counter (incremented only when a new keyframe is added to the detector database)
        self.map_img_count_to_kf_img_id = {}
        self.map_kf_img_id_to_img = {}
        self.map_kf_img_id_to_img_count = {} # not always used
        
        self.global_descriptor_type = None              # to be set by loop_detector_factory
        self.local_descriptor_aggregation_type = None   # to be set by loop_detector_factory
        self.local_feature_manager = None               # to be set by loop_detector_factory
        self.vocabulary_data = None                     # to be set by loop_detector_factory
        
        self.voc = None                                 # to be set by derived classes        
                
        # init the similarity matrix        
        if Parameters.kLoopClosingDebugWithSimmetryMatrix:
            self.max_num_kfs = 200            
            self.S_float = np.empty([self.max_num_kfs, self.max_num_kfs], 'float32')
            #self.S_color = np.empty([self.max_num_kfs, self.max_num_kfs, 3], 'uint8')
            self.S_color = np.full([self.max_num_kfs, self.max_num_kfs, 3], 0, 'uint8') # loop closures are found with a small score, this will make them disappear    
        else: 
            self.S_float = None 
            self.S_color = None
            
        # to nicely visualize current loop candidates in a single image
        self.loop_detection_imgs = LoopCandidateImgs() if Parameters.kLoopClosingDebugWithLoopDetectionImages else None 
        
    def using_torch_mp(self):
        return False
        
    def reset(self):
        pass 
        
    def init(self):
        pass
        
    # Check and compute if requested the image local descriptors by using the potentially allocated independent local feature manager.
    # This feature manager may have be allocated since we want to use different local descriptors in the loop detector (different from the extracted ones in the frontend).
    # If the local feature manager is allocated then compute the local descriptors and replace the "keyframe_data.des" field in the task data structure.
    def compute_local_des_if_needed(self, task: LoopDetectorTask):
        if self.local_feature_manager is not None:
            kps, des = self.local_feature_manager.compute(task.keyframe_data.img, task.keyframe_data.kps)
            task.keyframe_data.des = des
            print(f'LoopDetectorBase: re-computed {des.shape[0]} local descriptors ({self.local_feature_manager.descriptor_type.name}) for keyframe {task.keyframe_data.id}')
            
            
    def compute_global_des(self, local_des, img): 
        return None
                            
    def run_task(self, task: LoopDetectorTask):
        return None
    
    def compute_reference_similarity_score(self, task: LoopDetectorTask, vector_type, score_fun):
        # Compute reference BoW similarity score.
        # This is the lowest score to a connected keyframe in the covisibility graph.
        # Loop candidates must have a higher similarity than this
        keyframe = task.keyframe_data 
        min_score = 1
        #print(f'LoopDetectorBase: computing reference similarity score for keyframe {keyframe.id} with covisible keyframes {[cov_kf.id for cov_kf in task.covisible_keyframes_data]}')
        if len(task.covisible_keyframes_data) == 0:
            return 0
        for cov_kf in task.covisible_keyframes_data:
            if cov_kf.g_des is None:
                try:
                    if cov_kf.img is None: 
                        cov_kf.img = self.map_kf_img_id_to_img[cov_kf.id]
                except:
                    print(f'LoopDetectorBase: covisible keyframe {cov_kf.id} has no img')
                # if we don't have the global descriptor yet, we need to compute it
                cov_kf.g_des = self.compute_global_des(cov_kf.des, cov_kf.img)
            if cov_kf.g_des is not None:             
                if not isinstance(cov_kf.g_des, vector_type):
                    cov_kf.g_des = vector_type(cov_kf.g_des) # transform back from vec to specialized vector (this is used for bow vector) 
                print(f'LoopDetectorBase: here')
                score = score_fun(cov_kf.g_des, keyframe.g_des)
                min_score = min(min_score, score)
            else: 
                print(f'LoopDetectorBase: covisible keyframe {cov_kf.id} has no g_des')
        return min_score      
    
    def resize_similary_matrix_if_needed(self):
        if self.S_float is None:
            return
        if self.img_count >= self.max_num_kfs:
            self.max_num_kfs += 100
            # self.S_float.resize([self.max_num_kfs, self.max_num_kfs])
            # self.S_color.resize([self.max_num_kfs, self.max_num_kfs, 3])
            S_float = np.pad(self.S_float, ((0, self.max_num_kfs - self.S_float.shape[0]), \
                                            (0, self.max_num_kfs - self.S_float.shape[1])),\
                                            mode='constant', constant_values=0)
            self.S_float = S_float
            S_color = np.pad(self.S_color, ((0, self.max_num_kfs - self.S_color.shape[0]), \
                                            (0, self.max_num_kfs - self.S_color.shape[1]), \
                                            (0, 0)),\
                                            mode='constant', constant_values=0)
            self.S_color = S_color
                
    def update_similarity_matrix(self, score, img_count, other_img_count): 
        color_value = float_to_color(score)
        if self.S_float is not None:
            self.S_float[img_count, other_img_count] = score
            self.S_float[other_img_count, img_count] = score
        if self.S_color is not None:                     
            self.S_color[img_count, other_img_count] = color_value
            self.S_color[other_img_count, img_count] = color_value
            
    def update_loop_closure_imgs(self, score, other_img_id): 
        if self.loop_detection_imgs is not None:
            loop_img = self.map_kf_img_id_to_img[other_img_id]
            self.loop_detection_imgs.add(loop_img.copy(), other_img_id, score)             
                            
    def draw_loop_detection_imgs(self, img_cur, img_id, detection_output: LoopDetectorOutput):          
        if self.S_color is not None:
            detection_output.similarity_matrix = self.S_color#.copy()
        
        if self.loop_detection_imgs is not None and self.loop_detection_imgs.candidates is not None:
            detection_output.loop_detection_img_candidates = self.loop_detection_imgs.candidates#.copy()
