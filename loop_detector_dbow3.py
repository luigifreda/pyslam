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
from multiprocessing import Process, Manager, Value, Condition
import numpy as np
import cv2
from enum import Enum

from utils_files import gdrive_download_lambda 
from utils_sys import getchar, Printer 
from utils_features import transform_float_to_binary_descriptor

from parameters import Parameters
from feature_types import FeatureInfo

from timer import TimerFps

from keyframe import KeyFrame
from frame import Frame

import traceback
from loop_detector_base import LoopDetectorTaskType, LoopDetectKeyframeData, LoopDetectorTask, LoopDetectorOutput, LoopDetectorBase
from loop_detector_vocabulary import VocabularyData

import config
config.cfg.set_lib('pydbow3')
import pydbow3 as dbow3


kVerbose = True

kMinDeltaFrameForMeaningfulLoopClosure = Parameters.kMinDeltaFrameForMeaningfulLoopClosure
kMaxResultsForLoopClosure = Parameters.kMaxResultsForLoopClosure

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kDataFolder = kRootFolder + '/data'
kOrbVocabFile = kDataFolder + '/ORBvoc.txt'


if Parameters.kLoopClosingDebugAndPrintToFile:
    from loop_detector_base import print   
                 
class LoopDetectorDBoW3(LoopDetectorBase): 
    def __init__(self, vocabulary_data: VocabularyData, local_feature_manager=None):
        super().__init__()
        self.local_feature_manager = local_feature_manager              
        self.voc = dbow3.Vocabulary()  
        print(f'LoopDetectorDBoW3: loading vocabulary...')
        vocabulary_data.check_download()
        self.voc.load(vocabulary_data.vocab_file_path)
        print(f'...done')
        self.db = dbow3.Database()
        self.db.setVocabulary(self.voc)      
        
    def compute_global_des(self, local_des, img):  
        #print(f'computing global descriptors... voc empty: {self.voc.empty()}')     
        global_des = self.voc.transform(local_des) # this returns a bow vector
        return global_des
    
    # query with global descriptors
    def db_query(self, global_des: dbow3.BowVector, img_id, max_num_results=5): 
        results = self.db.query(global_des, max_results=max_num_results+1) # we need plus one to eliminate the best trivial equal to img_id
        return results    
                
    def run_task(self, task: LoopDetectorTask):                        
        print(f'LoopDetectorDBoW3: running task {task.keyframe_data.id}, img_count = {self.img_count}, task_type = {task.task_type.name}')      
        keyframe = task.keyframe_data             
        img_id = keyframe.id
        
        if self.loop_detection_imgs is not None:        
            self.map_kf_img_id_to_img[keyframe.id] = keyframe.img
            self.loop_detection_imgs.reset()
            
        self.resize_similary_matrix_if_needed()
        
        # compute global descriptor
        if keyframe.g_des is None:
            keyframe.g_des = self.compute_global_des(keyframe.des, keyframe.img) # get bow vector
            g_des_vec = keyframe.g_des.toVec() # transform it to a vector(numpy array) to make it pickable
        else: 
            if not isinstance(keyframe.g_des, dbow3.BowVector):
                g_des_vec = keyframe.g_des
                keyframe.g_des = dbow3.BowVector(keyframe.g_des) # transform back from vec to bow vector
        
        # add image descriptors to global_des_database
        self.db.addBowVector(keyframe.g_des)
        
        # the img_ids are mapped to img_counts (entry ids) inside the database management
        self.map_img_count_to_kf_img_id[self.img_count] = img_id        
        #print(f'LoopDetectorDBoW3: mapping img_id: {img_id} to img_count: {self.img_count}')
                    
        detection_output = LoopDetectorOutput(task_type=task.task_type, g_des_vec=g_des_vec, img_id=img_id, img=keyframe.img)
        
        if task.task_type == LoopDetectorTaskType.LOOP_CLOSURE:
                            
            # Compute reference BoW similarity score as the lowest score to a connected keyframe in the covisibility graph.
            min_score = self.compute_reference_similarity_score(task, dbow3.BowVector, score_fun=self.voc.score)
            print(f'LoopDetectorDBoW3: min_score = {min_score}')
                            
            candidate_idxs = []
            candidate_scores = []
                                        
            if self.img_count >= 1:
                results = self.db_query(keyframe.g_des, img_id, max_num_results=kMaxResultsForLoopClosure) 
                #print(f'connected keyframes: {[kf_id for kf_id in task.connected_keyframes_ids]}')
                for r in results:
                    r_img_id = self.map_img_count_to_kf_img_id[r.id] # get the image id of the keyframe from it's internal image count
                    #print(f'r_img_id = {r_img_id}, r.id = {r.id}')
                    self.update_similarity_matrix(score=r.score, img_count=self.img_count, other_img_count=r.id)
                    if abs(r_img_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure and \
                        r.score >= min_score and \
                        r_img_id not in task.connected_keyframes_ids:
                        candidate_idxs.append(r_img_id)
                        candidate_scores.append(r.score)
                        self.update_loop_closure_imgs(score=r.score, other_img_id = r_img_id)
              
            self.draw_loop_detection_imgs(keyframe.img, img_id, detection_output)  
                
            detection_output.candidate_idxs = candidate_idxs
            detection_output.candidate_scores = candidate_scores 
            detection_output.covisible_ids = [cov_kf.id for cov_kf in task.covisible_keyframes_data]
            detection_output.covisible_gdes_vecs = [cov_kf.g_des.toVec() for cov_kf in task.covisible_keyframes_data]
            
        else:
            # if we just wanted to compute the global descriptor (LoopDetectorTaskType.COMPUTE_GLOBAL_DES), we don't have to do anything
            pass
        
        self.img_count += 1        
        return detection_output