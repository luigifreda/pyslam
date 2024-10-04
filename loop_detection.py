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
from multiprocessing import Process, Queue, Value, Condition
import numpy as np
import cv2

from config import Config
config = Config()

from utils_files import gdrive_download_lambda 
from utils_sys import getchar, Printer 
from utils_img import float_to_color, convert_float_to_colored_uint8_image, LoopDetectionCandidateImgs
from utils_features import transform_float_to_binary_descriptor

from parameters import Parameters
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs
from feature_types import FeatureInfo

from timer import TimerFps, Timer

from keyframe import KeyFrame
from frame import Frame

config.set_lib('pyobindex2')
import pyobindex2 as obindex2

config.set_lib('pydbow3')
import pydbow3 as dbow3

config.set_lib('pydbow2')
import pydbow2 as dbow2


kMinDeltaFrameForMeaningfulLoopClosure = 10
kMaxResultsForLoopClosure = 5

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kDataFolder = kRootFolder + '/data'
kVocabFile = kDataFolder + '/ORBvoc.txt'


if Parameters.kLoopClosingDebugAndPrintToFile:
    # redirect the prints of local mapping to the file local_mapping.log 
    # you can watch the output in separate shell by running:
    # $ tail -f loop_closing.log 
    import builtins as __builtin__
    logging_file=open('loop_closing.log','w')
    def print(*args, **kwargs):
        return __builtin__.print(*args, **kwargs,file=logging_file,flush=True)
else:
    def print(*args, **kwargs):
        return
        
        
class LoopDetectorKeyFrameDataInput(object): 
    def __init__(self, keyframe: KeyFrame, img):
        # keyframe data
        self.id = keyframe.id
        self.kps = keyframe.kps
        self.angles = keyframe.angles
        self.sizes = keyframe.sizes
        self.octaves = keyframe.octaves
        self.des = keyframe.des
        self.kid = keyframe.kid
        self.img = img if img is not None else keyframe.img
        self.g_des = keyframe.g_des        
        # for loop closing 
        self.loop_query_id = None
        self.num_loop_words = 0
        self.loop_score = None
        # for relocalization 
        self.reloc_query_id = None
        self.num_reloc_words = 0
        self.reloc_score = None             
        
    def __str__(self) -> str:
        return f'LoopDetectorKeyFrameDataInput: id = {self.id}, kid = {self.kid}, img_count = {self.img_count}'
        
        
class LoopDetectorOutput(object): 
    def __init__(self, candidate_idxs, candidate_scores, g_des_vec, img_id):
        # candidates information + input keyframe data
        self.candidate_idxs = candidate_idxs
        self.candidate_scores = candidate_scores
        self.g_des_vec = g_des_vec  # we use g_des_vec instead of g_des since the first can be used with multiprocessing and transported in the queue
        self.img_id = img_id
        
    def __str__(self) -> str:
        return f'LoopDetectorOutput: candidate_idxs = {self.candidate_idxs}, candidate_scores = {self.candidate_scores}, img_id = {self.img_id}'
        
        
class LoopDetectorBase:
    def __init__(self):
        self.img_count = 0
        self.map_img_count_to_kf_id = {}
        self.map_kf_id_to_img = {}
                
        # init the similarity matrix        
        if Parameters.kLoopClosingDebugWithSimmetryMatrix:
            self.max_num_kfs = 200            
            self.S_float = np.empty([self.max_num_kfs, self.max_num_kfs], 'float32')
            #self.S_color = np.empty([self.max_num_kfs, self.max_num_kfs, 3], 'uint8')
            self.S_color = np.full([self.max_num_kfs, self.max_num_kfs, 3], 0, 'uint8') # loop closures are found with a small score, this will make them disappear    
        else: 
            self.S_float = None 
            self.S_color = None
        self.S_init = False
            
        # to nicely visualize current loop candidates in a single image
        self.loop_closure_imgs = LoopDetectionCandidateImgs() if Parameters.kLoopClosingDebugWithLoopClosureImages else None 
        self.loop_closure_imgs_init = False
            
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
                  
    def compute_global_des(self, local_des): 
        return None
                            
    def add_keyframe(self, keyframe: LoopDetectorKeyFrameDataInput):
        return None
    
    def update_similarity_matrix_and_loop_closure_imgs(self, score, img_count, img_id, other_img_count, other_img_id): 
        color_value = float_to_color(score)
        if self.S_float is not None:
            self.S_float[img_count, other_img_count] = score
            self.S_float[other_img_count, img_count] = score
        if self.S_color is not None:                     
            self.S_color[img_count, other_img_count] = color_value
            self.S_color[other_img_count, img_count] = color_value

        # visualize non-trivial loop closures: we check the query results are not too close to the current image
        if self.loop_closure_imgs is not None:
            if abs(other_img_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure: 
                print(f'result - best id: {other_img_id}, score: {score}')
                loop_img = self.map_kf_id_to_img[other_img_id]
                self.loop_closure_imgs.add(loop_img.copy(), other_img_id, score) 
                            
    def draw_loop_closure_imgs(self, img_cur, img_id):
        font_pos = (50, 50)                   
        cv2.putText(img_cur, f'id: {img_id}', font_pos, LoopDetectionCandidateImgs.kFont, LoopDetectionCandidateImgs.kFontScale, \
                    LoopDetectionCandidateImgs.kFontColor, LoopDetectionCandidateImgs.kFontThickness, cv2.LINE_AA)     
        cv2.imshow('loop detection input img', img_cur)
          
        if self.S_color is not None:
            if not self.S_init:
                cv2.namedWindow('Similarity matrix', cv2.WINDOW_NORMAL) # to get a resizable window
                self.S_init = True             
            cv2.imshow('Similarity matrix', self.S_color)            
            #cv2.imshow('Similarity matrix', convert_float_to_colored_uint8_image(S_float))
        
        if self.loop_closure_imgs.candidates is not None:
            if not self.loop_closure_imgs_init:
                cv2.namedWindow('loop detection candidates', cv2.WINDOW_NORMAL) # to get a resizable window
                self.loop_closure_imgs_init = True
            cv2.imshow('loop detection candidates', self.loop_closure_imgs.candidates)
        
        cv2.waitKey(1)


class LoopDetectorIBoW(LoopDetectorBase): 
    def __init__(self):
        super().__init__()
        # Creating a new index of images
        self.index = obindex2.ImageIndex(16, 150, 4, obindex2.MERGE_POLICY_AND, True)               
        
    def add_keyframe(self, keyframe: LoopDetectorKeyFrameDataInput):
        print(f'LoopDetectorIBoW: adding keyframe {keyframe.id}, img_count = {self.img_count}')        
        img_id = keyframe.id
        self.map_img_count_to_kf_id[self.img_count] = img_id        
        self.map_kf_id_to_img[keyframe.id] = keyframe.img
        
        kps, des = keyframe.kps, keyframe.des 
        # kp.response is not actually used
        #kps_ = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps]  # tuple_x_y_size_angle_response_octave
        kps_ = [(kp[0], kp[1], keyframe.sizes[i], keyframe.angles[i], 1, keyframe.octaves[i]) for i,kp in enumerate(kps)]  # tuple_x_y_size_angle_response_octave
        
        des_ = des
        if FeatureInfo.norm_type[Frame.feature_manager.descriptor_type] != cv2.NORM_HAMMING:
            des_ = transform_float_to_binary_descriptor(des)
                                
        if self.loop_closure_imgs is not None:
            self.loop_closure_imgs.reset()
            
        self.resize_similary_matrix_if_needed()
               
        candidate_idxs = []
        candidate_scores = []
        g_des = None
                                    
        if self.img_count == 0:
            self.index.addImage(self.img_count, kps_, des_)                       
        elif self.img_count >= 1:
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
                    other_img_id = self.map_img_count_to_kf_id[m.image_id]
                    other_img_count=m.image_id   
                    if abs(other_img_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure:  
                        candidate_idxs.append(other_img_id)
                        candidate_scores.append(m.score)                         
                    self.update_similarity_matrix_and_loop_closure_imgs(score=m.score, \
                                                                       img_count=self.img_count, \
                                                                       img_id=img_id, \
                                                                       other_img_count=other_img_count, \
                                                                       other_img_id = other_img_id) 
                    count_valid_candidates += 1                     
                    if count_valid_candidates >= kMaxResultsForLoopClosure: break   
        
            # Add the image to the index. Matched descriptors are used 
            # to update the index and the remaining ones are added as new visual words
            self.index.addImage(self.img_count, kps_, des_, matches)
                                        
        # Reindex features every 500 images
        if self.img_count % 250 == 0 and self.img_count>0:
            print("------ Rebuilding indices ------")
            self.index.rebuild() 
            
        if self.loop_closure_imgs:            
            self.draw_loop_closure_imgs(keyframe.img, img_id)
        self.img_count += 1      
        return candidate_idxs, candidate_scores, g_des, img_id          

        

class LoopDetectorDBoW2(LoopDetectorBase): 
    def __init__(self, vocab_path=kVocabFile):
        super().__init__()
        self.voc = dbow2.BinaryVocabulary()
        print(f'loading vocabulary...')
        if not os.path.exists(kVocabFile):
            gdrive_url = 'https://drive.google.com/uc?id=1-4qDFENJvswRd1c-8koqt3_5u1jMR4aF'
            gdrive_download_lambda(url=gdrive_url, path=vocab_path)
        self.voc.load(vocab_path)
        print(f'...done')
        self.global_des_database = []            
    
    def compute_global_des(self, local_des):
        # Feature vector associate features with nodes in the 4th level (from leaves up)
        # We assume the vocabulary tree has 6 levels, change the 4 otherwise        
        des_transform_result = self.voc.transform(local_des,4)
        global_des = des_transform_result.bowVector
        # at present, we don't use the featureVector des_transform_result.featureVector
        return global_des

    def db_query(self, global_des, img_id, max_num_results=5): 
        score = []
        for i, global_des_i in enumerate(self.global_des_database):
            score.append(self.voc.score(global_des, global_des_i))           
        score = np.array(score, 'float32')
        best_idxs = np.argsort(-score)[:max_num_results+1] # we need plus one since we eliminate the best trivial equal to img_id
        best_scores = score[best_idxs]  
        return best_idxs, best_scores
                
    def add_keyframe(self, keyframe: LoopDetectorKeyFrameDataInput):                                  
        print(f'LoopDetectorDBoW2: adding keyframe {keyframe.id}, img_count = {self.img_count}')        
        img_id = keyframe.id
        self.map_img_count_to_kf_id[self.img_count] = img_id        
        self.map_kf_id_to_img[keyframe.id] = keyframe.img
        
        des = keyframe.des 
        g_des = self.compute_global_des(des)
        self.global_des_database.append(g_des) # add image descriptors to global_des_database after having computed the scores
                
        if self.loop_closure_imgs is not None:
            self.loop_closure_imgs.reset()
            
        self.resize_similary_matrix_if_needed()

        candidate_idxs = []
        candidate_scores = []
                                    
        if self.img_count >= 1:
            best_idxs, best_scores = self.db_query(g_des, img_id, max_num_results=kMaxResultsForLoopClosure)
            for other_img_count, score in zip(best_idxs, best_scores):                            
                other_img_id = self.map_img_count_to_kf_id[other_img_count]
                if abs(other_img_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure:
                    candidate_idxs.append(other_img_id)
                    candidate_scores.append(score)
                self.update_similarity_matrix_and_loop_closure_imgs(score=score, \
                                                                    img_count=self.img_count, \
                                                                    img_id=img_id, \
                                                                    other_img_count=other_img_count, \
                                                                    other_img_id = other_img_id)                 
        if self.loop_closure_imgs: 
            self.draw_loop_closure_imgs(keyframe.img, img_id)       
        self.img_count += 1 
        return candidate_idxs, candidate_scores, g_des, img_id          
                 
                 

class LoopDetectorDBoW3(LoopDetectorBase): 
    kUseLocalDes = False
    def __init__(self, vocab_path=kVocabFile):
        super().__init__()
        self.voc = dbow3.Vocabulary()
        print(f'loading vocabulary...')
        if not os.path.exists(kVocabFile):
            gdrive_url = 'https://drive.google.com/uc?id=1-4qDFENJvswRd1c-8koqt3_5u1jMR4aF'
            gdrive_download_lambda(url=gdrive_url, path=vocab_path)
        self.voc.load(vocab_path)
        print(f'...done')
        self.db = dbow3.Database()
        self.db.setVocabulary(self.voc)      
        
    def compute_global_des(self, local_des):  
        #print(f'computing global descriptors... voc empty: {self.voc.empty()}')     
        global_des = self.voc.transform(local_des) # this returns a bow vector
        return global_des
            
    # query with local descriptors 
    def db_query(self, local_des: np.ndarray, img_id, max_num_results=5): 
        results = self.db.query(local_des, max_results=max_num_results+1) # we need plus one to eliminate the best trivial equal to img_id
        return results
    
    # query with global descriptors
    def db_query(self, global_des: dbow3.BowVector, img_id, max_num_results=5): 
        results = self.db.query(global_des, max_results=max_num_results+1) # we need plus one to eliminate the best trivial equal to img_id
        return results    
                
    def add_keyframe(self, keyframe: LoopDetectorKeyFrameDataInput):                       
        print(f'LoopDetectorDBoW3: adding keyframe {keyframe.id}, img_count = {self.img_count}')        
        img_id = keyframe.id
        self.map_img_count_to_kf_id[self.img_count] = img_id        
        self.map_kf_id_to_img[keyframe.id] = keyframe.img
        
        des = keyframe.des 
        
        # add image descriptors to database
        if LoopDetectorDBoW3.kUseLocalDes:
            self.db.addFeatures(des) # add local descriptors 
        else: 
            g_des = self.compute_global_des(des) # get bow vector
            self.db.addBowVector(g_des) # add directly global descriptors
                                
        if self.loop_closure_imgs is not None:
            self.loop_closure_imgs.reset()
            
        self.resize_similary_matrix_if_needed()
                      
        candidate_idxs = []
        candidate_scores = []
                                    
        if self.img_count >= 1:
            if LoopDetectorDBoW3.kUseLocalDes:            
                results = self.db_query(des, img_id, max_num_results=kMaxResultsForLoopClosure) 
            else:
                results = self.db_query(g_des, img_id, max_num_results=kMaxResultsForLoopClosure) 
            for r in results:
                r_img_id = self.map_img_count_to_kf_id[r.Id]
                if abs(r_img_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure:
                    candidate_idxs.append(r_img_id)
                    candidate_scores.append(r.Score)
                self.update_similarity_matrix_and_loop_closure_imgs(score=r.Score, \
                                                                    img_count=self.img_count, \
                                                                    img_id=img_id, \
                                                                    other_img_count=r.Id, \
                                                                    other_img_id = r_img_id)                 
        if self.loop_closure_imgs: 
            self.draw_loop_closure_imgs(keyframe.img, img_id)       
        self.img_count += 1  
        return candidate_idxs, candidate_scores, g_des, img_id  
    
    

class LoopDetection(object):
    def __init__(self, slam):
        #self.loop_detector = LoopDetectorIBoW()
        #self.loop_detector = LoopDetectorDBoW2()         
        self.loop_detector = LoopDetectorDBoW3()       
        
        self.timer_verbose = kTimerVerbose  # set this to True if you want to print timing     
        self.time_loop_detection = Value('d',0.0)       
        
        self.last_input = None
        self.q_in_condition = Condition()
        self.q_in = Queue(maxsize=100)
        self.q_out_condition = Condition()        
        self.q_out = Queue(maxsize=100)
        
        self.is_running  = Value('i',1)
        self.process = Process(target=self.run,
                          args=(self.q_in, self.q_in_condition, self.q_out, self.q_out_condition, self.is_running, self.time_loop_detection,))
        self.process.daemon = True
        self.process.start()

    def quit(self):
        if self.is_running.value == 1:
            print('LoopDetection: quitting...')
            self.is_running.value = 0            
            with self.q_in_condition:
                self.q_in.put(None)  # put a None in the queue to signal we have to exit
                self.q_in_condition.notify_all()       
            with self.q_out_condition:
                self.q_out_condition.notify_all()                           
            self.process.join(timeout=5)
            if self.process.is_alive():
                Printer.orange("Warning: Loop detection process did not terminate in time, forced kill.")  
                self.process.terminate()      
            print('LoopDetection: done')   
    
    def run(self, q_in, q_in_condition, q_out, q_out_condition, is_running, time_loop_detection):
        while is_running.value == 1:
            with q_in_condition:
                while q_in.empty() and is_running.value == 1:
                    q_in_condition.wait()
                if not q_in.empty():            
                    self.loop_detection(q_in, q_out, q_out_condition, is_running, time_loop_detection)                         
        print('LoopDetection: loop exit...')         

    def loop_detection(self, q_in, q_out, q_out_condition, is_running, time_loop_detection):
        if is_running.value == 1:
            timer = TimerFps()
            timer.start()
            self.last_input = q_in.get()
            if self.last_input is None: # got a None to exit
                is_running.value = 0
            else:
                candidate_idxs, candidate_scores, g_des, img_id = self.loop_detector.add_keyframe(self.last_input)
                g_des_vec = g_des.toVec() if g_des is not None else None
                last_output = LoopDetectorOutput(candidate_idxs, candidate_scores, g_des_vec, img_id)
                if is_running.value == 1:
                    with q_out_condition:
                        q_out.put(last_output)
                        q_out_condition.notify_all()
            timer.refresh()
            time_loop_detection.value = timer.last_elapsed
            print(f'LoopDetection: loop_detection time: {time_loop_detection.value}')

    def add_keyframe(self, keyframe: KeyFrame, img):   
        if self.is_running.value == 1:
            with self.q_in_condition:
                self.q_in.put(LoopDetectorKeyFrameDataInput(keyframe, img))
                self.q_in_condition.notify_all()

    def pop_output(self): 
        if self.is_running.value == 0:
            return None
        with self.q_out_condition:        
            while self.q_out.empty() and self.is_running.value == 1:
                self.q_out_condition.wait()
            if self.q_out.empty():
                return None
            else:               
                return self.q_out.get()