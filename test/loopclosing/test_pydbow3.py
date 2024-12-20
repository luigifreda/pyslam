import os
import sys 
sys.path.append("../../")

from config import Config
config = Config()

from utils_files import gdrive_download_lambda 
from utils_sys import getchar, Printer 
from utils_img import float_to_color, convert_float_to_colored_uint8_image, LoopCandidateImgs, ImgWriter

import math
import cv2 
import numpy as np

from config_parameters import Parameters
from dataset import dataset_factory
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

config.set_lib('pydbow3')
import pydbow3 as dbow3


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kDataFolder = kRootFolder + '/data'
kOrbVocabFile = kDataFolder + '/ORBvoc.txt'
#kOrbVocabFile = kDataFolder + '/orbvoc.dbow3'


kMinDeltaFrameForMeaningfulLoopClosure = 10
kMaxResultsForLoopClosure = 5


class LoopCloserBase:
    def __init__(self):
        self.entry_id = 0
        self.map_entry_id_to_frame_id = {}
        self.map_frame_id_to_img = {}
                
        # init the similarity matrix        
        if Parameters.kLoopClosingDebugWithSimmetryMatrix:
            self.max_num_kfs = 200            
            self.S_float = np.empty([self.max_num_kfs, self.max_num_kfs], 'float32')
            self.S_color = np.empty([self.max_num_kfs, self.max_num_kfs, 3], 'uint8')
            #self.S_color = np.full([self.max_num_kfs, self.max_num_kfs, 3], 0, 'uint8') # loop closures are found with a small score, this will make them disappear    
        else: 
            self.S_float = None 
            self.S_color = None
            
        # to nicely visualize current loop candidates in a single image
        self.loop_closure_imgs = LoopCandidateImgs() if Parameters.kLoopClosingDebugWithLoopDetectionImages else None 
            
    def resize_similary_matrix_if_needed(self):
        if self.S_float is None:
            return
        if self.entry_id >= self.max_num_kfs:
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
    
    def update_similarity_matrix_and_loop_closure_imgs(self, score, entry_id, img_id, other_entry_id, other_frame_id): 
        color_value = float_to_color(score)
        if self.S_float is not None:
            self.S_float[entry_id, other_entry_id] = score
            self.S_float[other_entry_id, entry_id] = score
        if self.S_color is not None:                     
            self.S_color[entry_id, other_entry_id] = color_value
            self.S_color[other_entry_id, entry_id] = color_value

        # visualize non-trivial loop closures: we check the query results are not too close to the current image
        if self.loop_closure_imgs is not None:
            if abs(other_frame_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure: 
                print(f'result - best id: {other_frame_id}, score: {score}')
                loop_img = self.map_frame_id_to_img[other_frame_id]
                self.loop_closure_imgs.add(loop_img.copy(), other_frame_id, score) 
                            
    def draw_loop_closure_imgs(self, img_cur, img_id):
        if self.S_color is not None or self.loop_closure_imgs.candidates is not None:
            font_pos = (50, 50)                   
            cv2.putText(img_cur, f'id: {img_id}', font_pos, LoopCandidateImgs.kFont, LoopCandidateImgs.kFontScale, \
                        LoopCandidateImgs.kFontColor, LoopCandidateImgs.kFontThickness, cv2.LINE_AA)     
            cv2.imshow('loop img', img_cur)
            
            if self.S_color is not None:
                cv2.imshow('S', self.S_color)            
                #cv2.imshow('S', convert_float_to_colored_uint8_image(S_float))
            
            if self.loop_closure_imgs.candidates is not None:
                cv2.imshow('loop_closure_imgs', self.loop_closure_imgs.candidates)
            
            cv2.waitKey(1)
        
        
class LoopCloserDBoW3(LoopCloserBase): 
    def __init__(self, vocab_path=kOrbVocabFile):
        super().__init__()
        self.voc = dbow3.Vocabulary()
        print(f'loading vocabulary...')
        if not os.path.exists(kOrbVocabFile):
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
                
    def add_keyframe(self, img, img_id, des):
        
        use_local_des = False
                
        print(f'LoopCloserDBoW3: adding frame {img_id}, entry_id = {self.entry_id}')        
        self.map_entry_id_to_frame_id[self.entry_id] = img_id        
        self.map_frame_id_to_img[img_id] = img
        
        # add image descriptors to database
        if use_local_des:
            self.db.addFeatures(des) # add local descriptors 
        else: 
            g_des = self.compute_global_des(des)
            #print(f'global_des: {g_des}')
            self.db.addBowVector(g_des) # add directly global descriptors
                                
        if self.loop_closure_imgs is not None:
            self.loop_closure_imgs.reset()
            
        self.resize_similary_matrix_if_needed()
                            
        if self.entry_id >= 1:
            if use_local_des:            
                results = self.db_query(des, img_id, max_num_results=kMaxResultsForLoopClosure+1) # we need plus one to eliminate the best trivial equal to img_id
            else:
                results = self.db_query(g_des, img_id, max_num_results=kMaxResultsForLoopClosure+1) # we need plus one to eliminate the best trivial equal to img_id
            for r in results:
                r_frame_id = self.map_entry_id_to_frame_id[r.id]
                self.update_similarity_matrix_and_loop_closure_imgs(score=r.score, \
                                                                    entry_id=self.entry_id, \
                                                                    img_id=img_id, \
                                                                    other_entry_id=r.id, \
                                                                    other_frame_id = r_frame_id)                 
        if self.loop_closure_imgs: 
            self.draw_loop_closure_imgs(img, img_id)       
        self.entry_id += 1   


# online loop closure detection by using DBoW3        
if __name__ == '__main__':
    
    dataset = dataset_factory(config)
    
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = 2000

    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    loop_closer = LoopCloserDBoW3()
    
    img_writer = ImgWriter(font_scale=0.7)
        
    # voc = dbow.Vocabulary()
    # print(f'loading vocabulary...')
    # if not os.path.exists(kOrbVocabFile):
    #     gdrive_url = 'https://drive.google.com/uc?id=1-4qDFENJvswRd1c-8koqt3_5u1jMR4aF'
    #     gdrive_download_lambda(url=gdrive_url, path=kOrbVocabFile)
    # voc.load(kOrbVocabFile)
    # print(f'...done')
    # db = dbow.Database()
    # db.setVocabulary(voc)
    
    # # to nicely visualize current loop candidates in a single image
    # loop_closure_imgs = LoopCandidateImgs()
    
    # # init the similarity matrix
    # S_float = np.empty([dataset.num_frames, dataset.num_frames], 'float32')
    # S_color = np.empty([dataset.num_frames, dataset.num_frames, 3], 'uint8')
    # #S_color = np.full([dataset.num_frames, dataset.num_frames, 3], 0, 'uint8') # loop closures are found with a small score, this will make them disappear    
    
    # cv2.namedWindow('S', cv2.WINDOW_NORMAL)
        
    #entry_id = 0
    img_id = 0   #180, 340, 400   # you can start from a desired frame id if needed 
    while dataset.isOk():

        timestamp = dataset.getTimestamp()          # get current timestamp 
        img = dataset.getImageColor(img_id)

        if img is not None:
            print('----------------------------------------')
            print(f'processing img {img_id}')
                        
            # loop_closure_imgs.reset()
                       
            # Find the keypoints and descriptors in img1
            kps, des = feature_tracker.detectAndCompute(img)   # with DL matchers this a null operation 
            
            loop_closer.add_keyframe(img, img_id, des)
            img_id += 1
            
            
            img_writer.write(img, f'id: {img_id}', (30, 30))
            cv2.imshow('img', img)
                        
            # # add image descriptors to database
            # db.add(des)
                       
        #     if entry_id >= 1:
        #         results = db.query(des, max_results=kMaxResultsForLoopClosure+1) # we need plus one to eliminate the best trivial equal to img_id
        #         for r in results:
        #             float_value = r.score * 255
        #             color_value = float_to_color(r.score)
        #             S_float[img_id, r.id] = float_value
        #             S_float[r.id, img_id] = float_value
        #             S_color[img_id, r.id] = color_value
        #             S_color[r.id, img_id] = color_value
                    
        #             # visualize non-trivial loop closures: we check the query results are not too close to the current image
        #             if abs(r.id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure: 
        #                 print(f'result - best id: {r.id}, score: {r.score}')
        #                 loop_img = dataset.getImageColor(r.id)
        #                 loop_closure_imgs.add(loop_img, r.id, r.score)

        #     font_pos = (50, 50)                   
        #     cv2.putText(img, f'id: {img_id}', font_pos, LoopCandidateImgs.kFont, LoopCandidateImgs.kFontScale, \
        #                 LoopCandidateImgs.kFontColor, LoopCandidateImgs.kFontThickness, cv2.LINE_AA)     
        #     cv2.imshow('img', img)
            
        #     cv2.imshow('S', S_color)            
        #     #cv2.imshow('S', convert_float_to_colored_uint8_image(S_float))
            
        #     if loop_closure_imgs.candidates is not None:
        #         cv2.imshow('loop_closure_imgs', loop_closure_imgs.candidates)
            
            cv2.waitKey(1)
        else: 
            getchar()
            
        # img_id += 1
        # entry_id += 1