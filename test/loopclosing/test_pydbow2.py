import os
import sys 
sys.path.append("../../")

from pyslam.config import Config
config = Config()

from pyslam.utilities.utils_files import gdrive_download_lambda 
from pyslam.utilities.utils_sys import getchar, Printer 
from pyslam.utilities.utils_img import float_to_color, convert_float_to_colored_uint8_image, LoopCandidateImgs, ImgWriter

import math
import cv2 
import numpy as np

from pyslam.io.dataset_factory import dataset_factory
from pyslam.local_features.feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

config.set_lib('pydbow2')
import pydbow2 as dbow


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kDataFolder = kRootFolder + '/data'
kOrbVocabFile = kDataFolder + '/ORBvoc.txt'


kMinDeltaFrameForMeaningfulLoopClosure = 10
kMaxResultsForLoopClosure = 5


class GlobalFeatureDatabase:
    def __init__(self, voc_filename, max_num_frames):       
        self.global_des_database = [] 
                               
        self.voc = dbow.BinaryVocabulary()
        print(f'loading vocabulary...')
        if not os.path.exists(voc_filename):
            gdrive_url = 'https://drive.google.com/uc?id=1-4qDFENJvswRd1c-8koqt3_5u1jMR4aF'
            gdrive_download_lambda(url=gdrive_url, path=voc_filename)
        self.voc.load(voc_filename)
        print(f'...done')
                                   
        # init the similarity matrix
        self.S_float = np.empty([dataset.num_frames, dataset.num_frames], 'float32')
        self.S_color = np.empty([dataset.num_frames, dataset.num_frames, 3], 'uint8')
        #self.S_color = np.full([dataset.num_frames, dataset.num_frames, 3], 0, 'uint8') # loop closures are found with a small score, this will make them disappear    
                                

    def get_max_score(self, g_des, img_id, max_num_results=5): 
        score = []
        for i, g_des_i in enumerate(self.global_des_database):
            score.append(self.voc.score(g_des, g_des_i))           
        score = np.array(score, 'float32')
        best_idxs = np.argsort(-score)[:max_num_results+1] # we need plus one since we eliminate the best trivial equal to img_id
        best_scores = score[best_idxs]
        for idx, score in zip(best_idxs, best_scores):
            score_color = float_to_color(score)
            self.S_float[img_id, idx] = score
            self.S_float[idx, img_id] = score
            self.S_color[img_id, idx] = score_color
            self.S_color[idx, img_id] = score_color        
        return best_idxs, best_scores
    
    def compute_global_des(self, local_des):
        # Feature vector associate features with nodes in the 4th level (from leaves up)
        # We assume the vocabulary tree has 6 levels, change the 4 otherwise        
        des_transform_result = self.voc.transform(local_des,4)
        global_des = des_transform_result.bowVector
        # at present, we don't use the featureVector des_transform_result.featureVector
        return global_des
    
    def add(self, g_des):
        self.global_des_database.append(g_des)

# online loop closure detection by using DBoW3        
if __name__ == '__main__':
    
    dataset = dataset_factory(config)
    
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = 2000

    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    global_des_database = GlobalFeatureDatabase(kOrbVocabFile, dataset.num_frames)

    # to nicely visualize current loop candidates in a single image
    loop_closure_imgs = LoopCandidateImgs()
    
    img_writer = ImgWriter(font_scale=0.7)    
    
    cv2.namedWindow('S', cv2.WINDOW_NORMAL)
        
    entry_id = 0
    img_id = 0   #180, 340, 400   # you can start from a desired frame id if needed 
    while dataset.isOk():

        timestamp = dataset.getTimestamp()          # get current timestamp 
        img = dataset.getImageColor(img_id)

        if img is not None:
            print('----------------------------------------')
            print(f'processing img {img_id}')
            
            loop_closure_imgs.reset()
                       
            # Find the keypoints and descriptors in img1
            kps, des = feature_tracker.detectAndCompute(img)   # with DL matchers this a null operation 

            g_des = global_des_database.compute_global_des(des)
            global_des_database.add(g_des) # add image descriptors to global_des_database after having computed the scores
                       
            if entry_id >= 1:
                best_idxs, best_scores = global_des_database.get_max_score(g_des, img_id, max_num_results=kMaxResultsForLoopClosure)
                
                # visualize non-trivial loop closures: we check the query results are not too close to the current image
                for idx, score in zip(best_idxs, best_scores):
                    if abs(idx - img_id) > kMinDeltaFrameForMeaningfulLoopClosure: 
                        print(f'result - best id: {idx}, score: {score}')
                        loop_img = dataset.getImageColor(idx)
                        loop_closure_imgs.add(loop_img, idx, score)

            img_writer.write(img, f'id: {img_id}', (30, 30))
            cv2.imshow('img', img)
            
            cv2.imshow('S', global_des_database.S_color)            
            #cv2.imshow('S', convert_float_to_colored_uint8_image(global_des_database.S_float))
            
            if loop_closure_imgs.candidates is not None:
                cv2.imshow('loop_closure_imgs', loop_closure_imgs.candidates)
            
            cv2.waitKey(1)
        else: 
            getchar()
            
        img_id += 1
        entry_id += 1