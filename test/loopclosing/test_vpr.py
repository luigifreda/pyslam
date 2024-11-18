import argparse
import configparser
import os

import os
import sys 
sys.path.append("../../")

from config import Config
config = Config()

import numpy as np
import cv2

from utils_sys import getchar, Printer 
from utils_img import float_to_color, float_to_color_array, convert_float_to_colored_uint8_image, LoopCandidateImgs, ImgWriter


from dataset import dataset_factory
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

config.set_lib('vpr', prepend=True)


def scoreSAD(g_des1, g_des2):
    diff = g_des1-g_des2
    is_nan_diff = np.isnan(diff)
    nan_count_per_row = np.count_nonzero(is_nan_diff, axis=1)
    dim = diff.shape[1] - nan_count_per_row
    #print(f'dim: {dim}, diff.shape: {diff.shape}')
    diff[is_nan_diff] = 0
    return -np.sum(np.abs(diff),axis=1) / dim    

def scoreCosine(g_des1, g_des2):
    norm_g_des1 = np.linalg.norm(g_des1, axis=1, keepdims=True)  # g_des1 is [1, D], so norm is scalar
    norm_g_des2 = np.linalg.norm(g_des2, axis=1, keepdims=True)  # g_des2 is [M, D]
    dot_product = np.dot(g_des2, g_des1.T).ravel()
    cosine_similarity = dot_product / (norm_g_des1 * norm_g_des2.ravel())
    return cosine_similarity.ravel()



class GlobalFeatureDatabase:
    def __init__(self, global_descriptor_type, max_num_frames):
        self.global_descriptor_type = global_descriptor_type        
        self.global_feature_extractor = self.init_global_feature(global_descriptor_type)
        self.global_des_database = [] 
                
        self.score = None           
        if global_descriptor_type == 'SAD':
            self.score = scoreSAD
            self.min_score = -100
            self.S_float = np.full([max_num_frames, max_num_frames], self.min_score, 'float32')    
            self.S_color = np.full([max_num_frames, max_num_frames, 3], 0, 'uint8') 
        else:
            self.score = scoreCosine
            self.min_score = 0            
            self.S_float = np.full([max_num_frames, max_num_frames], 0, 'float32')  
            self.S_color = np.full([max_num_frames, max_num_frames, 3],  0, 'uint8')                         

    def init_global_feature(self, global_descriptor_type):
        global_feature_extractor = None
        if global_descriptor_type == 'HDC-DELF':
            from feature_extraction.feature_extractor_holistic import HDCDELF
            global_feature_extractor = HDCDELF()
        elif global_descriptor_type == 'AlexNet':
            from feature_extraction.feature_extractor_holistic import AlexNetConv3Extractor
            global_feature_extractor = AlexNetConv3Extractor()
        elif global_descriptor_type == 'SAD':
            from feature_extraction.feature_extractor_holistic import SAD
            global_feature_extractor = SAD()
        elif global_descriptor_type == 'NetVLAD':
            from feature_extraction.feature_extractor_patchnetvlad import PatchNetVLADFeatureExtractor
            from patchnetvlad.tools import PATCHNETVLAD_ROOT_DIR
            print(f'PatchNetVLADFeatureExtractor: {PATCHNETVLAD_ROOT_DIR}')
            configfile = os.path.join(PATCHNETVLAD_ROOT_DIR, 'configs/netvlad_extract.ini')
            assert os.path.isfile(configfile)
            config = configparser.ConfigParser()
            config.read(configfile)
            global_feature_extractor = PatchNetVLADFeatureExtractor(config)
        elif global_descriptor_type == 'CosPlace':
            from feature_extraction.feature_extractor_cosplace import CosPlaceFeatureExtractor
            global_feature_extractor = CosPlaceFeatureExtractor()
        elif global_descriptor_type == 'EigenPlaces':
            from feature_extraction.feature_extractor_eigenplaces import EigenPlacesFeatureExtractor
            global_feature_extractor = EigenPlacesFeatureExtractor()
        else:
            raise ValueError('Unknown descriptor: ' + global_descriptor_type)
        return global_feature_extractor

    def get_max_score(self, g_des, img_id, max_num_results=5): 
        descriptor_dim = g_des.shape[1]
        global_des_database = np.array(self.global_des_database).reshape(-1, descriptor_dim)
        score = self.score(g_des, global_des_database)
        # update the similarity matrix
        score_color = float_to_color_array(score)    
        self.S_float[img_id, :global_des_database.shape[0]] = score      
        self.S_float[:global_des_database.shape[0], img_id] = score   
        if self.global_descriptor_type == 'SAD':
            self.S_color[:global_des_database.shape[0],:global_des_database.shape[0]] =  convert_float_to_colored_uint8_image(self.S_float[:global_des_database.shape[0],:global_des_database.shape[0]])
        else:
            self.S_color[img_id, :global_des_database.shape[0]] = score_color
            self.S_color[:global_des_database.shape[0], img_id] = score_color  
        best_idxs = np.argsort(-score)[:max_num_results+1]
        best_scores = score[best_idxs[1:]]
        return best_idxs, best_scores
        
    def compute_global_des(self, img):
        g_des = self.global_feature_extractor.compute_features_step(img)            
        return g_des
    
    def add(self, g_des):
        self.global_des_database.append(g_des)


kMinDeltaFrameForMeaningfulLoopClosure = 10
kMaxResultsForLoopClosure = 5

if __name__ == "__main__":
    
    dataset = dataset_factory(config)
    
    #global_descriptor_type = 'HDC-DELF'    # very slow
    #global_descriptor_type = 'SAD'          # fast 
    #global_descriptor_type = 'AlexNet'     # very slow
    #global_descriptor_type = 'NetVLAD'     # decently fast
    global_descriptor_type = 'CosPlace'    # decently fast
    #global_descriptor_type = 'EigenPlaces' # decently fast    
    
    global_feature_extractor = None

    global_des_database = GlobalFeatureDatabase(global_descriptor_type, dataset.num_frames)
    
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

            g_des = global_des_database.compute_global_des(img)
            global_des_database.add(g_des) # add image descriptors to database after having computed the scores
                        
            if entry_id > 1:            
                best_idxs, best_scores = global_des_database.get_max_score(g_des, img_id, max_num_results=kMaxResultsForLoopClosure+1) # we need plus one since we eliminate the best trivial equal to img_id

                # visualize non-trivial loop closures: we check the query results are not too close to the current image
                for idx, score in zip(best_idxs, best_scores):
                    if abs(idx - img_id) > kMinDeltaFrameForMeaningfulLoopClosure: 
                        print(f'result - best id: {idx}, score: {score}')
                        loop_img = dataset.getImageColor(idx)
                        loop_closure_imgs.add(loop_img, idx, score)
                                        
            img_writer.write(img, f'id: {img_id}', (30, 30))
            cv2.imshow('img', img)
            
            #cv2.imshow('S', convert_float_to_colored_uint8_image(global_des_database.S_float))
            cv2.imshow('S', global_des_database.S_color)
            
            if loop_closure_imgs.candidates is not None:
                cv2.imshow('loop_closure_imgs', loop_closure_imgs.candidates)
            
            cv2.waitKey(1)
        else: 
            getchar()
            
        img_id += 1
        entry_id += 1