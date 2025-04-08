import os
import sys 
sys.path.append("../../")

from config import Config
config = Config()

from utils_files import gdrive_download_lambda 
from utils_sys import getchar, Printer 
from utils_img import float_to_color, convert_float_to_colored_uint8_image, LoopCandidateImgs
from utils_features import transform_float_to_binary_descriptor

import math
import cv2 
import numpy as np

from dataset_factory import dataset_factory
from dataset_types import SensorType
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs
from feature_types import FeatureInfo

config.set_lib('pyibow')
import pyibow as ibow


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/../..'
kDataFolder = kRootFolder + '/data'
kOrbVocabFile = kDataFolder + '/ORBvoc.txt'


kMinDeltaFrameForMeaningfulLoopClosure = 10
kMaxResultsForLoopClosure = 5


# online loop closure detection by using DBoW3        
if __name__ == '__main__':
    
    dataset = dataset_factory(config)
    
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = 2000

    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)

    lc_detector_parameters = ibow.LCDetectorParams()
    lc_detector = ibow.LCDetector(lc_detector_parameters)
    
    # to nicely visualize current loop candidates in a single image
    loop_closure_imgs = LoopCandidateImgs()
    
    # init the similarity matrix
    S_float = np.empty([dataset.num_frames, dataset.num_frames], 'float32')
    S_color = np.empty([dataset.num_frames, dataset.num_frames, 3], 'uint8')
    #S_color = np.full([dataset.num_frames, dataset.num_frames, 3], 0, 'uint8') # loop closures are found with a small score, this will make them disappear    
    
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
            kps_ = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps]  # tuple_x_y_size_angle_response_octave
            
            des_ = des
            if FeatureInfo.norm_type[feature_tracker.descriptor_type] != cv2.NORM_HAMMING:
                des_ = transform_float_to_binary_descriptor(des)
            
            result = lc_detector.process(img_id, kps_, des_)
            
            float_value = result.score
            color_value = float_to_color(result.score)
            S_float[img_id, result.train_id] = float_value
            S_float[result.train_id, img_id] = float_value
            S_color[img_id, result.train_id] = color_value
            S_color[result.train_id, img_id] = color_value
            if result.isLoop():
                # visualize non-trivial loop closures: we check the query results are not too close to the current image
                if abs(result.train_id - img_id) > kMinDeltaFrameForMeaningfulLoopClosure: 
                    print(f'result - best id: {result.train_id}, score: {result.score}')
                    loop_img = dataset.getImageColor(result.train_id)
                    loop_closure_imgs.add(loop_img, result.train_id, result.score)

            if result.status == ibow.LCDetectorStatus.LC_DETECTED:
                print(f'Loop detected: {result.train_id}, #inliers: {result.inliers}, score: {result.score}')
            elif result.status == ibow.LCDetectorStatus.LC_NOT_DETECTED:
                print('No loop found')
            elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_IMAGES:
                print('Not enough images to found a loop')
            elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_ISLANDS:
                print('Not enough islands to found a loop')
            elif result.status == ibow.LCDetectorStatus.LC_NOT_ENOUGH_INLIERS:
                print('Not enough inliers')
            elif result.status == ibow.LCDetectorStatus.LC_TRANSITION:
                print('Transitional loop closure')
            else:
                print('No status information')
        
            font_pos = (50, 50)                   
            cv2.putText(img, f'id: {img_id}', font_pos, LoopCandidateImgs.kFont, LoopCandidateImgs.kFontScale, \
                        LoopCandidateImgs.kFontColor, LoopCandidateImgs.kFontThickness, cv2.LINE_AA)     
            cv2.imshow('img', img)
            
            cv2.imshow('S', S_color)            
            #cv2.imshow('S', convert_float_to_colored_uint8_image(S_float))
            
            if loop_closure_imgs.candidates is not None:
                cv2.imshow('loop_closure_imgs', loop_closure_imgs.candidates)
            
            cv2.waitKey(1)
        else: 
            getchar()
            
        img_id += 1
        entry_id += 1