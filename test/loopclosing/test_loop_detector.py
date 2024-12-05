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

from dataset import dataset_factory
from frame import Frame
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

from parameters import Parameters
Parameters.kLoopClosingDebugAndPrintToFile = False
Parameters.kLoopClosingDebugWithSimmetryMatrix = True
Parameters.kLoopClosingDebugWithLoopDetectionImages = True

from loop_detector_dbow2 import LoopDetectorDBoW2
from loop_detector_configs import LoopDetectorConfigs, loop_detector_factory, SlamFeatureManagerInfo
from loop_detector_base import LoopDetectorTask, LoopDetectorTaskType, LoopDetectKeyframeData


# online loop closure detection by using DBoW3        
if __name__ == '__main__':
    
    dataset = dataset_factory(config)
    
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = 2000
    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    # This is normally done by the Slam class we don't have here. We need to set the static field of the class Frame and FrameShared. 
    Frame.set_tracker(feature_tracker)     
    
    # Select your loop closing configuration (see the file loop_detector_configs.py). Set it to None to disable loop closing. 
    # LoopDetectorConfigs: DBOW2, DBOW3, etc.
    loop_detection_config = LoopDetectorConfigs.DBOW3  
    Printer.green('loop_detection_config: ',loop_detection_config)
    loop_detector = loop_detector_factory(**loop_detection_config, slam_info=SlamFeatureManagerInfo(feature_manager=feature_tracker.feature_manager))
    
    img_writer = ImgWriter(font_scale=0.7)    
    
    cv2.namedWindow('similarity matrix', cv2.WINDOW_NORMAL) # to get a resizable window
    cv2.namedWindow('loop detection candidates', cv2.WINDOW_NORMAL) # to get a resizable window
        
    img_id = 0   #180, 340, 400   # you can start from a desired frame id if needed 
    while True:

        timestamp, img = None, None 
        
        if dataset.isOk():
            timestamp = dataset.getTimestamp()          # get current timestamp 
            img = dataset.getImageColor(img_id)

        if img is not None:
            print('----------------------------------------')
            print(f'processing img {img_id}')
            
            # Find the keypoints and descriptors in img1
            kps, des = feature_tracker.detectAndCompute(img)   # with DL matchers this a null operation 
            kps_ = [(kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps]  # tuple_x_y_size_angle_response_octave
            
            task_type = LoopDetectorTaskType.LOOP_CLOSURE
            covisible_keyframes = []
            connected_keyframes = []
            keyframe = LoopDetectKeyframeData()
            keyframe.id = img_id
            keyframe.img = img
            keyframe.kps = kps_
            keyframe.des = des
            task = LoopDetectorTask(keyframe, img, task_type, covisible_keyframes=covisible_keyframes, connected_keyframes=connected_keyframes)        
                        
            # check and compute if needed the local descriptors by using the independent local feature manager (if present). 
            loop_detector.compute_local_des_if_needed(task)
            # run the loop detection task
            detection_output = loop_detector.run_task(task)
                    
            img_writer.write(img, f'id: {img_id}', (30, 30))
            cv2.imshow('img', img)
            
            if detection_output.similarity_matrix is not None:
                cv2.imshow('similarity matrix', detection_output.similarity_matrix)            
        
            if detection_output.loop_detection_img_candidates is not None:
                cv2.imshow('loop detection candidates', detection_output.loop_detection_img_candidates)
            
            cv2.waitKey(1)
        else: 
            cv2.waitKey(100)
            
        img_id += 1