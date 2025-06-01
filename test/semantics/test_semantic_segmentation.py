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

from dataset_factory import dataset_factory
from frame import Frame, FeatureTrackerShared
from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_tracker_configs import FeatureTrackerConfigs

from semantic_segmentation_factory import semantic_segmentation_factory, SemanticSegmentationType
from semantic_utils import SemanticDatasetType, labels_color_map_factory
from semantic_types import SemanticFeatureType

from semantic_segmentation_deep_lab_v3 import SemanticSegmentationDeepLabV3
from semantic_segmentation_segformer import SemanticSegmentationSegformer
from semantic_segmentation_clip import SemanticSegmentationCLIP


from config_parameters import Parameters
Parameters.kLoopClosingDebugAndPrintToFile = False
Parameters.kLoopClosingDebugWithSimmetryMatrix = True
Parameters.kLoopClosingDebugWithLoopDetectionImages = True


# online loop closure detection by using DBoW3        
if __name__ == '__main__':
    
    dataset = dataset_factory(config)
    
    tracker_config = FeatureTrackerConfigs.ORB2
    tracker_config['num_features'] = 2000
    print('tracker_config: ',tracker_config)    
    feature_tracker = feature_tracker_factory(**tracker_config)
    
    # This is normally done by the Slam class we don't have here. We need to set the static field of the class Frame and FeatureTrackerShared. 
    FeatureTrackerShared.set_feature_tracker(feature_tracker)     
    
    # Select your semantic segmentation configuration (see the file semantics/semantic_segmentation_factory.py)
    
    semantic_segmentation_type=SemanticSegmentationType.SEGFORMER
    semantic_feature_type=SemanticFeatureType.LABEL
    semantic_dataset_type=SemanticDatasetType.CITYSCAPES
    image_size = (512, 512)
    device = None # autodetect
    semantic_segmentation = semantic_segmentation_factory(semantic_segmentation_type=semantic_segmentation_type,
                               semantic_feature_type=semantic_feature_type,
                               semantic_dataset_type=semantic_dataset_type, image_size=image_size, device=device)
    Printer.green(f'semantic_segmentation_type: {semantic_segmentation_type.name}')
    
    semantics_color_map = None
    if semantic_dataset_type != SemanticDatasetType.FEATURE_SIMILARITY:
        semantics_color_map = labels_color_map_factory(semantic_dataset_type)

    
        
    img_writer = ImgWriter(font_scale=0.7)    
    
    cv2.namedWindow('semantic prediction', cv2.WINDOW_NORMAL) # to get a resizable window
        
    img_id = 0   #180, 340, 400   # you can start from a desired frame id if needed 
    key = None
    while True:

        timestamp, img = None, None 
        
        if dataset.isOk():
            timestamp = dataset.getTimestamp()          # get current timestamp 
            img = dataset.getImageColor(img_id)

        if img is not None:
            print('----------------------------------------')
            print(f'processing img {img_id}')
            
            curr_semantic_prediction = semantic_segmentation.infer(img)  
            semantic_color_img = semantic_segmentation.to_rgb(curr_semantic_prediction, bgr=True)
                    
            img_writer.write(img, f'id: {img_id}', (30, 30))
            cv2.imshow('img', img)
            
            cv2.imshow('semantic prediction', semantic_color_img)
            
            key = cv2.waitKey(1)
        else: 
            key = cv2.waitKey(100)
            
        if key == ord('q') or key == 27:
            break
            
        img_id += 1