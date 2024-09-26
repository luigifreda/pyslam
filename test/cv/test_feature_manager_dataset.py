#!/usr/bin/env -S python3 -O
import sys 
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append("../../")
from config import Config

from mplot_figure import MPlotFigure
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from dataset import dataset_factory, SensorType

from collections import defaultdict, Counter

from feature_manager_configs import FeatureManagerConfigs
from feature_tracker_configs import FeatureTrackerConfigs

from timer import TimerFps


# ==================================================================================================
# N.B.: here we test feature manager detectAndCompute()
# ==================================================================================================

if __name__ == "__main__":

    config = Config()

    dataset = dataset_factory(config)
    
    timer = TimerFps()

    num_features=2000

    
    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, FAST_TFEAT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
    feature_tracker_config = FeatureTrackerConfigs.ORB2
    feature_tracker_config['num_features'] = num_features

    feature_manager_config = FeatureManagerConfigs.extract_from(feature_tracker_config)
    print('feature_manager_config: ',feature_manager_config)
    feature_manager = feature_manager_factory(**feature_manager_config)

    des = None 

    img_id = 0  #180, 340, 400   # you can start from a desired frame id if needed 
    while dataset.isOk():
        
        print('..................................')
        print('image: ', img_id)                
        img = dataset.getImageColor(img_id)        
        timer.start()
        
        # just detect keypoints 
        #kps = feature_manager.detect(img) 
        
        # detect keypoints and compute descriptors 
        kps, des = feature_manager.detectAndCompute(img) 
            
        timer.refresh()

        print('#kps: ', len(kps))
        if des is not None: 
            print('des shape: ', des.shape)

        #print('octaves: ', [p.octave for p in kps])
        # count points for each octave
        kps_octaves = [k.octave for k in kps]
        kps_octaves = Counter(kps_octaves)
        print('kps levels-histogram: \n', kps_octaves.most_common())    

        imgDraw = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

        cv2.imshow('image', imgDraw)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_id += 1