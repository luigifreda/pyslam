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
from utils_features import ssc_nms

from collections import defaultdict, Counter

from feature_manager_configs import FeatureManagerConfigs
from feature_tracker_configs import FeatureTrackerConfigs

from timer import TimerFps

from feature_keynet import KeyNetDescFeature2D


# ==================================================================================================
# N.B.: here we test feature manager detectAndCompute()
# ==================================================================================================


timer = TimerFps()

#img = cv2.imread('../data/kitti06-12.png',cv2.IMREAD_COLOR)
#img = cv2.imread('../data/kitti06-435.png',cv2.IMREAD_COLOR)
img = cv2.imread('../data/kitti06-12-color.png',cv2.IMREAD_COLOR)
#img = cv2.imread('../data/mars1.png')

num_features=2000

 
# select your tracker configuration (see the file feature_tracker_configs.py) 
# FeatureTrackerConfigs: SHI_TOMASI_ORB, FAST_ORB, ORB, ORB2, ORB2_FREAK, ORB2_BEBLID, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, KEYNET, SUPERPOINT, CONTEXTDESC, LIGHTGLUE, XFEAT, XFEAT_XFEAT
feature_tracker_config = FeatureTrackerConfigs.ROOT_SIFT
feature_tracker_config['num_features'] = num_features

feature_manager_config = FeatureManagerConfigs.extract_from(feature_tracker_config)
print('feature_manager_config: ',feature_manager_config)
feature_manager = feature_manager_factory(**feature_manager_config)

des = None 

img_old = img.copy()
    
# loop for measuring time performance 
N=1
for i in range(N):
    timer.start()
    
    # just detect keypoints 
    #kps = feature_manager.detect(img) 
    
    # detect keypoints and compute descriptors 
    kps, des = feature_manager.detectAndCompute(img) 
    
    timer.refresh()

#sizes = np.array([x.size for x in kps], dtype=np.float32) 

print('#kps: ', len(kps))
des = np.array(des)
if des is not None: 
    print('des shape: ', des.shape)

#print('octaves: ', [p.octave for p in kps])
# count points for each octave
kps_octaves = [k.octave for k in kps]
kps_octaves = Counter(kps_octaves)
print('kps levels-histogram: \n', kps_octaves.most_common())    

imgDraw = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig = MPlotFigure(imgDraw[:,:,[2,1,0]], title='features')
MPlotFigure.show()