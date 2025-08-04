#!/usr/bin/env -S python3 -O
import sys 
import numpy as np
import cv2
from matplotlib import pyplot as plt


from pyslam.config import Config

from pyslam.viz.mplot_figure import MPlotFigure
from pyslam.local_features.feature_manager import feature_manager_factory
from pyslam.local_features.feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from pyslam.utilities.utils_features import ssc_nms

from pyslam.local_features.feature_manager_configs import FeatureManagerConfigs
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs

from collections import defaultdict, Counter

from pyslam.utilities.timer import TimerFps


# ==================================================================================================
# N.B.: here we test feature manager detect() and compute() separately; 
#       results are shown in two separate windows  
# ==================================================================================================

timer = TimerFps()

#img = cv2.imread('../data/kitti06-12.png',cv2.IMREAD_COLOR) 
#img = cv2.imread('../data/kitti06-435.png',cv2.IMREAD_COLOR)
img = cv2.imread('../data/kitti06-12-color.png',cv2.IMREAD_COLOR) 
#img = cv2.imread('../data/mars1.png')

num_features=2000

# select your tracker configuration (see the file feature_tracker_configs.py) 
feature_tracker_config = FeatureTrackerConfigs.TEST
feature_tracker_config['num_features'] = num_features

feature_manager_config = FeatureManagerConfigs.extract_from(feature_tracker_config)
print('feature_manager_config: ',feature_manager_config)
feature_manager = feature_manager_factory(**feature_manager_config)

des = None 


print('img:', img.shape)

# loop for measuring time performance 
N=1
for i in range(N):
    timer.start()
                    
    # first, just detect keypoints 
    kps = feature_manager.detect(img, filter=True) 
    
    # then, compute descriptors (this operation may filter out some detected keypoints)
    kpsd, des = feature_manager.compute(img, kps, filter=False) 
        
    timer.refresh()

print('#kps: ', len(kps))
print('#kps after description: ', len(kpsd))
if des is not None: 
    print('des info:')
    np.info(des)
    print('des[0]',des[0])

#print('octaves: ', [p.octave for p in kps])

# count points for each octave
kps_octaves = [k.octave for k in kps]
kps_octaves = Counter(kps_octaves)
print('kps levels-histogram: \n', kps_octaves.most_common())    

kps_sizes = [kp.size for kp in kps] 
kps_sizes_histogram = np.histogram(kps_sizes, bins=10)
print('kps sizes-histogram: \n', list(zip(kps_sizes_histogram[1],kps_sizes_histogram[0])))


if False: 
    print('after computing descriptors:')

    kpsd_octaves = [k.octave for k in kpsd]
    kpsd_octaves = Counter(kpsd_octaves)
    print('kpsd octaves-histogram: \n', kpsd_octaves.most_common())    

    kpsd_sizes = [kp.size for kp in kpsd] 
    kpsd_sizes_histogram = np.histogram(kpsd_sizes, bins=10)
    print('kpsd sizes-histogram: \n', list(zip(kpsd_sizes_histogram[1],kpsd_sizes_histogram[0])))    


img_kps = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
img_kpsd = cv2.drawKeypoints(img, kpsd, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

fig1 = MPlotFigure(img_kps[:,:,[2,1,0]], title='detected keypoints')
fig2 = MPlotFigure(img_kpsd[:,:,[2,1,0]], title='computed features (keypoints after computing descriptors)')
MPlotFigure.show()