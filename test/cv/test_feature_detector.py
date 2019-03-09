import sys 
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append("../../")
from feature_detector import feature_detector_factory, FeatureDetectorTypes, FeatureDescriptorTypes

img = cv2.imread('kitti06-12.png',cv2.IMREAD_COLOR)

num_features=2000
# N.B.: here you can use just ORB descriptors!
feature_detector = feature_detector_factory(min_num_features=num_features, num_levels=8, detector_type = FeatureDetectorTypes.SHI_TOMASI, descriptor_type = FeatureDescriptorTypes.ORB)    
#feature_detector = feature_detector_factory(min_num_features=num_features, num_levels=3, detector_type = FeatureDetectorTypes.FAST, descriptor_type = FeatureDescriptorTypes.ORB)
#feature_detector = feature_detector_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.BRISK, descriptor_type = FeatureDescriptorTypes.ORB)
#feature_detector = feature_detector_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.ORB, descriptor_type = FeatureDescriptorTypes.ORB)

kps, des = feature_detector.detectAndCompute(img) 

imgDraw = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

#fig1 = plt.figure()
#plt.imshow(img[:,:,[2,1,0]], cmap = 'gray', interpolation = 'bicubic')
#plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

fig = plt.figure()
plt.imshow(imgDraw[:,:,[2,1,0]])
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis

figManager = plt.get_current_fig_manager() 
#figManager.full_screen_toggle() 
figManager

plt.show()