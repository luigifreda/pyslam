import sys 
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append("../../")
from config import Config

from mplot_figure import MPlotFigure
from feature_manager import feature_manager_factory, FeatureDetectorTypes, FeatureDescriptorTypes

img = cv2.imread('../data/kitti06-12.png',cv2.IMREAD_COLOR)

num_features=2000
#feature_detector = feature_manager_factory(min_num_features=num_features, num_levels=4, detector_type = FeatureDetectorTypes.SHI_TOMASI, descriptor_type = FeatureDescriptorTypes.ORB)    
#feature_detector = feature_manager_factory(min_num_features=num_features, num_levels=2, detector_type = FeatureDetectorTypes.FAST, descriptor_type = FeatureDescriptorTypes.ORB)
#feature_detector = feature_manager_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.BRISK, descriptor_type = FeatureDescriptorTypes.ORB)
#feature_detector = feature_manager_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.ORB, descriptor_type = FeatureDescriptorTypes.ORB)
#feature_detector = feature_manager_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.FREAK, descriptor_type = FeatureDescriptorTypes.FREAK)
feature_detector = feature_manager_factory(min_num_features=num_features, num_levels=1, detector_type = FeatureDetectorTypes.SUPERPOINT, descriptor_type = FeatureDescriptorTypes.SUPERPOINT)
#feature_detector = feature_manager_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.ROOT_SIFT, descriptor_type = FeatureDescriptorTypes.ROOT_SIFT)

kps, des = feature_detector.detectAndCompute(img) 
print('kps length: ', len(kps))
print('des shape: ', des.shape)

#print('octaves: ', [p.octave for p in kps])

imgDraw = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

fig = MPlotFigure(imgDraw[:,:,[2,1,0]], title='features')
MPlotFigure.show()