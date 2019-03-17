import sys 
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append("../../")

from mplot_figure import MPlotFigure
from feature_detector import feature_detector_factory, FeatureDetectorTypes, FeatureDescriptorTypes

img = cv2.imread('kitti06-12.png',cv2.IMREAD_COLOR)

num_features=2000
# N.B.: here you can use just ORB descriptors!
#feature_detector = feature_detector_factory(min_num_features=num_features, num_levels=4, detector_type = FeatureDetectorTypes.SHI_TOMASI, descriptor_type = FeatureDescriptorTypes.ORB)    
feature_detector = feature_detector_factory(min_num_features=num_features, num_levels=2, detector_type = FeatureDetectorTypes.FAST, descriptor_type = FeatureDescriptorTypes.ORB)
#feature_detector = feature_detector_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.BRISK, descriptor_type = FeatureDescriptorTypes.ORB)
#feature_detector = feature_detector_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.ORB, descriptor_type = FeatureDescriptorTypes.ORB)

kps, des = feature_detector.detectAndCompute(img) 

#print('octaves: ', [p.octave for p in kps])

imgDraw = cv2.drawKeypoints(img, kps, None, color=(0,255,0), flags=0)

fig = MPlotFigure(imgDraw[:,:,[2,1,0]], title='features')
MPlotFigure.show()