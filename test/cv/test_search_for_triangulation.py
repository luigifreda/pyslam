import sys 
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append("../../")

from config import Config

from mplot_figure import MPlotFigure

from frame import Frame, match_frames
from geom_helpers import triangulate_points, add_ones, poseRt, skew, draw_points2
from search_points import search_map_by_projection, search_frame_by_projection, search_local_frames_by_projection, search_frame_for_triangulation
from map_point import MapPoint
from slam import SLAM
from pinhole_camera import Camera, PinholeCamera
from initializer import Initializer
from timer import TimerFps

from helpers import Printer
import optimizer_g2o

from feature_tracker import feature_tracker_factory, TrackerTypes 
from feature_manager import feature_manager_factory, FeatureDetectorTypes, FeatureDescriptorTypes
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from ground_truth import groundtruth_factory
from dataset import dataset_factory
from timer import Timer

import parameters  

#------------- 

config = Config()
# forced camera settings to be kept coherent with the input file below 
config.config_parser[config.dataset_type]['cam_settings'] = 'settings/KITTI04-12.yaml'
config.get_cam_settings()

dataset = dataset_factory(config.dataset_settings)

grountruth = groundtruth_factory(config.dataset_settings)


cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                    config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                    config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                    config.DistCoef)


num_features=parameters.kNumFeatures 
"""
select your feature tracker 
N.B.: ORB detector (not descriptor) does not work as expected!
"""
tracker_type = TrackerTypes.DES_BF
#tracker_type = TrackerTypes.DES_FLANN
feature_tracker = feature_tracker_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.SHI_TOMASI, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)    
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 1, detector_type = FeatureDetectorTypes.FAST, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)
#feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 3, detector_type = FeatureDetectorTypes.BRISK, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)    
#feature_tracker = feature_tracker_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.ORB, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)


slam = SLAM(cam, feature_tracker, grountruth)

timer = Timer()

#------------- 

# N.B.: keep this coherent with the above forced camera settings 
img_ref = cv2.imread('../data/kitti06-12.png',cv2.IMREAD_COLOR)
#img_cur = cv2.imread('../data/kitti06-12-01.png',cv2.IMREAD_COLOR)
img_cur = cv2.imread('../data/kitti06-13.png',cv2.IMREAD_COLOR)


slam.track(img_ref, frame_id=0)
slam.track(img_cur, frame_id=1)

f_ref = slam.map.frames[-2]
f_cur = slam.map.frames[-1]

print('search for triangulation...')
timer.start()
idx_ref, idx_cur, num_found_matches, img_cur_epi = search_frame_for_triangulation(f_ref, f_cur, img_cur, img1=img_ref) 
elapsed = timer.elapsed()
print('time: ', elapsed)
print('# found matches: ', num_found_matches)

N = len(idx_ref)

pts_ref = f_ref.kpsu[idx_ref[:N]] 
pts_cur = f_cur.kpsu[idx_cur[:N]] 
    
img_ref, img_cur = draw_points2(img_ref, img_cur, pts_ref, pts_cur)    


if img_cur_epi is not None:
    fig1 = MPlotFigure(img_cur_epi, title='points and epipolar lines')

fig_ref = MPlotFigure(img_ref, title='image ref')
fig_cur = MPlotFigure(img_cur, title='image cur')

MPlotFigure.show()