import sys 
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt

sys.path.append("../../")

from config import Config

from mplot_figure import MPlotFigure

from frame import Frame, match_frames
from utils_geom import triangulate_points, add_ones, poseRt, skew
from utils_draw import draw_points2, draw_feature_matches
from search_points import search_map_by_projection, search_frame_by_projection, search_frame_for_triangulation
from search_points_test import search_frame_for_triangulation_test
from map_point import MapPoint
from slam import Slam
from camera  import Camera, PinholeCamera
from initializer import Initializer
from timer import TimerFps

from utils import Printer
import optimizer_g2o

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from ground_truth import groundtruth_factory
from dataset import dataset_factory
from timer import Timer

from parameters import Parameters  

from feature_tracker_configs import FeatureTrackerConfigs

#------------- 

config = Config()
# forced camera settings to be kept coherent with the input file below 
config.config_parser[config.dataset_type]['cam_settings'] = 'settings/KITTI04-12.yaml'
config.get_cam_settings()

#dataset = dataset_factory(config.dataset_settings)
#grountruth = groundtruth_factory(config.dataset_settings)


cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                    config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                    config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                    config.DistCoef,config.cam_settings['Camera.fps'])


#============================================
# Init Feature Tracker   
#============================================  

num_features=2000 

tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
#tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

# select your tracker configuration (see the file feature_tracker_configs.py) 
tracker_config = FeatureTrackerConfigs.TEST
tracker_config['num_features'] = num_features
tracker_config['match_ratio_test'] = 0.8        # 0.7 is the default
tracker_config['tracker_type'] = tracker_type

feature_tracker = feature_tracker_factory(**tracker_config)

#============================================
# create SLAM object 
#============================================
slam = Slam(cam, feature_tracker) #, grountruth)


timer = Timer()

#============================================

# N.B.: keep this coherent with the above forced camera settings 
img_ref = cv2.imread('../data/kitti06-12.png',cv2.IMREAD_COLOR)
#img_cur = cv2.imread('../data/kitti06-12-01.png',cv2.IMREAD_COLOR)
img_cur = cv2.imread('../data/kitti06-13.png',cv2.IMREAD_COLOR)


slam.track(img_ref, frame_id=0)
slam.track(img_cur, frame_id=1)

f_ref = slam.map.get_frame(-2)
f_cur = slam.map.get_frame(-1)

print('search for triangulation...')
timer.start()

idxs_ref, idxs_cur, num_found_matches, img_cur_epi = search_frame_for_triangulation_test(f_ref, f_cur, img_cur, img1=img_ref)  # test
#idxs_ref, idxs_cur, num_found_matches, img_cur_epi = search_frame_for_triangulation(f_ref, f_cur)

elapsed = timer.elapsed()
print('time:',elapsed)
print('# found matches:',num_found_matches)

N = len(idxs_ref)

pts_ref = f_ref.kpsu[idxs_ref[:N]] 
pts_cur = f_cur.kpsu[idxs_cur[:N]] 
    
img_ref, img_cur = draw_points2(img_ref, img_cur, pts_ref, pts_cur)    
img_matches = draw_feature_matches(img_ref, img_cur, pts_ref, pts_cur, horizontal=False)


if img_cur_epi is not None:
    fig1 = MPlotFigure(img_cur_epi, title='points and epipolar lines')

fig_ref = MPlotFigure(img_ref, title='image ref')
fig_cur = MPlotFigure(img_cur, title='image cur')
fig_matches = MPlotFigure(img_matches, title='image matches')

MPlotFigure.show()