import sys 
import os
import numpy as np
import cv2
from matplotlib import pyplot as plt


from pyslam.config import Config

from pyslam.viz.mplot_figure import MPlotFigure

from pyslam.utilities.utils_geom import add_ones, poseRt, skew
from pyslam.utilities.utils_draw import draw_points2, draw_feature_matches
from pyslam.slam.search_points_test import search_frame_for_triangulation_test
from pyslam.slam.map_point import MapPoint
from pyslam.slam.slam import Slam
from pyslam.slam.camera  import Camera, PinholeCamera
from pyslam.slam.initializer import Initializer
from pyslam.utilities.timer import TimerFps

from pyslam.utilities.utils_sys import Printer

from pyslam.local_features.feature_tracker import FeatureTrackerTypes 

from pyslam.io.ground_truth import groundtruth_factory
from pyslam.io.dataset_factory import dataset_factory
from pyslam.utilities.timer import Timer

from pyslam.config_parameters import Parameters  

from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs


if __name__ == "__main__":

    config = Config()
    # forced camera settings to be kept coherent with the input file below 
    config.config[config.dataset_type]['settings'] = 'settings/KITTI04-12.yaml'
    config.sensor_type='mono'
    config.get_general_system_settings()  # parse again the settings file

    #dataset = dataset_factory(config)
    #grountruth = groundtruth_factory(config.dataset_settings)


    cam = PinholeCamera(config)

    #============================================
    # Init Feature Tracker   
    #============================================  

    num_features=2000 

    tracker_type = FeatureTrackerTypes.DES_BF      # descriptor-based, brute force matching with knn 
    #tracker_type = FeatureTrackerTypes.DES_FLANN  # descriptor-based, FLANN-based matching 

    # select your tracker configuration (see the file feature_tracker_configs.py) 
    feature_tracker_config = FeatureTrackerConfigs.TEST
    feature_tracker_config['num_features'] = num_features
    feature_tracker_config['match_ratio_test'] = 0.8        # 0.7 is the default
    feature_tracker_config['tracker_type'] = tracker_type

    #============================================
    # create SLAM object 
    #============================================
    slam = Slam(cam, feature_tracker_config) 

    timer = Timer()

    #============================================

    # N.B.: keep this coherent with the above forced camera settings 
    img_ref = cv2.imread('../data/kitti06-12.png',cv2.IMREAD_COLOR)
    #img_cur = cv2.imread('../data/kitti06-12-01.png',cv2.IMREAD_COLOR)
    img_cur = cv2.imread('../data/kitti06-13.png',cv2.IMREAD_COLOR)


    print(f'camera: {slam.tracking.camera.width}x{slam.tracking.camera.height}')

    slam.track(img_ref, img_id=0, img_right=None, depth=None)
    slam.track(img_ref, img_id=1, img_right=None, depth=None) # fake input to get an id-distance of 2 frames in the initializer
    slam.track(img_cur, img_id=2, img_right=None, depth=None)

    f_ref = slam.map.get_frame(-2)
    f_cur = slam.map.get_frame(-1)

    print('search for triangulation...')
    timer.start()
    
    print(f'max descriptor distance: {Parameters.kMaxDescriptorDistance}')

    img_cur_epi = None

    idxs_ref, idxs_cur, num_found_matches, img_cur_epi = search_frame_for_triangulation_test(f_ref, f_cur, img_cur, img1=img_ref)  # test
    #idxs_ref, idxs_cur, num_found_matches = search_frame_for_triangulation(f_ref, f_cur)

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
