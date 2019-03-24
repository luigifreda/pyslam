"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np
import cv2
import math

from config import Config

from slam import SLAM
from pinhole_camera import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

from display2D import Display2D
from viewer3D import Viewer3D
from helpers import getchar

from feature_tracker import feature_tracker_factory, TrackerTypes 
from feature_detector import feature_detector_factory, FeatureDetectorTypes, FeatureDescriptorTypes
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

import parameters  

if __name__ == "__main__":

    config = Config()

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

    feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 3, detector_type = FeatureDetectorTypes.SHI_TOMASI, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)    
    #feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 4, detector_type = FeatureDetectorTypes.FAST, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)
    #feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 4, detector_type = FeatureDetectorTypes.BRISK, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)    
    #feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 4, detector_type = FeatureDetectorTypes.BRISK, descriptor_type = FeatureDescriptorTypes.BRISK, tracker_type = tracker_type)     
    #feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 4, detector_type = FeatureDetectorTypes.AKAZE, descriptor_type = FeatureDescriptorTypes.AKAZE, tracker_type = tracker_type)    
    #feature_tracker = feature_tracker_factory(min_num_features=num_features, num_levels = 4, detector_type = FeatureDetectorTypes.ORB, descriptor_type = FeatureDescriptorTypes.ORB, tracker_type = tracker_type)
    #feature_tracker = feature_tracker_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.SIFT, descriptor_type = FeatureDescriptorTypes.SIFT, tracker_type = tracker_type)
    #feature_tracker = feature_tracker_factory(min_num_features=num_features, detector_type = FeatureDetectorTypes.SURF, descriptor_type = FeatureDescriptorTypes.SURF, tracker_type = tracker_type)

    # create SLAM object 
    slam = SLAM(cam, feature_tracker, grountruth)

    viewer3D = Viewer3D()
    display2d = Display2D(cam.width, cam.height) 

    is_draw_matched_points = True  
    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')    

    img_id = 0 #200   # you can start from a desired frame id if needed 
    while dataset.isOk():
            
        print('..................................')
        print('frame: ', img_id)    

        img = dataset.getImageColor(img_id)

        if img is not None:
            slam.track(img, img_id)

            # 3D display (map display)
            if viewer3D is not None:
                viewer3D.drawMap(slam.map)

            # 2D display (image display)
            if display2d is not None:
                img_draw = slam.map.annotate(img)
                display2d.draw(img_draw)

            if is_draw_matched_points and slam.num_matched_kps is not None: 
                if slam.num_matched_kps is not None: 
                    matched_kps_signal = [img_id, slam.num_matched_kps]     
                    matched_points_plt.draw(matched_kps_signal,'# keypoint matches',color='r')                         
                if slam.num_inliers is not None: 
                    inliers_signal = [img_id, slam.num_inliers]                    
                    matched_points_plt.draw(inliers_signal,'# inliers',color='g')
                valid_matched_map_points_signal = [img_id, slam.num_vo_map_points]   # valid matched map points (in current pose optimization)                                       
                matched_points_plt.draw(valid_matched_map_points_signal,'# matched map pts', color='b')                                           
                matched_points_plt.refresh()                   

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_id += 1

        step = False    
        if step and img_id > 1:
            getchar()  # stop at each frame

    cv2.waitKey(0)

    cv2.destroyAllWindows()