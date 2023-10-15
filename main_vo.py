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

# -----------------------------------------------------------
# fix from https://github.com/tum-vision/fastfusion/issues/21
import gi
gi.require_version('Gtk', '2.0')
# -----------------------------------------------------------

import numpy as np
import cv2
import math

from config import Config

from visual_odometry import VisualOdometry
from camera  import PinholeCamera
from ground_truth import groundtruth_factory
from dataset import dataset_factory

#from mplot3d import Mplot3d
#from mplot2d import Mplot2d
from mplot_thread import Mplot2d, Mplot3d

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from feature_tracker_configs import FeatureTrackerConfigs



"""
use or not pangolin (if you want to use it then you need to install it by using the script install_thirdparty.sh)
"""
kUsePangolin = False  

if kUsePangolin:
    from viewer3D import Viewer3D




if __name__ == "__main__":

    config = Config()

    dataset = dataset_factory(config.dataset_settings)

    groundtruth = groundtruth_factory(config.dataset_settings)

    cam = PinholeCamera(config.cam_settings['Camera.width'], config.cam_settings['Camera.height'],
                        config.cam_settings['Camera.fx'], config.cam_settings['Camera.fy'],
                        config.cam_settings['Camera.cx'], config.cam_settings['Camera.cy'],
                        config.DistCoef, config.cam_settings['Camera.fps'])


    num_features=2000  # how many features do you want to detect and track?

    # select your tracker configuration (see the file feature_tracker_configs.py) 
    # LK_SHI_TOMASI, LK_FAST
    # SHI_TOMASI_ORB, FAST_ORB, ORB, BRISK, AKAZE, FAST_FREAK, SIFT, ROOT_SIFT, SURF, SUPERPOINT, FAST_TFEAT
    tracker_config = FeatureTrackerConfigs.LK_SHI_TOMASI
    tracker_config['num_features'] = num_features
    
    feature_tracker = feature_tracker_factory(**tracker_config)

    # create visual odometry object 
    vo = VisualOdometry(cam, groundtruth, feature_tracker)

    is_draw_traj_img = True
    traj_img_size = 800
    traj_img = np.zeros((traj_img_size, traj_img_size, 3), dtype=np.uint8)
    half_traj_img_size = int(0.5*traj_img_size)
    draw_scale = 1

    is_draw_3d = True
    if kUsePangolin:
        viewer3D = Viewer3D()
    else:
        plt3d = Mplot3d(title='3D trajectory')

    is_draw_err = True 
    err_plt = Mplot2d(xlabel='img id', ylabel='m',title='error')

    is_draw_matched_points = True 
    matched_points_plt = Mplot2d(xlabel='img id', ylabel='# matches',title='# matches')

    img_id = 0
    while dataset.isOk():

        img = dataset.getImage(img_id)

        if img is not None:

            vo.track(img, img_id)  # main VO function 

            if(img_id > 2):	       # start drawing from the third image (when everything is initialized and flows in a normal way)

                x, y, z = vo.traj3d_est[-1]
                x_true, y_true, z_true = vo.traj3d_gt[-1]

                if is_draw_traj_img:      # draw 2D trajectory (on the plane xz)
                    draw_x, draw_y = int(draw_scale*x) + half_traj_img_size, half_traj_img_size - int(draw_scale*z)
                    true_x, true_y = int(draw_scale*x_true) + half_traj_img_size, half_traj_img_size - int(draw_scale*z_true)
                    cv2.circle(traj_img, (draw_x, draw_y), 1,(img_id*255/4540, 255-img_id*255/4540, 0), 1)   # estimated from green to blue
                    cv2.circle(traj_img, (true_x, true_y), 1,(0, 0, 255), 1)  # groundtruth in red
                    # write text on traj_img
                    cv2.rectangle(traj_img, (10, 20), (600, 60), (0, 0, 0), -1)
                    text = "Coordinates: x=%2fm y=%2fm z=%2fm" % (x, y, z)
                    cv2.putText(traj_img, text, (20, 40), cv2.FONT_HERSHEY_PLAIN, 1, (255, 255, 255), 1, 8)
                    # show 		
                    cv2.imshow('Trajectory', traj_img)

                if is_draw_3d:           # draw 3d trajectory 
                    if kUsePangolin:
                        viewer3D.draw_vo(vo)   
                    else:
                        plt3d.drawTraj(vo.traj3d_gt,'ground truth',color='r',marker='.')
                        plt3d.drawTraj(vo.traj3d_est,'estimated',color='g',marker='.')
                        plt3d.refresh()

                if is_draw_err:         # draw error signals 
                    errx = [img_id, math.fabs(x_true-x)]
                    erry = [img_id, math.fabs(y_true-y)]
                    errz = [img_id, math.fabs(z_true-z)] 
                    err_plt.draw(errx,'err_x',color='g')
                    err_plt.draw(erry,'err_y',color='b')
                    err_plt.draw(errz,'err_z',color='r')
                    err_plt.refresh()    

                if is_draw_matched_points:
                    matched_kps_signal = [img_id, vo.num_matched_kps]
                    inliers_signal = [img_id, vo.num_inliers]                    
                    matched_points_plt.draw(matched_kps_signal,'# matches',color='b')
                    matched_points_plt.draw(inliers_signal,'# inliers',color='g')                    
                    matched_points_plt.refresh()                    


            # draw camera image 
            cv2.imshow('Camera', vo.draw_img)				

        # press 'q' to exit!
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        img_id += 1

    #print('press a key in order to exit...')
    #cv2.waitKey(0)

    if is_draw_traj_img:
        print('saving map.png')
        cv2.imwrite('map.png', traj_img)
    if is_draw_3d:
        if not kUsePangolin:
            plt3d.quit()
        else: 
            viewer3D.quit()
    if is_draw_err:
        err_plt.quit()
    if is_draw_matched_points is not None:
        matched_points_plt.quit()
                
    cv2.destroyAllWindows()
