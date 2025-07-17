import sys 
sys.path.append("../../")

from pyslam.config import Config
config = Config()
config.set_lib('raft_stereo') 

import torch
import numpy as np
import cv2
import time

from pyslam.depth_estimation.depth_estimator_raft_stereo import DepthEstimatorRaftStereo
from pyslam.utilities.utils_depth import img_from_depth

data_path = '../data'

if __name__ == "__main__":

    depth_estimator = DepthEstimatorRaftStereo()

    #imgfile1 = data_path + '/stereo_bicycle/im0.png'
    #imgfile2 = data_path + '/stereo_bicycle/im1.png'
    
    imgfile1 = data_path + '/kitti06-12-color.png'
    imgfile2 = data_path + '/kitti06-12-R-color.png'
    
    image1 = cv2.imread(imgfile1)
    image2 = cv2.imread(imgfile2)

    start_time = time.time()    
    depth_prediction, pts3d_prediction = depth_estimator.infer(image1, image2)
    
    end_time = time.time()
    print('Time taken for stereo depth prediction: ', end_time-start_time)
    
    depth_img = img_from_depth(depth_prediction)
    
    stereo_pair = np.concatenate([image1, image2], axis=1)
    cv2.imshow("stereo pair", stereo_pair)
    cv2.imshow("depth", depth_img)
    cv2.waitKey(0)