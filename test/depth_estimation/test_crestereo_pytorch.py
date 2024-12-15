import sys 
sys.path.append("../../")

from config import Config

import numpy as np
import cv2
import time

from camera import Camera
from depth_estimator_factory import depth_estimator_factory, DepthEstimatorType
from utils_depth import img_from_depth

data_path = '../data'

if __name__ == "__main__":

    camera = Camera(config=None)
    camera.fx = 707.0912
    camera.bf = 379.8145
    camera.b = camera.bf / camera.fx

    #depth_estimator = DepthEstimatorCrestereoPytorch(camera=camera)

    #Select your depth estimator (see the file depth_estimator_configs.py).
    depth_estimator_type = DepthEstimatorType.DEPTH_CRESTEREO
    depth_estimator = depth_estimator_factory(depth_estimator_type=depth_estimator_type, camera=camera)
    
    #imgfile1 = data_path + '/stereo_bicycle/im0.png'
    #imgfile2 = data_path + '/stereo_bicycle/im1.png'
    
    imgfile1 = data_path + '/kitti06-12-color.png'
    imgfile2 = data_path + '/kitti06-12-R-color.png'
    
    image1 = cv2.imread(imgfile1)
    image2 = cv2.imread(imgfile2)

    start_time = time.time()    
    depth_map = depth_estimator.infer(image1, image2)
    
    end_time = time.time()
    print('Time taken for stereo depth prediction: ', end_time-start_time)
    
    depth_img = img_from_depth(depth_map)
    
    stereo_pair = np.concatenate([image1, image2], axis=1)
    cv2.imshow("stereo pair", stereo_pair)
    cv2.imshow("depth", depth_img)
    cv2.waitKey(0)