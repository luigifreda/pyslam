import sys
sys.path.append("../../")
from pyslam.config import Config

from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType, DatasetType, DatasetEnvironmentType
from pyslam.io.dataset import Dataset

import numpy as np
import os
import cv2
    
    
if __name__ == "__main__":

    config = Config()
    dataset = dataset_factory(config)

    frame_id = 0
    while True: 
        
        img, img_right, depth = None, None, None
        if dataset.isOk():
            img = dataset.getImageColor(frame_id)
            img_right = dataset.getImageColorRight(frame_id) if dataset.sensor_type == SensorType.STEREO else None
            depth = dataset.getDepth(frame_id)
            timestamp = dataset.getTimestamp()
            frame_id += 1
        else: 
            print(f'Dataset end: {dataset.name}, path: {dataset.path}')
            break 
        
        if img is not None:
            cv2.imshow("color_image", img)
        if img_right is not None:
            cv2.imshow("right_color_image", img_right)
        if depth is not None:
            cv2.imshow("depth_image", depth)
        cv2.waitKey(30)