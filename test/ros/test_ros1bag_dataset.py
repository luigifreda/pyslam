import sys
sys.path.append("../../")
from config import Config

from dataset_factory import dataset_factory
from dataset_types import SensorType, DatasetType, DatasetEnvironmentType
from dataset import Dataset

from ros1bag_dataset import Ros1bagSyncReaderATS, Ros1bagDataset

import numpy as np
import os
import cv2

import rosbag
import rospy
from cv_bridge import CvBridge

if not rospy.core.is_initialized():
    rospy.Time.now = lambda: rospy.Time.from_sec(0)
from message_filters import ApproximateTimeSynchronizer, SimpleFilter
from collections import defaultdict


    
    
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