import sys
sys.path.append("../../")
from pyslam.config import Config

from pyslam.io.dataset_factory import dataset_factory
from pyslam.io.dataset_types import SensorType, DatasetType, DatasetEnvironmentType
from pyslam.io.dataset import Dataset

from pyslam.io.ros1bag_dataset import Ros1bagSyncReaderATS, Ros1bagDataset

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
    #dataset = dataset_factory(config)
    
    bag_path = config.ros_settings['bag_path']
    if not os.path.exists(bag_path):
        raise ValueError(f"Bag path {bag_path} does not exist")
    topics_dict = config.ros_settings['topics']
    topics_list = list(topics_dict.values())

    color_image_topic = topics_dict['color_image'] if 'color_image' in topics_dict else None
    depth_image_topic = topics_dict['depth_image'] if 'depth_image' in topics_dict else None
    left_color_image_topic = topics_dict['left_color_image'] if 'left_color_image' in topics_dict else None
    
    reader = Ros1bagSyncReaderATS(bag_path, topics_list, slop=0.05)
    bridge = CvBridge()
    
    num_colored_images = reader.topic_counts[topics_dict['color_image']]
    print(f'num_colored_images: {num_colored_images}')

    #for ts, synced in reader.read():
    while True:
        result = reader.read_step()
        if result is None:
            break
        ts, synced = result
        synced_msgs = {}
        timestamp = None
        if color_image_topic:
            color_img_msg = synced[color_image_topic]
            timestamp = color_img_msg.header.stamp
            synced_msgs[color_image_topic] = color_img_msg
            color_image = bridge.imgmsg_to_cv2(color_img_msg, desired_encoding="bgr8")
            cv2.imshow('color image', color_image)
        if depth_image_topic:
            depth_msg = synced[depth_image_topic]
            synced_msgs[depth_image_topic] = depth_msg
            depth_image = bridge.imgmsg_to_cv2(depth_msg, desired_encoding="passthrough")
            cv2.imshow('depth image', depth_image)
        if left_color_image_topic:
            left_color_img_msg = synced[left_color_image_topic]
            synced_msgs[left_color_image_topic] = left_color_img_msg
            left_color_image = bridge.imgmsg_to_cv2(left_color_img_msg, desired_encoding="bgr8")
            cv2.imshow('left color image', left_color_image)
        print(f'Synced @ {ts}: {[(name, msg.header.stamp) for name, msg in synced_msgs.items()]}')
        cv2.waitKey(30)