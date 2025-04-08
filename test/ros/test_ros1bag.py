import sys
sys.path.append("../../")
from config import Config


import numpy as np
import subprocess
import os
import rosbag


if __name__ == "__main__":

    config = Config()
    
    bag_path = config.ros_settings['bag_path']
    if not os.path.exists(bag_path):
        raise ValueError(f"Bag path {bag_path} does not exist")
    topics = config.ros_settings['topics'].values()
    print(f'topics: {topics}')
    
    bag = rosbag.Bag(bag_path)

    for topic, msg, t in bag.read_messages(topics=topics):
        print(f"Topic: {topic}")
        print(f"Message: {msg}")
        print(f"Time: {t}")

    bag.close()



