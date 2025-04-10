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



