
# How to convert ros1 bag to ros2 bag 

from https://docs.ros.org/en/noetic/api/ov_core/html/dev-ros1-to-ros2.html

```bash
pip3 install rosbags>=0.9.11
rosbags-convert V1_01_easy.bag --dst <ros2_bag_folder>
```

This is using the package https://pypi.org/project/rosbags/