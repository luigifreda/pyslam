# ros2_pybindings

**Author**: Luigi Freda

This package provides **Python bindings** for synchronized reading of ROS 2 bag files using C++ and pybind11. It enables fast and flexible access to `rosbag2` data from Python, **without relying on ROS 2's native Python interface**, which is strictly tied to Python 3.8 in ROS 2.

## 🔧 Problem Solved

One of the major limitations of ROS 2 (Foxy and others) is its **strict dependency on Python 3.8**, due to precompiled rclpy binaries and message types.

**This package solves that problem.**  
By wrapping the C++ APIs of `rosbag2_cpp`, `rclcpp`, and `message_filters` using `pybind11`, it allows:

- Using **any modern Python version** (e.g. Python 3.9, 3.10, 3.11+) in your project
- Full control from Python without needing to launch ROS nodes
- Seamless integration with modern Python tools, packages, and environments

## ✅ Features

- Compatible with **Python ≥3.8+** (even outside of ROS workspaces)
- High-performance C++ backend using `rosbag2_cpp` and `rclcpp::Serialization`
- **Approximate time synchronization** for 1, 2, or 3 topics via `message_filters`
- Access to `sensor_msgs::msg::Image` messages from Python
- Built-in support for reading and timestamping all messages per topic
- Can be easily adapted to your needs 


## Build 

Just run the script `build.sh`. It will automatically detect your ros version by using the script `find_ros.sh`.


## Test 

See the provided `test.py` file.