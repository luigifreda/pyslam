cmake_minimum_required(VERSION 3.14)
project(ros2_pybindings)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter")
set(WITH_OPENCV_SHOW OFF CACHE BOOL "Debug with opencv")
set(WITH_JAZZY OFF CACHE BOOL "Build with ROS2 Jazzy")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -DNDEBUG")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")

# Force your Python 3.11 interpreter
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python3: ${Python3_EXECUTABLE}")

# Make sure your pybind11 is used, not ROS 2's
add_subdirectory(pybind11)
include_directories(BEFORE ${PROJECT_SOURCE_DIR}/pybind11/include)

# ROS 2 dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(rosbag2_storage REQUIRED)

if(WITH_OPENCV_SHOW)
  message(STATUS "WITH_OPENCV_SHOW: ${WITH_OPENCV_SHOW}")
  add_definitions(-DWITH_OPENCV_SHOW)
  find_package(OpenCV REQUIRED)
endif()

if(WITH_JAZZY)
  message(STATUS "WITH_JAZZY: ${WITH_JAZZY}")
  add_definitions(-DWITH_JAZZY)
endif()

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

# Python module
pybind11_add_module(ros2_pybindings
  ros2_pybindings.cpp
  ros2_bag_sync_reader.cpp
)

target_include_directories(ros2_pybindings
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
    ${Python3_INCLUDE_DIRS}
    ${rclcpp_INCLUDE_DIRS}
    ${sensor_msgs_INCLUDE_DIRS}
    ${rosbag2_cpp_INCLUDE_DIRS}
    ${message_filters_INCLUDE_DIRS}
    ${rosbag2_storage_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ros2_pybindings
  PRIVATE
    rclcpp::rclcpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp
    rosbag2_cpp::rosbag2_cpp
    message_filters::message_filters
    rosbag2_storage::rosbag2_storage
    ${OpenCV_LIBS}
)

ament_package()
