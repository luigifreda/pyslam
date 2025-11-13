cmake_minimum_required(VERSION 3.14)
project(ros2_pybindings)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SET_RPATH               ON  CACHE BOOL   "Set RPATH instead of the default RUNPATH")
set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter")
set(WITH_JAZZY OFF CACHE BOOL "Build with ROS2 Jazzy")
set(WITH_ROLLING OFF CACHE BOOL "Build with ROS2 Rolling")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -DNDEBUG")
if(SET_RPATH)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--disable-new-dtags")
endif()

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
find_package(cv_bridge REQUIRED)

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_DIR: ${OpenCV_DIR}")
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV_LIBS: ${OpenCV_LIBS}")

if(WITH_JAZZY)
  message(STATUS "WITH_JAZZY: ${WITH_JAZZY}")
  add_definitions(-DWITH_JAZZY)
endif()

if(WITH_ROLLING)
  message(STATUS "WITH_ROLLING: ${WITH_ROLLING}")
  add_definitions(-DWITH_ROLLING)
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
    ${cv_bridge_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)

target_link_libraries(ros2_pybindings
  PRIVATE
    rclcpp::rclcpp
    sensor_msgs::sensor_msgs__rosidl_typesupport_cpp 
    rosbag2_cpp::rosbag2_cpp
    message_filters::message_filters
    rosbag2_storage::rosbag2_storage
    ${cv_bridge_LIBRARIES}
    ${OpenCV_LIBS}
)

ament_package()
