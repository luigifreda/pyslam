cmake_minimum_required(VERSION 3.14)
project(ros2_pybindings)


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 
set(WITH_OPENCV_SHOW OFF CACHE BOOL "Debug with opencv")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -DNDEBUG")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")


# the following 2 lines are added to correctly detect the python version 
if(WITH_PYTHON_INTERP_CHECK)
  message(STATUS "WITH_PYTHON_INTERP_CHECK: ${WITH_PYTHON_INTERP_CHECK}")
  find_package(PythonInterp) 
  #find_package(PythonLibs)
  message(STATUS "PythonInterp: ${PythonInterp}")
  #message(STATUS "PythonLibs: ${PythonLibs}")
endif()

# Find dependencies
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(rosbag2_cpp REQUIRED)
find_package(message_filters REQUIRED)
find_package(rosbag2_storage REQUIRED)

if (WITH_OPENCV_SHOW)
  message(STATUS "WITH_OPENCV_SHOW: ${WITH_OPENCV_SHOW}")
  add_definitions(-DWITH_OPENCV_SHOW)
  find_package(OpenCV REQUIRED)
endif()


set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(pybind11)

# Add Python module
pybind11_add_module(ros2_pybindings
ros2_pybindings.cpp
ros2_bag_sync_reader.cpp
)

# For message_filters headers
target_include_directories(ros2_pybindings
  PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}
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

# # Link message_filters manually if needed
# ament_target_dependencies(ros2_pybindings
#   rclcpp
#   sensor_msgs
#   rosbag2_cpp
#   message_filters
# )

# # Install for Python
# install(TARGETS ros2_pybindings
#   LIBRARY DESTINATION lib/${PROJECT_NAME}
# )

ament_package()
