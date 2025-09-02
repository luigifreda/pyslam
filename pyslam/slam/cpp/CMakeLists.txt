# CMakeLists.txt for PYSLAM C++ Core Module
# This file builds the cpp_core pybind11 module with all core SLAM classes

cmake_minimum_required(VERSION 3.12)
project(cpp_core)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for better IDE navigation (clangd / cpptools)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -march=native -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC -DNDEBUG")


# Find required packages
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)

set(PYSLAM_ROOT ${PROJECT_SOURCE_DIR}/../../../)
set(PYSLAM_SRC ${PYSLAM_ROOT}/pyslam)
set(PYSLAM_SLAM_SRC ${PYSLAM_SRC}/slam/cpp)
set(PYSLAM_SEMANTICS_SRC ${PYSLAM_SRC}/semantics/cpp)


include_directories(${PYSLAM_SLAM_SRC})
include_directories(${PYSLAM_SEMANTICS_SRC})


# json
set(CMAKE_PREFIX_PATH ${PYSLAM_ROOT}/thirdparty/json/install/share/cmake/nlohmann_json/)
set(JSON_DIR ${PYSLAM_ROOT}/thirdparty/json/install/)
message(STATUS "set custom json folder: ${JSON_DIR}")
find_package(nlohmann_json REQUIRED)

# Add pybind11 from thirdparty
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/pybind11" pybind11)

# Add g2o from thirdparty
set(G2O_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/g2opy")
set(G2O_INCLUDE_DIRS "${G2O_ROOT};${G2O_ROOT}/build/")  # the second for g2o/config.h
set(G2O_LIBRARIES "${G2O_ROOT}/lib")

message(STATUS "G2O_ROOT: ${G2O_ROOT}")
message(STATUS "G2O_INCLUDE_DIRS: ${G2O_INCLUDE_DIRS}")
message(STATUS "G2O_LIBRARIES: ${G2O_LIBRARIES}")

message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${G2O_INCLUDE_DIRS})

# cpp core

set(SLAM_CPP_CORE_SOURCES
    ${PYSLAM_SLAM_SRC}/cpp_core_module.cpp
    ${PYSLAM_SLAM_SRC}/map_point.cpp
    ${PYSLAM_SLAM_SRC}/map_point_serialization.cpp
    ${PYSLAM_SLAM_SRC}/frame.cpp
    ${PYSLAM_SLAM_SRC}/frame_serialization.cpp
    ${PYSLAM_SLAM_SRC}/keyframe.cpp
    ${PYSLAM_SLAM_SRC}/keyframe_serialization.cpp
    ${PYSLAM_SLAM_SRC}/map.cpp
    ${PYSLAM_SLAM_SRC}/map_serialization.cpp
    ${PYSLAM_SLAM_SRC}/map_points.cpp
    ${PYSLAM_SLAM_SRC}/camera_pose.cpp
    ${PYSLAM_SLAM_SRC}/camera.cpp
    ${PYSLAM_SLAM_SRC}/feature_shared_info.cpp
    ${PYSLAM_SLAM_SRC}/cpp_core_module.cpp
    ${PYSLAM_SLAM_SRC}/utils/features.cpp    
)

set(SLAM_CPP_CORE_HEADERS
    ${PYSLAM_SLAM_SRC}/map_point.h
    ${PYSLAM_SLAM_SRC}/frame.h
    ${PYSLAM_SLAM_SRC}/keyframe.h
    ${PYSLAM_SLAM_SRC}/map.h
    ${PYSLAM_SLAM_SRC}/camera_pose.h
    ${PYSLAM_SLAM_SRC}/camera.h
    ${PYSLAM_SLAM_SRC}/feature_shared_info.h
    ${PYSLAM_SLAM_SRC}/utils/descriptors.h    
    ${PYSLAM_SLAM_SRC}/utils/dictionary.h    
    ${PYSLAM_SLAM_SRC}/utils/features.h
    ${PYSLAM_SLAM_SRC}/utils/messages.h    
    ${PYSLAM_SLAM_SRC}/utils/serialization.h    
    ${PYSLAM_SLAM_SRC}/utils/cv_ops.h
    ${PYSLAM_SLAM_SRC}/casters/dictionary_casters.h
    ${PYSLAM_SLAM_SRC}/casters/opencv_type_casters.h
    ${PYSLAM_SLAM_SRC}/eigen_aliases.h
)

# semantics 

set(SEMANTICS_CPP_CORE_SOURCES
    ${PYSLAM_SEMANTICS_SRC}/semantic_fusion_methods.cpp
    ${PYSLAM_SEMANTICS_SRC}/semantic_serialization.cpp
)

set(SEMANTICS_CPP_CORE_HEADERS
    ${PYSLAM_SEMANTICS_SRC}/semantic_fusion_methods.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_serialization.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_types.h
)


# all source and header files

set(CPP_CORE_SOURCES
    ${SLAM_CPP_CORE_SOURCES}
    ${SEMANTICS_CPP_CORE_SOURCES}
)

set(CPP_CORE_HEADERS
    ${SLAM_CPP_CORE_HEADERS}
    ${SEMANTICS_CPP_CORE_HEADERS}
)

set(CPP_CORE_LINK_LIBRARIES
    ${OpenCV_LIBS}
    ${EIGEN3_LIBRARIES}
    ${G2O_ROOT}/lib/libg2o_core.a
    ${G2O_ROOT}/lib/libg2o_stuff.a
    ${G2O_ROOT}/lib/libg2o_types_slam3d.a
    ${G2O_ROOT}/lib/libg2o_solver_eigen.a
    ${G2O_ROOT}/lib/libg2o_solver_dense.a
    nlohmann_json::nlohmann_json
)

# Create the pybind11 module
pybind11_add_module(cpp_core ${CPP_CORE_SOURCES} ${CPP_CORE_HEADERS})

# Link libraries
target_link_libraries(cpp_core PRIVATE
    ${CPP_CORE_LINK_LIBRARIES}
)

# Compiler flags
target_compile_definitions(cpp_core PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
    PYBIND11_DETAILED_ERROR_MESSAGES
)

# Set output directory
set_target_properties(cpp_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Install target
install(TARGETS cpp_core
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

# Print configuration info
message(STATUS "Building PYSLAM C++ Core Module")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output directory: ${CMAKE_CURRENT_SOURCE_DIR}/lib")
