# CMakeLists.txt for PYSLAM C++ Core Module
# This file builds the cpp_core pybind11 module with all core SLAM classes

cmake_minimum_required(VERSION 3.12)
project(cpp_core)

# Set C++ standard
set(CMAKE_CXX_STANDARD 17)
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

# json
set(PYSLAM_ROOT ${PROJECT_SOURCE_DIR}/../../../)
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

# Source files
set(CPP_CORE_SOURCES
    map_point.cpp
    frame.cpp
    keyframe.cpp
    map.cpp
    camera_pose.cpp
    camera.cpp
    cpp_core_module.cpp
)

# Header files
set(CPP_CORE_HEADERS
    map_point.h
    frame.h
    keyframe.h
    map.h
    camera_pose.h
    camera.h
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
pybind11_add_module(cpp_core ${CPP_CORE_SOURCES})

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
