# CMakeLists.txt for PYSLAM C++ Core Module
# This file builds the cpp_core pybind11 module with all core SLAM classes

cmake_minimum_required(VERSION 3.12)
project(cpp_core)

set(PYSLAM_CPP_CORE_VERSION_INFO 0.1.0)


OPTION(USE_PYTHON "Use Python" ON)

# Set C++ standard
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Export compile commands for better IDE navigation (clangd / cpptools)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)


IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

message(STATUS "C++ core build type: " ${CMAKE_BUILD_TYPE})

if (CMAKE_BUILD_TYPE MATCHES Release)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -march=native -fPIC -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC -DNDEBUG")
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O1 -march=native -fPIC -pg")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -march=native -fPIC -pg")
endif()

if (USE_PYTHON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PYTHON=1")
endif()


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
set(pybind11_INCLUDE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/pybind11/include")
include_directories(${pybind11_INCLUDE_DIRS})

# Add Python development
if (USE_PYTHON)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
    message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
    include_directories(${Python_INCLUDE_DIRS})
endif()

# Add g2o from thirdparty
set(G2O_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../../thirdparty/g2opy")
set(G2O_INCLUDE_DIRS "${G2O_ROOT};${G2O_ROOT}/build/")  # the second for g2o/config.h

# find all the libs in the g2o lib directory
file(GLOB G2O_LIBRARIES "${G2O_ROOT}/lib/*.a")
# exclude g2o.cython* 
list(FILTER G2O_LIBRARIES EXCLUDE REGEX "g2o.cython.*")
# G2O_LIBRARIES already contains full paths, so we can use them directly
set(G2O_LIBRARIES "${G2O_LIBRARIES}")

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
    ${PYSLAM_SLAM_SRC}/map_point.cpp
    ${PYSLAM_SLAM_SRC}/map_point_serialization_json.cpp
    ${PYSLAM_SLAM_SRC}/frame.cpp
    ${PYSLAM_SLAM_SRC}/frame_serialization_json.cpp
    ${PYSLAM_SLAM_SRC}/keyframe.cpp
    ${PYSLAM_SLAM_SRC}/keyframe_serialization_json.cpp
    ${PYSLAM_SLAM_SRC}/map.cpp
    ${PYSLAM_SLAM_SRC}/map_serialization_json.cpp
    ${PYSLAM_SLAM_SRC}/map_points.cpp
    ${PYSLAM_SLAM_SRC}/camera_pose.cpp
    ${PYSLAM_SLAM_SRC}/camera.cpp
    ${PYSLAM_SLAM_SRC}/camera_serialization_json.cpp
    ${PYSLAM_SLAM_SRC}/feature_shared_resources.cpp
    ${PYSLAM_SLAM_SRC}/utils/features.cpp    
    ${PYSLAM_SLAM_SRC}/utils/geom_2views.cpp
    ${PYSLAM_SLAM_SRC}/optimizer_g2o.cpp
    ${PYSLAM_SLAM_SRC}/tracking_utils.cpp
    ${PYSLAM_SLAM_SRC}/config_parameters.cpp
    ${PYSLAM_SLAM_SRC}/geometry_matchers.cpp
)

if (USE_PYTHON)
    set(SLAM_CPP_CORE_SOURCES
        ${SLAM_CPP_CORE_SOURCES}
        ${PYSLAM_SLAM_SRC}/camera_serialization_py.cpp
        ${PYSLAM_SLAM_SRC}/map_point_serialization_py.cpp
        ${PYSLAM_SLAM_SRC}/frame_serialization_py.cpp
        ${PYSLAM_SLAM_SRC}/keyframe_serialization_py.cpp
    )
endif()

set(SLAM_CPP_CORE_HEADERS
    ${PYSLAM_SLAM_SRC}/ckdtree_eigen.h
    ${PYSLAM_SLAM_SRC}/map_point.h
    ${PYSLAM_SLAM_SRC}/frame.h
    ${PYSLAM_SLAM_SRC}/keyframe.h
    ${PYSLAM_SLAM_SRC}/map.h
    ${PYSLAM_SLAM_SRC}/camera_pose.h
    ${PYSLAM_SLAM_SRC}/camera.h
    ${PYSLAM_SLAM_SRC}/feature_shared_resources.h
    ${PYSLAM_SLAM_SRC}/dictionary.h
    ${PYSLAM_SLAM_SRC}/optimizer_g2o.h
    ${PYSLAM_SLAM_SRC}/utils/descriptor_helpers.h
    ${PYSLAM_SLAM_SRC}/utils/eigen_helpers.h
    ${PYSLAM_SLAM_SRC}/utils/features.h
    ${PYSLAM_SLAM_SRC}/utils/messages.h    
    ${PYSLAM_SLAM_SRC}/utils/image_processing.h
    ${PYSLAM_SLAM_SRC}/utils/serialization_json.h    
    ${PYSLAM_SLAM_SRC}/utils/cv_ops.h
    ${PYSLAM_SLAM_SRC}/utils/accessors.h
    ${PYSLAM_SLAM_SRC}/utils/geom_2views.h
    ${PYSLAM_SLAM_SRC}/eigen_aliases.h
    ${PYSLAM_SLAM_SRC}/tracking_utils.h
    ${PYSLAM_SLAM_SRC}/rotation_histogram.h
    ${PYSLAM_SLAM_SRC}/config_parameters.h
    ${PYSLAM_SLAM_SRC}/geometry_matchers.h
)

if (USE_PYTHON)
    set(SLAM_CPP_CORE_HEADERS
        ${SLAM_CPP_CORE_HEADERS}
        ${PYSLAM_SLAM_SRC}/utils/serialization_numpy.h        
        ${PYSLAM_SLAM_SRC}/utils/numpy_helpers.h  # for serialization    
    )
endif()


# cpp core semantics 

set(SEMANTICS_CPP_CORE_SOURCES
    ${PYSLAM_SEMANTICS_SRC}/semantic_fusion_methods.cpp
    ${PYSLAM_SEMANTICS_SRC}/semantic_serialization.cpp
)

set(SEMANTICS_CPP_CORE_HEADERS
    ${PYSLAM_SEMANTICS_SRC}/semantic_fusion_methods.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_serialization.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_types.h
)


# pybind11 module

set(SLAM_CPP_PYBIND_MODULE_SOURCES
    ${PYSLAM_SLAM_SRC}/py_module/cpp_core_module.cpp
)

set(SLAM_CPP_PYBIND_MODULE_HEADERS
    ${PYSLAM_SLAM_SRC}/casters/dictionary_casters.h
    ${PYSLAM_SLAM_SRC}/casters/opencv_type_casters.h
    ${PYSLAM_SLAM_SRC}/py_module/map_module.h
    ${PYSLAM_SLAM_SRC}/py_module/ckdtree_module.h
    ${PYSLAM_SLAM_SRC}/py_module/camera_module.h
    ${PYSLAM_SLAM_SRC}/py_module/map_point_module.h
    ${PYSLAM_SLAM_SRC}/py_module/feature_shared_resources_module.h
    ${PYSLAM_SLAM_SRC}/py_module/frame_module.h
    ${PYSLAM_SLAM_SRC}/py_module/keyframe_module.h
    ${PYSLAM_SLAM_SRC}/py_module/optimizer_g2o_module.h
    ${PYSLAM_SLAM_SRC}/py_module/sim3_pose_module.h
    ${PYSLAM_SLAM_SRC}/py_module/eigen_module.h
    ${PYSLAM_SLAM_SRC}/py_module/tracking_utils_module.h
    ${PYSLAM_SLAM_SRC}/py_module/rotation_histogram_module.h
    ${PYSLAM_SLAM_SRC}/py_module/geometry_matchers_module.h
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
    ${G2O_LIBRARIES}
    nlohmann_json::nlohmann_json
)

# Ensure lib directory exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

#=======================================
# Add C++ core library 
#=======================================

add_library(cpp_core_lib STATIC ${CPP_CORE_SOURCES} ${CPP_CORE_HEADERS})

# Link libraries
target_link_libraries(cpp_core_lib PRIVATE ${CPP_CORE_LINK_LIBRARIES})

# Include directories
target_include_directories(cpp_core_lib SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})

if (USE_PYTHON)
target_link_libraries(cpp_core_lib PRIVATE ${Python_LIBRARIES})
target_include_directories(cpp_core_lib SYSTEM PRIVATE ${Python_INCLUDE_DIRS})
endif()

# Compiler flags
target_compile_definitions(cpp_core_lib PRIVATE 
    VERSION_INFO=${PYSLAM_CPP_CORE_VERSION_INFO}
    PYBIND11_DETAILED_ERROR_MESSAGES
)

# Set output directory for static lib
set_target_properties(cpp_core_lib PROPERTIES 
    ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)


#=======================================
# Create the pybind11 module using the static library
#=======================================
if (USE_PYTHON)

    pybind11_add_module(cpp_core ${SLAM_CPP_PYBIND_MODULE_SOURCES} ${SLAM_CPP_PYBIND_MODULE_HEADERS})
    add_dependencies(cpp_core cpp_core_lib)

    # Link the static library instead of compiling sources again
    target_link_libraries(cpp_core PRIVATE
        cpp_core_lib  # Link the static library
        ${CPP_CORE_LINK_LIBRARIES}
        ${Python_LIBRARIES}
    )

    # Treat Eigen headers as SYSTEM to silence warnings originating from Eigen itself
    target_include_directories(cpp_core SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR} ${Python_INCLUDE_DIRS})

    # Compiler flags
    target_compile_definitions(cpp_core PRIVATE 
        VERSION_INFO=${PYSLAM_CPP_CORE_VERSION_INFO}
        PYBIND11_DETAILED_ERROR_MESSAGES
    )

    # Set output directory
    set_target_properties(cpp_core PROPERTIES
        LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
    )

endif()

#=======================================
# Add C++ test examples
#=======================================
add_subdirectory("${CMAKE_CURRENT_SOURCE_DIR}/tests_cpp" tests_cpp)


#=======================================
# Install target
#=======================================

if (USE_PYTHON)
    install(TARGETS cpp_core
        LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
    )
endif()

install(TARGETS cpp_core_lib
    LIBRARY DESTINATION ${CMAKE_CURRENT_SOURCE_DIR}/lib
)


# Print configuration info
message(STATUS "Building PYSLAM C++ Core Module")
message(STATUS "OpenCV version: ${OpenCV_VERSION}")
if(NOT EIGEN3_VERSION AND EIGEN3_VERSION_STRING)
    set(EIGEN3_VERSION ${EIGEN3_VERSION_STRING})
endif()
message(STATUS "Eigen3 version: ${EIGEN3_VERSION}")
message(STATUS "C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output directory: ${CMAKE_CURRENT_SOURCE_DIR}/lib")
