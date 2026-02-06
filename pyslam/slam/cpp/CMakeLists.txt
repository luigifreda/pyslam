# CMakeLists.txt for PYSLAM C++ Core Module
# This file builds the cpp_core pybind11 module with all core SLAM classes

cmake_minimum_required(VERSION 3.12)
project(cpp_core)

set(PYSLAM_CPP_CORE_VERSION_INFO 0.1.0)


set(WITH_MARCH_NATIVE ON  CACHE BOOL   "Build with \"-march native\"")
set(USE_PYTHON        ON CACHE BOOL    "Use Python")
set(SET_RPATH         ON  CACHE BOOL   "Set RPATH instead of the default RUNPATH") # This is needed for correct linking of the gtsam libraries.

#=======================================
# Build options and variables
#=======================================

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
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -fPIC -DNDEBUG")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -DNDEBUG")
elseif (CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O1 -fPIC -pg")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O1 -fPIC -pg")
endif()

# Add -march=native if requested and not on macOS
# (macOS system clang doesn't properly support -march=native on Apple Silicon)
if(WITH_MARCH_NATIVE AND NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    message(STATUS "Using -march=native for optimization")
elseif(WITH_MARCH_NATIVE)
    message(STATUS "Skipping -march=native on macOS (system clang limitation)")
endif()

if (USE_PYTHON)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DUSE_PYTHON=1")
endif()

# In order to set RPATH (instead of RUNPATH) for gtsam libraries linking issues.
if(SET_RPATH)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wl,--disable-new-dtags")
endif()

#=======================================
# Set project source directory
#=======================================

set(PYSLAM_ROOT ${PROJECT_SOURCE_DIR}/../../..)
set(PYSLAM_SRC ${PYSLAM_ROOT}/pyslam)
set(PYSLAM_SLAM_SRC ${PYSLAM_SRC}/slam/cpp)
set(PYSLAM_SEMANTICS_SRC ${PYSLAM_SRC}/semantics/cpp)


#=======================================
# Find required packages
#=======================================

find_package(OpenCV REQUIRED)
message(STATUS "OpenCV_INCLUDE_DIRS: ${OpenCV_INCLUDE_DIRS}")

find_package(Eigen3 REQUIRED)
# Modern CMake provides Eigen3::Eigen target, older versions use variables
if(TARGET Eigen3::Eigen)
    # Use the modern CMake target
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    if(EIGEN3_INCLUDE_DIR)
        # If it's a list, get the first element
        if(EIGEN3_INCLUDE_DIR MATCHES ";")
            list(GET EIGEN3_INCLUDE_DIR 0 EIGEN3_INCLUDE_DIR)
        endif()
    endif()
elseif(Eigen3_INCLUDE_DIR)
    # Fallback to variable-based approach for older CMake
    set(EIGEN3_INCLUDE_DIR ${Eigen3_INCLUDE_DIR})
elseif(Eigen3_INCLUDE_DIRS)
    list(GET Eigen3_INCLUDE_DIRS 0 EIGEN3_INCLUDE_DIR)
endif()

if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Eigen3 include directory not found. Please install Eigen3 or ensure it's in your system paths.")
endif()
message(STATUS "EIGEN3_INCLUDE_DIR: ${EIGEN3_INCLUDE_DIR}")
find_package(TBB REQUIRED)

# boost 
find_package(Boost REQUIRED COMPONENTS thread filesystem)


include_directories(${PYSLAM_SLAM_SRC})
include_directories(${PYSLAM_SEMANTICS_SRC})


# json
set(CMAKE_PREFIX_PATH ${PYSLAM_ROOT}/thirdparty/json/install/share/cmake/nlohmann_json/)
set(JSON_DIR ${PYSLAM_ROOT}/thirdparty/json/install/)
message(STATUS "set custom json folder: ${JSON_DIR}")
find_package(nlohmann_json REQUIRED)

# Add pybind11 from thirdparty
add_subdirectory("${PYSLAM_ROOT}/thirdparty/pybind11" pybind11)
set(pybind11_INCLUDE_DIRS "${PYSLAM_ROOT}/thirdparty/pybind11/include")
include_directories(${pybind11_INCLUDE_DIRS})

# Add Python development
if (USE_PYTHON)
    # On macOS, prefer the Python executable from PATH to ensure we use the correct version
    # This is especially important when using virtual environments
    if(APPLE AND NOT Python_EXECUTABLE)
        find_program(PYTHON_EXECUTABLE_FROM_PATH python python3)
        if(PYTHON_EXECUTABLE_FROM_PATH)
            message(STATUS "Using Python executable from PATH: ${PYTHON_EXECUTABLE_FROM_PATH}")
            set(Python_EXECUTABLE ${PYTHON_EXECUTABLE_FROM_PATH} CACHE PATH "Python executable")
        endif()
    endif()
    
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    message(STATUS "Python_EXECUTABLE: ${Python_EXECUTABLE}")
    message(STATUS "Python_INCLUDE_DIRS: ${Python_INCLUDE_DIRS}")
    message(STATUS "Python_LIBRARIES: ${Python_LIBRARIES}")
    include_directories(${Python_INCLUDE_DIRS})
    
    # Find NumPy include directory
    # Try multiple Python executables in order of preference
    set(PYTHON_CANDIDATES)
    if(Python_EXECUTABLE)
        list(APPEND PYTHON_CANDIDATES "${Python_EXECUTABLE}")
    endif()
    if(Python3_EXECUTABLE)
        list(APPEND PYTHON_CANDIDATES "${Python3_EXECUTABLE}")
    endif()
    # Try python3 from PATH as fallback
    find_program(PYTHON3_FROM_PATH python3)
    if(PYTHON3_FROM_PATH)
        list(APPEND PYTHON_CANDIDATES "${PYTHON3_FROM_PATH}")
    endif()
    
    set(NUMPY_FOUND FALSE)
    set(NUMPY_INCLUDE_DIR "")
    
    foreach(PYTHON_FOR_NUMPY IN LISTS PYTHON_CANDIDATES)
        execute_process(
            COMMAND "${PYTHON_FOR_NUMPY}" -c "import numpy; print(numpy.get_include())"
            OUTPUT_VARIABLE NUMPY_INCLUDE_DIR_TMP
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
            RESULT_VARIABLE NUMPY_FOUND_RESULT
        )
        if(NUMPY_FOUND_RESULT EQUAL 0 AND EXISTS "${NUMPY_INCLUDE_DIR_TMP}")
            set(NUMPY_INCLUDE_DIR "${NUMPY_INCLUDE_DIR_TMP}")
            set(NUMPY_FOUND TRUE)
            message(STATUS "NumPy_INCLUDE_DIR: ${NUMPY_INCLUDE_DIR} (found via ${PYTHON_FOR_NUMPY})")
            break()
        endif()
    endforeach()
    
    if(NUMPY_FOUND)
        include_directories(${NUMPY_INCLUDE_DIR})
    else()
        message(WARNING "NumPy not found. Some features may not work correctly.")
        message(WARNING "Tried Python executables: ${PYTHON_CANDIDATES}")
        message(FATAL_ERROR "Please install NumPy in your Python environment: pip install numpy")
    endif()
endif()

# Add g2o from thirdparty
set(G2O_ROOT "${PYSLAM_ROOT}/thirdparty/g2opy")
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


# gtsam 
set(GTSAM_ROOT "${PYSLAM_ROOT}/thirdparty/gtsam_local")
set(CMAKE_PREFIX_PATH ${GTSAM_ROOT}/install/lib/cmake/GTSAM/)
set(GTSAM_DIR ${GTSAM_ROOT}/install/)
set(GTSAM_LIB_DIR ${GTSAM_ROOT}/install/lib)
message(STATUS "set custom gtsam folder: ${GTSAM_DIR}")
find_package(GTSAM REQUIRED)
message(STATUS "gtsam include: ${GTSAM_INCLUDE_DIR}")

#gtsam factors
set(GTSAM_FACTORS_ROOT "${PYSLAM_ROOT}/thirdparty/gtsam_factors")
set(GTSAM_FACTORS_INCLUDE_DIRS "${GTSAM_FACTORS_ROOT}/include")
message(STATUS "set custom gtsam factors folder: ${GTSAM_FACTORS_ROOT}")
message(STATUS "gtsam factors include: ${GTSAM_FACTORS_INCLUDE_DIRS}")


# Include directories
include_directories(${OpenCV_INCLUDE_DIRS})
include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${G2O_INCLUDE_DIRS})
include_directories(${TBB_INCLUDE_DIRS})
include_directories(${GTSAM_INCLUDE_DIR})
include_directories(${GTSAM_FACTORS_INCLUDE_DIRS})
include_directories(${Boost_INCLUDE_DIRS})

set(CPP_CORE_LINK_LIBRARIES
    ${OpenCV_LIBS}
    ${EIGEN3_LIBRARIES}
    ${G2O_LIBRARIES}
    ${TBB_LIBRARIES}
    gtsam  
    ${Boost_LIBRARIES}
    nlohmann_json::nlohmann_json
)

link_directories( 
  ${GTSAM_LIB_DIR}
)

#=======================================
# Add C++ core sources and headers
#=======================================

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
    ${PYSLAM_SLAM_SRC}/optimizer_gtsam.cpp
    ${PYSLAM_SLAM_SRC}/tracking_core.cpp
    ${PYSLAM_SLAM_SRC}/config_parameters.cpp
    ${PYSLAM_SLAM_SRC}/geometry_matchers.cpp
    ${PYSLAM_SLAM_SRC}/local_mapping_core.cpp
)

if (USE_PYTHON)
    set(SLAM_CPP_CORE_SOURCES
        ${SLAM_CPP_CORE_SOURCES}
        ${PYSLAM_SLAM_SRC}/camera_serialization_py.cpp
        ${PYSLAM_SLAM_SRC}/map_point_serialization_py.cpp
        ${PYSLAM_SLAM_SRC}/frame_serialization_py.cpp
        ${PYSLAM_SLAM_SRC}/keyframe_serialization_py.cpp
        ${PYSLAM_SLAM_SRC}/map_serialization_py.cpp
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
    ${PYSLAM_SLAM_SRC}/optimizer_gtsam.h
    ${PYSLAM_SLAM_SRC}/optimizer_common.h
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
    ${PYSLAM_SLAM_SRC}/tracking_core.h
    ${PYSLAM_SLAM_SRC}/rotation_histogram.h
    ${PYSLAM_SLAM_SRC}/config_parameters.h
    ${PYSLAM_SLAM_SRC}/geometry_matchers.h
    ${PYSLAM_SLAM_SRC}/local_mapping_core.h
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
    ${PYSLAM_SEMANTICS_SRC}/semantic_types.cpp
    ${PYSLAM_SEMANTICS_SRC}/semantic_utils.cpp
    ${PYSLAM_SEMANTICS_SRC}/semantic_colormap.cpp
    ${PYSLAM_SEMANTICS_SRC}/semantic_labels.cpp
    ${PYSLAM_SEMANTICS_SRC}/semantic_mapping_shared_resources.cpp
)

set(SEMANTICS_CPP_CORE_HEADERS
    ${PYSLAM_SEMANTICS_SRC}/semantic_fusion_methods.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_serialization.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_types.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_utils.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_colormap.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_labels.h
    ${PYSLAM_SEMANTICS_SRC}/semantic_mapping_shared_resources.h
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
    ${PYSLAM_SLAM_SRC}/py_module/tracking_core_module.h
    ${PYSLAM_SLAM_SRC}/py_module/rotation_histogram_module.h
    ${PYSLAM_SLAM_SRC}/py_module/geometry_matchers_module.h
    ${PYSLAM_SLAM_SRC}/py_module/local_mapping_core_module.h
    # semantics
    ${PYSLAM_SEMANTICS_SRC}/py_module/semantic_mapping_shared_resources_module.h
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


# Ensure lib directory exists
file(MAKE_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib)

#=======================================
# Add C++ core library 
#=======================================

add_library(cpp_core_lib STATIC ${CPP_CORE_SOURCES} ${CPP_CORE_HEADERS})

# Link libraries
target_link_libraries(cpp_core_lib PRIVATE ${CPP_CORE_LINK_LIBRARIES})

# Include directories
if(TARGET Eigen3::Eigen)
    # Use modern CMake target if available
    target_link_libraries(cpp_core_lib PRIVATE Eigen3::Eigen)
else()
    # Fallback to include directory for older CMake
    target_include_directories(cpp_core_lib SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
endif()

if (USE_PYTHON)
    target_link_libraries(cpp_core_lib PRIVATE ${Python_LIBRARIES})
    target_include_directories(cpp_core_lib SYSTEM PRIVATE ${Python_INCLUDE_DIRS})
    if(NUMPY_INCLUDE_DIR)
        target_include_directories(cpp_core_lib SYSTEM PRIVATE ${NUMPY_INCLUDE_DIR})
    endif()
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

    # Link the static library and dependencies
    # On macOS, force-load to include all symbols (macOS linker is more aggressive about removing unused symbols)
    if(APPLE)
        target_link_options(cpp_core PRIVATE "-Wl,-force_load,$<TARGET_FILE:cpp_core_lib>")
    else()
        target_link_libraries(cpp_core PRIVATE cpp_core_lib)
    endif()
    target_link_libraries(cpp_core PRIVATE
        ${CPP_CORE_LINK_LIBRARIES}
        ${Python_LIBRARIES}
    )

    # Treat Eigen headers as SYSTEM to silence warnings originating from Eigen itself
    if(TARGET Eigen3::Eigen)
        target_link_libraries(cpp_core PRIVATE Eigen3::Eigen)
    else()
        target_include_directories(cpp_core SYSTEM PRIVATE ${EIGEN3_INCLUDE_DIR})
    endif()
    target_include_directories(cpp_core SYSTEM PRIVATE ${Python_INCLUDE_DIRS})
    if(NUMPY_INCLUDE_DIR)
        target_include_directories(cpp_core SYSTEM PRIVATE ${NUMPY_INCLUDE_DIR})
    endif()

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
message(STATUS "\t CMAKE_BUILD_TYPE: ${CMAKE_BUILD_TYPE}")
message(STATUS "\t WITH_MARCH_NATIVE: ${WITH_MARCH_NATIVE}")
message(STATUS "\t USE_PYTHON: ${USE_PYTHON}")
message(STATUS "\t CMAKE_C_FLAGS: ${CMAKE_C_FLAGS}")
message(STATUS "\t CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "\t OpenCV version: ${OpenCV_VERSION}")
if(NOT EIGEN3_VERSION AND EIGEN3_VERSION_STRING)
    set(EIGEN3_VERSION ${EIGEN3_VERSION_STRING})
endif()
message(STATUS "\t Eigen3 version: ${EIGEN3_VERSION}")
message(STATUS "\t C++ standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "\t Output directory: ${CMAKE_CURRENT_SOURCE_DIR}/lib")
