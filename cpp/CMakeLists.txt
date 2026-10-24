cmake_minimum_required(VERSION 3.5)
project(pyslam_utils)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for clangd and other tools
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 
set(WITH_MARCH_NATIVE ON CACHE BOOL "Build with -march=native")

set(PYSLAM_ROOT ${PROJECT_SOURCE_DIR}/..)
set(PYSLAM_SLAM_CPP_SRC ${PYSLAM_ROOT}/cpp/core/slam)

# Set base optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -DNDEBUG")

# Enable C++20 concepts for GCC (required for GCC < 10)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
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

# Find Eigen3 - try CONFIG mode first, fallback to MODULE mode
find_package(Eigen3 CONFIG QUIET)
if(NOT TARGET Eigen3::Eigen)
    find_package(Eigen3 REQUIRED)
endif()

# Get Eigen include directory
if(TARGET Eigen3::Eigen)
    get_target_property(EIGEN3_INCLUDE_DIR Eigen3::Eigen INTERFACE_INCLUDE_DIRECTORIES)
    if(EIGEN3_INCLUDE_DIR)
        list(GET EIGEN3_INCLUDE_DIR 0 EIGEN3_INCLUDE_DIR)
    endif()
endif()

# EIGEN3_INCLUDE_DIR is set by find_package in MODULE mode if target doesn't exist
if(NOT EIGEN3_INCLUDE_DIR)
    message(FATAL_ERROR "Could not find Eigen3 include directory")
endif()

message(STATUS "Found Eigen3 at: ${EIGEN3_INCLUDE_DIR}")

find_package(TBB QUIET)
if(TBB_FOUND)
    message(STATUS "TBB found: ${TBB_VERSION}")
    message(STATUS "TBB include: ${TBB_INCLUDE_DIRS}")
    message(STATUS "TBB libs: ${TBB_LIBRARIES}")
    # Try to use modern TBB targets if available
    if(TARGET TBB::tbb)
        message(STATUS "Using TBB::tbb target")
        set(TBB_LINK_LIBS TBB::tbb)
    elseif(TBB_LIBRARIES)
        message(STATUS "Using TBB_LIBRARIES: ${TBB_LIBRARIES}")
        set(TBB_LINK_LIBS ${TBB_LIBRARIES})
    else()
        # Fallback: try to find TBB libraries manually
        find_library(TBB_LIB tbb PATHS ${TBB_ROOT}/lib ${TBB_ROOT}/lib64 /usr/lib /usr/local/lib)
        if(TBB_LIB)
            message(STATUS "Found TBB library: ${TBB_LIB}")
            set(TBB_LINK_LIBS ${TBB_LIB})
        else()
            message(WARNING "TBB found but libraries not located. Trying pkg-config...")
            find_package(PkgConfig QUIET)
            if(PKG_CONFIG_FOUND)
                pkg_check_modules(TBB_PKG QUIET tbb)
                if(TBB_PKG_FOUND)
                    set(TBB_LINK_LIBS ${TBB_PKG_LIBRARIES})
                    message(STATUS "Found TBB via pkg-config: ${TBB_LINK_LIBS}")
                endif()
            endif()
        endif()
    endif()
    # If TBB_INCLUDE_DIRS is empty but we have a modern target, query its
    # interface include directories so that headers like <tbb/blocked_range.h>
    # are visible to the compiler (this is important on macOS/Homebrew).
    if(TARGET TBB::tbb AND NOT TBB_INCLUDE_DIRS)
        get_target_property(_tbb_includes TBB::tbb INTERFACE_INCLUDE_DIRECTORIES)
        if(_tbb_includes)
            set(TBB_INCLUDE_DIRS ${_tbb_includes})
            message(STATUS "Resolved TBB_INCLUDE_DIRS from TBB::tbb target: ${TBB_INCLUDE_DIRS}")
        endif()
    endif()
else()
    message(WARNING "TBB not found. Parallelization will be disabled.")
    set(TBB_LINK_LIBS "")
endif()

find_package(OpenCV ${OPENCV_VERSION} QUIET)
message(STATUS "found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dirs: ${OpenCV_INSTALL_PATH}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

# Find qhull - try to find it in the thirdparty directory first
set(QHULL_ROOT "${PYSLAM_ROOT}/thirdparty/qhull/install")
if(EXISTS "${QHULL_ROOT}/lib/cmake/Qhull")
    set(Qhull_DIR "${QHULL_ROOT}/lib/cmake/Qhull")
    message(STATUS "Setting Qhull_DIR to: ${Qhull_DIR}")
endif()

find_package(Qhull QUIET)
if(Qhull_FOUND)
    message(STATUS "Qhull found: ${Qhull_VERSION}")
    
    # Helper function to find and add C++ interface library (removes code duplication)
    function(find_qhull_cpp_lib)
        if(TARGET Qhull::qhullcpp)
            message(STATUS "Also linking Qhull::qhullcpp for C++ interface")
            list(APPEND QHULL_LINK_LIBS Qhull::qhullcpp PARENT_SCOPE)
        else()
            find_library(QHULL_CPP_LIB qhullcpp PATHS ${QHULL_ROOT}/lib NO_DEFAULT_PATH)
            if(QHULL_CPP_LIB)
                message(STATUS "Found Qhull C++ library: ${QHULL_CPP_LIB}")
                list(APPEND QHULL_LINK_LIBS ${QHULL_CPP_LIB} PARENT_SCOPE)
            else()
                message(WARNING "Qhull C++ library (libqhullcpp) not found. C++ interface features will not work.")
                message(WARNING "  To fix: Rebuild Qhull with C++ support enabled.")
                message(WARNING "  Run: cd thirdparty/qhull && rm -rf build install && cd ../.. && ./scripts/install_qhull.sh")
            endif()
        endif()
    endfunction()
    
    # Set up base library and include directory
    if(TARGET Qhull::qhull_r)
        message(STATUS "Using Qhull::qhull_r target")
        get_target_property(QHULL_INCLUDE_DIR Qhull::qhull_r INTERFACE_INCLUDE_DIRECTORIES)
        set(QHULL_LINK_LIBS Qhull::qhull_r)
        find_qhull_cpp_lib()
    elseif(Qhull_LIBRARIES)
        message(STATUS "Using Qhull_LIBRARIES: ${Qhull_LIBRARIES}")
        set(QHULL_INCLUDE_DIR ${Qhull_INCLUDE_DIRS})
        set(QHULL_LINK_LIBS ${Qhull_LIBRARIES})
        find_qhull_cpp_lib()
    else()
        # Fallback: manually find libraries
        find_library(QHULL_LIB qhull_r PATHS ${QHULL_ROOT}/lib NO_DEFAULT_PATH)
        if(QHULL_LIB)
            message(STATUS "Found Qhull library manually: ${QHULL_LIB}")
            set(QHULL_INCLUDE_DIR "${QHULL_ROOT}/include")
            set(QHULL_LINK_LIBS ${QHULL_LIB})
            find_qhull_cpp_lib()
        else()
            message(WARNING "Qhull found but libraries not located")
            set(QHULL_LINK_LIBS "")
        endif()
    endif()
else()
    message(WARNING "Qhull not found. Some features may be disabled.")
    set(QHULL_LINK_LIBS "")
    set(QHULL_INCLUDE_DIR "")
endif()

if(_LINUX_)
    set(OpenGL_GL_PREFERENCE "GLVND")
endif()
find_package(OpenGL REQUIRED)

if(APPLE)
  include_directories(/System/Library/Frameworks)
  find_library(OpenGL_LIBRARY OpenGL)
  mark_as_advanced(OpenGL_LIBRARY)
  set(OPENGL_LIBRARIES ${OpenGL_LIBRARY})
endif()


# the following 2 lines are added to correctly detect the python version 
if(WITH_PYTHON_INTERP_CHECK)
  message(STATUS "WITH_PYTHON_INTERP_CHECK: ${WITH_PYTHON_INTERP_CHECK}")
  find_package(PythonInterp) 
  #find_package(PythonLibs)
  message(STATUS "PythonInterp: ${PythonInterp}")
  #message(STATUS "PythonLibs: ${PythonLibs}")
endif()


include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
    ${TBB_INCLUDE_DIRS}
    ${QHULL_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(pybind11)


include_directories(casters)

#=======================================
# Add pyslam_utils module
#=======================================

pybind11_add_module(pyslam_utils modules/utils/utils_module.cpp modules/utils/utils.cpp)
target_link_libraries(pyslam_utils PRIVATE 
    ${OpenCV_LIBS}
)

#=======================================
# Add data_thread_reader module
#=======================================

find_package(Threads REQUIRED)
pybind11_add_module(data_thread_reader
    modules/data_thread_reader/data_thread_reader_module.cpp
    modules/data_thread_reader/image_prefetcher.cpp)
target_link_libraries(data_thread_reader PRIVATE
    ${OpenCV_LIBS}
    Threads::Threads
)

#=======================================
# Add sim3solver module
#=======================================

add_library(sim3solver_lib STATIC 
    modules/solvers/Sim3Solver.cpp
    modules/solvers/Sim3PointRegistrationSolver.cpp
    modules/solvers/Random.cpp)
target_link_libraries(sim3solver_lib PRIVATE 
    ${OpenCV_LIBS}
)

pybind11_add_module(sim3solver 
modules/solvers/sim3solver_module.cpp)
target_link_libraries(sim3solver PRIVATE sim3solver_lib)

#=======================================
# Add pnpsolver module
#=======================================

add_library(pnpsolver_lib STATIC 
    modules/solvers/PnPsolver.cpp
    modules/solvers/MLPnPsolver.cpp
    modules/solvers/Random.cpp)
target_link_libraries(pnpsolver_lib PRIVATE 
    ${OpenCV_LIBS}
)

pybind11_add_module(pnpsolver 
modules/solvers/pnpsolver_module.cpp)
target_link_libraries(pnpsolver PRIVATE pnpsolver_lib)

#=======================================
# Add volumetric module
#=======================================

add_library(volumetric_lib STATIC 
    modules/volumetric/camera_frustrum.cpp
    modules/volumetric/bounding_boxes_3d.cpp
    modules/volumetric/bounding_boxes_2d.cpp)
target_link_libraries(volumetric_lib PRIVATE 
    ${Open3D_LIBRARIES}
    ${TBB_LINK_LIBS}
    ${OpenCV_LIBS}
    ${QHULL_LINK_LIBS}
)
if(TBB_FOUND)
    target_compile_definitions(volumetric_lib PRIVATE TBB_FOUND)
endif()
if(Qhull_FOUND AND QHULL_LINK_LIBS)
    target_compile_definitions(volumetric_lib PRIVATE QHULL_FOUND)
endif()

pybind11_add_module(volumetric 
modules/volumetric/volumetric_module.cpp)
if(TBB_FOUND)
    target_compile_definitions(volumetric PRIVATE TBB_FOUND)
endif()
if(Qhull_FOUND AND QHULL_LINK_LIBS)
    target_compile_definitions(volumetric PRIVATE QHULL_FOUND)
endif()
target_link_libraries(volumetric PRIVATE volumetric_lib)

#=======================================
# Add glutils module
#=======================================

add_library(glutils_lib STATIC 
    modules/glutils/glutils_drawing.cpp)
target_link_libraries(glutils_lib PRIVATE 
    ${OPENGL_LIBRARIES}
    volumetric_lib
)


pybind11_add_module(glutils 
    modules/glutils/glutils_module.cpp
    modules/glutils/glutils_bindings.cpp)
target_link_libraries(glutils PRIVATE glutils_lib)

#=======================================
# Add trajectory_tools module
#=======================================

pybind11_add_module(trajectory_tools 
modules/trajectory/trajectory_tools_module.cpp)
target_link_libraries(trajectory_tools PRIVATE
)

#=======================================
# Add color_utils module
#=======================================

add_library(color_utils_lib STATIC 
    modules/color_utils/color_utils.cpp)
target_link_libraries(color_utils_lib PRIVATE 
    ${OpenCV_LIBS}
)
# Include directory for color_helpers.h
target_include_directories(color_utils_lib PRIVATE
    ${PYSLAM_SLAM_CPP_SRC}
)

pybind11_add_module(color_utils 
modules/color_utils/color_utils_module.cpp)
target_link_libraries(color_utils PRIVATE color_utils_lib)
# Include directory for color_helpers.h
target_include_directories(color_utils PRIVATE
    ${PYSLAM_SLAM_CPP_SRC}
)

# pybind11_add_module(cv2_pyslam cv2_pyslam_module.cpp)
# target_link_libraries(cv2_pyslam PRIVATE
#     ${OpenCV_LIBS}    
# )


#=======================================
# Add hamming module
#=======================================

pybind11_add_module(hamming 
modules/hamming/hamming_module.cpp)
target_link_libraries(hamming PRIVATE)

#=======================================
# Camera models used by the Python core.
# Formulas stay in cpp/core/slam/camera_model_math.h. The camera type lives in
# one shared library so the factor modules share its type info. g2o and gtsam
# are separate modules: no translation unit includes both, and importing
# camera_models does not load either optimizer library.
#=======================================

set(CAMERA_MODELS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/modules/camera_models")

add_library(camera_model_core SHARED modules/camera_models/camera_model.cpp)
target_include_directories(camera_model_core PUBLIC
    ${CAMERA_MODELS_DIR}
    ${PYSLAM_SLAM_CPP_SRC}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera_model_core PUBLIC ${OpenCV_LIBS})
set_target_properties(camera_model_core PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

function(camera_models_module_rpath target)
    set_target_properties(${target} PROPERTIES
        BUILD_RPATH "\$ORIGIN"
        INSTALL_RPATH "\$ORIGIN")
endfunction()

pybind11_add_module(camera_models modules/camera_models/camera_models_module.cpp)
target_include_directories(camera_models PRIVATE
    ${CAMERA_MODELS_DIR}
    ${PYSLAM_SLAM_CPP_SRC}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS})
target_link_libraries(camera_models PRIVATE camera_model_core ${OpenCV_LIBS})
camera_models_module_rpath(camera_models)

set(G2O_ROOT "${PYSLAM_ROOT}/thirdparty/g2opy")
set(G2O_INCLUDE_DIRS "${G2O_ROOT};${G2O_ROOT}/build")
file(GLOB CAMERA_MODELS_G2O_LIBRARIES "${G2O_ROOT}/lib/*.a")
list(FILTER CAMERA_MODELS_G2O_LIBRARIES EXCLUDE REGEX "g2o.cython.*")
if(CAMERA_MODELS_G2O_LIBRARIES)
    pybind11_add_module(camera_models_g2o modules/camera_models/camera_models_g2o_module.cpp)
    target_include_directories(camera_models_g2o PRIVATE
        ${CAMERA_MODELS_DIR}
        ${PYSLAM_SLAM_CPP_SRC}
        ${G2O_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS})
    target_link_libraries(camera_models_g2o PRIVATE
        camera_model_core
        ${CAMERA_MODELS_G2O_LIBRARIES}
        ${OpenCV_LIBS})
    camera_models_module_rpath(camera_models_g2o)
else()
    message(WARNING "camera_models_g2o was not built (g2o not found).")
endif()

set(GTSAM_ROOT "${PYSLAM_ROOT}/thirdparty/gtsam_local")
set(GTSAM_DIR "${GTSAM_ROOT}/install/lib/cmake/GTSAM")
find_package(GTSAM QUIET CONFIG)
set(GTSAM_FACTORS_INCLUDE_DIRS "${PYSLAM_ROOT}/thirdparty/gtsam_factors/include")
if(GTSAM_FOUND)
    pybind11_add_module(camera_models_gtsam modules/camera_models/camera_models_gtsam_module.cpp)
    target_include_directories(camera_models_gtsam PRIVATE
        ${CAMERA_MODELS_DIR}
        ${PYSLAM_SLAM_CPP_SRC}
        ${GTSAM_INCLUDE_DIR}
        ${GTSAM_FACTORS_INCLUDE_DIRS}
        ${EIGEN3_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS})
    target_link_directories(camera_models_gtsam PRIVATE "${GTSAM_ROOT}/install/lib")
    target_link_libraries(camera_models_gtsam PRIVATE camera_model_core gtsam ${OpenCV_LIBS})
    set_target_properties(camera_models_gtsam PROPERTIES
        BUILD_RPATH "\$ORIGIN;${GTSAM_ROOT}/install/lib"
        INSTALL_RPATH "\$ORIGIN;${GTSAM_ROOT}/install/lib")
    if(NOT APPLE)
        # libgtsam has no RPATH of its own: only DT_RPATH (not DT_RUNPATH) lets the loader
        # resolve its libmetis-gtsam dependency without importing the gtsam package first.
        target_link_options(camera_models_gtsam PRIVATE "-Wl,--disable-new-dtags")
    endif()
else()
    message(WARNING "camera_models_gtsam was not built (gtsam not found).")
endif()
