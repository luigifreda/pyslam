cmake_minimum_required(VERSION 3.5)
project(pyslam_utils)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 
set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "Build with -march=native")

# Set base optimization flags
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3 -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -DNDEBUG")

# Enable C++20 concepts for GCC (required for GCC < 10)
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fconcepts")
endif()


# Add -march=native if requested and not on macOS
# (macOS system clang doesn't properly support -march=native on Apple Silicon)
if(BUILD_WITH_MARCH_NATIVE AND NOT APPLE)
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
    message(STATUS "Using -march=native for optimization")
elseif(BUILD_WITH_MARCH_NATIVE)
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
else()
    message(WARNING "TBB not found. Parallelization will be disabled.")
    set(TBB_LINK_LIBS "")
endif()

find_package(OpenCV ${OPENCV_VERSION} QUIET)
message(STATUS "found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "OpenCV include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "OpenCV lib dirs: ${OpenCV_INSTALL_PATH}")
message(STATUS "OpenCV libs: ${OpenCV_LIBS}")

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
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(pybind11)


include_directories(casters)

pybind11_add_module(pyslam_utils 
utils/utils_module.cpp 
utils/utils.cpp)
target_link_libraries(pyslam_utils PRIVATE
    ${OpenCV_LIBS}    
)

pybind11_add_module(sim3solver 
solvers/sim3solver_module.cpp 
solvers/Sim3Solver.cpp 
solvers/Sim3PointRegistrationSolver.cpp 
solvers/Random.cpp)
target_link_libraries(sim3solver PRIVATE
    ${OpenCV_LIBS}
)

pybind11_add_module(pnpsolver 
solvers/pnpsolver_module.cpp 
solvers/PnPsolver.cpp 
solvers/MLPnPsolver.cpp 
solvers/Random.cpp)
target_link_libraries(pnpsolver PRIVATE
    ${OpenCV_LIBS}
)

pybind11_add_module(glutils 
glutils/glutils_module.cpp)
target_link_libraries(glutils PRIVATE
    ${OPENGL_LIBRARIES}    
)

pybind11_add_module(trajectory_tools 
trajectory/trajectory_tools_module.cpp)
target_link_libraries(trajectory_tools PRIVATE
)

pybind11_add_module(volumetric 
volumetric/volumetric_module.cpp)
if(TBB_FOUND)
    target_compile_definitions(volumetric PRIVATE TBB_FOUND)
endif()
target_link_libraries(volumetric PRIVATE
    ${Open3D_LIBRARIES}
    ${TBB_LINK_LIBS}
)

# pybind11_add_module(cv2_pyslam cv2_pyslam_module.cpp)
# target_link_libraries(cv2_pyslam PRIVATE
#     ${OpenCV_LIBS}    
# )
