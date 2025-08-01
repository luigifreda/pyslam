cmake_minimum_required(VERSION 3.10)
project(gtsam_factors)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(SET_RPATH               ON  CACHE BOOL   "Set RPATH instead of the default RUNPATH") # for solving the linking issue with g2o binaries (see the related node in the compiler options section)
set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 


# In order to set RPATH (instead of RUNPATH) for g2o and other libraries linking issue.
if(LINUX)
  if(SET_RPATH)
    set(MY_FLAGS "${MY_FLAGS} -Wl,--disable-new-dtags")
  endif()
endif()

if(APPLE)
  set(CMAKE_THREAD_PREFER_PTHREAD TRUE)
  set(THREADS_PREFER_PTHREAD_FLAG TRUE)
  find_package(Threads REQUIRED)
endif()

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -fPIC -DNDEBUG ${MY_FLAGS}")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -fPIC -DNDEBUG ${MY_FLAGS}")

message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "CXX flags: ${CMAKE_CXX_FLAGS}")

# eigen 
find_package(Eigen3 3.1.0 REQUIRED)
message(STATUS "eigen include: ${EIGEN3_INCLUDE_DIR}")

# gtsam
set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/../gtsam_local/install/lib/cmake/GTSAM/)
set(GTSAM_DIR ${PROJECT_SOURCE_DIR}/../gtsam_local/install/)
message(STATUS "set custom gtsam folder: ${GTSAM_DIR}")
find_package(GTSAM REQUIRED)
message(STATUS "gtsam include: ${GTSAM_INCLUDE_DIR}")
message(STATUS "gtsam lib dirs: ${GTSAM_LIBRARY_DIRS}")
message(STATUS "gtsam libs: ${GTSAM_LIBS}")

find_package(Boost REQUIRED COMPONENTS system serialization filesystem)
message(STATUS "Boost_VERSION: ${Boost_VERSION}")
message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}")


# if (APPLE)
#   find_package(TBB REQUIRED)
#   if (TBB_FOUND)
#     message(STATUS "TBB found")
#     if (NOT TBB_LIBRARIES)
#       set(TBB_LIBRARIES TBB::tbb)
#     endif()
#   endif()
#   message(STATUS "TBB include: ${TBB_INCLUDE_DIRS}")
#   message(STATUS "TBB libs: ${TBB_LIBRARIES}")
#   message(STATUS "TBB library dirs: ${TBB_LIBRARY_DIRS}")
# endif()

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
    ${GTSAM_INCLUDE_DIR}
    ${Boost_INCLUDE_DIRS}
    ${TBB_INCLUDE_DIRS}
)

# link_directories( 
#   ${GTSAM_LIBRARY_DIRS}
# )

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(pybind11)

pybind11_add_module(gtsam_factors 
  gtsam_factors.cpp 
  optimizers.h 
  numerical_derivative.h 
  resectioning.h
  weighted_projection_factors.h)
target_link_libraries(gtsam_factors PRIVATE
  gtsam
  ${Boost_LIBRARIES}
  ${TBB_LIBRARIES}
)