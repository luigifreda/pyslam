cmake_minimum_required(VERSION 2.6)

project (vertigo)

# we have our own find script for g2o
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${CMAKE_SOURCE_DIR}/cmake-modules/")

# check if we have gtsam and / or g2o
find_package(GTSAM)
find_package(G2O)

# set compiler options depending on build type
SET(CMAKE_BUILD_TYPE Release)
if("${CMAKE_BUILD_TYPE}" STREQUAL "")
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -mfpmath=sse -msse3 -fPIC -O3 -DNDEBUG")
set(CMAKE_LD_FLAGS_RELEASE "${CMAKE_LD_FLAGS_RELEASE} -O3")
set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -fPIC -ggdb")


add_subdirectory(src)
add_subdirectory(examples)
