cmake_minimum_required(VERSION 2.6)

project (robustISAM2)

# GTSAM is required
find_package(GTSAM REQUIRED)

find_package(Eigen3 REQUIRED)

# boost is required
find_package(Boost 1.46 COMPONENTS program_options REQUIRED)

# specify our own source files that need to be compiled and linked into the lib
include_directories(${PROJECT_SOURCE_DIR}/../../include)
include_directories(${PROJECT_SOURCE_DIR}/../../include/gtsam2)
include_directories(${GTSAM_INCLUDE_DIR} ${Boost_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})
set (LINK_LIBRARIES ${LINK_LIBRARIES} vertigo-gtsam ${GTSAM_LIBS} ${Boost_LIBRARIES} ${EIGEN3_LIBRARIES})

add_executable(robustISAM2 robustISAM2.cpp)
target_link_libraries(robustISAM2 ${LINK_LIBRARIES}) 

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}) # specify where the resulting binaries should be created

