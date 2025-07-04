cmake_minimum_required(VERSION 2.6)

project (vertigo-g2o)

# specify where the resulting shared library should be created
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../../lib)

# G2O is required
find_package(G2O REQUIRED)


# Eigen3 is required
find_package(Eigen3 REQUIRED)

# opengl is required
find_package(OpenGL REQUIRED)

# glut is required
find_package(GLUT REQUIRED)


# specify our own source files that need to be compiled and linked into the lib
set(SOURCEFILES
  types_g2o_robust.cpp
  edge_switchPrior.cpp
  edge_se2Switchable.cpp
  edge_se2MaxMixture.cpp
  edge_se3Switchable.cpp
  vertex_switchLinear.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/../../include/g2o)
include_directories(${G2O_INCLUDE_DIR} ${EIGEN3_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${SOURCEFILES})

set (LINKLIBRARIES ${G2O_LIBRARIES} ${EIGEN3_LIBRARIES} ${OpenGL_LIBRARIES} ${GLUT_LIBRARIES})

target_link_libraries(${PROJECT_NAME} ${LINKLIBRARIES}) 

