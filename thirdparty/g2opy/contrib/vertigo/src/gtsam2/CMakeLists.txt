cmake_minimum_required(VERSION 2.6)

project (vertigo-gtsam)

# specify where the resulting shared library should be created
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/../../lib)

# GTSAN is required
find_package(GTSAM REQUIRED)


# specify our own source files that need to be compiled and linked into the lib
set(SOURCEFILES
  ../timer.cpp
  betweenFactorSwitchable.cpp
)

include_directories(${PROJECT_SOURCE_DIR}/../../include)
include_directories(${PROJECT_SOURCE_DIR}/../../include/gtsam2)
include_directories(${GTSAM_INCLUDE_DIR})

add_library(${PROJECT_NAME} SHARED ${SOURCEFILES})

set (LINKLIBRARIES ${GTSAM_LIBS} )

target_link_libraries(${PROJECT_NAME} ${LINKLIBRARIES}) 



install (TARGETS ${PROJECT_NAME} LIBRARY DESTINATION lib)
