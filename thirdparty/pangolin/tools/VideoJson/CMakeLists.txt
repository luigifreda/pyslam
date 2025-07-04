# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(VideoJsonPrint main-print.cpp)
target_link_libraries(VideoJsonPrint ${Pangolin_LIBRARIES})

add_executable(VideoJsonTransform main-transform.cpp)
target_link_libraries(VideoJsonTransform ${Pangolin_LIBRARIES})

#######################################################
## Install

install(TARGETS VideoJsonPrint VideoJsonTransform
  RUNTIME DESTINATION ${CMAKE_INSTALL_PREFIX}/bin
  LIBRARY DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
  ARCHIVE DESTINATION ${CMAKE_INSTALL_PREFIX}/lib
)
