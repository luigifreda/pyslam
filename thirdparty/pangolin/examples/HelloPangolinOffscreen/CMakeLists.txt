# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.5 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(HelloPangolinOffscreen main.cpp)
target_link_libraries(HelloPangolinOffscreen ${Pangolin_LIBRARIES})
