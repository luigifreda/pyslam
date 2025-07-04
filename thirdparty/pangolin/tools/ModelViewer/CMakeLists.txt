# Find Pangolin (https://github.com/stevenlovegrove/Pangolin)
find_package(Pangolin 0.3 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(ModelViewer main.cpp)
target_link_libraries(ModelViewer ${Pangolin_LIBRARIES})
