if(UNIX)
find_package(Pangolin 0.4 REQUIRED)
include_directories(${Pangolin_INCLUDE_DIRS})

add_executable(SharedMemoryCamera main.cpp)
target_link_libraries(SharedMemoryCamera ${Pangolin_LIBRARIES})
endif()
