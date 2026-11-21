add_executable(test_direct_slam test_direct_slam.cpp)
target_link_libraries(test_direct_slam ${CORE_LIBS})

add_executable(test_direct_components test_direct_components.cpp)
target_link_libraries(test_direct_components ${CORE_LIBS})
