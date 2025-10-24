set(CORE_LIBS cpp_core_lib ${CPP_CORE_LINK_LIBRARIES} ${Python_LIBRARIES} pthread)


set(TESTS 
    test_kdtree_validation 
    test_kdtree_basic
    test_numpy_cv_mat
    test_numpy_map_point
    test_numpy_frame
    test_numpy_keyframe
    test_numpy_camera
    test_json_map_point
    test_json_eigen
    test_json_serialization
    test_json_frame
    test_json_keyframe
    test_json_map
    test_json_camera
    test_image_management
)

# add cycle for ${TESTS}
foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} ${CORE_LIBS})
endforeach()
