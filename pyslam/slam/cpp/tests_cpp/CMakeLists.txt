set(CORE_LIBS cpp_core_lib ${CPP_CORE_LINK_LIBRARIES} ${Python_LIBRARIES} pthread)


set(TESTS 
    test_kdtree_validation 
    test_kdtree_basic
)

# add cycle for ${TESTS}
foreach(TEST ${TESTS})
    add_executable(${TEST} ${TEST}.cpp)
    target_link_libraries(${TEST} ${CORE_LIBS})
endforeach()
