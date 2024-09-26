cmake_minimum_required(VERSION 3.4...3.18)
project(pydbow2)

LIST(APPEND CMAKE_MODULE_PATH 
    ${PROJECT_SOURCE_DIR}/cmake
)

add_subdirectory(modules/pybind11)

find_package(OpenCV REQUIRED)

set(DBOW2_DIR ${PROJECT_SOURCE_DIR}/modules/dbow2)
set(DBoW2_INCLUDE_DIRS ${DBOW2_DIR}/)
if(APPLE)
set(DBoW2_LIB ${DBOW2_DIR}/lib/libDBoW2.dylib)
else()
set(DBoW2_LIB ${DBOW2_DIR}/lib/libDBoW2.so)
endif()

include_directories(
    ${DBoW2_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
)
link_directories( 
    ${DBoW2_LIB}
)

find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

pybind11_add_module(pydbow2 
    src/py_dbow2.cpp
    src/ndarray_converter.cpp
)

target_link_libraries(pydbow2
PRIVATE
    ${OpenCV_LIBS}
    ${DBoW2_LIB}
    ${PYTHON_LIBRARIES} Python3::NumPy
)

target_compile_definitions(pydbow2 
PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
