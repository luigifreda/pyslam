cmake_minimum_required(VERSION 3.4...3.18)
project(pydbow3)

LIST(APPEND CMAKE_MODULE_PATH 
    ${PROJECT_SOURCE_DIR}/cmake
)

add_subdirectory(modules/pybind11)

find_package(OpenCV REQUIRED)

set(CMAKE_PREFIX_PATH ${PROJECT_SOURCE_DIR}/modules/dbow3/install/lib/cmake/)
find_package(DBoW3 REQUIRED)

find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

pybind11_add_module(pydbow3 
    src/py_dbow3.cpp
    src/ndarray_converter.cpp
)

target_link_libraries(pydbow3
PRIVATE
    ${OpenCV_LIBS}
    ${DBoW3_LIBS}
    ${PYTHON_LIBRARIES} Python3::NumPy
)

target_compile_definitions(pydbow3 
PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
