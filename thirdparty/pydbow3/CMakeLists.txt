cmake_minimum_required(VERSION 3.4...3.18)
project(pydbow3)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "Build with -march=native")

LIST(APPEND CMAKE_MODULE_PATH 
    ${PROJECT_SOURCE_DIR}/cmake
)

# Generate file compile_commands.json in our build folder: it contains the full command line to compile individual source files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(BUILD_WITH_MARCH_NATIVE AND NOT "${ARCH}" MATCHES "arm" AND "${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
    message(STATUS "Build with -march=native")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native -O3")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native -O3")
endif()


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
