cmake_minimum_required(VERSION 3.4...3.18)
project(pyibow)


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
find_package(Boost REQUIRED COMPONENTS system filesystem)

set(OBINDEX2_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/modules/obindex2/lib/include/)
set(OBINDEX2_LIB_DIR ${PROJECT_SOURCE_DIR}/modules/obindex2/lib/build/)
set(IBOW_LCD_INCLUDE_DIR ${PROJECT_SOURCE_DIR}/modules/ibow-lcd/include/)
set(IBOW_LCD_LIB_DIR ${PROJECT_SOURCE_DIR}/modules/ibow-lcd/build/)
set(IBOW_LCD_LIBS lcdetector obindex2)

include_directories(
    ${Boost_INCLUDE_DIRS}
    ${OBINDEX2_INCLUDE_DIR}
    ${IBOW_LCD_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)
link_directories( 
    ${OBINDEX2_LIB_DIR}
    ${IBOW_LCD_LIB_DIR}
)

find_package(Python3 COMPONENTS Interpreter NumPy REQUIRED)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)


pybind11_add_module(pyibow 
    src/py_ibow.cpp
    src/ndarray_converter.cpp
)
target_link_libraries(pyibow
PRIVATE
    ${OpenCV_LIBS}
    ${IBOW_LCD_LIBS}
    ${Boost_LIBRARIES}
    ${PYTHON_LIBRARIES} Python3::NumPy
)
target_compile_definitions(pyibow 
PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)


pybind11_add_module(pyobindex2 
    src/py_obindex2.cpp
    src/ndarray_converter.cpp
)
target_link_libraries(pyobindex2
PRIVATE
    ${OpenCV_LIBS}
    ${IBOW_LCD_LIBS}
    ${PYTHON_LIBRARIES} Python3::NumPy
)
target_compile_definitions(pyobindex2 
PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
