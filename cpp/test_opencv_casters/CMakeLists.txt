cmake_minimum_required(VERSION 3.5)
project(pyslam_utils)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -march=native -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC -DNDEBUG")

find_package(Eigen3 REQUIRED)

find_package(OpenCV ${OPENCV_VERSION} REQUIRED)
message(STATUS "found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv lib dirs: ${OpenCV_INSTALL_PATH}")
message(STATUS "opencv libs: ${OpenCV_LIBS}")


# the following 2 lines are added to correctly detect the python version 
if(WITH_PYTHON_INTERP_CHECK)
  message(STATUS "WITH_PYTHON_INTERP_CHECK: ${WITH_PYTHON_INTERP_CHECK}")
  find_package(PythonInterp) 
  #find_package(PythonLibs)
  message(STATUS "PythonInterp: ${PythonInterp}")
  #message(STATUS "PythonLibs: ${PythonLibs}")
endif()


include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(pybind11)

pybind11_add_module(cvcasters_test test_bindings.cpp)
target_include_directories(cvcasters_test PRIVATE ${OpenCV_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../casters)
target_link_libraries(cvcasters_test PRIVATE ${OpenCV_LIBS})

find_package(Python3 COMPONENTS Interpreter Development REQUIRED)

add_executable(bench_opencv_casters_current bench_opencv_casters.cpp)
target_include_directories(bench_opencv_casters_current PRIVATE ${OpenCV_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../casters ${CMAKE_CURRENT_SOURCE_DIR}/../casters/old_versions)
target_link_libraries(bench_opencv_casters_current PRIVATE ${OpenCV_LIBS} ${Python3_LIBRARIES} pybind11::embed)
target_compile_definitions(bench_opencv_casters_current PRIVATE OPENCV_CASTER_HEADER_PATH=opencv_type_casters.h)

add_executable(bench_opencv_casters_v1 bench_opencv_casters.cpp)
target_include_directories(bench_opencv_casters_v1 PRIVATE ${OpenCV_INCLUDE_DIRS} ${Python3_INCLUDE_DIRS} ${CMAKE_CURRENT_SOURCE_DIR}/../casters ${CMAKE_CURRENT_SOURCE_DIR}/../casters/old_versions)
target_link_libraries(bench_opencv_casters_v1 PRIVATE ${OpenCV_LIBS} ${Python3_LIBRARIES} pybind11::embed)
target_compile_definitions(bench_opencv_casters_v1 PRIVATE OPENCV_CASTER_HEADER_PATH=old_versions/opencv_type_casters_v1.h)
