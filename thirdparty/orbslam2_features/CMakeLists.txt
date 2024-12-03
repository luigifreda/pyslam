cmake_minimum_required(VERSION 2.8.12)
project(orbslam2_features)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 11)

set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -march=native -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC -DNDEBUG")

find_package(OpenCV 4 QUIET)
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
    ${OpenCV_INCLUDE_DIRS}    
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(pybind11)

pybind11_add_module(orbslam2_features orb_extractor.cpp ORBextractor.cpp)
target_link_libraries(orbslam2_features PRIVATE
    ${OpenCV_LIBS}    
)


add_executable(test test.cpp ORBextractor.cpp)
target_link_libraries(test PRIVATE
    ${OpenCV_LIBS}    
)