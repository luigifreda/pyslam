cmake_minimum_required(VERSION 3.5)
project(ibow-lcd LANGUAGES CXX)

set(OPENCV_VERSION          "4" CACHE STRING "Desired OpenCV version") # this is the first target version that is searched for 
set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "Build with -march=native")

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # CMake will generate an error if it is unable to compile the code with a compiler that supports C++17.

# Generate file compile_commands.json in our build folder: it contains the full command line to compile individual source files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Setting build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

# Setting the flags for profiling information or not
if(CMAKE_BUILD_TYPE MATCHES Release)
  message(STATUS "Setting Release options")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O3 -fPIC -DNDEBUG")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC -DNDEBUG")
elseif(CMAKE_BUILD_TYPE MATCHES Debug)
  message(STATUS "Setting Debug options")
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall -O1 -pg -fPIC")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O1 -pg -fPIC")
endif()

if(BUILD_WITH_MARCH_NATIVE)
    if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
        message(STATUS "Build with -march=native")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    else()
        include(CheckCXXCompilerFlag)
        unset(COMPILER_SUPPORTS_MARCH_NATIVE CACHE)
        CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE AND NOT APPLE)
            message(STATUS "Build with -march=native")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
        endif()
    endif()
endif() 

# Printing the compiling flags
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

# Catkin dependencies
#find_package(catkin REQUIRED COMPONENTS roscpp obindex2)

# Other packages

message(STATUS "Desired OpenCV version: ${OPENCV_VERSION}")
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)
include(CheckIncludeFileCXX) 
set (CMAKE_REQUIRED_INCLUDES ${OpenCV_INCLUDE_DIRS})
check_include_file_cxx(opencv2/xfeatures2d.hpp HAVE_OPENCV_CONTRIB)
if(HAVE_OPENCV_CONTRIB)
  message(STATUS "opencv_contrib found!")
  add_definitions(-DHAVE_OPENCV_CONTRIB)
endif()

if(APPLE)
  set(OpenMP_C "${CMAKE_C_COMPILER}")
  set(OpenMP_C_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/14.0.0/include")
  set(OpenMP_C_LIB_NAMES "libomp")
  set(OpenMP_libomp_LIBRARY "omp")

  set(OpenMP_CXX "${CMAKE_CXX_COMPILER}")
  set(OpenMP_CXX_FLAGS "-Xclang -fopenmp -I/usr/local/Cellar/libomp/14.0.0/include")
  set(OpenMP_CXX_LIB_NAMES "libomp")
  set(OpenMP_libomp_LIBRARY "omp")
endif()

find_package(Boost REQUIRED COMPONENTS serialization filesystem)

# Find OpenMP
find_package(OpenMP) # OpenMP
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fopenmp")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fopenmp")
endif()

# Defining the package
# catkin_package(
#     INCLUDE_DIRS include external
#     LIBRARIES lcdetector
#     CATKIN_DEPENDS roscpp obindex2
#     DEPENDS OpenCV Boost
# )

set(catkin_INCLUDE_DIRS ${PROJECT_SOURCE_DIR}/../obindex2/lib/include)
set(catkin_LIBRARIES obindex2)
link_directories(${PROJECT_SOURCE_DIR}/../obindex2/lib/build)

include_directories(include
                    external
                    external/nlohmann
                    ${catkin_INCLUDE_DIRS} 
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS}
                    ${OPENMP_INCLUDES}
)

# Copy test images to the binary directory
file(GLOB JSON "evaluation/*.json")
file(COPY ${JSON}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

### Targets ###
# Library
add_library(lcdetector
            include/ibow-lcd/island.h
            src/lcdetector.cc)
target_link_libraries(lcdetector
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES})

# Main / Demo
add_executable(demo
               src/main.cc)
target_link_libraries(demo
                      lcdetector
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES})

# Evaluation
add_executable(evaluator
               evaluation/lcevaluator.cc
               evaluation/main.cc)
target_link_libraries(evaluator
                      lcdetector
                      ${catkin_LIBRARIES}
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES})
