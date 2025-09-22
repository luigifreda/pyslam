cmake_minimum_required(VERSION 3.5)
project(DBoW2)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

set(CPP_STANDARD_VERSION "17" CACHE STRING "Desired C++ standard version") 
set(OPENCV_VERSION "4" CACHE STRING "Desired OpenCV version") # this is the first target version that is searched for 
set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "Build with -march=native")

# Set the C++ standard 
set(CMAKE_CXX_STANDARD ${CPP_STANDARD_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -Wall -O3 -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -O3 -fPIC -DNDEBUG")

# Generate file compile_commands.json in our build folder: it contains the full command line to compile individual source files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O3")

if(BUILD_WITH_MARCH_NATIVE)
    if("${CMAKE_SYSTEM_NAME}" MATCHES "Linux")
        message(STATUS "Build with -march=native")
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
        set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
    else()
        include(CheckCXXCompilerFlag)
        unset(COMPILER_SUPPORTS_MARCH_NATIVE CACHE)
        CHECK_CXX_COMPILER_FLAG(-march=native COMPILER_SUPPORTS_MARCH_NATIVE)
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            message(STATUS "Build with -march=native")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
        endif()
    endif()
endif() 

set(HDRS_DBOW2
  DBoW2/DBoW2.h
  DBoW2/BowVector.h
  DBoW2/FORB.h
  DBoW2/FClass.h
  DBoW2/FSurf64.h
  DBoW2/FBrief.h
  DBoW2/FeatureVector.h
  DBoW2/ScoringObject.h
  DBoW2/TemplatedVocabulary.h
  DBoW2/TemplatedDatabase.h  
  DBoW2/QueryResults.h
  DBoW2/KeyFrameOrbDatabase.h
)
set(SRCS_DBOW2
  DBoW2/BowVector.cpp
  DBoW2/FORB.cpp
  DBoW2/FSurf64.cpp
  DBoW2/FBrief.cpp
  DBoW2/FeatureVector.cpp
  DBoW2/ScoringObject.cpp
  DBoW2/QueryResults.cpp  
  DBoW2/KeyFrameOrbDatabase.cpp
)

set(HDRS_DUTILS
  DUtils/Random.h
  DUtils/Timestamp.h)
set(SRCS_DUTILS
  DUtils/Random.cpp
  DUtils/Timestamp.cpp)

message(STATUS "Desired OpenCV version: ${OPENCV_VERSION}")
find_package(OpenCV ${OPENCV_VERSION} REQUIRED)
message(STATUS "found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv lib dirs: ${OpenCV_INSTALL_PATH}")
message(STATUS "opencv libs: ${OpenCV_LIBS}")

find_package(Boost REQUIRED COMPONENTS serialization filesystem)
include_directories(${Boost_INCLUDE_DIRS})

set(EXTERNAL_CORE_LIBS   # core external libs 
${OpenCV_LIBS}
${Boost_LIBRARIES}
)

set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

include_directories(${OpenCV_INCLUDE_DIRS})
add_library(DBoW2 SHARED ${SRCS_DBOW2} ${SRCS_DUTILS})
target_link_libraries(DBoW2 ${EXTERNAL_CORE_LIBS})


# This option allows the generations of a file compile_commands.json in our build folder: that file contains the full command line to compile individual source files
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
