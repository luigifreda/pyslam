cmake_minimum_required(VERSION 3.5)
project(obindex2 LANGUAGES CXX)

# Compile tests or not
option(EnableTesting "EnableTesting" ON)

set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "Build with -march=native")

# Setting build type to Release
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
message(STATUS "Build type: " ${CMAKE_BUILD_TYPE})

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON) # CMake will generate an error if it is unable to compile the code with a compiler that supports C++17.


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
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            message(STATUS "Build with -march=native")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
        endif()
    endif()
endif() 

# Packages required to work with the library
find_package(OpenCV REQUIRED) # OpenCV
find_package(Boost REQUIRED COMPONENTS system serialization filesystem)

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
find_package(OpenMP REQUIRED) # OpenMP
if (OPENMP_FOUND)
  set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Printing the compiling flags
message(STATUS "Compiler flags: ${CMAKE_CXX_FLAGS}")

include_directories(include
                    external
                    ${OpenCV_INCLUDE_DIRS}
                    ${Boost_INCLUDE_DIRS})

link_directories(${OpenCV_LIBRARIES}
                 ${Boost_LIBRARIES})

# Copy test images to the binary directory
file(GLOB IMAGES "tests/*.jpg")
file(COPY ${IMAGES}
     DESTINATION ${CMAKE_CURRENT_BINARY_DIR})

### Targets ###
add_library(obindex2
            src/binary_descriptor.cc
            src/binary_tree_node.cc
            include/obindex2/priority_queues.h
            src/binary_tree.cc
            src/binary_index.cc)
target_link_libraries(obindex2
                      ${OpenCV_LIBRARIES}
                      ${Boost_LIBRARIES})

### Testing ###
if (${EnableTesting} STREQUAL "ON")
  # Test for BinaryDescriptor class
  #add_executable(test_bdesc tests/test_bdesc.cc)
  #target_link_libraries(test_bdesc obindex2)

  # Test for BinaryTree class
  add_executable(test_btree tests/test_btree.cc)
  target_link_libraries(test_btree obindex2)

  # Test for ImageIndex class
  add_executable(test_bindex tests/test_bindex.cc)
  target_link_libraries(test_bindex obindex2)

  # Test for searching images
  add_executable(test_search tests/test_search.cc)
  target_link_libraries(test_search obindex2)

  enable_testing(true)  # Enables unit-testing.
  #add_test(NAME test_bdesc COMMAND test_bdesc ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
  add_test(NAME test_btree COMMAND test_btree ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
  add_test(NAME test_bindex COMMAND test_bindex ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
  #add_test(NAME test_search COMMAND test_search ${OpenCV_LIBRARIES} ${Boost_LIBRARIES})
endif()
