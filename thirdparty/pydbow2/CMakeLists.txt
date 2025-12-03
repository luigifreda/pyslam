cmake_minimum_required(VERSION 3.4...3.18)
project(pydbow2)

set(CPP_STANDARD_VERSION "17" CACHE STRING "Desired C++ standard version") 

# Set the C++ standard 
set(CMAKE_CXX_STANDARD ${CPP_STANDARD_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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
        if(COMPILER_SUPPORTS_MARCH_NATIVE AND NOT APPLE)
            message(STATUS "Build with -march=native")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
        endif()
    endif()
endif() 

add_subdirectory(modules/pybind11)

find_package(OpenCV REQUIRED)

find_package(Boost REQUIRED COMPONENTS serialization filesystem)
message(STATUS "Boost_VERSION: ${Boost_VERSION}")
message(STATUS "Boost includes: ${Boost_INCLUDE_DIRS}")

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
    ${Boost_INCLUDE_DIRS}
)
link_directories( 
    ${DBoW2_LIB}
)

if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    # When building a Python extension we should use the Module component to
    # avoid pulling in an extra libpython runtime (which can crash on import).
    # Development.Module requires CMake 3.18+
    find_package(Python3 COMPONENTS Interpreter Development.Module NumPy REQUIRED)
else()
    find_package(Python3 COMPONENTS Interpreter Development NumPy REQUIRED)
endif()
message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")

# Force bundled pybind11 to use the same interpreter
set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

pybind11_add_module(pydbow2 
    src/py_dbow2.cpp
    src/ndarray_converter.cpp
)

target_link_libraries(pydbow2
PRIVATE
    ${OpenCV_LIBS}
    ${Boost_LIBRARIES}
    ${DBoW2_LIB}
    Python3::Module
    Python3::NumPy
)

target_compile_definitions(pydbow2 
PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
