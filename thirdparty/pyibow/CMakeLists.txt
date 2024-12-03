cmake_minimum_required(VERSION 3.4...3.18)
project(pyibow)

set(CPP_STANDARD_VERSION "17" CACHE STRING "Desired C++ standard version") 
set(BUILD_WITH_MARCH_NATIVE ON CACHE BOOL "Build with -march=native")

# Set the C++ standard 
set(CMAKE_CXX_STANDARD ${CPP_STANDARD_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING
      "Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
      FORCE)
endif()

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
        if(COMPILER_SUPPORTS_MARCH_NATIVE)
            message(STATUS "Build with -march=native")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -march=native")
            set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -march=native")
        endif()
    endif()
endif() 

add_subdirectory(modules/pybind11)

find_package(OpenCV REQUIRED)
find_package(Boost REQUIRED COMPONENTS system filesystem serialization)

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
    ${Boost_LIBRARIES}    
    ${PYTHON_LIBRARIES} Python3::NumPy
)
target_compile_definitions(pyobindex2 
PRIVATE 
    VERSION_INFO=${EXAMPLE_VERSION_INFO}
)
