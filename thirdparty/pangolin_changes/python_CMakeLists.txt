set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 

include_directories(${PROJECT_SOURCE_DIR}/include)

# Find Eigen3
find_package(Eigen3 REQUIRED)
include_directories(${EIGEN3_INCLUDE_DIR})

# the following 2 lines are added to correctly detect the python version 
if(WITH_PYTHON_INTERP_CHECK)
  message(STATUS "WITH_PYTHON_INTERP_CHECK: ${WITH_PYTHON_INTERP_CHECK}")
  find_package(PythonInterp) 
  find_package(PythonLibs)
  message(STATUS "PythonInterp: ${PythonInterp}")
  message(STATUS "PythonLibs: ${PythonLibs}")
endif()

#find_package(pybind11)

# pybind11 (version 2.2.1)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/pybind11/tools)
include_directories(${PROJECT_SOURCE_DIR}/external/pybind11/include)
include(pybind11Tools)


pybind11_add_module( pangolin pangolin.cpp )
SET_TARGET_PROPERTIES(pangolin PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(pangolin PRIVATE ${LIBRARY_NAME})
