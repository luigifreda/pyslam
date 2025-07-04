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
# LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/external/pybind11/tools)
# include_directories(${PROJECT_SOURCE_DIR}/external/pybind11/include)
# include(pybind11Tools)


add_subdirectory(
  ${PROJECT_SOURCE_DIR}/external/pybind11
  ${CMAKE_BINARY_DIR}/external/pybind11_build
)

set(PYTARGET_NAME pypangolin)
pybind11_add_module(${PYTARGET_NAME} pangolin.cpp )
SET_TARGET_PROPERTIES(${PYTARGET_NAME} PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR})
target_link_libraries(${PYTARGET_NAME} PRIVATE ${LIBRARY_NAME})
