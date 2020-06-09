set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 

include_directories(${PROJECT_SOURCE_DIR})

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(${CSPARSE_INCLUDE_DIR})

# the following 2 lines are added to correctly detect the python version 
if(WITH_PYTHON_INTERP_CHECK)
  message(STATUS "WITH_PYTHON_INTERP_CHECK: ${WITH_PYTHON_INTERP_CHECK}")
  find_package(PythonInterp) 
  find_package(PythonLibs)
  message(STATUS "PythonInterp: ${PythonInterp}")
  message(STATUS "PythonLibs: ${PythonLibs}")
endif()

# pybind11 (version 2.2.1)
LIST(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/EXTERNAL/pybind11/tools)
include_directories(${PROJECT_SOURCE_DIR}/EXTERNAL/pybind11/include)
include(pybind11Tools)


pybind11_add_module(g2o g2o.cpp)
target_link_libraries(g2o PRIVATE 
    core
    solver_cholmod
    solver_csparse
    solver_eigen
    solver_dense
    solver_pcg
    solver_slam2d_linear
    solver_structure_only
    types_data
    types_icp
    types_sba
    types_sclam2d
    types_sim3
    types_slam2d
    types_slam2d_addons
    types_slam3d
    types_slam3d_addons
    contrib
)
