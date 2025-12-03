set(WITH_PYTHON_INTERP_CHECK ON CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 

include_directories(${PROJECT_SOURCE_DIR})

include_directories(${EIGEN3_INCLUDE_DIR})
include_directories(${CHOLMOD_INCLUDE_DIR})
include_directories(${CSPARSE_INCLUDE_DIR})

# the following 2 lines are added to correctly detect the python version 
if(WITH_PYTHON_INTERP_CHECK)
  message(STATUS "WITH_PYTHON_INTERP_CHECK: ${WITH_PYTHON_INTERP_CHECK}")


  if(CMAKE_VERSION VERSION_GREATER_EQUAL 3.18)
    # When building a Python extension we should use the Module component to
    # avoid pulling in an extra libpython runtime (which can crash on import).
    # Development.Module requires CMake 3.18+
    find_package(Python3 COMPONENTS Interpreter Development.Module REQUIRED)
  else()
    find_package(Python3 COMPONENTS Interpreter Development REQUIRED)
  endif()
  
  message(STATUS "Python3_EXECUTABLE: ${Python3_EXECUTABLE}")
  message(STATUS "Python3_LIBRARIES: ${Python3_LIBRARIES}")  
endif()

# Add pybind11 as a proper target
add_subdirectory(${PROJECT_SOURCE_DIR}/EXTERNAL/pybind11 ${CMAKE_BINARY_DIR}/pybind11_build)

message(STATUS "OpenGL_LIBRARIES: ${OpenGL_LIBRARIES}")
message(STATUS "GLUT_LIBRARIES: ${GLUT_LIBRARIES}")

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
    csparse_extension
    opengl_helper
    stuff
    ${EIGEN3_LIBRARIES}
    ${CHOLMOD_LIBRARIES}
    ${CSPARSE_LIBRARY}
    Python3::Module
    ${OPENGL_LIBRARIES} 
    ${GLUT_LIBRARIES}
)
