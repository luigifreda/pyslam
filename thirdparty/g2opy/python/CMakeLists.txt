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

# Add pybind11 as a proper target
add_subdirectory(${PROJECT_SOURCE_DIR}/EXTERNAL/pybind11 ${CMAKE_BINARY_DIR}/pybind11_build)

message(STATUS "OpenGL_LIBRARIES: ${OpenGL_LIBRARIES}")
message(STATUS "GLUT_LIBRARIES: ${GLUT_LIBRARIES}")

pybind11_add_module(g2o g2o.cpp)
target_link_libraries(g2o PRIVATE 
    ${PROJECT_SOURCE_DIR}/lib/libg2o_core.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_cholmod.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_csparse.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_eigen.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_dense.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_pcg.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_slam2d_linear.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_solver_structure_only.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_data.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_icp.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_sba.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_sclam2d.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_sim3.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_slam2d.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_slam2d_addons.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_slam3d.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_types_slam3d_addons.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_contrib.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_csparse_extension.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_opengl_helper.a
    ${PROJECT_SOURCE_DIR}/lib/libg2o_stuff.a
    ${EIGEN3_LIBRARIES}
    ${CHOLMOD_LIBRARIES}
    ${CSPARSE_LIBRARY}
    ${PYTHON_LIBRARIES}
    ${OPENGL_LIBRARIES} 
    ${GLUT_LIBRARIES}
)
