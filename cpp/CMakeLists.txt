cmake_minimum_required(VERSION 3.5)
project(pyslam_utils)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE Release)
ENDIF()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(WITH_PYTHON_INTERP_CHECK OFF CACHE BOOL "Checking python interpreter") # to be activated when called within virtual python environment 

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}  -O3 -march=native -fPIC -DNDEBUG")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -march=native -fPIC -DNDEBUG")

find_package(Eigen3 REQUIRED)

find_package(OpenCV ${OPENCV_VERSION} QUIET)
message(STATUS "found OpenCV version: ${OpenCV_VERSION}")
message(STATUS "opencv include: ${OpenCV_INCLUDE_DIRS}")
message(STATUS "opencv lib dirs: ${OpenCV_INSTALL_PATH}")
message(STATUS "opencv libs: ${OpenCV_LIBS}")

if(_LINUX_)
    set(OpenGL_GL_PREFERENCE "GLVND")
endif()
find_package(OpenGL REQUIRED)

if(APPLE)
  include_directories(/System/Library/Frameworks)
  find_library(OpenGL_LIBRARY OpenGL)
  mark_as_advanced(OpenGL_LIBRARY)
  set(OPENGL_LIBRARIES ${OpenGL_LIBRARY})
endif()


# the following 2 lines are added to correctly detect the python version 
if(WITH_PYTHON_INTERP_CHECK)
  message(STATUS "WITH_PYTHON_INTERP_CHECK: ${WITH_PYTHON_INTERP_CHECK}")
  find_package(PythonInterp) 
  #find_package(PythonLibs)
  message(STATUS "PythonInterp: ${PythonInterp}")
  #message(STATUS "PythonLibs: ${PythonLibs}")
endif()


include_directories(
    ${PYTHON_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIR}
    ${OpenCV_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}
)

set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib)

add_subdirectory(pybind11)

pybind11_add_module(pyslam_utils utils_module.cpp utils.cpp)
target_link_libraries(pyslam_utils PRIVATE
    ${OpenCV_LIBS}    
)

pybind11_add_module(sim3solver sim3solver_module.cpp Sim3Solver.cpp Sim3PointRegistrationSolver.cpp Random.cpp)
target_link_libraries(sim3solver PRIVATE
    ${OpenCV_LIBS}    
)

pybind11_add_module(pnpsolver pnpsolver_module.cpp PnPsolver.cpp MLPnPsolver.cpp Random.cpp)
target_link_libraries(pnpsolver PRIVATE
    ${OpenCV_LIBS}    
)

pybind11_add_module(glutils glutils_module.cpp)
target_link_libraries(glutils PRIVATE
    ${OPENGL_LIBRARIES}    
)

pybind11_add_module(trajectory_tools trajectory_tools_module.cpp)
target_link_libraries(trajectory_tools PRIVATE
)

# pybind11_add_module(cv2_pyslam cv2_pyslam_module.cpp)
# target_link_libraries(cv2_pyslam PRIVATE
#     ${OpenCV_LIBS}    
# )
