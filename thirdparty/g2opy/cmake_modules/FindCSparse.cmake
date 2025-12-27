# Look for csparse; note the difference in the directory specifications!
# First check if paths are already set (from command line - they take precedence)
if(NOT CSPARSE_INCLUDE_DIR)
  # Try to detect pixi/conda environment from Python executable
  set(PIXI_CONDA_PREFIX "")
  if(Python3_EXECUTABLE)
    get_filename_component(PYTHON_DIR ${Python3_EXECUTABLE} DIRECTORY)
    get_filename_component(PYTHON_BIN_DIR ${PYTHON_DIR} DIRECTORY)
    get_filename_component(ENV_PREFIX ${PYTHON_BIN_DIR} DIRECTORY)
    
    # Check if this looks like a pixi/conda environment
    if(EXISTS "${ENV_PREFIX}/include/suitesparse" OR EXISTS "${ENV_PREFIX}/lib/libcxsparse.so" OR EXISTS "${ENV_PREFIX}/lib/libcxsparse.a")
      set(PIXI_CONDA_PREFIX ${ENV_PREFIX})
    endif()
  endif()
  
  # Also check CONDA_PREFIX environment variable
  if(NOT PIXI_CONDA_PREFIX AND DEFINED ENV{CONDA_PREFIX})
    set(PIXI_CONDA_PREFIX $ENV{CONDA_PREFIX})
  endif()
  
  FIND_PATH(CSPARSE_INCLUDE_DIR NAMES cs.h
    PATHS
    ${PIXI_CONDA_PREFIX}/include/suitesparse  # pixi/conda first
    /usr/include/suitesparse
    /usr/include
    /opt/homebrew/include/suitesparse
    /opt/local/include
    /usr/local/include
    /sw/include
    /usr/include/ufsparse
    /opt/local/include/ufsparse
    /usr/local/include/ufsparse
    /sw/include/ufsparse
    )
endif()

if(NOT CSPARSE_LIBRARY)
  # Try to detect pixi/conda environment from Python executable
  set(PIXI_CONDA_PREFIX "")
  if(Python3_EXECUTABLE)
    get_filename_component(PYTHON_DIR ${Python3_EXECUTABLE} DIRECTORY)
    get_filename_component(PYTHON_BIN_DIR ${PYTHON_DIR} DIRECTORY)
    get_filename_component(ENV_PREFIX ${PYTHON_BIN_DIR} DIRECTORY)
    
    # Check if this looks like a pixi/conda environment
    if(EXISTS "${ENV_PREFIX}/include/suitesparse" OR EXISTS "${ENV_PREFIX}/lib/libcxsparse.so" OR EXISTS "${ENV_PREFIX}/lib/libcxsparse.a")
      set(PIXI_CONDA_PREFIX ${ENV_PREFIX})
    endif()
  endif()
  
  # Also check CONDA_PREFIX environment variable
  if(NOT PIXI_CONDA_PREFIX AND DEFINED ENV{CONDA_PREFIX})
    set(PIXI_CONDA_PREFIX $ENV{CONDA_PREFIX})
  endif()
  
  FIND_LIBRARY(CSPARSE_LIBRARY NAMES cxsparse
    PATHS
    ${PIXI_CONDA_PREFIX}/lib  # pixi/conda first
    /usr/lib
    /usr/local/lib
    /opt/local/lib
    /sw/lib
    /opt/homebrew/lib
    )
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(CSPARSE DEFAULT_MSG
  CSPARSE_INCLUDE_DIR CSPARSE_LIBRARY)
