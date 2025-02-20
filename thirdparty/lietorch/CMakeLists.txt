cmake_minimum_required(VERSION 3.10)
project(lietorch LANGUAGES CXX CUDA)

set(CPP_STANDARD_VERSION "17" CACHE STRING "Desired C++ standard version") # We need C++17 since NVCC does not support C++20


# Let's detect which NVCC standards are supported
# Run "nvcc -h | grep -- '--std'" inside a Bash shell
execute_process(
    COMMAND bash -c "nvcc -h | grep -- '--std'"
    OUTPUT_VARIABLE NVCC_OUTPUT
    ERROR_VARIABLE NVCC_ERROR
    RESULT_VARIABLE NVCC_RESULT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
if(NVCC_RESULT)
    message(WARNING "Failed to get NVCC supported standards: ${NVCC_ERROR}")
    set(NVCC_OUTPUT "")
endif()
message(STATUS "Filtered NVCC output:\n${NVCC_OUTPUT}")

# Detect GCC version
execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
    OUTPUT_VARIABLE GCC_VERSION
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
string(REPLACE "." ";" VERSION_LIST ${GCC_VERSION})
list(GET VERSION_LIST 0 GCC_MAJOR_VERSION)
message(STATUS "Detected GCC version: ${GCC_VERSION}")

# Check available C++ standards
if(NVCC_OUTPUT MATCHES "c\\+\\+20" AND GCC_MAJOR_VERSION GREATER_EQUAL 10)
    set(CPP_STANDARD_VERSION "20")
elseif(NVCC_OUTPUT MATCHES "c\\+\\+17")
    set(CPP_STANDARD_VERSION "17")
elseif(NVCC_OUTPUT MATCHES "c\\+\\+14")
    set(CPP_STANDARD_VERSION "14")
else()
    message(WARNING "No valid C++ standard found, defaulting to C++${CPP_STANDARD_VERSION}")
endif()

message(STATUS "Using C++ standard: C++${CPP_STANDARD_VERSION}")

# Set default build type to Release
# if(NOT CMAKE_BUILD_TYPE)
#   set(CMAKE_BUILD_TYPE Release)
# endif()

# Set CMake policies
# cmake_policy(SET CMP0148 NEW)
# cmake_policy(SET CMP0146 NEW)

# Set the C++ standard
set(CMAKE_CXX_STANDARD ${CPP_STANDARD_VERSION})
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Generate compile_commands.json for tooling
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# Set basic compiler flags
set(CMAKE_C_FLAGS "-O2")
set(CMAKE_CXX_FLAGS "-O2")

# Function to detect CUDA architectures
function(detect_cuda_architectures ARCHS)
  execute_process(
    COMMAND nvcc --list-gpu-arch
    OUTPUT_VARIABLE GPU_ARCHS_OUTPUT
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )

  # Parse the output and extract architectures
  string(REPLACE "\n" ";" GPU_ARCHS_LIST "${GPU_ARCHS_OUTPUT}")
  set(DETECTED_ARCHS "")
  foreach(ARCH ${GPU_ARCHS_LIST})
    string(REGEX MATCH "compute_([0-9]+)" _ ${ARCH})
    if(NOT "${CMAKE_MATCH_1}" STREQUAL "")
      list(APPEND DETECTED_ARCHS "${CMAKE_MATCH_1}")
    endif()
  endforeach()

  if(DETECTED_ARCHS)
    set(${ARCHS} ${DETECTED_ARCHS} PARENT_SCOPE)
  else()
    message(WARNING "No CUDA architectures detected. Falling back to default.")
    set(${ARCHS} "70" PARENT_SCOPE) # Default to a commonly supported architecture
  endif()
endfunction()

# Detect CUDA architectures and set them
detect_cuda_architectures(CUDA_ARCHITECTURES)

# Set CUDA architectures
set(CMAKE_CUDA_ARCHITECTURES ${CUDA_ARCHITECTURES})
message(STATUS "Detected CUDA architectures: ${CMAKE_CUDA_ARCHITECTURES}")

# Use the newer CUDA toolkit package
find_package(CUDAToolkit QUIET)
if (NOT CUDAToolkit_FOUND)
  find_package(CUDA REQUIRED)
endif()

# Find Python (and its development files)
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# Determine the Python site-packages directory
execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import site; print(site.getsitepackages()[0])"
    OUTPUT_VARIABLE SITE_PACKAGES_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Python site-packages directory: ${SITE_PACKAGES_DIR}")

#find_package(Torch REQUIRED)

# Query PyTorch’s include and library paths (as done in setup.py)
execute_process(
  COMMAND python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[0])"
  OUTPUT_VARIABLE TORCH_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND python3 -c "from torch.utils import cpp_extension; print(cpp_extension.library_paths()[0])"
  OUTPUT_VARIABLE TORCH_LIBRARY_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND python3 -c "from torch.utils import cpp_extension; print(cpp_extension.include_paths()[1])"
  OUTPUT_VARIABLE TORCH_API_INCLUDE_DIR
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

execute_process(
  COMMAND python3 -c "import torch; import glob; print(';'.join(glob.glob(torch.__path__[0] + '/lib/*.so')))"
  OUTPUT_VARIABLE TORCH_LIBRARIES
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

message(STATUS "PyTorch include directory: ${TORCH_INCLUDE_DIR}")
message(STATUS "PyTorch library directory: ${TORCH_LIBRARY_DIR}")
message(STATUS "PyTorch API include directory: ${TORCH_API_INCLUDE_DIR}")
message(STATUS "PyTorch libraries: ${TORCH_LIBRARIES}")


execute_process(
  COMMAND python3 -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))"
  OUTPUT_VARIABLE GLIBCXX_USE_CXX11_ABI
  OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS "Detected _GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI}")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${GLIBCXX_USE_CXX11_ABI})


# Include directories for Python, PyTorch, and CUDA
include_directories(${Python3_INCLUDE_DIRS})
include_directories(${TORCH_INCLUDE_DIR})
include_directories(${TORCH_API_INCLUDE_DIR})
#include_directories(${Torch_INCLUDE_DIRS})
include_directories(${CUDA_INCLUDE_DIRS})

include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lietorch/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/eigen)

# Add PyTorch and Python library directories
link_directories(${CUDA_LIBRARY_DIRS})
link_directories(${TORCH_LIBRARY_DIR})

# Set up extra compile arguments (here we simply append -std=c++17 and -O2)
# You might add more flags (e.g., -allow-unsupported-compiler) as needed.
list(APPEND CUDA_NVCC_FLAGS "-std=c++${CMAKE_CXX_STANDARD}")
list(APPEND CUDA_NVCC_FLAGS "-O2")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

# Build the shared libraries for the two extension modules.
# Note: We remove the default "lib" prefix so that the modules are named 
# exactly as Python expects (e.g. "lietorch_backends.so")
add_library(lietorch_backends SHARED
  lietorch/src/lietorch_gpu.cu
  lietorch/src/lietorch.cpp
  lietorch/src/lietorch_cpu.cpp
)
set_target_properties(lietorch_backends PROPERTIES PREFIX "")

add_library(lietorch_extras SHARED
  lietorch/extras/altcorr_kernel.cu
  lietorch/extras/corr_index_kernel.cu
  lietorch/extras/se3_builder.cu
  lietorch/extras/se3_inplace_builder.cu
  lietorch/extras/se3_solver.cu
  lietorch/extras/extras.cpp
)
set_target_properties(lietorch_extras PROPERTIES PREFIX "")

# Apply compile options to both targets
target_compile_options(lietorch_backends PUBLIC ${CUDA_FLAGS} ${CUDA_NVCC_FLAGS})
target_compile_options(lietorch_extras PUBLIC ${CUDA_FLAGS} ${CUDA_NVCC_FLAGS})

# Link with PyTorch, Python, and CUDA libraries
target_link_libraries(lietorch_backends PUBLIC ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${CUDA_LIBRARIES})
target_link_libraries(lietorch_extras PUBLIC ${TORCH_LIBRARIES} ${Python3_LIBRARIES} ${CUDA_LIBRARIES})

# Specify output directories (this is optional and can be adjusted)
set_target_properties(lietorch_backends PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set_target_properties(lietorch_extras PROPERTIES LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)

# Define the TORCH_EXTENSION_NAME for each module
target_compile_definitions(lietorch_backends PRIVATE TORCH_EXTENSION_NAME=lietorch_backends)
target_compile_definitions(lietorch_extras PRIVATE TORCH_EXTENSION_NAME=lietorch_extras)

# Install the built extension modules into a package directory.
# We also install the entire "lietorch" folder (which should include __init__.py and other Python files).
install(TARGETS lietorch_backends DESTINATION ${SITE_PACKAGES_DIR})
install(TARGETS lietorch_extras DESTINATION ${SITE_PACKAGES_DIR})
install(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lietorch/ DESTINATION ${SITE_PACKAGES_DIR}/lietorch)


message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
message(STATUS "CMAKE_CXX_STANDARD: ${CMAKE_CXX_STANDARD}")
message(STATUS "Output directory: ${CMAKE_BINARY_DIR}")

# Display GCC build flags
message(STATUS "GCC CXX flags: ${CMAKE_CXX_FLAGS}")

# If CUDA is used, display NVCC flags
if(CMAKE_CUDA_COMPILER)
    message(STATUS "NVCC flags: ${CMAKE_CUDA_FLAGS} ${CUDA_FLAGS} ${CUDA_NVCC_FLAGS}")
endif()

