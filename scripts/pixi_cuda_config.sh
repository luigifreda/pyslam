#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam
#
# This script configures CUDA paths and environment variables for pixi environments.
# It can be sourced by other build scripts to set up CUDA properly.
#
# Usage:
#   source scripts/pixi_cuda_config.sh
#   # or
#   . scripts/pixi_cuda_config.sh
#
# After sourcing, the following variables and functions are available:
#   - PIXI_CUDA_ROOT: Path to pixi CUDA installation
#   - PIXI_NVCC: Path to nvcc compiler
#   - PIXI_CUDA_LIB_DIR: Path to CUDA libraries
#   - PIXI_CUDA_INCLUDE_DIR: Path to CUDA headers
#   - get_pixi_cuda_cmake_options: Function that returns CMake options for CUDA
#
# Example usage in a build script:
#   source scripts/pixi_cuda_config.sh
#   if [ "$CUDA_ON" = "ON" ]; then
#       CMAKE_OPTS="$CMAKE_OPTS $(get_pixi_cuda_cmake_options)"
#   fi

# Get the script directory
SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR_=$(readlink -f "$SCRIPT_DIR_")  # this reads the actual path if a symbolic directory is used
ROOT_DIR="$SCRIPT_DIR_/.."

# Check if we're in a pixi environment
if [ -z "$PIXI_ACTIVATED" ]; then
    # Try to detect pixi environment
    if [ -d "$ROOT_DIR/.pixi/envs/default" ] || [ -d ".pixi/envs/default" ]; then
        export PIXI_ACTIVATED=true
    else
        # Not in pixi environment, exit silently
        return 0 2>/dev/null || exit 0
    fi
fi

# Only configure if pixi is activated
if [ "$PIXI_ACTIVATED" = true ]; then
    # Find pixi CUDA installation
    PIXI_NVVM_BIN=""
    PIXI_CUDA_ROOT=""
    PIXI_NVCC=""
    PIXI_NVCC_BIN_DIR=""
    
    # Find nvvm/bin directory (contains cicc)
    if [ -d "$ROOT_DIR/.pixi/envs/default/nvvm/bin" ]; then
        PIXI_NVVM_BIN="$ROOT_DIR/.pixi/envs/default/nvvm/bin"
        PIXI_CUDA_ROOT="$ROOT_DIR/.pixi/envs/default"
    elif [ -d ".pixi/envs/default/nvvm/bin" ]; then
        PIXI_NVVM_BIN=".pixi/envs/default/nvvm/bin"
        PIXI_CUDA_ROOT=".pixi/envs/default"
    fi
    
    # Find nvcc compiler
    if [ -f "$ROOT_DIR/.pixi/envs/default/targets/x86_64-linux/bin/nvcc" ]; then
        PIXI_NVCC="$ROOT_DIR/.pixi/envs/default/targets/x86_64-linux/bin/nvcc"
        PIXI_NVCC_BIN_DIR="$ROOT_DIR/.pixi/envs/default/targets/x86_64-linux/bin"
    elif [ -f ".pixi/envs/default/targets/x86_64-linux/bin/nvcc" ]; then
        PIXI_NVCC=".pixi/envs/default/targets/x86_64-linux/bin/nvcc"
        PIXI_NVCC_BIN_DIR=".pixi/envs/default/targets/x86_64-linux/bin"
    elif command -v nvcc &> /dev/null; then
        PIXI_NVCC=$(which nvcc)
        PIXI_NVCC_BIN_DIR=$(dirname "$PIXI_NVCC")
    fi
    
    # Add nvcc binary directory to PATH so CUDA detection can find it
    if [ -n "$PIXI_NVCC_BIN_DIR" ] && [ -d "$PIXI_NVCC_BIN_DIR" ]; then
        export PATH="$PIXI_NVCC_BIN_DIR${PATH:+:$PATH}"
        echo "[pixi_cuda_config] Added CUDA nvcc bin directory to PATH: $PIXI_NVCC_BIN_DIR"
    fi
    
    # Add nvvm/bin to PATH so nvcc can find cicc (CUDA Internal Compiler)
    if [ -n "$PIXI_NVVM_BIN" ] && [ -d "$PIXI_NVVM_BIN" ]; then
        export PATH="$PIXI_NVVM_BIN${PATH:+:$PATH}"
        echo "[pixi_cuda_config] Added CUDA nvvm/bin to PATH: $PIXI_NVVM_BIN"
    fi
    
    # Configure CUDA paths if CUDA root is found
    if [ -n "$PIXI_CUDA_ROOT" ] && [ -d "$PIXI_CUDA_ROOT" ]; then
        # Export CUDA root for use in other scripts
        export PIXI_CUDA_ROOT
        
        # Find CUDA library directory
        PIXI_CUDA_LIB_DIR=""
        if [ -d "$PIXI_CUDA_ROOT/targets/x86_64-linux/lib" ]; then
            PIXI_CUDA_LIB_DIR="$PIXI_CUDA_ROOT/targets/x86_64-linux/lib"
        elif [ -d "$PIXI_CUDA_ROOT/lib" ]; then
            PIXI_CUDA_LIB_DIR="$PIXI_CUDA_ROOT/lib"
        fi
        
        # Find CUDA include directory
        PIXI_CUDA_INCLUDE_DIR=""
        if [ -d "$PIXI_CUDA_ROOT/targets/x86_64-linux/include" ]; then
            PIXI_CUDA_INCLUDE_DIR="$PIXI_CUDA_ROOT/targets/x86_64-linux/include"
        elif [ -d "$PIXI_CUDA_ROOT/include" ]; then
            PIXI_CUDA_INCLUDE_DIR="$PIXI_CUDA_ROOT/include"
        fi
        
        # Add library directory to paths
        if [ -n "$PIXI_CUDA_LIB_DIR" ] && [ -d "$PIXI_CUDA_LIB_DIR" ]; then
            export CMAKE_LIBRARY_PATH="$PIXI_CUDA_LIB_DIR${CMAKE_LIBRARY_PATH:+:${CMAKE_LIBRARY_PATH}}"
            export LD_LIBRARY_PATH="$PIXI_CUDA_LIB_DIR${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
            export PIXI_CUDA_LIB_DIR
            echo "[pixi_cuda_config] Added CUDA library directory: $PIXI_CUDA_LIB_DIR"
        fi
        
        # Add include directory to paths
        if [ -n "$PIXI_CUDA_INCLUDE_DIR" ] && [ -d "$PIXI_CUDA_INCLUDE_DIR" ]; then
            export CMAKE_INCLUDE_PATH="$PIXI_CUDA_INCLUDE_DIR${CMAKE_INCLUDE_PATH:+:${CMAKE_INCLUDE_PATH}}"
            export PIXI_CUDA_INCLUDE_DIR
            echo "[pixi_cuda_config] Added CUDA include directory: $PIXI_CUDA_INCLUDE_DIR"
        fi
        
        echo "[pixi_cuda_config] CUDA root: $PIXI_CUDA_ROOT"
    fi
    
    # Export nvcc path
    if [ -n "$PIXI_NVCC" ] && [ -f "$PIXI_NVCC" ]; then
        export PIXI_NVCC
        echo "[pixi_cuda_config] CUDA compiler: $PIXI_NVCC"
    fi
    
    # Function to generate CMake options for CUDA
    # Usage: PIXI_CUDA_CMAKE_OPTS=$(get_pixi_cuda_cmake_options)
    get_pixi_cuda_cmake_options() {
        local opts=""
        
        if [ -n "$PIXI_CUDA_ROOT" ] && [ -d "$PIXI_CUDA_ROOT" ]; then
            # Set CUDA_TOOLKIT_ROOT_DIR for CMake to find CUDA tools (legacy)
            opts="$opts -DCUDA_TOOLKIT_ROOT_DIR=$PIXI_CUDA_ROOT"
            # Set CUDAToolkit_ROOT for modern CMake find_package(CUDAToolkit)
            opts="$opts -DCUDAToolkit_ROOT=$PIXI_CUDA_ROOT"
            # Set CUDA_ROOT for some detection scripts
            opts="$opts -DCUDA_ROOT=$PIXI_CUDA_ROOT"
            # Set CUDA_SDK_ROOT_DIR (legacy variable)
            opts="$opts -DCUDA_SDK_ROOT_DIR=$PIXI_CUDA_ROOT"
        fi
        
        if [ -n "$PIXI_NVCC" ] && [ -f "$PIXI_NVCC" ]; then
            # Set CMAKE_CUDA_COMPILER explicitly so CMake can find nvcc
            opts="$opts -DCMAKE_CUDA_COMPILER=$PIXI_NVCC"
            # Enable CUDA language support
            opts="$opts -DCMAKE_CUDA_STANDARD=17"
        fi
        
        if [ -n "$PIXI_CUDA_LIB_DIR" ] && [ -d "$PIXI_CUDA_LIB_DIR" ]; then
            # Also set via CMake variable for find_library
            opts="$opts -DCMAKE_LIBRARY_PATH=$PIXI_CUDA_LIB_DIR"
        fi
        
        if [ -n "$PIXI_CUDA_INCLUDE_DIR" ] && [ -d "$PIXI_CUDA_INCLUDE_DIR" ]; then
            # Also set via CMake variable for find_path
            opts="$opts -DCMAKE_INCLUDE_PATH=$PIXI_CUDA_INCLUDE_DIR"
        fi
        
        echo "$opts"
    }
    
    # Export the function so it can be used in other scripts
    export -f get_pixi_cuda_cmake_options
    
    echo "[pixi_cuda_config] CUDA configuration complete"
else
    echo "[pixi_cuda_config] Not in pixi environment, skipping CUDA configuration"
fi

