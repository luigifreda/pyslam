#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam
#
# This script configures Python paths and environment variables for pixi environments.
# It can be sourced by other build scripts to set up Python properly.
#
# Usage:
#   source scripts/pixi_python_config.sh
#   # or
#   . scripts/pixi_python_config.sh
#
# After sourcing, the following variables are available:
#   - PIXI_PYTHON: Path to pixi Python executable
#   - PIXI_PYTHON_VERSION: Python version (e.g., "3.11")
#   - PIXI_PYTHON_SITE_PACKAGES: Path to Python site-packages directory
#   - PYTHON_EXECUTABLE: Alias for PIXI_PYTHON (for compatibility)
#   - PYTHON_VERSION: Alias for PIXI_PYTHON_VERSION (for compatibility)
#   - PYTHON_SITE_PACKAGES: Alias for PIXI_PYTHON_SITE_PACKAGES (for compatibility)
#
# Example usage in a build script:
#   source scripts/pixi_python_config.sh
#   if [ -n "$PIXI_PYTHON" ]; then
#       $PIXI_PYTHON -m pip install somepackage
#   fi

# Get the script directory
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
SCRIPT_DIR=$(readlink -f "$SCRIPT_DIR")  # this reads the actual path if a symbolic directory is used
ROOT_DIR="$SCRIPT_DIR/.."

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
    # Try to find pixi's Python directly
    PIXI_PYTHON=""
    if [ -f "$ROOT_DIR/.pixi/envs/default/bin/python" ]; then
        PIXI_PYTHON="$ROOT_DIR/.pixi/envs/default/bin/python"
    elif [ -f ".pixi/envs/default/bin/python" ]; then
        PIXI_PYTHON=".pixi/envs/default/bin/python"
    elif command -v pixi &> /dev/null; then
        # Use pixi run to get the correct Python
        # Note: This is a command string, not a direct path
        PIXI_PYTHON="pixi run python"
    fi
    
    if [ -n "$PIXI_PYTHON" ]; then
        # Check if it's a direct path or a command
        if [ -f "$PIXI_PYTHON" ]; then
            # Direct path - get version and site-packages
            PIXI_PYTHON_VERSION=$("$PIXI_PYTHON" -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")" 2>/dev/null || echo "")
            PIXI_PYTHON_SITE_PACKAGES=$("$PIXI_PYTHON" -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null || echo "")
            echo "[pixi_python_config] Found pixi Python at: $PIXI_PYTHON"
        else
            # Command string - try to use it
            PIXI_PYTHON_VERSION=$(pixi run python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")" 2>/dev/null || echo "")
            PIXI_PYTHON_SITE_PACKAGES=$(pixi run python -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null || echo "")
            echo "[pixi_python_config] Using pixi run python"
        fi
        
        if [ -n "$PIXI_PYTHON_VERSION" ]; then
            echo "[pixi_python_config] Python version: $PIXI_PYTHON_VERSION"
        fi
        
        if [ -n "$PIXI_PYTHON_SITE_PACKAGES" ]; then
            echo "[pixi_python_config] Python site-packages: $PIXI_PYTHON_SITE_PACKAGES"
        else
            echo "[pixi_python_config] WARNING: Could not determine pixi site-packages"
        fi
    else
        # Fallback: try to use pixi run
        PIXI_PYTHON_VERSION=$(pixi run python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")" 2>/dev/null || python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")" 2>/dev/null || echo "")
        PIXI_PYTHON_SITE_PACKAGES=$(pixi run python -c "import sysconfig; print(sysconfig.get_path('purelib'))" 2>/dev/null || echo "")
        if [ -z "$PIXI_PYTHON_SITE_PACKAGES" ]; then
            echo "[pixi_python_config] WARNING: Could not determine pixi site-packages. Using PYTHONPATH method."
            PIXI_PYTHON_SITE_PACKAGES=""
        else
            PIXI_PYTHON="pixi run python"
            echo "[pixi_python_config] Using pixi run python (fallback)"
        fi
    fi
    
    # Export variables for use in other scripts
    if [ -n "$PIXI_PYTHON" ]; then
        export PIXI_PYTHON
        # Also export as PYTHON_EXECUTABLE for compatibility
        export PYTHON_EXECUTABLE="$PIXI_PYTHON"
    fi
    
    if [ -n "$PIXI_PYTHON_VERSION" ]; then
        export PIXI_PYTHON_VERSION
        # Also export as PYTHON_VERSION for compatibility
        export PYTHON_VERSION="$PIXI_PYTHON_VERSION"
    fi
    
    if [ -n "$PIXI_PYTHON_SITE_PACKAGES" ]; then
        export PIXI_PYTHON_SITE_PACKAGES
        # Also export as PYTHON_SITE_PACKAGES for compatibility
        export PYTHON_SITE_PACKAGES="$PIXI_PYTHON_SITE_PACKAGES"
    fi
    
    # Function to get Python include directory
    get_pixi_python_include_dir() {
        if [ -n "$PIXI_PYTHON" ]; then
            if [ -f "$PIXI_PYTHON" ]; then
                "$PIXI_PYTHON" -c "from sysconfig import get_paths; print(get_paths()['include'])" 2>/dev/null || echo ""
            else
                pixi run python -c "from sysconfig import get_paths; print(get_paths()['include'])" 2>/dev/null || echo ""
            fi
        fi
    }
    
    # Function to get Python library path
    get_pixi_python_library() {
        if [ -n "$PIXI_PYTHON" ]; then
            if [ -f "$PIXI_PYTHON" ]; then
                "$PIXI_PYTHON" -c "import sysconfig; import os; libdir=sysconfig.get_config_var('LIBDIR'); libname=sysconfig.get_config_var('LDLIBRARY'); print(os.path.join(libdir, libname)) if libdir and libname else ''" 2>/dev/null || echo ""
            else
                pixi run python -c "import sysconfig; import os; libdir=sysconfig.get_config_var('LIBDIR'); libname=sysconfig.get_config_var('LDLIBRARY'); print(os.path.join(libdir, libname)) if libdir and libname else ''" 2>/dev/null || echo ""
            fi
        fi
    }
    
    # Function to get CMake options for Python
    get_pixi_python_cmake_options() {
        local opts=""
        
        if [ -n "$PIXI_PYTHON" ]; then
            opts="$opts -DPYTHON_EXECUTABLE=$PIXI_PYTHON"
            opts="$opts -DPYTHON3_EXECUTABLE=$PIXI_PYTHON"
            opts="$opts -DPYTHON_DEFAULT_EXECUTABLE=$PIXI_PYTHON"
            
            local include_dir=$(get_pixi_python_include_dir)
            if [ -n "$include_dir" ] && [ -d "$include_dir" ]; then
                opts="$opts -DPYTHON_INCLUDE_DIR=$include_dir"
            fi
            
            local lib_path=$(get_pixi_python_library)
            if [ -n "$lib_path" ] && [ -f "$lib_path" ]; then
                opts="$opts -DPYTHON_LIBRARY=$lib_path"
            fi
        fi
        
        echo "$opts"
    }
    
    # Export the functions so they can be used in other scripts
    export -f get_pixi_python_include_dir
    export -f get_pixi_python_library
    export -f get_pixi_python_cmake_options
    
    echo "[pixi_python_config] Python configuration complete"
else
    echo "[pixi_python_config] Not in pixi environment, skipping Python configuration"
fi

