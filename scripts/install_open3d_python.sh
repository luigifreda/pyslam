#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

SCRIPTS_DIR="$SCRIPT_DIR_"
ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

# NOTE: this is required under mac where I got unexpected segmentation fault errors
#       on open3d dynamic library loading

STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  

print_blue "Installing open3d-python from source"

# check if HAVE_CUDA is set, if not, source the cuda_config.sh script
if [ -z "$HAVE_CUDA" ]; then
    source $ROOT_DIR/cuda_config.sh
else
    echo "HAVE_CUDA is already set to $HAVE_CUDA"
fi

# Detect architecture (works on both macOS and Linux)
ARCH=$(uname -m)

# Initialize EXTERNAL_OPTIONS (may be set by caller, but ensure it exists)
EXTERNAL_OPTIONS="${EXTERNAL_OPTIONS:-}"

# Add CMAKE_POLICY_VERSION_MINIMUM for all platforms (needed for CMake 4.2 compatibility)
EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DCMAKE_POLICY_VERSION_MINIMUM=3.5"

if [[ $OSTYPE == "darwin"* ]]; then

    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION
    unset PKG_CONFIG_PATH DYLD_LIBRARY_PATH

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")
    if [[ "$ARCH" == "arm64" ]]; then
        # Explicitly set architecture for Apple Silicon
        # Disable Intel-specific libraries (IPP, MKL) that don't support ARM64
        # Set APPLE_AARCH64=TRUE to ensure dependencies download correct ARM64 binaries
        MAC_OPTIONS="-DCMAKE_C_COMPILER=/usr/bin/clang \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DAPPLE_AARCH64=TRUE \
        -DCMAKE_PREFIX_PATH=/opt/homebrew \
        -DCMAKE_FIND_FRAMEWORK=LAST \
        -DCMAKE_FIND_APPBUNDLE=LAST \
        -DBUILD_ISPC_MODULE=OFF \
        -DBUILD_COMMON_ISPC_ISAS=OFF \
        -DWITH_IPP=OFF \
        -DUSE_BLAS=ON \
        -DBUILD_WEBRTC=OFF"
        
        # Set library paths for Homebrew on Apple Silicon
        export LDFLAGS="-L/opt/homebrew/opt/openblas/lib -L/opt/homebrew/lib"
        export CPPFLAGS="-I/opt/homebrew/opt/openblas/include -I/opt/homebrew/Cellar/minizip/1.3.1/include/minizip/ -I/opt/homebrew/include/libpng16"
        export CPLUS_INCLUDE_PATH="/opt/homebrew/opt/openblas/include:/opt/homebrew/Cellar/minizip/1.3.1/include/minizip/"
        export LIBRARY_PATH="/opt/homebrew/opt/openblas/lib"
    else
        # Intel Mac
        MAC_OPTIONS="-DCMAKE_C_COMPILER=/usr/bin/clang \
        -DCMAKE_CXX_COMPILER=/usr/bin/clang++ \
        -DCMAKE_PREFIX_PATH=/usr/local \
        -DCMAKE_FIND_FRAMEWORK=LAST \
        -DCMAKE_FIND_APPBUNDLE=LAST"
        
        export LDFLAGS="-L/usr/local/opt/openblas/lib -L/usr/local/lib"
        export CPPFLAGS="-I/usr/local/opt/openblas/include -I/usr/local/Cellar/minizip/1.3.1/include/minizip/ -I/usr/local/include/libpng16"
        export CPLUS_INCLUDE_PATH="/usr/local/opt/openblas/include:/usr/local/Cellar/minizip/1.3.1/include/minizip/"
        export LIBRARY_PATH="/usr/local/opt/openblas/lib"
    fi

    # Set CMAKE_SYSROOT if we have a valid SDK path
    if [[ -n "$MAC_SYSROOT" ]]; then
        MAC_OPTIONS="$MAC_OPTIONS -DCMAKE_SYSROOT=$MAC_SYSROOT"
    fi

    # Set deployment target to ensure compatibility (use current macOS version as minimum)
    MAC_OPTIONS="$MAC_OPTIONS -DCMAKE_OSX_DEPLOYMENT_TARGET=$(sw_vers -productVersion | cut -d. -f1-2)"

    echo "Using MAC_OPTIONS for open3d build: $MAC_OPTIONS"

fi

#pip3 install --upgrade pip
pip3 uninstall -y open3d

cd thirdparty
if [ ! -d open3d ]; then
    git clone https://github.com/isl-org/Open3D.git open3d

    cd open3d
    # pin to known good commit; do not reset user changes, just ensure detached state
    git checkout 02674268f706be4b004bbbf3d39b95fa9de35f74

    # Apply the patch with --3way to handle any minor conflicts gracefully
    # The patch includes fixes for:
    # - Embree ISA configuration for Apple Silicon
    # - ZeroMQ cppzmq header discovery
    # - ExternalProject CMAKE_OSX_ARCHITECTURES inheritance
    # - Poisson SparseMatrix.inl compilation fix
    # - Pybind export symbols cleanup
    print_blue "Applying Open3D patches..."
    # Use --3way to handle cases where patch might already be partially applied
    # Suppress trailing whitespace warnings (harmless) but preserve git apply exit code
    git apply --3way ../open3d.patch 2>&1 | grep -v "trailing whitespace" > /dev/null 2>&1
    APPLY_EXIT_CODE=${PIPESTATUS[0]}
    
    if [ $APPLY_EXIT_CODE -eq 0 ]; then
        print_blue "✓ Open3D patch applied successfully"
    else
        # Check if changes are already present (patch was already applied)
        if ! git diff --quiet HEAD; then
            print_blue "✓ Open3D patch already applied or changes present"
        else
            print_yellow "⚠ Patch application had issues, but continuing..."
        fi
    fi

    cd ..

fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    num_cores=$(sysctl -n hw.physicalcpu)
else 
    num_cores=$(nproc)
fi

cd open3d

if [ ! -d install ]; then

    mkdir -p build
    cd build

    if [ $HAVE_CUDA == 1 ]; then 
        EXTERNAL_OPTIONS+=" -DBUILD_CUDA_MODULE=ON -DBUILD_COMMON_CUDA_ARCHS=ON "
    else
        EXTERNAL_OPTIONS+=" -DBUILD_CUDA_MODULE=OFF -DBUILD_COMMON_CUDA_ARCHS=OFF -DBUILD_SYCL_MODULE=OFF -DBUILD_PYTORCH_OPS=OFF -DBUILD_TENSORFLOW_OPS=OFF"
    fi

    echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

    # Build additional compiler flags for macOS to exclude conda sysroot paths
    # Initialize empty for Linux
    MAC_CXX_FLAGS=""
    if [[ "$OSTYPE" == "darwin"* ]]; then
        MAC_CXX_FLAGS="-Wno-error=unused-variable"
        if [[ -n "$MAC_SYSROOT" ]]; then
            # Explicitly set sysroot and ensure we use libc++ (macOS standard library)
            MAC_CXX_FLAGS="$MAC_CXX_FLAGS -isysroot $MAC_SYSROOT -stdlib=libc++"
            # Exclude conda sysroot paths from header search
            MAC_CXX_FLAGS="$MAC_CXX_FLAGS -Wno-unused-command-line-argument"
        fi
    fi

    # On macOS, prefer system packages from Homebrew (built for correct architecture)
    # This avoids building problematic dependencies from source
    # Use system OpenBLAS to avoid CMake compatibility issues
    # Note: Open3D requires LAPACKE, so we need to ensure it's available
    USE_SYSTEM_BLAS_VALUE="ON"

    if [[ $OSTYPE == "darwin"* ]]; then
        # ====================================================
        # Start macOS specific code
        # macOS: Check for Homebrew OpenBLAS
        if [[ "$ARCH" == "arm64" ]]; then
            # Check if OpenBLAS is installed via Homebrew on Apple Silicon
            # Check both the opt symlink and the actual Cellar location
            OPENBLAS_BASE=""
            if [ -d "/opt/homebrew/opt/openblas" ]; then
                OPENBLAS_BASE="/opt/homebrew/opt/openblas"
            elif [ -d "/opt/homebrew/Cellar/openblas" ]; then
                # Find the latest version in Cellar
                OPENBLAS_BASE=$(ls -td /opt/homebrew/Cellar/openblas/* 2>/dev/null | head -1)
            fi
            
            if [ -n "$OPENBLAS_BASE" ] && [ -d "$OPENBLAS_BASE" ]; then
                print_blue "Using system OpenBLAS from Homebrew at $OPENBLAS_BASE (avoids CMake compatibility issues)"
                
                # Set environment variables for CMake to find OpenBLAS/LAPACKE
                export LAPACKE_DIR="$OPENBLAS_BASE"
                export LAPACKE_INCLUDE_DIR="$OPENBLAS_BASE/include"
                export LAPACKE_LIBRARY="$OPENBLAS_BASE/lib"
                
                # Also set OpenBLAS-specific variables
                export OpenBLAS_DIR="$OPENBLAS_BASE"
                export OpenBLAS_ROOT="$OPENBLAS_BASE"
                
                # Add to PKG_CONFIG_PATH so pkg-config can find it
                export PKG_CONFIG_PATH="$OPENBLAS_BASE/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
                
                # Add explicit CMake variables to pass to Open3D
                # LAPACKE_DIR is the key variable that CMake's find_package(LAPACKE) uses
                # We also need to set the library paths so the test can link against them
                # OpenBLAS includes LAPACKE in the same library, so we point to libopenblas.dylib
                OPENBLAS_CMAKE_VARS="-DLAPACKE_DIR=$OPENBLAS_BASE \
                -DLAPACKE_LIBRARY=$OPENBLAS_BASE/lib/libopenblas.dylib \
                -DBLAS_LIBRARIES=$OPENBLAS_BASE/lib/libopenblas.dylib \
                -DLAPACK_LIBRARIES=$OPENBLAS_BASE/lib/libopenblas.dylib"
            else
                print_yellow "WARNING: OpenBLAS not found at /opt/homebrew/opt/openblas or /opt/homebrew/Cellar/openblas"
                print_yellow "Install it with: brew install openblas"
                print_yellow "Falling back to building OpenBLAS from source (may have CMake compatibility issues)"
                USE_SYSTEM_BLAS_VALUE="OFF"
                OPENBLAS_CMAKE_VARS=""
            fi
        elif [[ "$ARCH" == "x86_64" ]]; then
            # Check if OpenBLAS is installed via Homebrew on Intel Mac
            OPENBLAS_BASE=""
            if [ -d "/usr/local/opt/openblas" ]; then
                OPENBLAS_BASE="/usr/local/opt/openblas"
            elif [ -d "/usr/local/Cellar/openblas" ]; then
                OPENBLAS_BASE=$(ls -td /usr/local/Cellar/openblas/* 2>/dev/null | head -1)
            fi
            
            if [ -n "$OPENBLAS_BASE" ] && [ -d "$OPENBLAS_BASE" ]; then
                print_blue "Using system OpenBLAS from Homebrew at $OPENBLAS_BASE"
                
                # Set environment variables for CMake to find OpenBLAS/LAPACKE
                export LAPACKE_DIR="$OPENBLAS_BASE"
                export LAPACKE_INCLUDE_DIR="$OPENBLAS_BASE/include"
                export LAPACKE_LIBRARY="$OPENBLAS_BASE/lib"
                export OpenBLAS_DIR="$OPENBLAS_BASE"
                export OpenBLAS_ROOT="$OPENBLAS_BASE"
                export PKG_CONFIG_PATH="$OPENBLAS_BASE/lib/pkgconfig:${PKG_CONFIG_PATH:-}"
                
                # Add explicit CMake variables to pass to Open3D
                # LAPACKE_DIR is the key variable that CMake's find_package(LAPACKE) uses
                # We also need to set the library paths so the test can link against them
                # OpenBLAS includes LAPACKE in the same library, so we point to libopenblas.dylib
                OPENBLAS_CMAKE_VARS="-DLAPACKE_DIR=$OPENBLAS_BASE \
                -DLAPACKE_LIBRARY=$OPENBLAS_BASE/lib/libopenblas.dylib \
                -DBLAS_LIBRARIES=$OPENBLAS_BASE/lib/libopenblas.dylib \
                -DLAPACK_LIBRARIES=$OPENBLAS_BASE/lib/libopenblas.dylib"
            else
                print_yellow "WARNING: OpenBLAS not found. Install it with: brew install openblas"
                USE_SYSTEM_BLAS_VALUE="OFF"
                OPENBLAS_CMAKE_VARS=""
            fi
        fi
    # ====================================================
    # End macos specific code
    else

        # Linux: Check for system OpenBLAS (typically installed via package manager)
        # Common locations: /usr/lib, /usr/local/lib, /usr/lib/x86_64-linux-gnu, /usr/lib/aarch64-linux-gnu
        OPENBLAS_FOUND=false
        
        # Check common Linux library paths
        if [ -f "/usr/lib/x86_64-linux-gnu/libopenblas.so" ] || [ -f "/usr/lib/libopenblas.so" ] || \
           [ -f "/usr/lib/aarch64-linux-gnu/libopenblas.so" ] || [ -f "/usr/local/lib/libopenblas.so" ]; then
            OPENBLAS_FOUND=true
        fi
        
        # Also check for liblapacke
        if [ "$OPENBLAS_FOUND" = true ]; then
            if [ -f "/usr/lib/x86_64-linux-gnu/liblapacke.so" ] || [ -f "/usr/lib/liblapacke.so" ] || \
               [ -f "/usr/lib/aarch64-linux-gnu/liblapacke.so" ] || [ -f "/usr/local/lib/liblapacke.so" ]; then
                print_blue "Using system OpenBLAS/LAPACKE from package manager"
                # On Linux, CMake should find these automatically via pkg-config or standard paths
                USE_SYSTEM_BLAS_VALUE="ON"
            else
                print_yellow "WARNING: OpenBLAS found but LAPACKE not found"
                print_yellow "Install it with: sudo apt-get install liblapacke-dev (Debian/Ubuntu) or equivalent"
                USE_SYSTEM_BLAS_VALUE="OFF"
            fi
        else
            print_yellow "WARNING: System OpenBLAS not found"
            print_yellow "Install it with: sudo apt-get install libopenblas-dev liblapacke-dev (Debian/Ubuntu) or equivalent"
            USE_SYSTEM_BLAS_VALUE="OFF"
        fi
    fi
    
    # Build cmake command - handle macOS-specific flags
    # MAC_OPTIONS is only set on macOS, empty on Linux
    MAC_OPTIONS="${MAC_OPTIONS:-}"
    
    # Initialize OPENBLAS_CMAKE_VARS if not set (for Linux or if OpenBLAS not found)
    OPENBLAS_CMAKE_VARS="${OPENBLAS_CMAKE_VARS:-}"
    
    # Build cmake command - handle macOS-specific flags
    # MAC_OPTIONS is only set on macOS, empty on Linux
    MAC_OPTIONS="${MAC_OPTIONS:-}"
    
    # Initialize OPENBLAS_CMAKE_VARS if not set (for Linux or if OpenBLAS not found)
    OPENBLAS_CMAKE_VARS="${OPENBLAS_CMAKE_VARS:-}"
    
    if [[ $OSTYPE == "darwin"* ]]; then

        USE_SYSTEM_CURL_VALUE="ON"
        USE_SYSTEM_JSONCPP_VALUE="ON"
        USE_SYSTEM_GLFW_VALUE="ON"
        USE_SYSTEM_FMT_VALUE="ON"
        USE_SYSTEM_JPEG_VALUE="ON"
        USE_SYSTEM_ZEROMQ_VALUE="ON"  # Use system package - patch finds cppzmq headers
        USE_SYSTEM_OPENSSL_VALUE="ON"  # Use system OpenSSL for MD5 functions

        # Set CMAKE_PREFIX_PATH to prioritize Homebrew over conda
        # This ensures CMake finds Homebrew's arm64 libraries instead of conda's x86_64 ones
        HOMEBREW_PREFIX="/opt/homebrew"
        if [[ -d "$HOMEBREW_PREFIX" ]]; then
            export CMAKE_PREFIX_PATH="$HOMEBREW_PREFIX:${CMAKE_PREFIX_PATH:-}"
            print_blue "Setting CMAKE_PREFIX_PATH to prioritize Homebrew: $CMAKE_PREFIX_PATH"
        fi
        
        # Exclude conda paths from library search to avoid x86_64 libraries
        # BUT keep Python paths since we need Python from conda
        if [[ -n "$CONDA_PREFIX" ]]; then
            # Remove conda paths from library search paths, but keep Python-specific paths
            # Python libraries are typically in $CONDA_PREFIX/lib/python3.x/config-*
            export LIBRARY_PATH=$(echo "$LIBRARY_PATH" | tr ':' '\n' | grep -v "$CONDA_PREFIX/lib" | grep -v "$CONDA_PREFIX/x86_64" | tr '\n' ':' | sed 's/:$//')
            export LD_LIBRARY_PATH=$(echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep -v "$CONDA_PREFIX/lib" | grep -v "$CONDA_PREFIX/x86_64" | tr '\n' ':' | sed 's/:$//')
            export DYLD_LIBRARY_PATH=$(echo "$DYLD_LIBRARY_PATH" | tr ':' '\n' | grep -v "$CONDA_PREFIX/lib" | grep -v "$CONDA_PREFIX/x86_64" | tr '\n' ':' | sed 's/:$//')
            print_blue "Excluded conda library paths (keeping Python): $CONDA_PREFIX"
        fi
        
        # Add Homebrew library paths explicitly
        export LIBRARY_PATH="$HOMEBREW_PREFIX/lib:${LIBRARY_PATH:-}"
        export DYLD_LIBRARY_PATH="$HOMEBREW_PREFIX/lib:${DYLD_LIBRARY_PATH:-}"

        # macOS: Use system packages for problematic dependencies
        CMAKE_CMD="cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON \
        -DUSE_SYSTEM_ASSIMP=ON -DUSE_SYSTEM_VTK=ON -DUSE_SYSTEM_BLAS=$USE_SYSTEM_BLAS_VALUE -DUSE_SYSTEM_EIGEN3=ON \
        -DUSE_SYSTEM_PNG=ON -DUSE_SYSTEM_TBB=ON -DUSE_SYSTEM_EMBREE=ON \
        -DUSE_SYSTEM_CURL=$USE_SYSTEM_CURL_VALUE \
        -DUSE_SYSTEM_JSONCPP=$USE_SYSTEM_JSONCPP_VALUE \
        -DUSE_SYSTEM_GLFW=$USE_SYSTEM_GLFW_VALUE \
        -DUSE_SYSTEM_FMT=$USE_SYSTEM_FMT_VALUE \
        -DUSE_SYSTEM_JPEG=$USE_SYSTEM_JPEG_VALUE \
        -DUSE_SYSTEM_ZEROMQ=$USE_SYSTEM_ZEROMQ_VALUE \
        -DUSE_SYSTEM_OPENSSL=$USE_SYSTEM_OPENSSL_VALUE \
        -DCMAKE_OSX_ARCHITECTURES=arm64 \
        -DCMAKE_FIND_FRAMEWORK=LAST \
        -DCMAKE_FIND_APPBUNDLE=LAST"
        
        # Exclude conda library/include paths from CMake search
        if [[ -n "$CONDA_PREFIX" ]]; then
            CMAKE_CMD="$CMAKE_CMD -DCMAKE_IGNORE_PATH=\"$CONDA_PREFIX/lib;$CONDA_PREFIX/include;$CONDA_PREFIX/x86_64-conda-linux-gnu\""
        fi
        
        # Explicitly set OpenSSL paths to ensure CMake finds and links it correctly
        # Open3D uses find_package(OpenSSL) which looks for OpenSSL::Crypto target
        # We need to ensure OpenSSL is in CMAKE_PREFIX_PATH and set OPENSSL_ROOT_DIR
        OPENSSL_ROOT=$(brew --prefix openssl@3 2>/dev/null || brew --prefix openssl 2>/dev/null || echo "")
        if [[ -n "$OPENSSL_ROOT" ]] && [[ -d "$OPENSSL_ROOT" ]]; then
            # Add OpenSSL to CMAKE_PREFIX_PATH so find_package can find it
            export CMAKE_PREFIX_PATH="$OPENSSL_ROOT:$CMAKE_PREFIX_PATH"
            # Also set OPENSSL_ROOT_DIR for find_package(OpenSSL)
            CMAKE_CMD="$CMAKE_CMD -DOPENSSL_ROOT_DIR=\"$OPENSSL_ROOT\""
            print_blue "Setting OpenSSL root: $OPENSSL_ROOT (added to CMAKE_PREFIX_PATH)"
        fi
        
        # Explicitly set Python paths to use the active Python (not conda's)
        # This ensures CMake uses the correct Python with development libraries
        PYTHON_EXECUTABLE=$(which python3)
        PYTHON_PREFIX=$(python3 -c "import sys; print(sys.prefix)" 2>/dev/null)
        PYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null)
        PYTHON_LIBRARY_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null)
        
        if [[ -n "$PYTHON_EXECUTABLE" ]] && [[ -n "$PYTHON_PREFIX" ]]; then
            CMAKE_CMD="$CMAKE_CMD -DPython3_EXECUTABLE=\"$PYTHON_EXECUTABLE\" -DPython3_ROOT_DIR=\"$PYTHON_PREFIX\""
            if [[ -n "$PYTHON_INCLUDE_DIR" ]]; then
                CMAKE_CMD="$CMAKE_CMD -DPython3_INCLUDE_DIR=\"$PYTHON_INCLUDE_DIR\""
            fi
            if [[ -n "$PYTHON_LIBRARY_DIR" ]]; then
                # Find the actual Python library file
                PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null)
                if [[ -f "$PYTHON_LIBRARY_DIR/libpython${PYTHON_VERSION}.dylib" ]]; then
                    CMAKE_CMD="$CMAKE_CMD -DPython3_LIBRARY=\"$PYTHON_LIBRARY_DIR/libpython${PYTHON_VERSION}.dylib\""
                elif [[ -f "$PYTHON_LIBRARY_DIR/libpython${PYTHON_VERSION}m.dylib" ]]; then
                    CMAKE_CMD="$CMAKE_CMD -DPython3_LIBRARY=\"$PYTHON_LIBRARY_DIR/libpython${PYTHON_VERSION}m.dylib\""
                fi
            fi
            print_blue "Setting Python paths: EXECUTABLE=$PYTHON_EXECUTABLE, ROOT=$PYTHON_PREFIX"
        fi
        
        CMAKE_CMD="$CMAKE_CMD -DBUILD_WEBRTC=OFF \
        -DCMAKE_INSTALL_PREFIX=\"\`pwd\`/../install\""
        
        # Add OpenBLAS CMake variables if system OpenBLAS is being used
        if [[ -n "$OPENBLAS_CMAKE_VARS" ]]; then
            print_blue "Adding OpenBLAS/LAPACKE CMake hints: $OPENBLAS_CMAKE_VARS"
            CMAKE_CMD="$CMAKE_CMD $OPENBLAS_CMAKE_VARS"
        fi
        
        # Add CXX flags for macOS
        if [[ -n "$MAC_CXX_FLAGS" ]]; then
            CMAKE_CMD="$CMAKE_CMD -DCMAKE_CXX_FLAGS=\"$MAC_CXX_FLAGS\""
        fi
        
        # Add external options and MAC_OPTIONS
        CMAKE_CMD="$CMAKE_CMD $EXTERNAL_OPTIONS $MAC_OPTIONS"
    else
        # Linux: Use original configuration (unchanged from before)
        CMAKE_CMD="cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_EXAMPLES=OFF -DBUILD_SHARED_LIBS=ON \
        -DUSE_SYSTEM_ASSIMP=ON -DUSE_SYSTEM_VTK=ON -DUSE_SYSTEM_BLAS=ON -DUSE_SYSTEM_EIGEN3=ON \
        -DUSE_SYSTEM_PNG=ON -DUSE_SYSTEM_TBB=ON \
        -DCMAKE_CXX_FLAGS=\"-Wno-error=unused-variable\" \
        $EXTERNAL_OPTIONS -DCMAKE_INSTALL_PREFIX=\"\`pwd\`/../install\""
    fi
    
    # Execute cmake command
    eval $CMAKE_CMD

    print_blue "Starting build..."
    make -j$num_cores

    # Activate the virtualenv first
    # Install pip package in the current python environment
    make install-pip-package -j$num_cores

    # Create Python package in build/lib
    make python-package -j$num_cores

    # Create pip wheel in build/lib
    # This creates a .whl file that you can install manually.
    make pip-package -j$num_cores

    $SUDO make install
fi


cd "$STARTING_DIR"
