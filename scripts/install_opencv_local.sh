#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

# ====================================================

function print_blue(){
	printf "\033[34;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function print_red(){
	printf "\033[31;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function print_yellow(){
	printf "\033[33;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function print_green(){
	printf "\033[32;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function check_package(){
    package_name=$1
    PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $package_name |grep "install ok installed")
    #echo "checking for $package_name: $PKG_OK"
    if [ "" == "$PKG_OK" ]; then
      #echo "$package_name is not installed"
      echo 1
    else
      echo 0
    fi
}

function install_package(){
    do_install=$(check_package $1)
    if [ $do_install -eq 1 ] ; then
        sudo apt-get install -y $1
    fi 
}

function install_packages(){
    for var in "$@"
    do
        install_package "$var"
    done
}

function get_usable_cuda_version(){
    version="$1"
    if [[ "$version" != *"cuda"* ]]; then
        version="cuda-${version}"      
    fi 
    # check if we have two dots in the version, check if the folder exists otherwise remove last dot
    if [[ $version =~ ^[a-zA-Z0-9-]+\.[0-9]+\.[0-9]+$ ]]; then
        if [ ! -d /usr/local/$version ]; then 
            version="${version%.*}"  # remove last dot        
        fi     
    fi    
    echo $version
}

# Helper function to find library in a directory (handles versioned .so files)
find_lib_in_dir() {
    local libname=$1
    local lib_dir=$2
    local found=$(find "$lib_dir" -maxdepth 1 -name "${libname}.so*" -type f 2>/dev/null | head -n 1)
    if [ -n "$found" ]; then
        echo "$found"
    else
        echo ""
    fi
}

# ====================================================

export TARGET_FOLDER=thirdparty

export OPENCV_VERSION="4.11.0"   # OpenCV version to download and install. See tags in https://github.com/opencv/opencv 


SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used
STARTING_DIR=`pwd`

ROOT_DIR="$SCRIPT_DIR_/.."

cd "$ROOT_DIR"
TARGET_FOLDER="$ROOT_DIR/$TARGET_FOLDER"

# ====================================================
print_blue  "Configuring and building $TARGET_FOLDER/opencv ..."

#pip3 install --upgrade pip
pip3 uninstall -y opencv-python
pip3 uninstall -y opencv-contrib-python

pip3 install --upgrade numpy

#set -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# Check if conda is installed
if command -v conda &> /dev/null; then
    echo "Conda is installed"
    CONDA_INSTALLED=true
else
    echo "Conda is not installed"
    CONDA_INSTALLED=false
fi

# Check if pixi is activated
if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
    echo "Pixi environment detected: $PIXI_PROJECT_NAME"

    source "$SCRIPT_DIR_/pixi_cuda_config.sh"
    source "$SCRIPT_DIR_/pixi_python_config.sh"
else
    PIXI_ACTIVATED=false
fi

if [ ! -d $TARGET_FOLDER ]; then 
    mkdir -p $TARGET_FOLDER
fi 

# set CUDA 
#export CUDA_VERSION="cuda-11.8"  # must be an installed CUDA path in /usr/local; 
                                  # if available, you can use the simple path "/usr/local/cuda" which should be a symbolic link to the last installed cuda version 
CUDA_ON=ON
if [[ -n "$CUDA_VERSION" ]]; then
    CUDA_VERSION=$(get_usable_cuda_version $CUDA_VERSION)
    echo using CUDA $CUDA_VERSION
	if [ ! -d /usr/local/$CUDA_VERSION ]; then 
		echo CUDA $CUDA_VERSION does not exist
		CUDA_ON=OFF
	fi 
else
    if [ -d /usr/local/cuda ]; then
        CUDA_VERSION="cuda"  # use last installed CUDA path 
        echo using CUDA $CUDA_VERSION        
    else
        print_red "Warning: CUDA $CUDA_VERSION not found and will not be used!"
        CUDA_ON=OFF
    fi 
fi 



echo CUDA_ON: $CUDA_ON


WITH_DNN=ON                # this can be used to turn off the DNN module
WITH_PROTOBUF=ON           # this can be used to turn off the protobuf module (it is required for DNN)
WITH_APPLE_FRAMEWORK=OFF
if [[ "$version" == *"darwin"* ]]; then
    #WITH_APPLE_FRAMEWORK=ON   # this will make opencv generate a single libopencv_world.so without the separate modules
    CUDA_ON=OFF
    WITH_PROTOBUF=OFF  # I am getting a protobuf version error on my mac
fi


WITH_NEON=OFF
arch=$(uname -m)
if [[ "$arch" == "arm64" || "$arch" == "aarch64" || "$arch" == arm* ]]; then
    WITH_NEON=ON
fi

WITH_QT=OFF  # it seems to interfere with python packages that use Qt
WITH_GTK=ON
WITH_OPENGL=ON
if [ "$CONDA_INSTALLED" = true ]; then
    WITH_QT=OFF
    WITH_GTK=ON
fi

# pre-installing some required packages 

export BUILD_SFM_OPTION="ON"
if [[ $version == *"24.04"* ]] ; then
    BUILD_SFM_OPTION="OFF"  # it seems this module brings some build issues with Ubuntu 24.04
fi

if [[ ! -d "$TARGET_FOLDER/opencv" ]]; then
    if [[ $version != *"darwin"* ]]; then
        sudo apt-get update
        sudo apt-get install -y pkg-config libglew-dev libtiff5-dev zlib1g-dev libjpeg-dev libeigen3-dev libtbb-dev libgtk2.0-dev libopenblas-dev libgtk-3-dev
        sudo apt-get install -y curl software-properties-common unzip
        sudo apt-get install -y libicu-dev        
        sudo apt-get install -y build-essential cmake 
        if [[ "$CUDA_ON" == "ON" ]]; then 
            if [[ $version == *"24.04"* ]] ; then
                sudo apt-get install -y libcudnn-dev
            else 
                sudo apt-get install -y libcudnn8 libcudnn8-dev  # check and install otherwise this is going to update to the latest version (and that's not we necessary want to do)
            fi
        fi 

        if [[ $version == *"22.04"* || $version == *"24.04"* ]] ; then
            sudo apt install -y libtbb-dev libeigen3-dev 
            sudo apt install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev
            sudo add-apt-repository -y "deb http://security.ubuntu.com/ubuntu xenial-security main"  # for libjasper-dev 
            sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys 3B4FE6ACC0B21F32 # for libjasper-dev 
            sudo apt update
            sudo apt install -y libjasper-dev
            sudo apt install -y libv4l-dev libdc1394-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
                                    libopencore-amrnb-dev libopencore-amrwb-dev libxine2-dev libva-dev           
        fi
        if [[ $version == *"20.04"* ]] ; then
            sudo apt install -y libtbb-dev libeigen3-dev 
            sudo apt install -y zlib1g-dev libjpeg-dev libwebp-dev libpng-dev libtiff5-dev 
            sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"  # for libjasper-dev 
            sudo apt install -y libjasper-dev
            sudo apt install -y libv4l-dev libdc1394-22-dev libtheora-dev libvorbis-dev libxvidcore-dev libx264-dev yasm \
                                    libopencore-amrnb-dev libopencore-amrwb-dev libxine2-dev            
        fi        
        if [[ $version == *"18.04"* ]] ; then
            sudo apt-get install -y libpng-dev 
            sudo add-apt-repository "deb http://security.ubuntu.com/ubuntu xenial-security main"  # for libjasper-dev 
            sudo apt-get install -y libjasper-dev
        fi
        if [[ $version == *"16.04"* ]] ; then
            sudo apt-get install -y libpng12-dev libjasper-dev 
        fi        

        DO_INSTALL_FFMPEG=$(check_package ffmpeg)
        if [ $DO_INSTALL_FFMPEG -eq 1 ] ; then
            echo "installing ffmpeg and its dependencies"
            sudo apt-get install -y libavcodec-dev libavformat-dev libavutil-dev libpostproc-dev libswscale-dev 
        fi 

        if [ "$CONDA_INSTALLED" = true ]; then
            # NOTE: these are the "system" packages that are needed within conda to build opencv from source
            conda install -y -c conda-forge \
                pkg-config \
                glew \
                cmake \
                suitesparse \
                lapack \
                libtiff zlib jpeg eigen tbb glew libpng \
                x264 ffmpeg \
                freetype cairo \
                pygobject gtk2 gtk3 glib xorg-xorgproto \
                libwebp expat \
                boost openblas \
                glog gflags

            # Linux-specific compilers
            if [[ "$OSTYPE" == "linux-gnu"* ]]; then
                conda install -y -c conda-forge \
                    compilers gcc_linux-64 gxx_linux-64 tbb tbb-devel
            fi

            if [[ "$CUDA_ON" == "ON" ]]; then 
                conda install -y -c conda-forge cudnn libcudnn libcudnn-dev
                # check if last command failed 
                if [[ $? -ne 0 ]]; then
                    print_yellow "Warning: failed to install libcudnn-dev. Setting WITH_DNN=OFF."
                    WITH_DNN=OFF
                fi
            fi
        fi        
    else
        brew install pkg-config 
        brew install glew
        brew install cmake
        brew install suitesparse 
        brew install lapack
        brew install libtiff zlib jpeg eigen tbb glew libpng webp 
        brew install x264 ffmpeg
        brew install protobuf
    fi 
fi

export CONDA_OPTIONS=""
export MAC_OPTIONS=""
export PIXI_OPTIONS=""

if [ "$CONDA_INSTALLED" = true ]; then

    # This linker flag only makes sense on Linux, not on macOS
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        CPPFLAGS="$CPPFLAGS -Wl,--disable-new-dtags" # enable RPATH support in conda environments
    fi
  
    CONDA_OPTIONS="-DOPENCV_FFMPEG_USE_FIND_PACKAGE=OFF \
    -DPKG_CONFIG_EXECUTABLE=$(which pkg-config) \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_CXX_STANDARD=17 \
    -DWITH_WEBP=ON -DBUILD_PROTOBUF=OFF -DPROTOBUF_UPDATE_FILES=ON \
    -DOPENCV_FFMPEG_SKIP_BUILD_CHECK=ON"

    echo "Using CONDA_OPTIONS for opencv build: $CONDA_OPTIONS"
fi

# When using pixi with CUDA, explicitly set CUDA library paths to help CMake find them
if [ "$PIXI_ACTIVATED" = true ]; then 

    # Configure pixi CUDA if CUDA is enabled
    # The pixi_cuda_config.sh script should have been sourced earlier
    # and set up all the necessary environment variables
    if [ "$CUDA_ON" = "ON" ] && [ "$PIXI_ACTIVATED" = true ]; then
        # Use the function from pixi_cuda_config.sh to get CMake options
        if command -v get_pixi_cuda_cmake_options &> /dev/null; then
            PIXI_CUDA_CMAKE_OPTS=$(get_pixi_cuda_cmake_options)
            PIXI_OPTIONS="$PIXI_OPTIONS $PIXI_CUDA_CMAKE_OPTS"
            echo "Added pixi CUDA CMake options: $PIXI_CUDA_CMAKE_OPTS"
        else
            echo "Warning: get_pixi_cuda_cmake_options not found. CUDA CMake options may not be set correctly."
        fi
    fi

    # OpenGL is disabled by default in pixi to avoid linking issues
    # Skip OpenGL configuration if it's disabled
    WITH_OPENGL=OFF
    if [ "$WITH_OPENGL" = "OFF" ]; then
        echo "OpenGL is disabled in pixi - skipping OpenGL library detection"
    fi

    # Fix compiler issues in pixi: unset conda-style compiler variables and use system compilers
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        # Unset any conda-style compiler variables that pixi might have set
        unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION
        
        # Find system compilers
        SYSTEM_CC=$(which gcc 2>/dev/null || echo "/usr/bin/gcc")
        SYSTEM_CXX=$(which g++ 2>/dev/null || echo "/usr/bin/g++")
        
        # Verify compilers exist
        if [ ! -f "$SYSTEM_CC" ]; then
            print_red "Warning: System gcc not found at $SYSTEM_CC. Trying to find alternative..."
            SYSTEM_CC=$(find /usr/bin /usr/local/bin -name gcc 2>/dev/null | head -n 1)
        fi
        if [ ! -f "$SYSTEM_CXX" ]; then
            print_red "Warning: System g++ not found at $SYSTEM_CXX. Trying to find alternative..."
            SYSTEM_CXX=$(find /usr/bin /usr/local/bin -name g++ 2>/dev/null | head -n 1)
        fi
        
        if [ -f "$SYSTEM_CC" ] && [ -f "$SYSTEM_CXX" ]; then
            PIXI_OPTIONS="$PIXI_OPTIONS -DCMAKE_C_COMPILER=$SYSTEM_CC -DCMAKE_CXX_COMPILER=$SYSTEM_CXX"
            echo "Using system compilers for pixi: CC=$SYSTEM_CC, CXX=$SYSTEM_CXX"
        else
            print_red "Error: Could not find system compilers (gcc/g++). Please install build-essential."
            exit 1
        fi
    fi

    # Fix Python detection in pixi: use pixi's Python instead of system Python
    # The pixi_python_config.sh script should have been sourced earlier
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        # Use the function from pixi_python_config.sh to get CMake options
        if command -v get_pixi_python_cmake_options &> /dev/null; then
            PIXI_PYTHON_CMAKE_OPTS=$(get_pixi_python_cmake_options)
            PIXI_OPTIONS="$PIXI_OPTIONS $PIXI_PYTHON_CMAKE_OPTS"
            if [ -n "$PIXI_PYTHON" ]; then
                echo "Using pixi Python for OpenCV build: $PIXI_PYTHON"
                if [ -f "$PIXI_PYTHON" ]; then
                    echo "Python version: $($PIXI_PYTHON --version 2>&1)"
                fi
            fi
        else
            print_yellow "Warning: get_pixi_python_cmake_options not found. CMake will try to auto-detect Python."
        fi
    fi

    # Fix protobuf conflicts in pixi: force OpenCV to build its own protobuf
    # Pixi's protobuf headers can conflict with OpenCV's bundled protobuf
    if [[ "$OSTYPE" == "linux-gnu"* ]] || [[ "$OSTYPE" == "darwin"* ]]; then
        PIXI_OPTIONS="$PIXI_OPTIONS -DBUILD_PROTOBUF=ON -DProtobuf_FOUND=OFF"
        
        # Temporarily hide pixi's protobuf headers to prevent conflicts
        PIXI_PROTOBUF_INCLUDE=""
        if [ -d "$ROOT_DIR/.pixi/envs/default/include/google/protobuf" ]; then
            PIXI_PROTOBUF_INCLUDE="$ROOT_DIR/.pixi/envs/default/include/google/protobuf"
        elif [ -d ".pixi/envs/default/include/google/protobuf" ]; then
            PIXI_PROTOBUF_INCLUDE=".pixi/envs/default/include/google/protobuf"
        fi
        
        if [ -n "$PIXI_PROTOBUF_INCLUDE" ] && [ -d "$PIXI_PROTOBUF_INCLUDE" ]; then
            PIXI_PROTOBUF_BACKUP="${PIXI_PROTOBUF_INCLUDE}.backup"
            if [ ! -d "$PIXI_PROTOBUF_BACKUP" ]; then
                echo "Temporarily hiding pixi protobuf headers to avoid conflicts with OpenCV's bundled protobuf"
                mv "$PIXI_PROTOBUF_INCLUDE" "$PIXI_PROTOBUF_BACKUP" 2>/dev/null || true
                # Create a marker file to track that we hid it
                touch "${PIXI_PROTOBUF_INCLUDE}.hidden_by_opencv_build" 2>/dev/null || true
            fi
        fi
    fi
    
    # Export PIXI_OPTIONS so it's available in the cmake command
    export PIXI_OPTIONS

fi # end of PIXI_ACTIVATED = true

# For non-pixi environments, set up Python options
if [ "$PIXI_ACTIVATED" != true ]; then
    PYTHON_OPTIONS="-DPYTHON_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") -DPYTHON_LIBRARY=$(python3 -c "import sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))") -DPYTHON_EXECUTABLE=$(which python3) -DPYTHON3_EXECUTABLE=$(which python3)"
else
    PYTHON_OPTIONS=""
fi

if [[ $version == *"24.04"* ]] ; then
    export CMAKE_ARGS="$CMAKE_ARGS -DBUILD_opencv_sfm=OFF" # It seems this module brings some build issues with Ubuntu 24.04
fi

if [[ "$OSTYPE" == "darwin"* ]]; then
    # Make sure we don't accidentally use a Linux cross-compiler or Linux sysroot from conda
    unset CC CXX CFLAGS CXXFLAGS LDFLAGS CPPFLAGS SDKROOT CONDA_BUILD_SYSROOT CONDA_BUILD_CROSS_COMPILATION

    # Ask Xcode for the proper macOS SDK path (fallback to default if unavailable)
    MAC_SYSROOT=$(xcrun --show-sdk-path 2>/dev/null || echo "")

    MAC_OPTIONS="-DBUILD_PROTOBUF=OFF -DPROTOBUF_UPDATE_FILES=ON \
    -DVIDEOIO_ENABLE_STRICT=ON -DWITH_FFMPEG=OFF \
    -DWITH_OPENGL=OFF \
    -DBUILD_opencv_hdf=OFF -DWITH_HDF5=OFF \
    -DWITH_OPENEXR=OFF -DWITH_AVIF=OFF \
    -DBUILD_opencv_sfm=OFF \
    -DCMAKE_C_COMPILER=/usr/bin/clang \
    -DCMAKE_CXX_COMPILER=/usr/bin/clang++"

    echo "Using MAC_OPTIONS for opencv build: $MAC_OPTIONS"
fi

export CMAKE_ARGS="$CONDA_OPTIONS $CMAKE_ARGS" # -DCMAKE_CXX_FLAGS=$CPPFLAGS"
export CMAKE_CXX_FLAGS="$CPPFLAGS"
export CMAKE_INCLUDE_PATH="$CPP_INCLUDE_PATH"
export CMAKE_LIBRARY_PATH="$LIBRARY_PATH"

# now let's download and compile opencv and opencv_contrib
# N.B: if you want just to update cmake settings and recompile then remove "opencv/install" and "opencv/build/CMakeCache.txt"

cd $TARGET_FOLDER

# Choose correct core library name based on platform
if [[ "$OSTYPE" == "darwin"* ]]; then
    OPENCV_CORE_LIB="opencv/install/lib/libopencv_core.dylib"
else
    OPENCV_CORE_LIB="opencv/install/lib/libopencv_core.so"
fi

if [ ! -f $OPENCV_CORE_LIB ]; then
    if [ ! -d opencv ]; then
      wget https://github.com/opencv/opencv/archive/$OPENCV_VERSION.zip
      sleep 1
      unzip $OPENCV_VERSION.zip
      rm $OPENCV_VERSION.zip
      cd opencv-$OPENCV_VERSION

      wget https://github.com/opencv/opencv_contrib/archive/$OPENCV_VERSION.zip
      sleep 1
      unzip $OPENCV_VERSION.zip
      rm $OPENCV_VERSION.zip

      cd ..
      mv opencv-$OPENCV_VERSION opencv
    fi
    echo "entering opencv"
    cd opencv
    mkdir -p build
    mkdir -p install
    cd build
    echo "I am in "$(pwd)
    machine="$(uname -m)"
    echo OS: $version
    if [[ "$machine" == "x86_64" || "$machine" == "x64" || "$version" == "darwin"* ]]; then
		# standard configuration 
        echo "building laptop/desktop config under $version"
        # as for the flags and consider this nice reference https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
        cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=17 \
          -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
          -DOPENCV_EXTRA_MODULES_PATH="`pwd`/../opencv_contrib-$OPENCV_VERSION/modules" \
          -DWITH_FFMPEG=ON \
          -DWITH_QT=$WITH_QT \
          -DWITH_GTK=$WITH_GTK \
          -DWITH_OPENGL=$WITH_OPENGL \
          -DWITH_TBB=ON \
          -DWITH_V4L=ON \
          -DWITH_CUDA=$CUDA_ON \
          -DWITH_CUBLAS=$CUDA_ON \
          -DWITH_CUFFT=$CUDA_ON \
          -DCUDA_FAST_MATH=$CUDA_ON \
          -DWITH_CUDNN=$CUDA_ON \
          -DBUILD_opencv_dnn=$WITH_DNN \
          -DOPENCV_DNN_CUDA=$CUDA_ON \
          -DCUDA_ARCH_BIN="5.3 6.0 6.1 7.0 7.5 8.6" \
          -DBUILD_opencv_cudacodec=OFF \
          -DENABLE_FAST_MATH=1 \
          -DBUILD_opencv_sfm=$BUILD_SFM_OPTION \
          -DBUILD_NEW_PYTHON_SUPPORT=ON \
          -DBUILD_DOCS=OFF \
          -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF \
          -DINSTALL_PYTHON_EXAMPLES=OFF \
          -DINSTALL_C_EXAMPLES=OFF \
          -DBUILD_EXAMPLES=OFF \
          -DBUILD_opencv_apps=OFF \
          -DOPENCV_ENABLE_NONFREE=ON \
          -DBUILD_opencv_java=OFF \
          -DBUILD_opencv_python3=ON \
          -Wno-deprecated-gpu-targets \
          -DBUILD_PROTOBUF=${WITH_PROTOBUF:-OFF} \
          -DAPPLE_FRAMEWORK=${WITH_APPLE_FRAMEWORK:-OFF} \
          $CONDA_OPTIONS $MAC_OPTIONS $PIXI_OPTIONS $PYTHON_OPTIONS ..
    else
        # Nvidia Jetson aarch64
        echo "building NVIDIA Jetson config"
        cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_CXX_STANDARD=17 \
          -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
          -DOPENCV_EXTRA_MODULES_PATH="`pwd`/../opencv_contrib-$OPENCV_VERSION/modules" \
          -DWITH_QT=$WITH_QT \
          -DWITH_GTK=$WITH_GTK \
          -DWITH_OPENGL=$WITH_OPENGL \
          -DWITH_TBB=ON \
          -DWITH_V4L=ON \
          -DWITH_CUDA=ON \
          -DWITH_CUBLAS=ON \
          -DWITH_CUFFT=ON \
          -DCUDA_FAST_MATH=ON \
          -DCUDA_ARCH_BIN="6.2" \
          -DCUDA_ARCH_PTX="" \
          -DBUILD_opencv_cudacodec=OFF \
          -DENABLE_NEON=ON \
          -DENABLE_FAST_MATH=ON \
          -DBUILD_NEW_PYTHON_SUPPORT=ON \
          -DBUILD_DOCS=OFF \
          -DBUILD_TESTS=OFF \
          -DBUILD_PERF_TESTS=OFF \
          -DINSTALL_PYTHON_EXAMPLES=OFF \
          -DINSTALL_C_EXAMPLES=OFF \
          -DBUILD_EXAMPLES=OFF \
          -Wno-deprecated-gpu-targets ..
    fi
    make -j$(nproc)  # use nproc to get the number of available cores
    make install -j$(nproc)
    
    # Restore pixi's protobuf headers if we hid them
    if [ "$PIXI_ACTIVATED" = true ]; then
        PIXI_PROTOBUF_INCLUDE=""
        if [ -d "$ROOT_DIR/.pixi/envs/default/include/google/protobuf.backup" ]; then
            PIXI_PROTOBUF_INCLUDE="$ROOT_DIR/.pixi/envs/default/include/google"
        elif [ -d ".pixi/envs/default/include/google/protobuf.backup" ]; then
            PIXI_PROTOBUF_INCLUDE=".pixi/envs/default/include/google"
        fi
        
        if [ -n "$PIXI_PROTOBUF_INCLUDE" ] && [ -f "${PIXI_PROTOBUF_INCLUDE}/protobuf.hidden_by_opencv_build" ]; then
            echo "Restoring pixi protobuf headers"
            rm -rf "${PIXI_PROTOBUF_INCLUDE}/protobuf" 2>/dev/null || true
            mv "${PIXI_PROTOBUF_INCLUDE}/protobuf.backup" "${PIXI_PROTOBUF_INCLUDE}/protobuf" 2>/dev/null || true
            rm -f "${PIXI_PROTOBUF_INCLUDE}/protobuf.hidden_by_opencv_build" 2>/dev/null || true
        fi
    fi
fi

cd $TARGET_FOLDER
if [[ -d opencv/install ]]; then
    cd opencv
    echo "deploying built cv2 python module"
    
    # For pixi environments, use pixi's Python directly to avoid pyenv interference
    # The pixi_python_config.sh script should have been sourced earlier
    if [ "$PIXI_ACTIVATED" = true ]; then
        # Use variables from pixi_python_config.sh
        if [ -n "$PIXI_PYTHON_VERSION" ]; then
            PYTHON_VERSION="$PIXI_PYTHON_VERSION"
        fi
        if [ -n "$PIXI_PYTHON_SITE_PACKAGES" ]; then
            PYTHON_SITE_PACKAGES="$PIXI_PYTHON_SITE_PACKAGES"
        fi
        if [ -n "$PIXI_PYTHON" ]; then
            PYTHON_EXECUTABLE="$PIXI_PYTHON"
        fi
        
        if [ -n "$PIXI_PYTHON" ]; then
            echo "Pixi environment detected - using pixi Python: $PIXI_PYTHON"
            if [ -n "$PYTHON_VERSION" ]; then
                echo "Python version: $PYTHON_VERSION"
            fi
            if [ -n "$PYTHON_SITE_PACKAGES" ]; then
                echo "Pixi site-packages: $PYTHON_SITE_PACKAGES"
            fi
        fi
    else
        # For non-pixi environments, use regular python
        PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")" 2>/dev/null || echo "")
        PYTHON_SITE_PACKAGES=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())" 2>/dev/null || python -c "import site; print(site.getsitepackages()[0])" 2>/dev/null || echo "")
    fi
    
    PYTHON_SOURCE_FOLDER=$(pwd)/install/lib/python$PYTHON_VERSION/site-packages/cv2
    
    echo "PYTHON_SITE_PACKAGES: $PYTHON_SITE_PACKAGES"
    echo "PYTHON_SOURCE_FOLDER: $PYTHON_SOURCE_FOLDER"
    
    if [[ -d "$PYTHON_SOURCE_FOLDER" ]]; then
        if [[ -n "$PYTHON_SITE_PACKAGES" ]] && [[ -d "$PYTHON_SITE_PACKAGES" ]] && [[ -w "$PYTHON_SITE_PACKAGES" ]]; then
            echo "copying built python cv2 module from $PYTHON_SOURCE_FOLDER to $PYTHON_SITE_PACKAGES"
            cp -r $PYTHON_SOURCE_FOLDER $PYTHON_SITE_PACKAGES
            if [[ $? -eq 0 ]]; then
                print_green "Successfully deployed cv2 module to $PYTHON_SITE_PACKAGES"
            else
                print_yellow "WARNING: Failed to copy cv2 module. Using PYTHONPATH method instead."
                PYTHON_SITE_PACKAGES=""  # Trigger fallback
            fi
        fi
        
        # If site-packages copy failed or wasn't attempted, use PYTHONPATH method
        if [[ -z "$PYTHON_SITE_PACKAGES" ]] || [[ ! -d "$PYTHON_SITE_PACKAGES" ]] || [[ ! -w "$PYTHON_SITE_PACKAGES" ]]; then
            PYTHONPATH_DIR=$(dirname $PYTHON_SOURCE_FOLDER)
            print_yellow "NOTE: cv2 module will be available via PYTHONPATH."
            print_green "The cv2 module is available at: $PYTHON_SOURCE_FOLDER"
            echo ""
            echo "To use it, add to your environment:"
            echo "  export PYTHONPATH=$PYTHONPATH_DIR:\$PYTHONPATH"
            echo ""
            echo "Or add this line to your ~/.bashrc or pixi environment setup:"
            echo "  export PYTHONPATH=$PYTHONPATH_DIR:\$PYTHONPATH"
        fi
    else
        print_red "ERROR: Built cv2 module not found at $PYTHON_SOURCE_FOLDER"  
    fi 
fi

cd $STARTING_DIR

# Install supported numpy version <2 to avoid conflicts
pip3 install "numpy<2"

echo "...done with opencv"