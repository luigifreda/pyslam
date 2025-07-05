#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR"

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 


STARTING_DIR=`pwd`  
cd "$ROOT_DIR"  


# ====================================================

print_blue '================================================'
print_blue "Installing opencv-python from source"
print_blue '================================================'

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
fi

# Check if conda is installed
if command -v conda &> /dev/null; then
    CONDA_INSTALLED=true
else
    CONDA_INSTALLED=false
fi

if [[ -n "$PIXI_PROJECT_NAME" ]]; then
    PIXI_ACTIVATED=true
else
    PIXI_ACTIVATED=false
fi


# Get the current Python environment's base directory
PYTHON_ENV=$(python3 -c "import sys; print(sys.prefix)")
PYTHON_VERSION=$(python3 -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")
echo "Using PYTHON_ENV: $PYTHON_ENV"


EXTERNAL_OPTIONS=""
if [ "$PIXI_ACTIVATED" = true ]; then
    PYTHON_EXECUTABLE=$(which python)
    PYTHON_INCLUDE_DIR=$(python -c "from sysconfig import get_paths as gp; print(gp()['include'])")
    PYTHON_LIBRARY=$(find $(dirname $PYTHON_EXECUTABLE)/../lib -name libpython$(python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')").so | head -n 1)
    echo PYTHON_EXECUTABLE: $PYTHON_EXECUTABLE
    echo PYTHON_INCLUDE_DIR: $PYTHON_INCLUDE_DIR
    echo PYTHON_LIBRARY: $PYTHON_LIBRARY

    # This is needed to make pybind11 find the correct python interpreter within the pixi environment
    EXTERNAL_OPTIONS="-DPYTHON_EXECUTABLE=\"${PYTHON_EXECUTABLE}\" \
    -DPYTHON_INCLUDE_DIR=\"${PYTHON_INCLUDE_DIR}\" \
    -DPYTHON_LIBRARY=\"$PYTHON_LIBRARY\""
fi


#pip3 install --upgrade pip
pip3 uninstall -y opencv-python
pip3 uninstall -y opencv-contrib-python

if [ "$PIXI_ACTIVATED" = true ]; then
    rm -rf $(python -c "import cv2; import os; print(os.path.dirname(cv2.__file__))")
    find $(python -c "import site; print(site.getsitepackages()[0])") -name "opencv*" -exec rm -rf {} +
fi


pip3 install --upgrade numpy


if [[ $version != *"darwin"* ]]; then
    sudo apt-get update
    sudo apt-get install -y pkg-config libglew-dev libtiff5-dev zlib1g-dev libjpeg-dev libeigen3-dev libtbb-dev libgtk2.0-dev libopenblas-dev
    sudo apt-get install -y curl software-properties-common unzip
    sudo apt-get install -y build-essential cmake 
    if [[ "$CUDA_ON" == "ON" ]]; then 
        if [[ $version == *"24.04"* ]] ; then
            install_packages libcudnn-dev
        else 
            install_packages libcudnn8 libcudnn8-dev  # check and install otherwise this is going to update to the latest version (and that's not we necessary want to do)
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
                                libopencore-amrnb-dev libopencore-amrwb-dev libxine2-dev            
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

    if [ $CONDA_INSTALLED = true ]; then
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
            pygobject gtk3 glib libwebp
    fi
else
    brew install pkg-config 
    brew install glew
    brew install cmake
    brew install suitesparse 
    brew install lapack
    brew install libtiff zlib jpeg eigen tbb glew libpng webp x264 ffmpeg
fi

cd thirdparty
if [ ! -d opencv-python ]; then
    git clone --recursive https://github.com/opencv/opencv-python.git
    cd opencv-python
    # This procedure worked on commit cce7c994d46406205eb39300bb7ca9c48d80185a  that corresponds to opencv 4.10.0.84 -> https://github.com/opencv/opencv-python/releases/tag/84 
    #git checkout cce7c994d46406205eb39300bb7ca9c48d80185a  # uncomment this if you get some issues in building opencv_python! 
    cd ..
fi

cd opencv-python
MY_OPENCV_PYTHON_PATH=`pwd`

export MAKEFLAGS="-j$(nproc)"
export CPPFLAGS+=""
export CPATH+=""
export CPP_INCLUDE_PATH+=""
export C_INCLUDE_PATH+=""


NUMPY_INCLUDE_PATH=$(python3 -c "import numpy; print(numpy.get_include())")


# Set include paths dynamically
#export CPPFLAGS="-I$NUMPY_INCLUDE_PATH $CPPFLAGS"
export CPATH="$NUMPY_INCLUDE_PATH:$CPATH"
export C_INCLUDE_PATH="$NUMPY_INCLUDE_PATH:$C_INCLUDE_PATH"
export CPP_INCLUDE_PATH="$NUMPY_INCLUDE_PATH:$CPP_INCLUDE_PATH"

# Set library paths dynamically
export LDFLAGS="-L$PYTHON_ENV/lib:$LDFLAGS"
export LIBRARY_PATH="$PYTHON_ENV/lib:$LIBRARY_PATH"

# Add NumPy library path if needed
NUMPY_LIB_PATH=$(python3 -c "import numpy; print(numpy.__path__[0] + '/core/lib')")
export LDFLAGS="-L$NUMPY_LIB_PATH:$LDFLAGS"
export LIBRARY_PATH="$NUMPY_LIB_PATH:$LIBRARY_PATH"


CONDA_OPTIONS=""
GTK_OPTIONS=""
if [ $CONDA_INSTALLED = true ]; then

    CPPFLAGS="$CPPFLAGS -Wl,--disable-new-dtags" # enable RPATH support in conda environments
  
    CONDA_OPTIONS="-DOPENCV_FFMPEG_USE_FIND_PACKAGE=OFF \
    -DPKG_CONFIG_EXECUTABLE=$(which pkg-config) \
    -DCMAKE_PREFIX_PATH=$CONDA_PREFIX -DCMAKE_CXX_STANDARD=17 -DWITH_WEBP=ON -DBUILD_PROTOBUF=OFF -DPROTOBUF_UPDATE_FILES=ON"
fi
echo "Using CONDA_OPTIONS for opencv build: $CONDA_OPTIONS"
echo "Using GTK_OPTIONS for opencv build: $GTK_OPTIONS"

export CMAKE_ARGS="-DOPENCV_ENABLE_NONFREE=ON \
-DOPENCV_EXTRA_MODULES_PATH=$MY_OPENCV_PYTHON_PATH/opencv_contrib/modules \
-DBUILD_SHARED_LIBS=OFF \
-DWITH_FFMPEG=ON  "$CONDA_OPTIONS" "$GTK_OPTIONS" \
-DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF ${EXTERNAL_OPTIONS}"

if [[ $version == *"24.04"* ]] ; then
    export CMAKE_ARGS="$CMAKE_ARGS -DBUILD_opencv_sfm=OFF" # It seems this module brings some build issues with Ubuntu 24.04
fi

export CMAKE_ARGS="$CMAKE_ARGS" # -DCMAKE_CXX_FLAGS=$CPPFLAGS"
export CMAKE_CXX_FLAGS="$CPPFLAGS"
export CMAKE_INCLUDE_PATH="$CPP_INCLUDE_PATH"
export CMAKE_LIBRARY_PATH="$LIBRARY_PATH"

# build opencv_python 
pip3 wheel . --verbose
# install built packages
pip3 install opencv*.whl --force-reinstall

# build opencv_contrib_python
# NOTE: A double "pip3 wheel" may seem redundant. I verified this is needed to ensure contrib modules are built and installed correctly
export ENABLE_CONTRIB=1
pip3 wheel . --verbose
# install built packages
pip3 install opencv*.whl --force-reinstall


# Extract the installed OpenCV version and wheel path
OPENCV_WHL_PATH=$(ls opencv*.whl | head -n 1)
OPENCV_PKG_NAME=$(basename "$OPENCV_WHL_PATH" | cut -d'-' -f1)  # usually "opencv_python"

# Write pip constraint file
echo "$OPENCV_PKG_NAME @ file://$MY_OPENCV_PYTHON_PATH/$OPENCV_WHL_PATH" > "$ROOT_DIR/constraints.txt"

print_green "Created constraints.txt:"
cat "$ROOT_DIR/constraints.txt"


cd "$ROOT_DIR"


# HACK for conda issue under linux:
# "ImportError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32"
#  See https://github.com/elerac/EasyPySpin/issues/12
if [[ "$OSTYPE" != "darwin"* ]]; then

    CV2_SO_PATH=$PYTHON_ENV/lib/python$PYTHON_VERSION/site-packages/cv2
    # Find the actual .so file (handles different naming variations)
    CV2_SO_FILE=$(ls "$CV2_SO_PATH"/cv2.*.so 2>/dev/null | head -n 1)
    # Check if a valid file was found
    if [[ -n "$CV2_SO_FILE" && -f "$CV2_SO_FILE" ]]; then
        if ldd "$CV2_SO_FILE" | grep -q "libffi"; then
            LIBFFI=$(ldconfig -p | grep libffi.so | awk '{print $NF}' | head -n1)
            #if [[ -f /lib/x86_64-linux-gnu/libffi.so ]]; then
            if [[ -f $LIBFFI ]]; then
                echo "Preloading libffi..."
                #export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so
                export LD_PRELOAD=$LIBFFI
            else
                echo "WARNING: /lib/x86_64-linux-gnu/libffi.so not found; check your libffi dependencies"
            fi
        fi
    fi

fi

# Install supported numpy version <2 to avoid conflicts
pip3 install "numpy<2"

cd "$STARTING_DIR"