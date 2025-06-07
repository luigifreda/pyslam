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

# ====================================================

export TARGET_FOLDER=thirdparty

export OPENCV_VERSION="4.10.0"   # OpenCV version to download and install. See tags in https://github.com/opencv/opencv 


SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
STARTING_DIR=`pwd`

cd "$SCRIPT_DIR"
TARGET_FOLDER="$SCRIPT_DIR/$TARGET_FOLDER"

# ====================================================
print_blue  "Configuring and building $TARGET_FOLDER/opencv ..."

#pip3 install --upgrade pip
pip3 uninstall -y opencv-python
pip3 uninstall -y opencv-contrib-python

pip3 install --upgrade numpy

set -e

if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    version=$(lsb_release -a 2>&1)  # ubuntu version
else 
    version=$OSTYPE
    echo "OS: $version"
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
export PATH=/usr/local/$CUDA_VERSION/bin${PATH:+:${PATH}}   # this is for having the right nvcc in the path
export LD_LIBRARY_PATH=/usr/local/$CUDA_VERSION/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}  # this is for libs 


WITH_APPLE_FRAMEWORK=OFF
WITH_PROTOBUF=ON
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

# pre-installing some required packages 

export BUILD_SFM_OPTION="ON"
if [[ $version == *"24.04"* ]] ; then
    BUILD_SFM_OPTION="OFF"  # it seems this module brings some build issues with Ubuntu 24.04
fi

if [[ ! -d "$TARGET_FOLDER/opencv" ]]; then
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
    else
        brew install pkg-config 
        brew install glew
        brew install cmake
        brew install suitesparse 
        brew install lapack
        brew install libtiff zlib jpeg eigen tbb glew libpng webp x264 ffmpeg
    fi 
fi

# now let's download and compile opencv and opencv_contrib
# N.B: if you want just to update cmake settings and recompile then remove "opencv/install" and "opencv/build/CMakeCache.txt"

cd $TARGET_FOLDER

if [ ! -f opencv/install/lib/libopencv_core.so ]; then
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
    if [[ "$machine" == "x86_64" || "$machine" == "x64" || $version == "darwin"* ]]; then
		# standard configuration 
        echo "building laptop/desktop config under $version"
        # as for the flags and consider this nice reference https://gist.github.com/raulqf/f42c718a658cddc16f9df07ecc627be7
        cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
          -DOPENCV_EXTRA_MODULES_PATH="`pwd`/../opencv_contrib-$OPENCV_VERSION/modules" \
          -DWITH_QT=ON \
          -DWITH_GTK=OFF \
          -DWITH_OPENGL=ON \
          -DWITH_TBB=ON \
          -DWITH_V4L=ON \
          -DWITH_CUDA=$CUDA_ON \
          -DWITH_CUBLAS=$CUDA_ON \
          -DWITH_CUFFT=$CUDA_ON \
          -DCUDA_FAST_MATH=$CUDA_ON \
          -DWITH_CUDNN=$CUDA_ON \
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
          -DOPENCV_ENABLE_NONFREE=ON \
          -DBUILD_opencv_java=OFF \
          -DBUILD_opencv_python3=ON \
          -Wno-deprecated-gpu-targets \
          -DBUILD_PROTOBUF=${WITH_PROTOBUF:-OFF} \
          -DAPPLE_FRAMEWORK=${WITH_APPLE_FRAMEWORK:-OFF} \
          -DPYTHON_INCLUDE_DIR=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])") \
          -DPYTHON_LIBRARY=$(python3 -c "import sysconfig; import os; print(os.path.join(sysconfig.get_config_var('LIBDIR'), sysconfig.get_config_var('LDLIBRARY')))") \
          ..
    else
        # Nvidia Jetson aarch64
        echo "building NVIDIA Jetson config"
        cmake \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX="`pwd`/../install" \
          -DOPENCV_EXTRA_MODULES_PATH="`pwd`/../opencv_contrib-$OPENCV_VERSION/modules" \
          -DWITH_QT=ON \
          -DWITH_GTK=OFF \
          -DWITH_OPENGL=ON \
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
fi

cd $TARGET_FOLDER
if [[ -d opencv/install ]]; then
    cd opencv
    echo "deploying built cv2 python module"
    PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")    
    PYTHON_SITE_PACKAGES=$(python -c "from distutils.sysconfig import get_python_lib; print(get_python_lib())")
    PYTHON_SOURCE_FOLDER=$(pwd)/install/lib/python$PYTHON_VERSION/site-packages/cv2
    if [[ -d "$PYTHON_SITE_PACKAGES" && -d "$PYTHON_SOURCE_FOLDER" ]]; then
        echo "copying built python cv2 module from $PYTHON_SOURCE_FOLDER to $PYTHON_SITE_PACKAGES"
        cp -r $PYTHON_SOURCE_FOLDER $PYTHON_SITE_PACKAGES
    else
        echo "ERROR: failed to copy build python cv2 module from $PYTHON_SOURCE_FOLDER to $PYTHON_SITE_PACKAGES"  
    fi 
fi

cd $STARTING_DIR

echo "...done with opencv"