#!/usr/bin/env bash

#set -e

# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

print_blue '================================================'
print_blue "Building Thirdparty"
print_blue '================================================'

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used
cd $SCRIPT_DIR
STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

# ====================================================
# check if want to use conda or venv
if [ -z $USING_CONDA_PYSLAM ]; then
    if [[ -z "${USE_PYSLAM_ENV}" ]]; then
        USE_PYSLAM_ENV=0
    fi
    if [ $USE_PYSLAM_ENV -eq 1 ]; then
        . pyenv-activate.sh
    fi  
else 
    echo "Using conda pyslam..."
    . pyenv-conda-activate.sh
fi 

# ====================================================
# check if we have external options
EXTERNAL_OPTIONS=$@
if [[ -n "$EXTERNAL_OPTIONS" ]]; then
    echo "external option: $EXTERNAL_OPTIONS" 
fi

# check if we want to add a python interpreter check
if [[ -n "$WITH_PYTHON_INTERP_CHECK" ]]; then
    echo "WITH_PYTHON_INTERP_CHECK: $WITH_PYTHON_INTERP_CHECK " 
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DWITH_PYTHON_INTERP_CHECK=$WITH_PYTHON_INTERP_CHECK"
fi

OpenCV_DIR="$SCRIPT_DIR/thirdparty/opencv/install/lib/cmake/opencv4"
if [[ -d "$OpenCV_DIR" ]]; then
    EXTERNAL_OPTIONS="$EXTERNAL_OPTIONS -DOpenCV_DIR=$OpenCV_DIR"
fi 

echo "EXTERNAL_OPTIONS: $EXTERNAL_OPTIONS"

# ====================================================

CURRENT_USED_PYENV=$(get_virtualenv_name)
print_blue "currently used pyenv: $CURRENT_USED_PYENV"

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/orbslam2_features ..."
cd thirdparty/orbslam2_features
. build.sh $EXTERNAL_OPTIONS
cd $STARTING_DIR

print_blue '================================================'
print_blue "Configuring and building thirdparty/Pangolin ..."

make_dir thirdparty

INSTALL_PANGOLIN_ORIGINAL=0
cd thirdparty
if [ $INSTALL_PANGOLIN_ORIGINAL -eq 1 ] ; then
    # N.B.: pay attention this will generate a module 'pypangolin' ( it does not have the methods dcam.SetBounds(...) and pangolin.DrawPoints(points, colors)  )
    if [ ! -d pangolin ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then
            sudo apt-get install -y libglew-dev
        fi     
        git clone https://github.com/stevenlovegrove/Pangolin.git pangolin
        cd pangolin
        git submodule init && git submodule update
        cd ..
    fi
    cd pangolin
    make_dir build 
    if [ ! -f build/src/libpangolin.so ]; then
        cd build
        cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON $EXTERNAL_OPTIONS
        make -j8
        cd build/src
        ln -s pypangolin.*-linux-gnu.so  pangolin.linux-gnu.so
    fi
else
    # N.B.: pay attention this will generate a module 'pangolin' 
    if [ ! -d pangolin ]; then
        if [[ "$OSTYPE" == "linux-gnu"* ]]; then    
            sudo apt-get install -y libglew-dev
            # git clone https://github.com/uoip/pangolin.git
            # cd pangolin
            # PANGOLIN_UOIP_REVISION=3ac794a
            # git checkout $PANGOLIN_UOIP_REVISION
            # cd ..      
            # # copy local changes 
            # rsync ./pangolin_changes/python_CMakeLists.txt ./pangolin/python/CMakeLists.txt 
            git clone --recursive https://gitlab.com/luigifreda/pypangolin.git pangolin
        fi 
        if [[ "$OSTYPE" == "darwin"* ]]; then
            git clone --recursive https://gitlab.com/luigifreda/pypangolin.git pangolin 
        fi 
        cd pangolin
        git apply ../pangolin.patch
        cd ..
    fi
    cd pangolin
    if [ ! -f pangolin.cpython-*.so ]; then   
        make_dir build   
        cd build
        cmake .. -DBUILD_PANGOLIN_LIBREALSENSE=OFF -DBUILD_PANGOLIN_FFMPEG=OFF $EXTERNAL_OPTIONS # disable realsense 
        make -j8
        cd ..
        #python setup.py install
    fi
fi
cd $STARTING_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/g2o ..."

cd thirdparty
if [ ! -d g2opy ]; then
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        sudo apt-get install -y libsuitesparse-dev libeigen3-dev
    fi     
	git clone https://github.com/uoip/g2opy.git
    cd g2opy
    G2OPY_REVISION=5587024
    git checkout $G2OPY_REVISION
    git apply ../g2opy.patch
    cd ..     
fi
cd g2opy
if [ ! -f lib/g2o.cpython-*.so ]; then  
    make_buid_dir
    cd build
    cmake .. $EXTERNAL_OPTIONS
    make -j8
    cd ..
    #python3 setup.py install --user
fi    
cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pydbow3 ..."

cd thirdparty/pydbow3
./build.sh $EXTERNAL_OPTIONS

cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pydbow2 ..."

cd thirdparty/pydbow2
./build.sh $EXTERNAL_OPTIONS

cd $STARTING_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pyibow ..."

cd thirdparty/pyibow
./build.sh $EXTERNAL_OPTIONS

cd $STARTING_DIR


if [[ "$OSTYPE" == "darwin"* ]]; then
    print_blue "=================================================================="
    print_blue "Configuring and building thirdparty/open3d ..."

    # NOTE: Under mac I got segmentation faults when trying to use open3d python bindings
    #       This happends when trying to load the open3d dynamic library.
    ./install_open3d_python.sh

    cd $STARTING_DIR
fi 



# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON