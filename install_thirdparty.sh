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

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

# ====================================================
# N.B.: this script requires that you have first run:
#./install_basic.sh 

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
EXTERNAL_OPTION=$1
if [[ -n "$EXTERNAL_OPTION" ]]; then
    echo "external option: $EXTERNAL_OPTION" 
fi

# check if we want to add a python interpreter check
if [[ -n "$WITH_PYTHON_INTERP_CHECK" ]]; then
    echo "WITH_PYTHON_INTERP_CHECK: $WITH_PYTHON_INTERP_CHECK " 
    EXTERNAL_OPTION="$EXTERNAL_OPTION -DWITH_PYTHON_INTERP_CHECK=$WITH_PYTHON_INTERP_CHECK"
fi
# ====================================================

CURRENT_USED_PYENV=$(get_virtualenv_name)
print_blue "currently used pyenv: $CURRENT_USED_PYENV"

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/orbslam2_features ..."
cd thirdparty/orbslam2_features
. build.sh $EXTERNAL_OPTION
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
        cmake ../ -DAVFORMAT_INCLUDE_DIR="" -DCPP11_NO_BOOST=ON $EXTERNAL_OPTION
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
        cmake .. -DBUILD_PANGOLIN_LIBREALSENSE=OFF -DBUILD_PANGOLIN_FFMPEG=OFF $EXTERNAL_OPTION # disable realsense 
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
    cmake .. $EXTERNAL_OPTION
    make -j8
    cd ..
    #python3 setup.py install --user
fi    
cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pydbow3 ..."

cd thirdparty/pydbow3
./build.sh

cd $STARTING_DIR

print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pydbow2 ..."

cd thirdparty/pydbow2
./build.sh

cd $STARTING_DIR


print_blue "=================================================================="
print_blue "Configuring and building thirdparty/pyibow ..."

cd thirdparty/pyibow
./build.sh

cd $STARTING_DIR

# NOTE: If you get build errors related to python interpreter check under Linux then run the following command:
# export WITH_PYTHON_INTERP_CHECK=ON