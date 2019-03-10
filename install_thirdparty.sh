#!/usr/bin/env bash


# ====================================================

function print_blue(){
	printf "\033[34;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function make_dir(){
if [ ! -d $1 ]; then
    mkdir $1
fi
}
function make_buid_dir(){
	make_dir build
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

function check_pip_package(){
    package_name=$1
    PKG_OK=$(pip3 list |grep $package_name)
    #echo "checking for $package_name: $PKG_OK"
    if [ "" == "$PKG_OK" ]; then
      #echo "$package_name is not installed"
      echo 1
    else
      echo 0
    fi
}
function install_pip_package(){
    do_install=$(check_pip_package $1)
    if [ $do_install -eq 1 ] ; then
        pip3 install --user $1
    fi 
}
function install_pip_packages(){
    for var in "$@"
    do
        install_pip_package "$var"
    done
}

# ====================================================

print_blue '================================================'
print_blue "Building Thirdparty"
print_blue '================================================'

set -e

STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

echo "=================================================================="
echo "Configuring and installing python packages ..."

# N.B.: it's required the use of python3 

install_pip_packages pygame numpy matplotlib pyopengl Pillow pybind11 scikit-image pyyaml termcolor
install_pip_packages opencv-python opencv-contrib-python 
install_package python3-sdl2 

make_dir lib

echo "=================================================================="
echo "Configuring and building lib/Pangolin ..."

ISTALL_PANGOLIN_ORIGINAL=0
cd lib
if [ $ISTALL_PANGOLIN_ORIGINAL -eq 1 ] ; then
    # N.B.: pay attention this will generate a module pypangolin ( it does not have the methods dcam.SetBounds(...) and pangolin.DrawPoints(points, colors)  )
    if [ ! -d pangolin ]; then
        sudo apt-get install -y libglew-dev
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
        make -j 8
        cd build/src
        ln -s pypangolin.*-linux-gnu.so  pangolin.linux-gnu.so
    fi
else
    # N.B.: pay attention this will generate a module pangolin 
    if [ ! -d pangolin ]; then
        sudo apt-get install -y libglew-dev
        git clone https://github.com/uoip/pangolin.git
    fi
    cd pangolin
    if [ ! -f pangolin.cpython-*-linux-gnu.so ]; then   
        make_dir build   
        cd build
        cmake .. -DBUILD_PANGOLIN_LIBREALSENSE=OFF # disable realsense 
        make -j8
        cd ..
        #python setup.py install
    fi
fi
cd $STARTING_DIR


echo "=================================================================="
echo "Configuring and building lib/g2o ..."

cd lib
if [ ! -d g2opy ]; then
    sudo apt-get install -y libsuitesparse-dev libeigen3-dev
	git clone https://github.com/uoip/g2opy.git
    cd g2opy
    G2OPY_REVISION=5587024
    git checkout $G2OPY_REVISION
    cd ..
    # copy local changes 
    cp ./g2opy_changes/types_six_dof_expmap.h ./g2opy/python/types/sba/types_six_dof_expmap.h
fi
cd g2opy
if [ ! -f lib/g2o.cpython-*-linux-gnu.so ]; then  
    make_buid_dir
    cd build
    cmake ..
    make -j8
    cd ..
    #python3 setup.py install --user
fi    
cd $STARTING_DIR
