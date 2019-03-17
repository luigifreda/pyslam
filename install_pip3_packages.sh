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

set -e

echo "=================================================================="
echo "Configuring and installing python packages ..."

# N.B.: it's required the use of python3 

install_pip_packages pygame numpy matplotlib pyopengl Pillow pybind11 scikit-image pyyaml termcolor
install_pip_packages opencv-python opencv-contrib-python 
install_package python3-sdl2 python3-tk