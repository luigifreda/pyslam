#!/usr/bin/env bash

# a collection of bash utils 

# ====================================================
function get_after_last_slash(){
    ret=$(echo $1 | sed 's:.*/::')
    echo $ret 
}
function get_virtualenv_name(){
    cmd_out=$(printenv | grep VIRTUAL_ENV)
    virtual_env_name=$(get_after_last_slash $cmd_out)
    echo $virtual_env_name
}

function print_blue(){
	printf "\033[34;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function make_dir(){
if [ ! -d $1 ]; then
    mkdir -p $1
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
    PKG_OK=$(pip3 list --format=legacy |grep $package_name)
    #echo "checking for $package_name: $PKG_OK"
    if [ "" == "$PKG_OK" ]; then
      #echo "$package_name is not installed"
      echo 1
    else
      echo 0
    fi
}
function check_pip_package2(){
    package_name=$1    
    if python3 -c "import "$package_name &> /dev/null; then
      echo 0
    else
      #echo "$package_name is not installed"
      echo 1
    fi
}
function install_pip_package(){
    do_install=$(check_pip_package $1)
    virtual_env=$(get_virtualenv_name)
    if [ $do_install -eq 1 ] ; then
        if [ "" == "$virtual_env" ]; then
            pip3 install --user $1          
        else
            pip3 install $1     # if you are in a virtual environment the option `--user` will install make pip3 install things outside the env 
        fi
    fi 
}
function install_pip_packages(){
    for var in "$@"
    do
        install_pip_package "$var"
    done
}

function set_git_modules() {
	#print_blue "setting up git submodules"
	git submodule init -- 
	git submodule sync --recursive 
	git submodule update --recursive
}

function update_git_modules() {
    git submodule update --recursive --remote
}

function pause(){
 read -s -n 1 -p "Press any key to continue . . ."
 echo ""
}
 
# ====================================================

