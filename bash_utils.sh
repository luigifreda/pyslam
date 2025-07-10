#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

# a collection of bash utils 


BASH_UTILS_SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of pySLAM)
BASH_UTILS_SCRIPT_DIR=$(readlink -f $BASH_UTILS_SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$BASH_UTILS_SCRIPT_DIR"

# ====================================================
function get_after_last_slash(){
    ret=$(echo $1 | sed 's:.*/::')
    echo $ret 
}
# function get_virtualenv_name(){
#     cmd_out=$(printenv | grep VIRTUAL_ENV)
#     virtual_env_name=$(get_after_last_slash $cmd_out)
#     echo $virtual_env_name
# }
function get_virtualenv_name(){
    if [ -n "$VIRTUAL_ENV" ]; then
        virtual_env_name=$(basename "$VIRTUAL_ENV")
    elif [ -n "$CONDA_DEFAULT_ENV" ]; then
        virtual_env_name="$CONDA_DEFAULT_ENV"
    else
        virtual_env_name=""
    fi
    echo $virtual_env_name
}

function print_green(){
	printf "\033[32;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function print_blue(){
	printf "\033[34;1m"
	printf "$@ \n"
	printf "\033[0m"
}

function print_yellow(){
    printf "\033[33;1m"
    printf "$@ \n"
    printf "\033[0m"
}

function print_red(){
    printf "\033[31;1m"
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
    local package_name="$1"
    local PKG_OK=$(dpkg-query -W --showformat='${Status}\n' "$package_name" 2>/dev/null | grep "install ok installed")

    if [ -z "$PKG_OK" ]; then
        echo 1  # Package is not installed
    else
        echo 0  # Package is installed
    fi
}
function install_package(){
    do_install=$(check_package $1)
    if [ $do_install -eq 1 ] ; then
        sudo apt-get install -y $1
    fi 
}

function install_packages(){
    for var in "$@"; do
        install_package "$var"
    done
}

function check_pip_package(){
    package_name=$1
    PKG_OK=$(python3 -m pip list |grep $package_name)
    #echo "checking for $package_name: $PKG_OK"
    if [ -z "$PKG_OK" ]; then
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
        if [ -z "$virtual_env" ]; then
            pip3 install --user $1          
        else
            pip3 install $1     # if you are in a virtual environment the option `--user` will install make pip3 install things outside the env 
        fi
    fi 
}
function install_pip_packages(){
    for var in "$@"; do
        install_pip_package "$var"
    done
}

function set_git_modules() {
	#print_blue "setting up git submodules"
	git submodule init -- 
	git submodule sync --recursive 
	git submodule update --init --recursive
}

function update_git_modules() {
    git submodule update --recursive --remote
}

function pause(){
    read -s -n 1 -p "Press any key to continue . . ."
    echo ""
}

function check_conda(){
    # check that conda is activated 
    if ! command -v conda &> /dev/null; then
        echo "ERROR: conda could not be found! Did you install/activate conda?"
        echo 1 
    else
        echo 0
    fi 
}

function gdrive_download () {
  if gdown -V >/dev/null 2>&1; then
    echo "" #"gdown is found in PATH"
  else
    if [[ -f $HOME/.local/bin/gdown ]]; then
      export PATH=$HOME/.local/bin:$PATH
    fi 
  fi  
  gdown https://drive.google.com/uc?id=$1
}
 
function extract_version(){
    #version=$(echo $1 | sed 's/[^0-9]*//g')
    #version=$(echo $1 | sed 's/[[:alpha:]|(|[:space:]]//g')
    version=$(echo $1 | sed 's/[[:alpha:]|(|[:space:]]//g' | sed s/://g)
    echo $version
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

function get_cuda_version(){
    if [ -d /usr/local/cuda ]; then 
        if [ -f /usr/local/cuda/version.txt ]; then 
            CUDA_STRING=$(cat /usr/local/cuda/version.txt)
            CUDA_VERSION=$(extract_version "$CUDA_STRING")
            echo $CUDA_VERSION
        else
            # Extract the CUDA version from the nvidia-smi output
            CUDA_VERSION=$(/usr/local/cuda/bin/nvcc --version | grep release | awk '{print $5}' | sed 's/,//')
            echo $CUDA_VERSION
        fi 
    else
        echo 0
    fi
}

function get_torch_cuda_version() {
    TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    echo $TORCH_CUDA_VERSION
}

function brew_install(){
    if brew ls --versions $1 > /dev/null; then
        # The package is installed
        echo $1 is already installed!
    else
    # The package is not installed
        brew install $1
    fi
}

# Function to load .env file
function load_env_file() {
    # Default to the .env file in the ROOT_DIR if not provided
    local file="${1:-$ROOT_DIR/.env}"

    # Check if the file exists
    if [ -f "$file" ]; then
        echo "Loading environment variables from $file..."

        # Export each line as an environment variable
        while IFS= read -r line || [ -n "$line" ]; do
            # Skip comments and empty lines
            if [[ "$line" =~ ^# ]] || [[ -z "$line" ]]; then
                continue
            fi

            # Export the variable
            echo "Exporting: $line"
            export "$line"
        done < "$file"

        echo "Environment variables loaded."
    else
        echo "File $file not found!"
    fi
}

# Function to set or update an environment variable in a file
function set_env_var() {
    local file="${1:-$ROOT_DIR/.env}"
    local key="$2"
    local value="$3"

    # Check if the file exists; create it if not
    if [ ! -f "$file" ]; then
        touch "$file"
    fi

    # Escape special characters in the key and value for safety
    local escaped_key=$(printf '%s' "$key" | sed 's/[].*^$[]/\\&/g')
    local escaped_value=$(printf '%s' "$value" | sed 's/[&/\]/\\&/g')

    # Check if the key exists in the file
    if grep -qE "^${escaped_key}=" "$file"; then
        # Update the existing key-value pair
        sed -i.bak -E "s|^(${escaped_key})=.*$|\1=${escaped_value}|" "$file"
        echo "Updated: $key=$value in $file"
    else
        # Add the key-value pair if it doesn't exist
        echo "${key}=${value}" >> "$file"
        echo "Added: $key=$value to $file"
    fi
}

function replace_dot_with_hyphen() {
    local input_string="$1"
    new_string=$(echo "$input_string" | tr '.' '-')
    echo "$new_string"
}

function remove_dots() {
    local input_string="$1"
    new_string="${input_string//./}"
    echo "$new_string"
}

# ===================== SUDO KEEP-ALIVE ========================

keep_sudo_alive() {
    sudo -v
    ( while true; do sudo -n true; sleep 60; done ) &
    SUDO_KEEP_ALIVE_PID=$!
}

stop_sudo_alive() {
    if [[ -n "$SUDO_KEEP_ALIVE_PID" ]]; then
        kill "$SUDO_KEEP_ALIVE_PID" 2>/dev/null
    fi
}

# ====================================================

if [[ "$OSTYPE" == "darwin"* ]]; then
    alias nproc="sysctl -n hw.logicalcpu"
fi 