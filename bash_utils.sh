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
 

# ====================================================
# CUDA
# ====================================================

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

# Function to detect CUDA toolkit version (not driver version)
function detect_cuda_toolkit_version() {
    local cuda_version=""
    
    # Method 1: Check nvcc (most reliable for toolkit version)
    if command -v nvcc &> /dev/null; then
        NVCC_OUTPUT=$(nvcc --version 2>&1)
        # Extract version from "release X.Y" pattern
        cuda_version=$(echo "$NVCC_OUTPUT" | grep -oP 'release \K[\d.]+' | head -1 || echo "")
        if [ -z "$cuda_version" ]; then
            cuda_version=$(echo "$NVCC_OUTPUT" | grep "release" | sed -n 's/.*release \([0-9]\+\.[0-9]\+\).*/\1/p' | head -1 || echo "")
        fi
    fi
    
    # Method 2: Check common CUDA installation paths
    if [ -z "$cuda_version" ]; then
        for cuda_path in /usr/local/cuda /usr/local/cuda-* /opt/cuda /opt/cuda-*; do
            if [ -d "$cuda_path" ] && [ -f "$cuda_path/version.txt" ]; then
                cuda_version=$(grep -oP '\d+\.\d+' "$cuda_path/version.txt" 2>/dev/null | head -1 || echo "")
                [ ! -z "$cuda_version" ] && break
            fi
        done
    fi
    
    # Method 3: Check CUDA_HOME environment variable
    if [ -z "$cuda_version" ] && [ ! -z "$CUDA_HOME" ] && [ -f "$CUDA_HOME/version.txt" ]; then
        cuda_version=$(grep -oP '\d+\.\d+' "$CUDA_HOME/version.txt" 2>/dev/null | head -1 || echo "")
    fi
    
    echo "$cuda_version"
}

# Return a list of supported SM codes from nvcc as "50 75 86" etc.
function get_supported_nvcc_archs() {
    if ! command -v nvcc &> /dev/null; then
        return 1
    fi
    nvcc --help 2>/dev/null | \
        grep -o "sm_[0-9][0-9]" | \
        sed 's/sm_//' | \
        sort -u
}


function detect_cuda_arch_bin() {
    local gpu_cap_raw="" gpu_cc=""
    local supported=()

    # 1) Try to read GPU compute capability via nvidia-smi
    if command -v nvidia-smi &> /dev/null; then
        gpu_cap_raw=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -n1)
        # gpu_cap_raw like "8.6" or "7.5"
        if [[ -n "$gpu_cap_raw" ]]; then
            gpu_cc="${gpu_cap_raw/./}"   # "8.6" -> "86"
        fi
    fi

    # 2) Try to get the list of archs supported by this CUDA toolkit (via nvcc)
    if get_supported_nvcc_archs > /dev/null; then
        mapfile -t supported < <(get_supported_nvcc_archs)
    fi

    # 2a) If we know GPU CC and also have nvcc's supported list, cross-check
    if [[ -n "$gpu_cc" && ${#supported[@]} -gt 0 ]]; then
        for a in "${supported[@]}"; do
            if [[ "$a" == "$gpu_cc" ]]; then
                # GPU is supported by this toolkit; return its exact capability (e.g. "8.6")
                echo "$gpu_cap_raw"
                return 0
            fi
        done
        # GPU is NOT in the supported list => typical case: old GPU + very new CUDA
        # Signal "no valid arch" so the caller can disable CUDA.
        echo ""
        return 1
    fi

    # 2b) If we know GPU CC but nvcc gave us nothing, just trust nvidia-smi
    if [[ -n "$gpu_cap_raw" ]]; then
        echo "$gpu_cap_raw"   # e.g. "8.6" on your RTX A3000
        return 0
    fi

    # 2c) No GPU info, but we have nvcc-supported archs: build a broad list from those
    if [[ ${#supported[@]} -gt 0 ]]; then
        local out=()
        for a in "${supported[@]}"; do
            case "$a" in
                50|52|53|60|61|70|72|75|80|86|89|90)
                    out+=("${a:0:1}.${a:1:1}")
                    ;;
            esac
        done
        if [[ ${#out[@]} -gt 0 ]]; then
            printf "%s " "${out[@]}"
            echo
            return 0
        fi
    fi

    # 3) Last-resort default if neither nvidia-smi nor nvcc gave anything useful
    # Architecture |  Example GPUs
    #  5.0         |  GTX 950M, GTX 960
    #  6.0         |  Tesla P100
    #  6.1         |  GTX 1080, 1070, 1060
    #  7.0         |  Tesla V100
    #  7.5         |  RTX 2080 / 2070 / 2060
    #  8.6         |  RTX 3080 / 3070 / 3060    
    echo "5.0 6.0 6.1 7.0 7.5 8.6"
    return 0
} 

# ====================================================

function get_torch_cuda_version() {
    TORCH_CUDA_VERSION=$(python3 -c "import torch; print(torch.version.cuda)")
    echo $TORCH_CUDA_VERSION
}

# Function to verify PyTorch has CUDA support
function verify_pytorch_cuda() {
    python -c "import torch; from torch.utils import cpp_extension; assert torch.cuda.is_available() or torch.version.cuda is not None, 'PyTorch CUDA not available'" 2>/dev/null
}


# Helper: get currently installed torch CUDA version if torch is already present; else empty
function get_installed_torch_cuda_ver() {
  python3 - <<'PY'
import importlib.util
spec = importlib.util.find_spec("torch")
if spec is None:
    print("")
else:
    import torch
    print(getattr(torch.version, "cuda", "") or "")
PY
}

# ====================================================


function brew_install(){
    if brew ls --versions $1 > /dev/null; then
        # The package is installed
        echo $1 is already installed!
    else
    # The package is not installed
        brew install $1
    fi
}

# ====================================================

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


# ====================================================
