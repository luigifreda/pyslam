#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

#set -e

# clean the old .env file if it exists
if [ -f "$ROOT_DIR/.env" ]; then
  rm "$ROOT_DIR/.env"
fi

# NOTE: in order to detect macOS use:  
# if [[ "$OSTYPE" == "darwin"* ]]; then 
# fi 

# 1. install required system packages 
brew update 
brew install wget 
brew install doxygen 
brew install eigen 
#brew install opencv 
brew install glew 
brew install pkg-config 
brew install suite-sparse 
brew install pyenv
brew install zlib bzip2
brew install rsync
brew install readline
brew install pyenv
brew install libomp   # need to add -L/opt/homebrew/opt/libomp/lib -I/opt/homebrew/opt/libomp/include
brew install boost    # for serialization
brew install tmux
brew install flann
brew install catch2
#brew install numpy
#brew install open3d     # built from source for different issues 
brew install x265 libjpeg libde265 libheif   # for pillow-heif

# 2. create a pyslam environment within pyenv and activate it 
./pyenv-venv-create-mac.sh  # NOTE: the use of ./ seems crucial for the correct identification of the python libs for C++ projects 

# 3. activate the created python virtual environment 
. pyenv-activate.sh   

# 4. set up git submodules (we need to install gdown before this) 
./install_git_modules.sh   

export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 

 # 5. install pip packages: some unresolved dep conflicts found in requirement-pip3.txt may be managed by the following command: 
. install_pip3_packages.sh 

# 6. build and install cpp stuff 
. install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# 7. build and install thirdparty 
. install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 
