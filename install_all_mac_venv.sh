#!/usr/bin/env bash

#N.B: this install script allows you to run main_slam.py and all the scripts 

# ====================================================
# import the utils 
. bash_utils.sh 

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

#set -e

# set up git submodules  
#. install_basic.sh 0 0 # the first '0' is an option for skipping pip3 packages installation (script install_pip3_packages.sh),
                       # the second '0' is for skipping the install_cpp.sh script therein (that will be called below) 

# set up git submodules  
./install_git_modules.sh   

# NOTE: in order to detect macOS use:  
# if [[ "$OSTYPE" == "darwin"* ]]; then 
# fi 

# install required packages 
brew_install wget 
brew_install doxygen 
brew_install eigen 
brew_install opencv 
brew_install glew 
brew_install pkg-config 
brew_install suite-sparse 
brew_install pyenv
brew_install zlib bzip2
brew_install rsync

# create a pyslam environment within pyenv and activate it 
./pyenv-mac-create.sh  # NOTE: the use of ./ seems crucial for the correct identification of the python libs for C++ projects 

# let's repeat it here! this seems to be crucial 
. pyenv-activate.sh   

#export WITH_PYTHON_INTERP_CHECK=ON  # in order to detect the correct python interpreter 
#export USE_PYSLAM_ENV=1

# build and install cpp stuff 
. install_cpp.sh                    # use . in order to inherit python env configuration and other environment vars 

# build and install thirdparty 
. install_thirdparty.sh             # use . in order to inherit python env configuration and other environment vars 

# N.B.:
# if your run into troubles with opencv xfeatures2D/SIFT/SURF then run the following commands into your pyslam environment 
# $ pip3 uninstall opencv-contrib-python
# $ pip3 install opencv-contrib-python==3.4.2.16

# HACK (this is actually run inside the created pyenv environment ) 
#pip uninstall opencv-contrib-python
#pip install opencv-contrib-python==3.4.2.16
