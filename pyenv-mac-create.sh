#!/usr/bin/env bash

#echo "usage: ./${0##*/} <env-name>"

STARTING_DIR=`pwd`

export ENV_NAME=$1

if [[ -z "${ENV_NAME}" ]]; then
    ENV_NAME='pyslam'
fi

ENVS_PATH=~/.python/venvs  # path where to group virtual environments 
ENV_PATH=$ENVS_PATH/$ENV_NAME        # path of the virtual environment we are creating 

# clean previous install 
if [ -d ~/.python/venvs/pyslam/ ]; then 
   rm -Rf ~/.python/venvs/pyslam/
   echo ""
fi  

# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================
# from https://github.com/pyenv/pyenv/issues/1740
export PATH="$HOME/.pyenv/bin:$PATH"
export PATH="/usr/local/bin:$PATH"

export LDFLAGS+="-L/usr/local/opt/zlib/lib -L/usr/local/opt/bzip2/lib"
export CPPFLAGS+="-I/usr/local/opt/zlib/include -I/usr/local/opt/bzip2/include"

# from Install python 3.8.0 via pyenv on BigSur
# from https://dev.to/kojikanao/install-python-3-8-0-via-pyenv-on-bigsur-4oee
export LDFLAGS+="-L/usr/local/opt/readline/lib -L/usr/local/opt/openssl@1.1/lib"
export CPPFLAGS+="-I/usr/local/opt/readline/include -I/usr/local/opt/openssl@1.1/include"
export PKG_CONFIG_PATH="/usr/local/opt/readline/lib/pkgconfig:/usr/local/opt/openssl@1.1/lib/pkgconfig:/usr/local/opt/zlib/lib/pkgconfig"

# CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib" pyenv install --patch 3.6.9 < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)

# ====================================================
# create folder for virtual environment and get into it 
make_dir $ENV_PATH
cd $ENVS_PATH

export PYSLAM_PYTHON_VERSION="3.6.9"    # <=  it works and it has tensorflow 1.5
#export PYSLAM_PYTHON_VERSION="3.7.9"   # <=  it works and it lacks tensorflow 1!
#export PYSLAM_PYTHON_VERSION="3.8.2"   # <=  it works and solves matplotlib open issue but it lacks tensorflow 1 !!! https://github.com/tensorflow/tensorflow/issues/39768 

# actually create the virtual environment 
if [ ! -d $ENV_PATH/bin ]; then 
    echo creating virtual environment $ENV_NAME with python version $PYSLAM_PYTHON_VERSION
    export PATH=~/.pyenv/shims:$PATH

    #PYTHON_CONFIGURE_OPTS="--enable-framework --with-openssl" pyenv install -v $PYSLAM_PYTHON_VERSION   # this does not work!
    
    # from Install python 3.8.0 via pyenv on BigSur
    # from https://dev.to/kojikanao/install-python-3-8-0-via-pyenv-on-bigsur-4oee
    PYTHON_CONFIGURE_OPTS="--enable-framework --with-openssl" CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib" pyenv install --patch $PYSLAM_PYTHON_VERSION < <(curl -sSL https://github.com/python/cpython/commit/8ea6353.patch\?full_index\=1)
    #PYTHON_CONFIGURE_OPTS="--enable-framework --with-openssl" CFLAGS="-I$(brew --prefix openssl)/include -I$(brew --prefix bzip2)/include -I$(brew --prefix readline)/include -I$(xcrun --show-sdk-path)/usr/include" LDFLAGS="-L$(brew --prefix openssl)/lib -L$(brew --prefix readline)/lib -L$(brew --prefix zlib)/lib -L$(brew --prefix bzip2)/lib" pyenv install -v $PYSLAM_PYTHON_VERSION
    
    pyenv local $PYSLAM_PYTHON_VERSION
    python3 -m venv $ENV_NAME
fi 

# activate the environment 
cd $STARTING_DIR
export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 
. $ENV_PATH/bin/activate  

# install required packages 

pip3 install --upgrade pip

#source install_pip3_packages.sh 
# or 
pip3 install -r requirements-mac-pip3.txt

pip3 install "tensorflow>=1.14,<2.0"      
pip3 install "tensorflow-estimator>=1.14,<2.0"  
pip3 install "tensorboard>=1.14,<2.0" 
pip3 install "tensorflow-estimator>=1.14,<2.0" 
#pip3 install "tf-estimator-nightly>=1.14,<2.0" 

# HACK to fix opencv-contrib-python version!
#pip3 uninstall opencv-contrib-python                # better to clean it before installing the right version 
#pip3 install opencv-contrib-python==3.4.2.16 

# N.B.: in order to activate the virtual environment run: 
# $ source pyenv-activate.sh 
# to deactivate 
# $ deactivate 
