#!/usr/bin/env bash


# ====================================================
# import the utils 
. bash_utils.sh 

# ====================================================

#set -e


STARTING_DIR=`pwd`  # this should be the main folder directory of the repo

print_blue '================================================'
print_blue "Checking and downloading git modules ..."
set_git_modules   


# download d2net models 
print_blue '================================================'
print_blue "Checking and downloading d2net models ..."
cd thirdparty/d2net
make_dir models 
if [ ! -f models/d2_ots.pth ]; then 
    echo downloading d2net model 
    wget https://dsmn.ml/files/d2-net/d2_ots.pth -O models/d2_ots.pth
    wget https://dsmn.ml/files/d2-net/d2_tf.pth -O models/d2_tf.pth
    wget https://dsmn.ml/files/d2-net/d2_tf_no_phototourism.pth -O models/d2_tf_no_phototourism.pth    
fi 
cd $STARTING_DIR


# download contextdesc models 
print_blue '================================================'
print_blue "Checking and downloading contextdesc models ..."
cd thirdparty/contextdesc
touch __init__.py 
make_dir pretrained 
cd pretrained
if [ ! -d retrieval_model ]; then 
    wget https://research.altizure.com/data/contextdesc_models/contextdesc_pp.tar 
    tar -xvf contextdesc_pp.tar
    rm contextdesc_pp.tar    
    wget https://research.altizure.com/data/contextdesc_models/retrieval_model.tar
    tar -xvf retrieval_model.tar    
    rm retrieval_model.tar 
fi 
cd $STARTING_DIR


# download lfnet models 
print_blue '================================================'
print_blue "Checking and downloading lfnet models ..."
cd thirdparty
# copy local changes 
cp ./lfnet_changes/inference.py ./lfnet/inference.py
# download pretrained model
cd lfnet 
touch __init__.py 
make_dir pretrained 
if [ ! -d pretrained/lfnet-norotaug ]; then 
    wget https://gfx.uvic.ca/pubs/2018/ono2018lfnet/lfnet-norotaug.tar.gz -O pretrained/lfnet-norotaug.tar.gz
    tar -C pretrained/ -xf pretrained/lfnet-norotaug.tar.gz
fi 
cd $STARTING_DIR


# setting keynet  
print_blue '================================================'
print_blue "Setting keynet ..."
cd thirdparty
# copy local changes 
cp ./keynet_changes/keynet_architecture.py ./keynet/keyNet/model/keynet_architecture.py
cd $STARTING_DIR


# install delf   
./install_delf.sh 
cd $STARTING_DIR
