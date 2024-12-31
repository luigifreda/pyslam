mkdir -p datasets/euroc
cd datasets/euroc
wget http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/machine_hall/MH_02_easy/MH_02_easy.zip
unzip MH_02_easy.zip -d mh02
rm -rf MH_02_easy.zip