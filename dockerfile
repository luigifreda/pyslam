# Usage: docker run -it -e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix --privileged --gpus all pyslam python3 main_vo.py

FROM ubuntu:20.04
ENV DEBIAN_FRONTEND=noninteractive

RUN apt update -y
RUN apt-get -y install git \
                       python3 \
                       python3-sdl2 \
                       python3-tk \
                       rsync \
                       libprotobuf-dev \
                       libeigen3-dev \
                       libopencv-dev \
                       python3-pip \
                       build-essential \
                       cmake \
                       libglew-dev \
                       libsuitesparse-dev \
                       libeigen3-dev

RUN git clone --recursive https://github.com/luigifreda/pyslam.git
WORKDIR /pyslam
RUN git checkout ubuntu20
RUN python3 -m pip install tensorflow-gpu

RUN ./install_basic.sh
RUN ./install_thirdparty.sh

WORKDIR /pyslam/thirdparty/g2opy 
RUN python3 setup.py install

FROM nvidia/cuda:11.0-base
COPY --from=0 . /
WORKDIR /pyslam