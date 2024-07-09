# Base Image
#FROM nvidia/cuda:12.1.1-devel-ubuntu22.04
FROM dustynv/l4t-pytorch:r36.2.0

# Arguments
ARG user
ARG uid
ARG home
ARG workspace
ARG shell
ARG container_name
ARG timezone=Etc/UTC

# Environment Variables for noninteractive installation and timezone
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ=${timezone}

# Update and Install Essential Packages
RUN apt-get update && apt-get -y install --no-install-recommends \
    tzdata  apt-utils keyboard-configuration git curl wget ca-certificates screen tree sudo ssh synaptic aptitude gedit geany mesa-utils cmake build-essential python3-pip \
    libqt5widgets5 libqt5gui5 libqt5core5a libxcb-xinerama0 libfontconfig1 libxrender1 libx11-6 libxcb1 libxext6 libx11-xcb1 libxi6 libxrandr2 libgl1-mesa-glx libdbus-1-3 x11-apps librsvg2-common

# Symbolic link for Python
RUN ln -s /usr/bin/python3 /usr/bin/python

# Install Python libraries
RUN pip install --upgrade pip && \
    pip install matplotlib numpy kornia tqdm pyyaml termcolor scipy

#RUN pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 --index-url https://download.pytorch.org/whl/cu121

# NVIDIA Container Runtime Configuration
ENV NVIDIA_VISIBLE_DEVICES ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics

# Expose SSH Port
EXPOSE 22

# Disable shared memory X11 affecting the container
ENV QT_X11_NO_MITSHM=1
ENV QT_QPA_PLATFORM_PLUGIN_PATH=/usr/lib/x86_64-linux-gnu/qt5/plugins/platforms

# Set up working directories and clone relevant repositories
WORKDIR ${home}
RUN git clone --branch ubuntu22_cuda --recursive https://github.com/AnielAlexa/pyslam.git
WORKDIR ${home}/pyslam
RUN git submodule update --init --recursive



