#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#echo "usage: ./${0##*/} <env-name>"

export ENV_NAME=$1

if [ -z "${ENV_NAME}" ]; then
    ENV_NAME='pyslam'
fi

# # Check if conda is already initialized
# if ! conda info -v &> /dev/null; then
#     conda init bash 
#     source ~/.bashrc
# fi


# This variable is used to indicate that we want to use conda
export USING_CONDA_PYSLAM=1

export PYTHONPATH=""   # clean python path => for me, remove ROS stuff 

if [ -z "$CONDA_PREFIX" ]; then
    CONDA_PREFIX=$(conda info --base)
fi
. "$CONDA_PREFIX"/bin/activate base   # from https://community.anaconda.cloud/t/unable-to-activate-environment-prompted-to-run-conda-init-before-conda-activate-but-it-doesnt-work/68677/10
conda activate $ENV_NAME

# N.B.: in order to deactivate the virtual environment run: 
# $ conda deactivate


PYTHON_ENV=$(python3 -c "import sys; print(sys.prefix)")
PYTHON_VERSION=$(python -c "import sys; print(f\"{sys.version_info.major}.{sys.version_info.minor}\")")

# HACK for "ImportError: /lib/x86_64-linux-gnu/libgobject-2.0.so.0: undefined symbol: ffi_type_uint32"
# See https://github.com/elerac/EasyPySpin/issues/12
CV2_SO_PATH=$PYTHON_ENV/lib/python$PYTHON_VERSION/site-packages/cv2
# Find the actual .so file (handles different naming variations)
CV2_SO_FILE=$(ls "$CV2_SO_PATH"/cv2.*.so 2>/dev/null | head -n 1)
# Check if a valid file was found
if [[ -n "$CV2_SO_FILE" && -f "$CV2_SO_FILE" ]]; then
    if ldd "$CV2_SO_FILE" | grep -q "libffi"; then
        if [[ -f /lib/x86_64-linux-gnu/libffi.so ]]; then
            echo "Preloading libffi..."
            export LD_PRELOAD=/lib/x86_64-linux-gnu/libffi.so
        else
            echo "WARNING: /lib/x86_64-linux-gnu/libffi.so not found; check your libffi dependencies"
        fi
    fi
fi