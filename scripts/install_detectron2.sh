#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

PYTHON_EXE=$(get_python_exe)
DETECTRON2_DIR="$ROOT_DIR/thirdparty/detectron2"

ensure_pip "$PYTHON_EXE" || exit 1
"$PYTHON_EXE" -m pip install wheel "setuptools<70"

if ! "$PYTHON_EXE" -c "import torch" 2>/dev/null; then
    print_red "ERROR: torch is not installed in ${PYTHON_EXE}."
    print_yellow "Install torch first, then rerun this script:"
    print_yellow "  $SCRIPT_DIR_/install_pip3_torch.sh"
    exit 1
fi

# detectron2/__init__.py runs setup_environment() before any submodule (including _C) loads;
# that path imports fvcore. This script only runs `build_ext --inplace`, so install the same
# import-time deps detectron2 declares in setup.py, or `import detectron2._C` fails even when
# the extension built correctly.
print_blue "Installing detectron2 Python dependencies (fvcore / iopath / PyYAML)..."
"$PYTHON_EXE" -m pip install -q "fvcore>=0.1.5,<0.1.6" "iopath>=0.1.7,<0.1.10" "PyYAML>=5.1" || exit 1

cd "$ROOT_DIR/thirdparty"

if [ ! -d "detectron2" ]; then
    print_blue "Cloning detectron2..."
    git clone https://github.com/facebookresearch/detectron2.git
    cd detectron2
    git checkout v0.6
    git apply ../detectron2.patch
else
    cd detectron2
    if git apply --check ../detectron2.patch &>/dev/null; then
        print_blue "Applying detectron2 patch..."
        git apply ../detectron2.patch
    fi
fi

# On recent macOS/Xcode, torch<=2.4 headers fail to compile detectron2 extensions unless
# -Wno-invalid-specialization is set (libc++ disallows std::is_arithmetic specializations).
if [[ "$OSTYPE" == darwin* ]]; then
    export CFLAGS="${CFLAGS:-} -Wno-invalid-specialization"
    export CXXFLAGS="${CXXFLAGS:-} -Wno-invalid-specialization"
fi

print_blue "Building detectron2 C++ extension for ${PYTHON_EXE}..."
"$PYTHON_EXE" setup.py build_ext --inplace

SITE_PACKAGES=$("$PYTHON_EXE" -c "import site; print(site.getsitepackages()[0])")
PTH_FILE="$SITE_PACKAGES/pyslam_detectron2.pth"
echo "$DETECTRON2_DIR" > "$PTH_FILE"
print_blue "Registered detectron2 path in $PTH_FILE"

if ! "$PYTHON_EXE" -c "import detectron2._C"; then
    print_red "ERROR: detectron2 C++ extension (detectron2._C) failed to load after build."
    print_yellow "If the traceback mentions a missing Python module, install detectron2 deps from"
    print_yellow "  thirdparty/detectron2 (e.g. pip install -e .) or rerun this script."
    print_yellow "Rebuild after any torch/torchvision change:"
    print_yellow "  $SCRIPT_DIR_/install_detectron2.sh"
    exit 1
fi

print_green "detectron2 installed successfully."
