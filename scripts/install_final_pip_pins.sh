#!/usr/bin/env bash
# Author: Luigi Freda
# This file is part of https://github.com/luigifreda/pyslam
#
# Final pip pins to be applied at the very end of the install process (venv/conda/pixi).
# Many packages installed along the way (e.g. wandb, nerfstudio/f3rm, detectron2, ...) silently
# change some shared dependencies. Here we restore the versions required by pyslam.

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used
ROOT_DIR="$SCRIPT_DIR_/.."

. "$ROOT_DIR"/bash_utils.sh

print_blue "Running install_final_pip_pins.sh"

# some legacy packages still import pkg_resources, which was removed in setuptools>=81
pip install --no-warn-conflicts "setuptools<81"

# torch and other recent packages need a recent typing_extensions (e.g. typing_extensions.TypeIs)
pip install --no-warn-conflicts -U "typing_extensions>=4.12"

# pyslam runs on numpy 2 (compiled modules are built against it). Some packages still declare numpy<2 in their
# metadata (e.g. depth-anything-3, nuscenes-devkit via nerfstudio) and may have downgraded it.
# Upper bound: numba requires numpy<2.6.
pip install --no-warn-conflicts "numpy>=2,<2.6"
