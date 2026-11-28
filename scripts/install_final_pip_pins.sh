#!/usr/bin/env bash
# Author: Luigi Freda
# This file is part of https://github.com/luigifreda/pyslam
#
# Final pip pins to be applied at the very end of the install process (venv/conda/pixi).
# Many packages installed along the way (e.g. wandb, nerfstudio/f3rm, detectron2, ...) silently
# upgrade some shared dependencies. Here we restore the versions required by the pinned
# tensorflow 2.13 stack and by pyslam.

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used
ROOT_DIR="$SCRIPT_DIR_/.."

. "$ROOT_DIR"/bash_utils.sh

print_blue "Running install_final_pip_pins.sh"

# Make sure a "latest" tensorflow (built against numpy 2) has not been pulled in (e.g. via tf-keras from tensorflow_hub>=0.16)
pip uninstall -y tf-keras > /dev/null 2>&1

# tensorflow_hub 0.15 (and other legacy packages) import pkg_resources, which was removed in setuptools>=81
pip install --no-warn-conflicts "setuptools<81"

# tensorflow 2.13 pins typing-extensions<4.6 but torch and other recent packages need a newer one (e.g. typing_extensions.TypeIs)
pip install --no-warn-conflicts -U "typing_extensions>=4.12"

# compiled modules (tensorflow 2.13, cpp core, etc.) are built against the numpy 1.x C-API
pip install --no-warn-conflicts "numpy<2" --force-reinstall
