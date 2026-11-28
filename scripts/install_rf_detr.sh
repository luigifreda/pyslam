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


cd $ROOT_DIR/thirdparty

if [ ! -d "rf_detr" ]; then
    echo "Cloning rf_detr..."
    git clone https://github.com/roboflow/rf-detr.git rf_detr

    cd rf_detr
    git checkout fd1295b8ccacba0fad2b4a40c8a35b67bc62c335
    git apply ../rf_detr.patch

    # Pin transformers/peft to compatible APIs used by rf_detr
    pip install --upgrade "transformers>=4.41.2" "peft==0.10.0"

    # Install the rf_detr dependencies (read from its pyproject.toml) except torch/torchvision, keeping the
    # already-installed torch stack untouched, then install rf_detr itself with --no-deps.
    # This avoids patching rf_detr and prevents pip from swapping torch behind our back.
    TORCH_BASE_VERSION=$(python3 -c "import torch; print(torch.__version__.split('+')[0])")
    TORCHVISION_BASE_VERSION=$(python3 -c "import torchvision; print(torchvision.__version__.split('+')[0])")
    RF_DETR_DEPS=$(python3 - <<'PY'
try:
    import tomllib
except ModuleNotFoundError:  # python < 3.11
    from pip._vendor import tomli as tomllib
from packaging.requirements import Requirement
with open("pyproject.toml", "rb") as f:
    deps = tomllib.load(f)["project"]["dependencies"]
print("\n".join(d for d in deps if Requirement(d).name not in ("torch", "torchvision")))
PY
)
    echo "rf_detr dependencies (without torch/torchvision):"
    echo "$RF_DETR_DEPS"
    echo "$RF_DETR_DEPS" | tr '\n' '\0' | xargs -0 pip install "torch==$TORCH_BASE_VERSION" "torchvision==$TORCHVISION_BASE_VERSION"

    # install rf_detr as a package
    pip install -e . --no-deps

    cd ..
fi

cd $ROOT_DIR
