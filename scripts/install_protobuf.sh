#!/usr/bin/env bash
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

SCRIPT_DIR_=$(cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f $SCRIPT_DIR_)  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

cd "$ROOT_DIR" 

# ====================================================

STARTING_DIR=`pwd`
cd "$ROOT_DIR"

print_blue "Running install_protobuf.sh"
echo "Installing a tensorflow compatible protobuf version..."

PROTOBUF_SPEC=$(
python - <<'PY'
from importlib.metadata import distribution, PackageNotFoundError
from packaging.requirements import Requirement
for pkg in ("tensorflow", "tensorflow-cpu"):
    try:
        reqs = distribution(pkg).requires or []
    except PackageNotFoundError:
        continue
    for req in reqs:
        r = Requirement(req)
        if r.name == "protobuf":
            print(r.specifier)
            raise SystemExit(0)
raise SystemExit("Could not determine protobuf requirement from installed TensorFlow")
PY
)

echo "PROTOBUF_SPEC: $PROTOBUF_SPEC"
pip install --force-reinstall "protobuf${PROTOBUF_SPEC}"