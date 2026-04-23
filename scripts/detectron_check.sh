#!/usr/bin/env bash
# Author: Luigi Freda 
# Author: Luigi Freda 
# This file is part of https://github.com/luigifreda/pyslam

#set -e

SCRIPT_DIR_=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir
SCRIPT_DIR_=$(readlink -f "$SCRIPT_DIR_")  # this reads the actual path if a symbolic directory is used

ROOT_DIR="$SCRIPT_DIR_/.."

# ====================================================
# import the bash utils 
. "$ROOT_DIR"/bash_utils.sh 

# ====================================================

PYTHON_BIN="${PYTHON:-python}"

if ! command -v "$PYTHON_BIN" &> /dev/null; then
    if command -v python3 &> /dev/null; then
        PYTHON_BIN="python3"
    else
        echo "ERROR: neither 'python' nor 'python3' was found in PATH."
        exit 1
    fi
fi

# This script avoids the following error by checking if the detectron2 C++ extension is loaded correctly:
#  Detectron2 C++ extension failed to load. This usually means it was compiled against a different PyTorch version. Rebuild detectron2 after installing the desired torch/torchvision versions, for example:
#   cd thirdparty/detectron2
#   python -m pip install --no-build-isolation -e . --force-reinstall

function print_detectron_fix_hint() {
    print_yellow "Recommended fix:"
    print_yellow "  cd \"$ROOT_DIR/thirdparty/detectron2\""
    print_yellow "  \"$PYTHON_BIN\" -m pip install --no-build-isolation -e . --force-reinstall"
    print_yellow "Make sure you rebuild detectron2 after changing torch/torchvision."
}

function reapply_detectron_patch_if_needed() {
    local patch_file="$ROOT_DIR/thirdparty/detectron2.patch"

    if [ ! -f "$patch_file" ]; then
        print_yellow "Patch file not found: $patch_file"
        return 0
    fi

    cd "$ROOT_DIR/thirdparty/detectron2"

    if git apply --check ../detectron2.patch &> /dev/null; then
        print_blue "Reapplying detectron2 patch..."
        git apply ../detectron2.patch
    else
        print_yellow "detectron2 patch already applied or not applicable, skipping patch step."
    fi
}

function check_detectron() {
    print_blue "=================================================================="
    print_blue "Checking detectron2 C++ extension..."

    cd "$ROOT_DIR/thirdparty"

    if [ ! -d "detectron2" ]; then
        print_yellow "Warning: thirdparty/detectron2 is missing."
        print_yellow "If detectron2 is not installed yet, run:"
        print_yellow "  $SCRIPT_DIR_/install_detectron2.sh"
    fi

    set +e
    "$PYTHON_BIN" - <<'PY'
import sys
import traceback

print(f"Python executable: {sys.executable}")
print(f"Python version: {sys.version.split()[0]}")

try:
    import torch
    print(f"torch version: {torch.__version__}")
    print(f"torch cuda version: {getattr(torch.version, 'cuda', None)}")
    print(f"torch cuda available: {torch.cuda.is_available()}")
except Exception as exc:
    print(f"Warning: failed to import torch: {exc}")

try:
    import detectron2
    print(f"detectron2 module: {getattr(detectron2, '__file__', '<unknown>')}")
    print(f"detectron2 version: {getattr(detectron2, '__version__', '<unknown>')}")
except Exception:
    print("ERROR: failed to import detectron2.")
    traceback.print_exc()
    sys.exit(2)

try:
    import detectron2._C as detectron2_cpp
    print(f"detectron2._C module: {detectron2_cpp.__file__}")
except Exception:
    print("ERROR: detectron2 Python package is present, but the compiled extension failed to load.")
    traceback.print_exc()
    sys.exit(3)

print("detectron2 import check passed.")
PY
    local status=$?
    set -e

    if [ $status -eq 0 ]; then
        print_green "detectron2 C++ extension is available and loadable."
    fi

    return $status
}

function rebuild_detectron() {
    print_blue "Rebuilding detectron2 C++ extension..."
    cd "$ROOT_DIR/thirdparty"

    if [ ! -d "detectron2" ]; then
        print_red "Cannot rebuild detectron2 because thirdparty/detectron2 is missing."
        print_yellow "Run $SCRIPT_DIR_/install_detectron2.sh first."
        return 1
    fi

    cd "$ROOT_DIR/thirdparty/detectron2"
    reapply_detectron_patch_if_needed
    "$PYTHON_BIN" -m pip install --no-build-isolation -e . --force-reinstall
    "$PYTHON_BIN" -m pip install  "numpy<2" --force-reinstall
    print_green "detectron2 C++ extension has been rebuilt."
}

if check_detectron; then
    exit 0
fi

print_red "detectron2 check failed."
print_detectron_fix_hint

rebuild_detectron

print_blue "Rechecking detectron2 after rebuild..."
check_detectron
