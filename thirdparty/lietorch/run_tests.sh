#!/bin/bash

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd ) # get script dir (this should be the main folder directory of PLVS)
SCRIPT_DIR=$(readlink -f $SCRIPT_DIR)  # this reads the actual path if a symbolic directory is used

# lietorch_backends links against torch/*.so; ensure the loader can find them.
if command -v python &>/dev/null; then
    _torch_lib=$(python -c "import torch, os; print(os.path.join(torch.__path__[0], 'lib'))" 2>/dev/null)
    if [[ -n "$_torch_lib" && -d "$_torch_lib" ]]; then
        export LD_LIBRARY_PATH="${_torch_lib}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
    fi
fi

python "$SCRIPT_DIR"/lietorch/run_tests.py
python "$SCRIPT_DIR"/lietorch/run_grad_tests.py --cuda --probe-embedding-grad
