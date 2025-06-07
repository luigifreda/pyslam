#!/usr/bin/env python3
# This script patches the pybind11 cast.h file to fix a specific issue with casting.
# It checks if the patch is already applied and only applies it if necessary.
# Usage: Run this script in the environment where pybind11 is installed.
# Note: This script assumes that the pybind11 library is installed in the standard location.
# It creates a backup of the original file before applying the patch.
import os
import torch

cast_h = os.path.join(torch.__path__[0], 'include', 'pybind11', 'cast.h')
backup = cast_h + '.bak'

if os.path.exists(cast_h):
    with open(cast_h, 'r') as f:
        content = f.read()
    if 'cast_op_type<T>()' in content:
        print(f"Patching: {cast_h}")
        with open(backup, 'w') as f:
            f.write(content)
        patched = content.replace(
            'return caster.operator typename make_caster<T>::template cast_op_type<T>();',
            'return static_cast<typename make_caster<T>::cast_op_type<T>>(caster);'
        )
        with open(cast_h, 'w') as f:
            f.write(patched)
    else:
        print("Already patched or no need to patch.")
