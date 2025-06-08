#!/usr/bin/env python3
# This script patches the pybind11 cast.h file to fix a specific issue with casting.
# It checks if the patch is already applied and only applies it if necessary.
# Usage: Run this script in the environment where pybind11 is installed.
# Note: This script assumes that the pybind11 library is installed in the standard location.
# It creates a backup of the original file before applying the patch.
import os
import torch
import platform

def is_ubuntu_24():
    try:
        # Ubuntu 24.04 returns ('Ubuntu', '24.04', ...)
        with open('/etc/os-release') as f:
            lines = f.readlines()
        id_line = next((l for l in lines if l.startswith('ID=')), '')
        version_line = next((l for l in lines if l.startswith('VERSION_ID=')), '')
        return 'ubuntu' in id_line.lower() and '24.04' in version_line
    except Exception as e:
        print(f"Could not determine OS version: {e}")
        return False

if not is_ubuntu_24():
    print("This patch is only applied on Ubuntu 24.04.")
    exit(0)

cast_h = os.path.join(torch.__path__[0], 'include', 'pybind11', 'cast.h')
backup = cast_h + '.bak'

original_line = 'return caster.operator typename make_caster<T>::template cast_op_type<T>();'
patched_line = 'return static_cast<typename make_caster<T>::cast_op_type<T>>(caster);'

if os.path.exists(cast_h):
    with open(cast_h, 'r') as f:
        content = f.read()
    if original_line in content:
        print(f"Patching: {cast_h}")
        with open(backup, 'w') as f:
            f.write(content)
        patched = content.replace(original_line, patched_line)
        with open(cast_h, 'w') as f:
            f.write(patched)
        print("Patch applied successfully.")
    else:
        print("Already patched or no need to patch.")
else:
    print("Could not find cast.h – is PyTorch properly installed?")
