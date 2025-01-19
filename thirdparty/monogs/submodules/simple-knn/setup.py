#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension
import os
import subprocess


# Initialize gcc major version
gcc_major_version = 0

# Get the version of g++
try: 
    # Run the command to get the g++ version
    result = subprocess.run(['g++', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        # Extract version from the output
        version_line = result.stdout.splitlines()[0]
        version = version_line.split()[-1]  # The last element is the version
        print(f"g++ version: {version}")

        # Check if the version supports C++20 (g++ 10 and above support it)
        gcc_major_version = int(version.split('.')[0])
        print(f"gcc major version: {gcc_major_version}")
    else:
        print("Failed to get g++ version")        
except Exception as e:
    print(f"Failed to get g++ version: {e}")
    

cxx_compiler_flags = ['-O2']

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

# Check nvcc version and set the appropriate flags
try:
    nvcc_std = subprocess.run("nvcc -h | grep -- '--std'", shell=True, capture_output=True, text=True)
    nvcc_std_output = nvcc_std.stdout
    
    nvcc_flags = ['-O2', '-allow-unsupported-compiler']
    if 'c++20' in nvcc_std_output and gcc_major_version >= 10:
        nvcc_flags.append('-std=c++20')
        cxx_compiler_flags.append('-std=c++20')
    elif 'c++17' in nvcc_std_output:
        nvcc_flags.append('-std=c++17')
        cxx_compiler_flags.append('-std=c++17')
    elif 'c++14' in nvcc_std_output:
        nvcc_flags.append('-std=c++14')
        cxx_compiler_flags.append('-std=c++14')
except Exception as e:
    print(f"Failed to get nvcc version: {e}")
    nvcc_flags = ['-O2', '-allow-unsupported-compiler']  # Default flags if nvcc check fails

    
print(f"nvcc flags: {nvcc_flags}")
print(f"cxx flags: {cxx_compiler_flags}")
    
setup(
    name="simple_knn",
    ext_modules=[
        CUDAExtension(
            name="simple_knn._C",
            sources=[
            "spatial.cu", 
            "simple_knn.cu",
            "ext.cpp"],
            extra_compile_args={"nvcc": nvcc_flags, "cxx": cxx_compiler_flags})
        ],
    cmdclass={
        'build_ext': BuildExtension
    }
)
