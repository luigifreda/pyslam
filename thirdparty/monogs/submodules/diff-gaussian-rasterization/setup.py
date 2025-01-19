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

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize gcc major version
gcc_major_version = 0

try:
    # Run the command to get the g++ version
    result = subprocess.run(['g++', '--version'], capture_output=True, text=True)
    if result.returncode == 0:
        version_line = result.stdout.splitlines()[0]
        version = version_line.split()[-1]  # Last element is the version
        print(f"g++ version: {version}")

        # Check if the version supports C++20 (g++ 10 and above support it)
        gcc_major_version = int(version.split('.')[0])
        print(f"gcc major version: {gcc_major_version}")
    else:
        print("Failed to get g++ version")
except Exception as e:
    print(f"Failed to get g++ version: {e}")

# Check for nvcc version and set appropriate flags
try:
    result = subprocess.run("nvcc -h | grep -- '--std'", shell=True, capture_output=True, text=True)
    nvcc_std = result.stdout
    nvcc_flags = ['-O2', '-allow-unsupported-compiler']

    # Check for supported C++ standard in nvcc
    if 'c++20' in nvcc_std and gcc_major_version >= 10:
        nvcc_flags.append('-std=c++20')
    elif 'c++17' in nvcc_std:
        nvcc_flags.append('-std=c++17')
    elif 'c++14' in nvcc_std:
        nvcc_flags.append('-std=c++14')

    # Add the path for third-party libraries (e.g., glm)
    glm_path = os.path.join(current_dir, "third_party/glm/")
    nvcc_flags.append('-I' + glm_path)
    
    print(f"nvcc flags: {nvcc_flags}")
except Exception as e:
    print(f"Failed to get nvcc version: {e}")
    nvcc_flags = ['-O2', '-allow-unsupported-compiler']  # Default flags if nvcc check fails

# Setup for building the CUDA extension
setup(
    name="diff_gaussian_rasterization",
    packages=['diff_gaussian_rasterization'],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp"
            ],
            extra_compile_args={"nvcc": nvcc_flags}
        ),
    ],
    cmdclass={
        'build_ext': BuildExtension
    }
)