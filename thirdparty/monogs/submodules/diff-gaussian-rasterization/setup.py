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
import re
import subprocess
import platform


NUM_PARALLEL_BUILD_JOBS = 4

# Get the current script's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Initialize gcc major version
gcc_major_version = 0

try:
    # Run the command to get the g++ version
    result = subprocess.run(["g++", "--version"], capture_output=True, text=True)
    if result.returncode == 0:
        version_line = result.stdout.splitlines()[0]
        version = version_line.split()[-1]  # Last element is the version
        print(f"g++ version: {version}")

        # Check if the version supports C++20 (g++ 10 and above support it)
        gcc_major_version = int(version.split(".")[0])
        print(f"gcc major version: {gcc_major_version}")
    else:
        print("Failed to get g++ version")
except Exception as e:
    print(f"Failed to get g++ version: {e}")


def get_os_release():
    # Python 3.11+ has this helper:
    try:
        return platform.freedesktop_os_release()
    except Exception:
        pass
    # Fallback: read /etc/os-release
    data = {}
    try:
        with open("/etc/os-release") as f:
            for line in f:
                if "=" in line:
                    k, v = line.rstrip().split("=", 1)
                    data[k] = v.strip().strip('"')
    except FileNotFoundError:
        pass
    return data


def is_ubuntu_22():
    rel = get_os_release()
    return (rel.get("ID") == "ubuntu") and rel.get("VERSION_ID", "").startswith("22")


def get_cuda_version_from_nvcc():
    try:
        out = subprocess.run(["nvcc", "--version"], capture_output=True, text=True)
        if out.returncode == 0:
            # e.g. "Cuda compilation tools, release 12.2, V12.2.128"
            m = re.search(r"release\s+(\d+)\.(\d+)", out.stdout)
            if m:
                return f"{m.group(1)}.{m.group(2)}"
    except FileNotFoundError:
        pass
    return None


def get_cuda_version():
    # prefer nvcc; fallback to torch
    v = get_cuda_version_from_nvcc()
    if v:
        return v
    try:
        import torch

        if torch.version.cuda:
            return torch.version.cuda
    except Exception:
        pass
    return ""


def get_supported_architectures():
    # Use `nvcc --list-gpu-arch` to list supported architectures
    try:
        result = subprocess.run(["nvcc", "--list-gpu-arch"], capture_output=True, text=True)
        if result.returncode == 0:
            architectures = result.stdout.splitlines()
            return [arch.split("_")[1] for arch in architectures if arch.startswith("compute_")]
        else:
            print("Could not retrieve architectures. Using defaults.")
    except FileNotFoundError:
        print("nvcc not found. Make sure CUDA is installed and in PATH.")
    # Return a default list if nvcc is unavailable
    return ["60", "61", "70", "75", "80", "86"]


def get_current_architecture():
    try:
        result = subprocess.run(["nvidia-smi", "-q"], capture_output=True, text=True)
        if result.returncode == 0:
            # Search for Compute Capability lines, e.g., "Compute Capability : 7.5"
            match = re.search(r"Compute Capability\s*:\s*(\d+)\.(\d+)", result.stdout)
            if match:
                major, minor = match.groups()
                return [f"{major}{minor}"]
            else:
                print("Compute Capability not found in nvidia-smi output.")
        else:
            print("nvidia-smi command failed.")
    except FileNotFoundError:
        print("nvidia-smi not found. Make sure NVIDIA drivers are installed.")

    # Fallback default if detection fails
    return ["75"]


# Check for nvcc version and set appropriate flags
try:
    result = subprocess.run("nvcc -h | grep -- '--std'", shell=True, capture_output=True, text=True)
    nvcc_std = result.stdout
    nvcc_flags = ["-O2", "-allow-unsupported-compiler"]

    cuda_ver = get_cuda_version()
    nvcc_flags = ["-O2", "-allow-unsupported-compiler"]

    if cuda_ver.startswith("12.2") and is_ubuntu_22():
        # Check for supported C++ standard in nvcc
        if "c++20" in nvcc_std and gcc_major_version >= 10:
            # force c++17 for CUDA 12.2 on Ubuntu 22.04
            nvcc_flags.append("-std=c++17")
        elif "c++17" in nvcc_std:
            nvcc_flags.append("-std=c++17")
        elif "c++14" in nvcc_std:
            nvcc_flags.append("-std=c++14")
    else:
        # Check for supported C++ standard in nvcc
        if "c++20" in nvcc_std and gcc_major_version >= 10:
            nvcc_flags.append("-std=c++20")
        elif "c++17" in nvcc_std:
            nvcc_flags.append("-std=c++17")
        elif "c++14" in nvcc_std:
            nvcc_flags.append("-std=c++14")

    # Add the path for third-party libraries (e.g., glm)
    glm_path = os.path.join(current_dir, "third_party/glm/")
    nvcc_flags.append("-I" + glm_path)

    print(f"nvcc flags: {nvcc_flags}")
except Exception as e:
    print(f"Failed to get nvcc version: {e}")
    nvcc_flags = ["-O2", "-allow-unsupported-compiler"]  # Default flags if nvcc check fails


supported_architectures = get_supported_architectures()
# supported_architectures = get_current_architecture()
for arch in supported_architectures:
    nvcc_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")

print(f"nvcc flags: {nvcc_flags}")


class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # Enable parallel builds
        self.parallel = NUM_PARALLEL_BUILD_JOBS
        print(f"Building with {self.parallel} parallel jobs")  # Debug message
        super().build_extensions()


# Setup for building the CUDA extension
setup(
    name="diff_gaussian_rasterization",
    packages=["diff_gaussian_rasterization"],
    ext_modules=[
        CUDAExtension(
            name="diff_gaussian_rasterization._C",
            sources=[
                "cuda_rasterizer/rasterizer_impl.cu",
                "cuda_rasterizer/forward.cu",
                "cuda_rasterizer/backward.cu",
                "rasterize_points.cu",
                "ext.cpp",
            ],
            extra_compile_args={"nvcc": nvcc_flags},
        ),
    ],
    cmdclass={"build_ext": CustomBuildExtension},
)
