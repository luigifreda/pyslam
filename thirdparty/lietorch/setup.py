from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

import subprocess
import os
import os.path as osp


NUM_PARALLEL_BUILD_JOBS = 1 # It seems setting more than 1 job does not work here! There seems to a race condition at build time for some reason.

ROOT = osp.dirname(osp.abspath(__file__))


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
    

def get_supported_architectures():
    # Use `nvcc --list-gpu-arch` to list supported architectures
    try:
        result = subprocess.run(["nvcc", "--list-gpu-arch"], capture_output=True, text=True)
        if result.returncode == 0:
            architectures = result.stdout.splitlines()
            return [arch.split('_')[1] for arch in architectures if arch.startswith("compute_")]
        else:
            print("Could not retrieve architectures. Using defaults.")
    except FileNotFoundError:
        print("nvcc not found. Make sure CUDA is installed and in PATH.")
    # Return a default list if nvcc is unavailable
    return ["60", "61", "70", "75", "80", "86"]



cxx_compiler_flags = ['-O2']

if os.name == 'nt':
    cxx_compiler_flags.append("/wd4624")

# Check nvcc version and set the appropriate flags.
# Make sure that the nvcc executable is available in $PATH variables,
# or find one according to the $CUDA_HOME variable
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
    
supported_architectures = get_supported_architectures()
for arch in supported_architectures:
    nvcc_flags.append(f"-gencode=arch=compute_{arch},code=sm_{arch}")

    
print(f"nvcc flags: {nvcc_flags}")
print(f"cxx flags: {cxx_compiler_flags}")
    

class CustomBuildExtension(BuildExtension):
    def build_extensions(self):
        # Enable parallel builds
        self.parallel = NUM_PARALLEL_BUILD_JOBS
        print(f"Building with {self.parallel} parallel jobs")  # Debug message        
        super().build_extensions()
        

setup(
    name='lietorch',
    version='0.2',
    description='Lie Groups for PyTorch',
    author='teedrz',
    packages=['lietorch'],
    ext_modules=[
        CUDAExtension('lietorch_backends', 
            include_dirs=[
                osp.join(ROOT, 'lietorch/include'), 
                osp.join(ROOT, 'eigen')],
            sources=[
                'lietorch/src/lietorch_gpu.cu',                
                'lietorch/src/lietorch.cpp', 
                'lietorch/src/lietorch_cpu.cpp'],
            extra_compile_args={
                'cxx': cxx_compiler_flags, 
                'nvcc': nvcc_flags
            }),

        CUDAExtension('lietorch_extras', 
            sources=[
                'lietorch/extras/altcorr_kernel.cu',
                'lietorch/extras/corr_index_kernel.cu',
                'lietorch/extras/se3_builder.cu',
                'lietorch/extras/se3_inplace_builder.cu',
                'lietorch/extras/se3_solver.cu',
                'lietorch/extras/extras.cpp',
            ],
            extra_compile_args={
                'cxx': cxx_compiler_flags, 
                'nvcc': nvcc_flags
            }),
    ],
    cmdclass={ 'build_ext': CustomBuildExtension }
)


