# Troubleshooting 

<!-- TOC -->

- [Troubleshooting](#troubleshooting)
  - [Submitting a git issue](#submitting-a-git-issue)
  - [Clean reset](#clean-reset)
  - [Bad tracking performances](#bad-tracking-performances)
  - [Errors](#errors)
    - [_ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts_](#error-pips-dependency-resolver-does-not-currently-take-into-account-all-the-packages-that-are-installed-this-behaviour-is-the-source-of-the-following-dependency-conflicts)
    - [_ImportError: /home/.../anaconda3/envs/pyslam/bin/../lib/libgcc\_s.so.1: version \`GCC\_12.0.0' not found (required by /lib/x86\_64-linux-gnu/libhwy.so.1)_](#importerror-homeanaconda3envspyslambinliblibgcc_sso1-version-gcc_1200-not-found-required-by-libx86_64-linux-gnulibhwyso1)
    - [_RuntimeError: The detected CUDA version (11.8) mismatches the version that was used to compile_](#runtimeerror-the-detected-cuda-version-118-mismatches-the-version-that-was-used-to-compile)
    - [_Gtk-ERROR \*\*: ... GTK+ 2.x symbols detected. Using GTK+ 2.x and GTK+ 3 in the same process is not supported_](#gtk-error---gtk-2x-symbols-detected-using-gtk-2x-and-gtk-3-in-the-same-process-is-not-supported)
    - [SURF error](#surf-error)
    - [g2o Errors](#g2o-errors)
      - [_AttributeError: 'g2o.EdgeSE3ProjectXYZ' object has no attribute 'fx'_](#attributeerror-g2oedgese3projectxyz-object-has-no-attribute-fx)
      - [Cannot properly import g2o library or other libs](#cannot-properly-import-g2o-library-or-other-libs)
    - [OrderedSet error](#orderedset-error)
    - [Import erros related to ROS and OpenCV](#import-erros-related-to-ros-and-opencv)
    - [Could not import PILLOW\_VERSION from PIL](#could-not-import-pillow_version-from-pil)
    - [_ValueError: ndarray is not C-contiguous_](#valueerror-ndarray-is-not-c-contiguous)
    - [_Error: python3: malloc.c:2401: sysmalloc: Assertion \`(old\_top == initial\_top (av) \&\& old\_size == 0) ..._](#error-python3-mallocc2401-sysmalloc-assertion-old_top--initial_top-av--old_size--0-)
  - [When loading a neural network with CUDA everything gets stuck](#when-loading-a-neural-network-with-cuda-everything-gets-stuck)
  - [Manual install instead of using the provided install scripts](#manual-install-instead-of-using-the-provided-install-scripts)

<!-- /TOC -->

This page contains a small collections of issues/errors that may be experienced along with their fixes. 

**FIRST OF ALL**:      
- Did you read the main [README](./../README.md) page? 
- Did you use the provided **INSTALL SCRIPTS**? 

If not then go back on the [README](./../README.md) page, read the few lines in the install section and launch the **REQUIRED** install script. The install scripts were created to perform all the required install operations for you and make the install process itself as smooth and painless as possible.   

If you work under **Ubuntu** or **MacOS**, check the specific installation procedures reported in the main [README](./../README.md) page. 

--- 

## Submitting a git issue

For faster support when opening a new git issue, please provide the following information:
- Git commit (ensure you are on lastest `master' commit).
- Dataset used (if not a standard one, it is expected the dataset is shared through a link)
- Code configuration:
  * Modified parameters in `config.yaml`
  * Modified parameters in `pyslam/config_parameters.py`
  * Any other changes made to the codebase
- System configuration: 
  * OS, CUDA version, etc.
  * Is your OS native, or are you using it within Windows, VirtualBox, or a similar virtualization tool?
- Full console log (including the error messages)

Providing this information is essential for reproducing and debugging the issue efficiently.

---

## Clean reset

If you want to perform a **clean reset** and launch a fresh new install (rebuilding everything from scratch), run the following commands:  
```bash
./clean.sh --hard   # clean build folders 
./pyenv-delete.sh   # delete the "pyslam" virtual environment
```

--- 

## Bad tracking performances

Due to the multi-threading system (tracking thread + local mapping thread) and the non-super-fast performances of the python implementations (indeed, [python is not actually multithreading](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/Is-Pythons-GIL-the-software-worlds-biggest-blunder)), bad tracking performances may occur and vary depending on your machine computation capabilities. In a few words, it may happen that the local mapping thread is not fast enough to spawn new map points in time for the tracking thread. In fact, new spawned map points are necessary to let the tracking thread find enough {keypoint}-{map point} correspondences, and hence stably grasp at the map and proceed along its estimated trajectory. Simply put, the local mapping thread continuously builds/unrolls the fundamental 'carpet' of points (the map) on which the tracking thread 'walks': no 'carpet', no party!

If you experience bad tracking performances, go in [config_parameters.py](./config_parameters.py) and try to set `kTrackingWaitForLocalMappingToGetIdle=True`.

--- 

## Errors

### _ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts_

I have verified through numerous installation tests that the errors reported by pip's dependency resolver are typically warnings rather than critical issues. These messages do not prevent the main and test scripts from running successfully.

### _ImportError: /home/.../anaconda3/envs/pyslam/bin/../lib/libgcc_s.so.1: version `GCC_12.0.0' not found (required by /lib/x86_64-linux-gnu/libhwy.so.1)_

If you hit this issue, please refer to this [discussion](https://github.com/luigifreda/pyslam/issues/133).

### _RuntimeError: The detected CUDA version (11.8) mismatches the version that was used to compile_

If you get the following error (or a similar one)
```
      RuntimeError:
      The detected CUDA version (11.8) mismatches the version that was used to compile
      PyTorch (12.1). Please make sure to use the same CUDA versions.
```
then your detected CUDA version should actually be 11.8 and the following command might help:
```
pip3 install torch==2.2.0+cu118 torchvision==0.17+cu118 --index-url https://download.pytorch.org/whl/cu118
```
Another solution is to install CUDA 12.1, as suggested by the error message.

To know which CUDA version is required by your installed torch version, you can run:
```
python -c "import torch; print(torch.__version__)"
```

In case your dected CUDA version is a different one, you can easily adjust the above command by changing the `cuXYZ` version code. See this reference for further details:
https://pytorch.org/get-started/previous-versions/


### _Gtk-ERROR **: ... GTK+ 2.x symbols detected. Using GTK+ 2.x and GTK+ 3 in the same process is not supported_

If you hit such an error then decomment (or add) the following code in both `main_vo.py` and `main_slam.py`
```
# fix from https://github.com/tum-vision/fastfusion/issues/21
import gi
gi.require_version('Gtk', '2.0')
```
this will solve the problem. 
 
### SURF error

In order to use [non-free OpenCV features](https://stackoverflow.com/questions/50467696/pycharm-installation-of-non-free-opencv-modules-for-operations-like-sift-surf) (i.e. **SURF**, etc.), you need to install the module `opencv-contrib-python` built with the enabled option `OPENCV_ENABLE_NONFREE`. 

The provided install scripts will install a recent opencv version (>=**4.10**) with non-free modules enabled (see the provided scripts [scripts/install_pip3_packages.sh](./../scripts/install_pip3_packages.sh) and [scripts/install_opencv_python.sh](./../scripts/install_opencv_python.sh)). To quickly verify your installed opencv version run:       
`$ . pyenv-activate.sh `        
`$ ./scripts/opencv_check.py`       
or use the following command:        
`$ python3 -c "import cv2; print(cv2.__version__)"`      
How to check if you have non-free OpenCV module support (no errors imply success):          
`$ python3 -c "import cv2; detector = cv2.xfeatures2d.SURF_create()"`    

### g2o Errors

#### _AttributeError: 'g2o.EdgeSE3ProjectXYZ' object has no attribute 'fx'_

If you run into the following error
```
 File ".../pyslam/optimizer_g2o.py", line 96, in optimization
    edge.fx = f.fx 
AttributeError: 'g2o.EdgeSE3ProjectXYZ' object has no attribute 'fx'
```
that's because you did not run the script
```
$ ./scripts/install_thirdparty.sh   
```
as explained in the main [README](./../README.md) file. That's required in order to properly build and install the required thirdparty libs. 
Please,follow these steps:
- check you are on the correct pyslam branch according to your OS
- use the pyslam install scripts
- open a terminal in the root folder of the repo clean with `$ ./clean.sh`
- launch a new install with `$ ./scripts/install_all_venv.sh`
- Please, double check that you have the file like `thirdparty/g2opy/lib/g2o.cpython-36m-x86_64-linux-gnu.so`. 


#### Cannot properly import g2o library or other libs 

If you get an error message like 
```
import g2o 
ModuleNotFoundError: No module named 'g2o' error
```
First of all, check if you have a compiled `thirdparty/g2opy/lib/g2o.cpython-*-linux-gnu.so`. If not, Did you use one of the install_all scripts? Depending on your selected working environment (native, conda, python3-venv), you need to launch its companion install_all script in order to actually install all the required libraries (including g2o). Please, read the install instruction in the main [README](./../README.md) file. 

On the other hand, if you already have a compiled `thirdparty/g2opy/lib/g2o.cpython-*-linux-gnu.so`, it's very likely you have libraries compiled in a 'mixed' way. Then, try to clean everything with the script `clean.sh`, and follow the installation procedure again (see the main [README](./../README.md) file). 

Last but not least, please recall that you need to activate your `pyenv`/`conda` environment before launching any pySLAM script.


### OrderedSet error 

Reference: https://github.com/luigifreda/pyslam/issues/48 

If you run `main_slam.py` and hit the following error
```
File "/home/dam/.local/lib/python3.5/site-packages/ordered_set.py", line 134, in copy
return self.class(self)
File "/home/dam/.local/lib/python3.5/site-packages/ordered_set.py", line 69, in init
self |= iterable
TypeError: unsupported operand type(s) for |=: 'OrderedSet' and 'OrderedSet'
``` 
You can solve such an issue by installing a lower version of OrderedSet
```
pip3 install ordered-set==3.1.1 --force-reinstall
``` 

### Import erros related to ROS and OpenCV

If you have ROS installed in your system and got the following error:
```
ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so:  
undefined symbol: PyCObject_Type
```
You can run the following command in your shell: 
```
$ export PYTHONPATH=""
```
this will remove the ROS OpenCV python modules from your python path and will solve the issue. 


### Could not import PILLOW_VERSION from PIL 

Or **ImportError: cannot import name 'PILLOW_VERSION'**

If you get this error, it is very likely that pillow 7.0.0 has some troubles with pytorch. In order to solve this error, run

```
$ pip3 uninstall pillow 
$ pip3 install pillow==6.2.2
```
(fix from this [page](https://stackoverflow.com/questions/59659146/could-not-import-pillow-version-from-pil))


### _ValueError: ndarray is not C-contiguous_

If the following error pops-up:
```
ValueError: ndarray is not C-contiguous
```
Open pyslam/search_points.py, find this line (should be line 79):
```
projs = f_cur.project_map_points(points)
```
and replace it with :
```
projs = f_cur.project_map_points(points)
projs = projs.copy(order='C')
```

(thanks to [naughtyStark](https://github.com/naughtyStark))

### _Error: python3: malloc.c:2401: sysmalloc: Assertion `(old_top == initial_top (av) && old_size == 0) ..._

I got this error after messing up with the installation of different python packages related to torch and torchvision. The result was that tfeat was generating segmentation faults. In order to check if this is actually your case, run
```
$ cd test/cv     # this is compulsory
$ python3 test_tfeat.py
```
and check if you get a segmenation fault. If this is the case, try to run 
```
$ pip3 uninstall torch torchvision 
$ pip3 install torch torchvision 
```
in order to get a clean installation of the torch packages. 

--- 

## When loading a neural network with CUDA everything gets stuck

I got this issue with a new NVIDIA GPU while loading `SuperPoint` neural network. The NN loading got stuck. This error arises when CUDA code was not compiled to target your GPU architecture. Two solutions: 
- Easy: turn off CUDA (for instance, with `SuperPointFeature2D()` set the default class option `do_cuda=False`). In this case, the computations will be moved to CPU. 
- You need to install a pytorch version that is compatible with your CUDA version and GPU architecture. See for instance these two links: 
https://stackoverflow.com/questions/75682385/runtimeerror-cuda-error-no-kernel-image-is-available-for-execution-on-the-devi
https://discuss.pytorch.org/t/failed-to-load-image-python-extension-could-not-find-module/140278   


---

## Manual install instead of using the provided install scripts

If you really want to install things manually instead of using the install scripts, follow the same steps of the install scripts, and good luck!
