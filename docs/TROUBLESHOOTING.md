# Troubleshooting 

<!-- TOC -->

- [Troubleshooting](#troubleshooting)
    - [1. Bad tracking performances](#1-bad-tracking-performances)
    - [2. Gtk-ERROR **: ... GTK+ 2.x symbols detected. Using GTK+ 2.x and GTK+ 3 in the same process is not supported](#2-gtk-error---gtk-2x-symbols-detected-using-gtk-2x-and-gtk-3-in-the-same-process-is-not-supported)
    - [3. SURF error](#3-surf-error)
    - [4. g2o Errors](#4-g2o-errors)
        - [4.1. AttributeError: 'g2o.EdgeSE3ProjectXYZ' object has no attribute 'fx'](#41-attributeerror-g2oedgese3projectxyz-object-has-no-attribute-fx)
        - [4.2. Cannot properly import g2o library or other libs](#42-cannot-properly-import-g2o-library-or-other-libs)
    - [5. When loading a neural network with CUDA everything gets stuck](#5-when-loading-a-neural-network-with-cuda-everything-gets-stuck)
    - [6. OrderedSet](#6-orderedset)
    - [7. Problems with ROS and OpenCV](#7-problems-with-ros-and-opencv)
    - [8. Could not import PILLOW_VERSION from PIL](#8-could-not-import-pillow_version-from-pil)
    - [9. ValueError: ndarray is not C-contiguous](#9-valueerror-ndarray-is-not-c-contiguous)
    - [10. Error: python3: malloc.c:2401: sysmalloc: Assertion old_top == initial_top av && old_size == 0 ...](#10-error-python3-mallocc2401-sysmalloc-assertion-old_top--initial_top-av--old_size--0-)
    - [11. Python version](#11-python-version)

<!-- /TOC -->

This page contains a small collections of issues/errors that may be experienced along with their fixes. 

**FIRST OF ALL**: did you read the main [README](./../README.md) page? did you use the provided **INSTALL SCRIPTS**? If not then go back on the [README](./../README.md) page, read the few lines in the install section and launch the **REQUIRED** install script. The install scripts were created in order to perform all the required install operations for you and make the install process itself as painless as possible.   

If you work under **Ubuntu 20.04** or **MacOS**, check the specific installation procedures reported in the main [README](./../README.md) page. 


## Bad tracking performances

Due to the multi-threading system (tracking thread + local mapping thread) and the non-super-fast performances of the python implementations (indeed, [python is not actually multithreading](https://www.theserverside.com/blog/Coffee-Talk-Java-News-Stories-and-Opinions/Is-Pythons-GIL-the-software-worlds-biggest-blunder)), bad tracking performances may occur and vary depending on your machine computation capabilities. In a few words, it may happen that the local mapping thread is not fast enough to spawn new map points in time for the tracking thread. In fact, new spawned map points are necessary to let the tracking thread find enough {keypoint}-{map point} correspondences, and hence stably grasp at the map and proceed along its estimated trajectory. Simply put, the local mapping thread continuously builds/unrolls the fundamental 'carpet' of points (the map) on which the tracking thread 'walks': no 'carpet', no party!

If you experience bad tracking performances, go in [config_parameters.py](./config_parameters.py) and try to set `kTrackingWaitForLocalMappingToGetIdle=True`.

## Gtk-ERROR **: ... GTK+ 2.x symbols detected. Using GTK+ 2.x and GTK+ 3 in the same process is not supported

If you hit such an error then decomment (or add) the following code in both `main_vo.py` and `main_slam.py`
```
# fix from https://github.com/tum-vision/fastfusion/issues/21
import gi
gi.require_version('Gtk', '2.0')
```
this will solve the problem. 
 
## SURF error

In order to use [non-free OpenCV features](https://stackoverflow.com/questions/50467696/pycharm-installation-of-non-free-opencv-modules-for-operations-like-sift-surf) (i.e. **SURF**, etc.), you need to install the module `opencv-contrib-python` built with the enabled option `OPENCV_ENABLE_NONFREE`. 

The provided install scripts will install a recent opencv version (>=**4.10**) with non-free modules enabled (see the provided scripts [install_pip3_packages.sh](./../install_pip3_packages.sh) and [install_opencv_python.sh](./../install_opencv_python.sh)). To quickly verify your installed opencv version run:       
`$ . pyenv-activate.sh `        
`$ ./scripts/opencv_check.py`       
or use the following command:        
`$ python3 -c "import cv2; print(cv2.__version__)"`      
How to check if you have non-free OpenCV module support (no errors imply success):          
`$ python3 -c "import cv2; detector = cv2.xfeatures2d.SURF_create()"`    

## g2o Errors

### AttributeError: 'g2o.EdgeSE3ProjectXYZ' object has no attribute 'fx'

If you run into the following error
```
 File ".../pyslam/optimizer_g2o.py", line 96, in optimization
    edge.fx = f.fx 
AttributeError: 'g2o.EdgeSE3ProjectXYZ' object has no attribute 'fx'
```
that's because you did not run the script
```
$ ./install_thirdparty.sh   
```
as explained in the main [README](./../README.md) file. That's required in order to properly build and install the required thirdparty libs. 
Please,follow these steps:
- check you are on the correct pyslam branch according to your OS
- use the pyslam install scripts
- open a terminal in the root folder of the repo clean with `$ ./clean.sh`
- launch a new install with `$ ./install_all_venv.sh`
- Please, double check that you have the file like `thirdparty/g2opy/lib/g2o.cpython-36m-x86_64-linux-gnu.so`. 


### Cannot properly import g2o library or other libs 

If you get an error message like 
```
import g2o 
ModuleNotFoundError: No module named 'g2o' error
```
First of all, check if you have a compiled `thirdparty/g2opy/lib/g2o.cpython-*-linux-gnu.so`. If not, Did you use one of the install_all scripts? Depending on your selected working environment (native, conda, python3-venv), you need to launch its companion install_all script in order to actually install all the required libraries (including g2o). Please, read the install instruction in the main [README](./../README.md) file. 

On the other hand, if you already have a compiled `thirdparty/g2opy/lib/g2o.cpython-*-linux-gnu.so`, it's very likely you have libraries compiled in a 'mixed' way. Then, try to clean everything with the script `clean.sh`, and follow the installation procedure again (see the main [README](./../README.md) file). 

Last but not least, please recall that you need to activate your `pyenv`/`conda` environment before launching any pySLAM script.

## When loading a neural network with CUDA everything gets stuck

I got this issue with a new NVIDIA GPU while loading `SuperPoint` neural network. The NN loading got stuck. This error arises when CUDA code was not compiled to target your GPU architecture. Two solutions: 
- Easy: turn off CUDA (for instance, with `SuperPointFeature2D()` set the default class option `do_cuda=False`). In this case, the computations will be moved to CPU. 
- You need to install a pytorch version that is compatible with your CUDA version and GPU architecture. See for instance these two links: 
https://stackoverflow.com/questions/75682385/runtimeerror-cuda-error-no-kernel-image-is-available-for-execution-on-the-devi
https://discuss.pytorch.org/t/failed-to-load-image-python-extension-could-not-find-module/140278   

## OrderedSet 

reference https://github.com/luigifreda/pyslam/issues/48 

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
pip install ordered-set==3.1.1 --force-reinstall
``` 

## Problems with ROS and OpenCV

If you have ROS installed in your system and got the following error:
```
ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so:  
undefined symbol: PyCObject_Type
```
you can run the following command in your shell: 
```
$ export PYTHONPATH=""
```
this will remove the ROS OpenCV python modules from your python path and will solve the issue. 


## Could not import PILLOW_VERSION from PIL 

Or **ImportError: cannot import name 'PILLOW_VERSION'**

If you get this error, it is very likely that pillow 7.0.0 has some troubles with pytorch. In order to solve this error, run

```
$ pip3 uninstall pillow 
$ pip3 install pillow==6.2.2
```
(fix from this [page](https://stackoverflow.com/questions/59659146/could-not-import-pillow-version-from-pil))


## ValueError: ndarray is not C-contiguous

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

## Error: python3: malloc.c:2401: sysmalloc: Assertion `(old_top == initial_top (av) && old_size == 0) ...

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



## Python version

Installation issues may happen if multiple python versions are mixed. All the instructions reported in this repository assume you are using python3. If you really want to install things manually instead of using the install scripts, follow the same steps of the install scripts, and good luck!
