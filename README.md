# pySLAM 

Author: [Luigi Freda](https://www.luigifreda.com)

**pySLAM** is a *'toy'* implementation of a monocular *Visual Odometry (VO)* pipeline in Python. I released it for **educational purposes**, for a [computer vision class](https://as-ai.org/visual-perception-and-spatial-computing/) I taught. I started developing it for fun as a python programming exercise, during my free time. I took inspiration from some python repos available on the web. 

Main Scripts:
* `main_vo.py` combines the simplest VO ingredients without performing any image point triangulation or windowed bundle adjustment. At each step $k$, `main_vo.py` estimates the current camera pose $C_k$ with respect to the previous one $C_{k-1}$. The inter frame pose estimation returns $[R_{k-1,k},t_{k-1,k}]$ with $||t_{k-1,k}||=1$. With this very basic approach, you need to use a ground truth in order to recover a correct inter-frame scale $s$ and estimate a valid trajectory by composing $C_k = C_{k-1} * [R_{k-1,k}, s t_{k-1,k}]$. This script is a first start to understand the basics of inter frame feature tracking and camera pose estimation.

* `main_slam.py` adds feature tracking along multiple frames, point triangulation and bundle adjustment in order to estimate the camera trajectory up-to-scale and a build a local map. It's still a VO pipeline but it shows some basic blocks which are necessary to develop a real visual SLAM pipeline. 

You can use this *'toy'* framework as a baseline to play with VO techniques or create your own (proof of concept) VO/SLAM pipeline in python. When you test it, please, consider that's intended as a simple *'toy'* framework, without any pretence of being considered peformant. Check the terminal warnings if you see something weird happening.  

**Enjoy it!**

<center> <img src="images/main-vo.png"
alt="VO" height="300" border="1" /> 
<img src="images/main-slam-kitti-map.png"
alt="SLAM" height="300" border="1" /> </center>

--- 
## Requirements


* Python 3  (tested under Python 3.5)
* Numpy
* OpenCV (see below for a suggested python installation)

The framework has been developed and tested under Ubuntu 16.04. You may need to install some required python3 packages. These packages can be easily and automatically installed by running:   

`$ ./install_pip3_packages.sh`   


If you want to run `main_slam.py` you have to install the libs: 

* [pangolin](https://github.com/stevenlovegrove/Pangolin) 
* [g2opy](https://github.com/uoip/g2opy)

These libs can be easily installed by running the script:    

`$ ./install_thirdparty.sh`   


#### How to install non-free OpenCV modules under Ubuntu 

In order to use non-free OpenCV modules ([link](https://stackoverflow.com/questions/50467696/pycharm-installation-of-non-free-opencv-modules-for-operations-like-sift-surf)) under Ubuntu, you can run    
`$ pip3 install opencv-contrib-python==3.4.0.12`

For a more advanced OpenCV installation procedure, you can take a look [here](https://www.pyimagesearch.com/2016/10/24/ubuntu-16-04-how-to-install-opencv/). 


#### Problems with ROS and OpenCV?

If you have tried to run the scripts and got the following error (having installed ROS in your system):
```
ImportError: /opt/ros/kinetic/lib/python2.7/dist-packages/cv2.so:  
undefined symbol: PyCObject_Type
```
you can run the following command in your shell: 
```
$ export PYTHONPATH=""
```
this will remove the ROS OpenCV python modules from your python path and will solve the issue. 

--- 
## Usage

You can test the code right away by running:
```
$ python3 -O main_vo.py
```
This will process a [KITTI]((http://www.cvlibs.net/datasets/kitti/eval_odometry.php)) video (available in the folder `videos`) by using its corresponding camera calibration file (available in the folder `settings`), and its groundtruth (available in the video folder). 

**N.B.**: as explained above, the script `main_vo.py` **strictly requires a ground truth**.  

In order to process a different **dataset**, you need to set the file `config.ini`:
* select your dataset `type` in the section `[DATASET]` (see the section *Datasets* below for further details) 
* the camera settings file accordingly (see the section *Camera Settings* below)
* the groudtruth file accordingly (see the section *Camera Settings* below)

If you want to test the script `main_slam.py`, you can run:
```
$ python3 -O main_slam.py
```

You can choose any detector/descriptor among *ORB*, *SIFT*, *SURF*, *BRISK*, *AKAZE* (see below for further information). 

**WARNING**: the available **KITTI videos** (due to information loss in video compression) make main_slam tracking peform worse than with the original KITTI *image sequences*. The available videos are intended to be used for a first quick test. Please, download and use the original KITTI image sequences as explained below. For instance, on the original KITTI sequence 06, main_slam successfully completes the round; at present time, this does not happen with the compressed video.

--- 
## Datasets

You can use 4 different types of datasets:

Dataset | type in `config.ini`
--- | --- 
[KITTI odometry data set (grayscale, 22 GB)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)  | `type=KITTI_DATASET` 
[TUM dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)                   | `type=TUM_DATASET` 
video file        | `type=VIDEO_DATASET` 
folder of images  | `type=FOLDER_DATASET` 

### KITTI Datasets

pySLAM code expects the following structure in the specified KITTI path folder (specified in the section `[KITTI_DATASET]` of the file `config.ini`). : 
```
├── sequences
    ├── 00
    ...
    ├── 21
├── poses
    ├── 00.txt
        ...
    ├── 10.txt

```
1. Download the dataset (grayscale images) from http://www.cvlibs.net/datasets/kitti/eval_odometry.php and prepare the KITTI folder as specified above

2. Select the corresponding calibration settings file (parameter `[KITTI_DATASET][cam_settings]` in the file `config.ini`)



### TUM Datasets 

pySLAM code expects a file `associations.txt` in each TUM dataset folder (specified in the section `[TUM_DATASET]` of the file `config.ini`). 

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). You can generate your `associations.txt` file by executing:

```
$ python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```
3. Select the corresponding calibration settings file (parameter `[TUM_DATASET][cam_settings]` in the file `config.ini`)


--- 
## Camera Settings

The folder `settings` contains the camera settings files which can be used for testing the code. These are the same used in the framework [ORBSLAM2](https://github.com/raulmur/ORB_SLAM2). You can easily modify one of those files for creating your own new calibration file (for your new datasets).

In order to calibrate your camera, you can use the scripts in the folder `calibration` and you may want to have a look [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html). In particular: 
1. use the script `grab_chessboard_images.py` to collect a sequence of images where the chessboard can be detected (set the chessboard size there)
2. use the script `calibrate.py` to process the collected images and compute the calibration parameters (set the chessboard size there)

---
## Detectors/Descriptors

At present time, the following feature **detectors** are supported: 
* *FAST*
* *Good features to track* [[ShiTo94]](https://ieeexplore.ieee.org/document/323794)
* *ORB*  
* *SIFT*
* *SURF*
* *AKAZE*
* *BRISK*

You can take a look at the file `feature_detector.py`. 

The following feature **descriptors** are supported: 
* *ORB*  
* *SIFT*
* *SURF*
* *AKAZE*
* *BRISK*

In both the scripts `main_vo.py` and `main_slam.py`, you can set which detector/descritor to use by means of the function *feature_tracker_factory()*. This function can be found in the file `feature_tracker.py`.

--- 
## References

Suggested books:
* *Multiple View Geometry in Computer Vision* by Richard Hartley and Andrew Zisserman
* *An Invitation to 3-D Vision* by Yi-Ma, Stefano Soatto, Jana Kosecka, S. Shankar Sastry 
* *Computer Vision: Algorithms and Applications*, by Richard Szeliski 

Suggested material:
* *[Vision Algorithms for Mobile Robotics](http://rpg.ifi.uzh.ch/teaching.html)* by Davide Scaramuzza 
* *[CS 682 Computer Vision](http://cs.gmu.edu/~kosecka/cs682.html)* by Jana Kosecka   

Moreover, you may want to have a look at the OpenCV [guide](https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) or [tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html).  

---
## TODO

Tons of things are still missing to attain a real SLAM pipeline: 

* keyframe generation and management 
* tracking w.r.t. previous keyframe 
* proper local map generation and management (covisibility)
* loop closure
* general relocalization 


---
## Credits 

* [twitchslam](https://github.com/geohot/twitchslam)
* [monoVO](https://github.com/uoip/monoVO-python)
* [pangolin](https://github.com/stevenlovegrove/Pangolin) 
* [g2opy](https://github.com/uoip/g2opy)
