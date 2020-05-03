# pySLAM 

Author: [Luigi Freda](https://www.luigifreda.com)

**pySLAM** is a *'toy'* implementation of a monocular *Visual Odometry (VO)* pipeline in Python. I released it for **educational purposes**, for a [computer vision class](https://as-ai.org/visual-perception-and-spatial-computing-12-hours/) I taught. I started developing it for fun as a python programming exercise, during my free time. I took inspiration from some python repos available on the web. 

Main Scripts:
* `main_vo.py` combines the simplest VO ingredients without performing any image point triangulation or windowed bundle adjustment. At each step $k$, `main_vo.py` estimates the current camera pose $C_k$ with respect to the previous one $C_{k-1}$. The inter frame pose estimation returns $[R_{k-1,k},t_{k-1,k}]$ with $||t_{k-1,k}||=1$. With this very basic approach, you need to use a ground truth in order to recover a correct inter-frame scale $s$ and estimate a valid trajectory by composing $C_k = C_{k-1} * [R_{k-1,k}, s t_{k-1,k}]$. This script is a first start to understand the basics of inter frame feature tracking and camera pose estimation.

* `main_slam.py` adds feature tracking along multiple frames, point triangulation and bundle adjustment in order to estimate the camera trajectory up-to-scale and build a map. It's still a VO pipeline but it shows some basic blocks which are necessary to develop a real visual SLAM pipeline. 

You can use this *'toy'* framework as a baseline to play with [features](#detectorsdescriptors), VO techniques and create your own (proof of concept) VO/SLAM pipeline in python. When you test it, please, consider that's intended as a  *'toy'* framework, without any pretence of having high localization accuracy or real-time performances. Check the terminal warnings if you see something weird happening.  

**Enjoy it!**

<center> <img src="images/main-vo.png"
alt="VO" height="300" border="1" /> 
<img src="images/main-slam-kitti-map.png"
alt="SLAM" height="300" border="1" /> </center>

--- 
## Install 

Clone this repo and its modules by running 
```
$ git clone --recursive https://github.com/luigifreda/pyslam.git
```

#### Requirements

* Python 3  (tested under Python 3.5)
* Numpy (1.18.1)
* OpenCV (>= 3.4, see [below](#how-to-install-non-free-opencv-modules) for a suggested python installation)
* PyTorch (>= 0.4, used by [SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork) and [Tfeat](https://github.com/vbalnt/tfeat) networks)

The framework has been developed and tested under Ubuntu (16.04 and 18.04). The required python3 packages can be automatically installed by running:   

`$ ./install_pip3_packages.sh`   

This allows you to run `main_vo.py`. If you want to run `main_slam.py`, you must install the libs [pangolin](https://github.com/stevenlovegrove/Pangolin) and [g2opy](https://github.com/uoip/g2opy) by running:    

`$ ./install_thirdparty.sh`   

#### How to install non-free OpenCV modules

In order to use [non-free OpenCV modules](https://stackoverflow.com/questions/50467696/pycharm-installation-of-non-free-opencv-modules-for-operations-like-sift-surf) (i.e. **SIFT** and **SURF**), you need `opencv-contrib-python`. This package can be installed by running     

`$ pip3 install opencv-contrib-python==3.4.2.16`

For a more advanced OpenCV installation procedure, you can take a look [here](https://www.pyimagesearch.com/opencv-tutorials-resources-guides/). 

How to check your installed OpenCV version:
```
$ python3 -c "import cv2; print(cv2.__version__)"
```

#### How to test under Anaconda

Please, see this [file](./CONDA.md).

#### Install Problems or Errors

If you run into install problems or run-time errors, please, check the file [TROUBLESHOOTING.md](./TROUBLESHOOTING.md).

--- 
## Usage

Once you have run the script `install_pip3_packages.sh`, you can immediately run:
```
$ python3 -O main_vo.py
```
This will process a [KITTI]((http://www.cvlibs.net/datasets/kitti/eval_odometry.php)) video (available in the folder `videos`) by using its corresponding camera calibration file (available in the folder `settings`), and its groundtruth (available in the video folder). 

**N.B.**: as explained above, the basic script `main_vo.py` **strictly requires a ground truth**.  

In order to process a different **dataset**, you need to set the file `config.ini`:
* select your dataset `type` in the section `[DATASET]` (see the section *[Datasets](#datasets)* below for further details) 
* the camera settings file accordingly (see the section *[Camera Settings](#camera-settings)* below)
* the groudtruth file accordingly (ee the section *[Datasets](#datasets)* below and check the files `ground_truth.py` and `convert_groundtruth.py` )

Once you have run the script `install_thirdparty.sh` (as required [above](#requirements)), you can test  `main_slam.py` by running:
```
$ python3 -O main_slam.py
```

You can choose any detector/descriptor among *ORB*, *SIFT*, *SURF*, *BRISK*, *AKAZE*, etc. (see the section *[Detectors/Descriptors](#detectorsdescriptors)* below for further information). 

Some basic **test files** are available in the subfolder `test`. In particular, you can start by taking a look at `test/test_feature_detector.py` and `test/test_feature_matching.py`.

**WARNING**: due to information loss in video compression, the available **KITTI videos** make `main_slam.py` tracking peform worse than with the original KITTI *image sequences*. The available videos are intended to be used for a first quick test. Please, download and use the original KITTI image sequences as explained [below](#datasets). 

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

In order to calibrate your camera, you can use the scripts in the folder `calibration`. In particular: 
1. use the script `grab_chessboard_images.py` to collect a sequence of images where the chessboard can be detected (set the chessboard size therein, you can use the calibration pattern `calib_pattern.pdf` in the same folder) 
2. use the script `calibrate.py` to process the collected images and compute the calibration parameters (set the chessboard size therein)

For further information about the calibration process, you may want to have a look [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html). 

If you want to use your camera, you have to:
* calibrate it and configure [WEBCAM.yaml](./settings/WEBCAM.yaml) accordingly
* record a video (for instance, by using `save_video.py` in the folder `calibration`)
* configure the `[VIDEO_DATASET]` section of `config.ini` in order to point to your video.

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
* *[SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)*

You can take a look at the file `feature_manager.py`. 

The following feature **descriptors** are supported: 
* *ORB*  
* *SIFT*   
* *ROOT SIFT*
* *SURF*   
* *AKAZE*   
* *BRISK*   
* *[SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)*
* *[Tfeat](https://github.com/vbalnt/tfeat)*

In both the scripts `main_vo.py` and `main_slam.py`, you can set which detector/descritor to use by means of the function *feature_tracker_factory()*. This function can be found in the file `feature_tracker.py`. Some examples (commented lines) are already present in both `main_vo.py` and `main_slam.py`.

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
## TODOs

Tons of things are still missing to attain a real SLAM pipeline: 

* keyframe generation and management 
* tracking w.r.t. previous keyframe 
* proper local map generation and management (e.g. covisibility graph)
* loop closure
* general relocalization 


---
## Credits 

* [twitchslam](https://github.com/geohot/twitchslam)
* [monoVO](https://github.com/uoip/monoVO-python)
* [pangolin](https://github.com/stevenlovegrove/Pangolin) 
* [g2opy](https://github.com/uoip/g2opy)
* [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)
* [Tfeat](https://github.com/vbalnt/tfeat)