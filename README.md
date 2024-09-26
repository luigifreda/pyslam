# pySLAM v2.2.5

Author: **[Luigi Freda](https://www.luigifreda.com)**

<!-- TOC -->

- [pySLAM v2.2.5](#pyslam-v225)
  - [Install](#install)
    - [Requirements](#requirements)
    - [Ubuntu](#ubuntu)
    - [MacOS](#macos)
    - [Docker](#docker)
    - [How to install non-free OpenCV modules](#how-to-install-non-free-opencv-modules)
    - [Troubleshooting](#troubleshooting)
  - [Usage](#usage)
    - [Save and reload a map](#save-and-reload-a-map)
    - [Trajectory saving](#trajectory-saving)
    - [GUI](#gui)
  - [Supported local features](#supported-local-features)
  - [Supported matchers](#supported-matchers)
  - [Supported global descriptors and local descriptor aggregation methods](#supported-global-descriptors-and-local-descriptor-aggregation-methods)
  - [Datasets](#datasets)
    - [KITTI Datasets](#kitti-datasets)
    - [TUM Datasets](#tum-datasets)
    - [EuRoC Dataset](#euroc-dataset)
  - [Camera Settings](#camera-settings)
  - [Comparison pySLAM vs ORB-SLAM3](#comparison-pyslam-vs-orb-slam3)
  - [Contributing to pySLAM](#contributing-to-pyslam)
  - [References](#references)
  - [Credits](#credits)
  - [TODOs](#todos)

<!-- /TOC -->

**pySLAM** is a python implementation of a *Visual Odometry (VO)* pipeline for **monocular**, **stereo** and **RGBD** cameras. 
- It supports many classical and modern **[local features](#supported-local-features)** and it offers a convenient interface for them.
- It implements loop-closing via descriptor aggregators such as visual Bag of Words (BoW), Vector of Locally Aggregated Descriptors (VLAD) and modern **[global descriptors](#supported-global-descriptors-and-local-descriptor-aggregation-methods)** (image-wise). 
- It collects other common and useful VO and SLAM tools. 

**Main Scripts**:
* `main_vo.py` combines the simplest VO ingredients without performing any image point triangulation or windowed bundle adjustment. At each step $k$, `main_vo.py` estimates the current camera pose $C_k$ with respect to the previous one $C_{k-1}$. The inter-frame pose estimation returns $[R_{k-1,k},t_{k-1,k}]$ with $\Vert t_{k-1,k} \Vert=1$. With this very basic approach, you need to use a ground truth in order to recover a correct inter-frame scale $s$ and estimate a valid trajectory by composing $C_k = C_{k-1} [R_{k-1,k}, s t_{k-1,k}]$. This script is a first start to understand the basics of inter-frame feature tracking and camera pose estimation.

* `main_slam.py` adds feature tracking along multiple frames, point triangulation, keyframe management and bundle adjustment in order to estimate the camera trajectory up-to-scale and build a map. It's still a VO pipeline but it shows some basic blocks which are necessary to develop a real visual SLAM pipeline. 

* `main_feature_matching.py` shows how to use the basic feature tracker capabilities (*feature detector* + *feature descriptor* + *feature matcher*) and allows to test the different available local features. Further details [here](./docs/basic_architecture.md).

* `main_map_viewer.py` allows to reload a saved map and visualize it.  

You can use this framework as a baseline to play with [local features](#supported-local-features), VO techniques and create your own (proof of concept) VO/SLAM pipeline in python. When you test it, consider that's a work in progress, a development framework written in Python, without any pretence of having state-of-the-art localization accuracy or real-time performances.   

**Enjoy it!**

<center> 
<img src="images/STEREO.png" alt="Visual Odometry" width="600" border="0" /> 
<img src="images/feature-matching.png" alt="Feature Matching" width="600" border="0" />  
<img src="images/RGBD2.png" alt="SLAM" width="600" border="0" />
<img src="images/main-rerun-vo-and-matching.png" alt="Feature matching and Visual Odometry" width="600" border="0" />  
</center>

--- 
## Install 

First, clone this repo and its modules by running 
```bash
$ git clone --recursive https://github.com/luigifreda/pyslam.git
$ cd pyslam 
```

Then, use the available specific install procedure according to your OS. The provided scripts will create a **single python environment** that is able to host all the [supported local features](#supported-local-features)!

- **Ubuntu**  [=>](#ubuntu)
- **MacOs** [=>](#macos) 
- **Windows** [=>](https://github.com/luigifreda/pyslam/issues/51)
- **Docker** [=>](#docker)


### Requirements

* Python 3.8.10
* OpenCV >=4.8.1 (see [below](#how-to-install-non-free-opencv-modules))
* PyTorch 2.3.1
* Tensorflow 2.13.1
* Kornia 0.7.3
* Rerun

If you run into troubles or performance issues, check this [TROUBLESHOOTING](./TROUBLESHOOTING.md) file.


### Ubuntu 

Follow the instructions reported [here](./PYTHON-VIRTUAL-ENVS.md) for creating a new virtual environment `pyslam` with **venv**.  The procedure has been tested on *Ubuntu 18.04*, *20.04*, *22.04* and *24.04*. 

If you prefer **conda**, run the scripts described in this other [file](./CONDA.md).


### MacOS

Follow the instructions in this [file](./MAC.md). The reported procedure was tested under *Sonoma 14.5* and *Xcode 15.4*.


### Docker

If you prefer docker or you have an OS that is not supported yet, you can use [rosdocker](https://github.com/luigifreda/rosdocker): 
- with its custom `pyslam` / `pyslam_cuda` docker files and follow the instructions [here](https://github.com/luigifreda/rosdocker#pyslam). 
- with one of the suggested docker images (*ubuntu\*_cuda* or *ubuntu\**), where you can build and run pyslam. 


### How to install non-free OpenCV modules

The provided install scripts take care of installing a recent opencv version (>=**4.8**) with its non-free modules enabled (see for instance [install_pip3_packages.sh](./install_pip3_packages.sh), which is used with venv under Ubuntu, or the [install_opencv_python.sh](./install_opencv_python.sh) under mac).

How to check your installed OpenCV version:      
`$ python3 -c "import cv2; print(cv2.__version__)"`

How to check if you have non-free OpenCV module support (no errors imply success):       
`$ python3 -c "import cv2; detector = cv2.xfeatures2d.SURF_create()"` 


### Troubleshooting

If you run into issues or errors during the installation process or at run-time, please, check the [TROUBLESHOOTING.md](./TROUBLESHOOTING.md) file.

--- 
## Usage

Once you have run the script `install_all_venv.sh` (follow the instructions above according to your OS), you can open a new terminal and run:
```bash
$ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is just needed once in a new terminal.
$ ./main_vo.py
```
This will process a default [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) video (available in the folder `videos`) by using its corresponding camera calibration file (available in the folder `settings`), and its groundtruth (available in the same `videos` folder). You can stop `main_vo.py` by focusing on one of the matplotlib windows and pressing the key 'Q'. 
**Note**: As explained above, the basic script `main_vo.py` **strictly requires a ground truth**.  

In order to process a different **dataset**, you need to set the file `config.yaml`:
* Select your dataset `type` in the section `DATASET` (further details in the section *[Datasets](#datasets)* below for further details). This identifies a corresponding dataset section (e.g. `KITTI_DATASET`, `TUM_DATASET`, etc). 
* Select the `sensor_type` (`mono`, `stereo`, `rgbd`) in the chosen dataset section.  
* Select the camera `settings` file in the dataset section (further details in the section *[Camera Settings](#camera-settings)* below).
* The `groudtruth_file` accordingly (further details in the section *[Datasets](#datasets)* below and check the files `ground_truth.py` and `convert_groundtruth.py`).

Similarly, you can test `main_slam.py` by running:
```bash
$ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is just needed once in a new terminal.
$ ./main_slam.py
```

This will process a default [KITTI]((http://www.cvlibs.net/datasets/kitti/eval_odometry.php)) video (available in the folder `videos`) by using its corresponding camera calibration file (available in the folder `settings`). You can stop it by focusing on the opened matplotlib window and pressing the key 'Q'. 
**Note**: Due to information loss in video compression, `main_slam.py` tracking may peform worse with the available KITTI videos than with the original KITTI image sequences. The available videos are intended to be used for a first quick test. Please, download and use the original KITTI image sequences as explained [below](#datasets).

If you just want to test the basic feature tracker capabilities (*feature detector* + *feature descriptor* + *feature matcher*) and get a tast of the different available local features, run
```bash
$ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is just needed once in a new terminal.
$ ./main_feature_matching.py
```

In any of the above scripts, you can choose any detector/descriptor among *ORB*, *SIFT*, *SURF*, *BRISK*, *AKAZE*, *SuperPoint*, etc. (see the section *[Supported Local Features](#supported-local-features)* below for further information). 

Some basic **test/example files** are available in the subfolder `test`. In particular, as for feature detection/description, you may want to take a look at [test/cv/test_feature_manager.py](./test/cv/test_feature_manager.py) too.

### Save and reload a map

When you run the script `main_slam.py`:
- The current map can be saved into the file `map.json` by pressing the button `Save` on the GUI. 
- The saved map can be reloaded and visualized into the GUI by running: 
```bash
$ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is just needed once in a new terminal.
$ ./main_map_viewer.py
```
Relocalization in a loaded map is a WIP.

### Trajectory saving

Estimated trajectories can be saved in three different formats: *TUM* (The Open Mapping format), *KITTI* (KITTI Odometry format), and *EuRoC* (EuRoC MAV format). To enable trajectory saving, open `config.yaml` and search for the `SAVE_TRAJECTORY`: set `save_trajectory: True`, select your `format_type` (`tum`, `kitti`, `euroc`), and the output filename. For instance for a `tum` format output:   
```
SAVE_TRAJECTORY:
  save_trajectory: True
  format_type: tum
  filename: kitti00_trajectory.txt
```

### GUI 

Some quick information about the non-trivial GUI buttons of `main_slam.py`: 
- `Step`: Enter in the *Step by step mode*. Press the button `Step` a first time to pause. Then, press it again to make the pipeline process a single new frame.
- `Save`: Save the map into the file `map.json`. You can visualize it back by using the script `/main_map_viewer.py` (as explained above). 
- `Draw GT`: In the case a groundtruth is loaded (e.g. with *KITTI*, *TUM*, *EUROC* datasets), you can visualize it by pressing this button. The groundtruth trajectory will be visualized and progressively aligned to the estimated trajectory. 

---
## Supported local features

At present time, the following feature **detectors** are supported: 
* *[FAST](https://www.edwardrosten.com/work/fast.html)*  
* *[Good features to track](https://ieeexplore.ieee.org/document/323794)* 
* *[ORB](http://www.willowgarage.com/sites/default/files/orb_final.pdf)*  
* *[ORB2](https://github.com/raulmur/ORB_SLAM2)* (improvements of ORB-SLAM2 to ORB detector) 
* *[SIFT](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)*   
* *[SURF](http://people.ee.ethz.ch/~surf/eccv06.pdf)*   
* *[KAZE](https://www.doc.ic.ac.uk/~ajd/Publications/alcantarilla_etal_eccv2012.pdf)*
* *[AKAZE](http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf)* 
* *[BRISK](http://www.margaritachli.com/papers/ICCV2011paper.pdf)*  
* *[AGAST](http://www.i6.in.tum.de/Main/ResearchAgast)*
* *[MSER](http://cmp.felk.cvut.cz/~matas/papers/matas-bmvc02.pdf)*
* *[StarDector/CenSurE](https://link.springer.com/content/pdf/10.1007%2F978-3-540-88693-8_8.pdf)*
* *[Harris-Laplace](https://www.robots.ox.ac.uk/~vgg/research/affine/det_eval_files/mikolajczyk_ijcv2004.pdf)* 
* *[SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)*
* *[D2-Net](https://github.com/mihaidusmanu/d2-net)*
* *[DELF](https://github.com/tensorflow/models/tree/master/research/delf)*
* *[Contextdesc](https://github.com/lzx551402/contextdesc)*
* *[LFNet](https://github.com/vcg-uvic/lf-net-release)*
* *[R2D2](https://github.com/naver/r2d2)*
* *[Key.Net](https://github.com/axelBarroso/Key.Net)*
* *[DISK](https://arxiv.org/abs/2006.13566)*
* *[ALIKED](https://arxiv.org/abs/2304.03608)*
* *[Xfeat](https://arxiv.org/abs/2404.19174)*
* *[KeyNetAffNetHardNet](https://github.com/axelBarroso/Key.Net)* (KeyNet detector + AffNet + HardNet descriptor).

The following feature **descriptors** are supported: 
* *[ORB](http://www.willowgarage.com/sites/default/files/orb_final.pdf)*  
* *[SIFT](https://www.cs.ubc.ca/~lowe/papers/iccv99.pdf)*
* *[ROOT SIFT](https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf)*
* *[SURF](http://people.ee.ethz.ch/~surf/eccv06.pdf)*    
* *[AKAZE](http://www.bmva.org/bmvc/2013/Papers/paper0013/paper0013.pdf)* 
* *[BRISK](http://www.margaritachli.com/papers/ICCV2011paper.pdf)*     
* *[FREAK](https://www.researchgate.net/publication/258848394_FREAK_Fast_retina_keypoint)* 
* *[SuperPoint](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)*
* *[Tfeat](https://github.com/vbalnt/tfeat)*
* *[BOOST_DESC](https://www.labri.fr/perso/vlepetit/pubs/trzcinski_pami15.pdf)*
* *[DAISY](https://ieeexplore.ieee.org/document/4815264)*
* *[LATCH](https://arxiv.org/abs/1501.03719)*
* *[LUCID](https://pdfs.semanticscholar.org/85bd/560cdcbd4f3c24a43678284f485eb2d712d7.pdf)*
* *[VGG](https://www.robots.ox.ac.uk/~vedaldi/assets/pubs/simonyan14learning.pdf)*
* *[Hardnet](https://github.com/DagnyT/hardnet.git)*
* *[GeoDesc](https://github.com/lzx551402/geodesc.git)*
* *[SOSNet](https://github.com/yuruntian/SOSNet.git)*
* *[L2Net](https://github.com/yuruntian/L2-Net)*
* *[Log-polar descriptor](https://github.com/cvlab-epfl/log-polar-descriptors)*
* *[D2-Net](https://github.com/mihaidusmanu/d2-net)*
* *[DELF](https://github.com/tensorflow/models/tree/master/research/delf)*
* *[Contextdesc](https://github.com/lzx551402/contextdesc)*
* *[LFNet](https://github.com/vcg-uvic/lf-net-release)*
* *[R2D2](https://github.com/naver/r2d2)*
* *[BEBLID](https://raw.githubusercontent.com/iago-suarez/BEBLID/master/BEBLID_Boosted_Efficient_Binary_Local_Image_Descriptor.pdf)*
* *[DISK](https://arxiv.org/abs/2006.13566)*
* *[ALIKED](https://arxiv.org/abs/2304.03608)*
* *[Xfeat](https://arxiv.org/abs/2404.19174)*
* *[KeyNetAffNetHardNet](https://github.com/axelBarroso/Key.Net)* (KeyNet detector + AffNet + HardNet descriptor).
  
You can find further information in the file [feature_types.py](./feature_types.py). Some of the local features consist of a *joint detector-descriptor*. You can start playing with the supported local features by taking a look at `test/cv/test_feature_manager.py` and `main_feature_matching.py`.

In both the scripts `main_vo.py` and `main_slam.py`, you can create your favourite detector-descritor configuration and feed it to the function `feature_tracker_factory()`. Some ready-to-use configurations are already available in the file [feature_tracker.configs.py](./feature_tracker_configs.py)

The function `feature_tracker_factory()` can be found in the file `feature_tracker.py`. Take a look at the file `feature_manager.py` for further details.

**N.B.**: you just need a *single* python environment to be able to work with all the [supported local features](#supported-local-features)!

---
## Supported matchers 

* *BF*: Brute force matcher on descriptors (with KNN)
* *[FLANN](https://www.semanticscholar.org/paper/Fast-Approximate-Nearest-Neighbors-with-Automatic-Muja-Lowe/35d81066cb1369acf4b6c5117fcbb862be2af350)* 
* *[XFeat](https://arxiv.org/abs/2404.19174)*      
* *[LightGlue](https://arxiv.org/abs/2306.13643)*
* *[LoFTR](https://arxiv.org/abs/2104.00680)*
  

---
## Supported global descriptors and local descriptor aggregation methods

**Local descriptor aggregation methods**

* Bag of Words (BoW) with TF-IDF: [DBoW2](https://github.com/dorian3d/DBoW2), [DBoW3](https://github.com/rmsalinas/DBow3)
* Vector of Locally Aggregated Descriptors: [VLAD](http://www.vlfeat.org/) 
* Incremental Bags of Binary Words (iBoW) via Online Binary Image Index: [iBoW](https://github.com/emiliofidalgo/ibow-lcd), [OBIndex2](https://github.com/emiliofidalgo/obindex2)
* Hyperdimensional Computing: [HDC](https://www.tu-chemnitz.de/etit/proaut/hdc_desc)


**Global descriptors**

* [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
* [NetVLAD](https://www.di.ens.fr/willow/research/netvlad/)
* [HDC-DELF](https://www.tu-chemnitz.de/etit/proaut/hdc_desc)
* [CosPlace](https://github.com/gmberton/CosPlace)
* [EigenPlaces](https://github.com/gmberton/EigenPlaces)


--- 
## Datasets

You can use 5 different types of datasets:

Dataset | type in `config.yaml`
--- | --- 
[KITTI odometry data set (grayscale, 22 GB)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)  | `type: KITTI_DATASET` 
[TUM dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)                   | `type: TUM_DATASET` 
[EUROC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)                   | `type: EUROC_DATASET` 
Video file        | `type: VIDEO_DATASET` 
Folder of images  | `type: FOLDER_DATASET` 

### KITTI Datasets

pySLAM code expects the following structure in the specified KITTI path folder (specified in the section `KITTI_DATASET` of the file `config.yaml`). : 
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

2. Select the corresponding calibration settings file (parameter `[KITTI_DATASET][cam_settings]` in the file `config.yaml`)



### TUM Datasets 

pySLAM code expects a file `associations.txt` in each TUM dataset folder (specified in the section `TUM_DATASET:` of the file `config.yaml`). 

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.

2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). You can generate your `associations.txt` file by executing:

```
$ python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```
3. Select the corresponding calibration settings file (parameter `TUM_DATASET: cam_settings:` in the file `config.yaml`)


### EuRoC Dataset

1. Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets (check this direct [link](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/))

2. Select the corresponding calibration settings file (parameter `EUROC_DATASET: cam_settings:` in the file `config.yaml`)

--- 
## Camera Settings

The folder `settings` contains the camera settings files which can be used for testing the code. These are the same used in the framework [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). You can easily modify one of those files for creating your own new calibration file (for your new datasets).

In order to calibrate your camera, you can use the scripts in the folder `calibration`. In particular: 
1. use the script `grab_chessboard_images.py` to collect a sequence of images where the chessboard can be detected (set the chessboard size therein, you can use the calibration pattern `calib_pattern.pdf` in the same folder) 
2. use the script `calibrate.py` to process the collected images and compute the calibration parameters (set the chessboard size therein)

For further information about the calibration process, you may want to have a look [here](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html). 

If you want to **use your camera**, you have to:
* calibrate it and configure [WEBCAM.yaml](./settings/WEBCAM.yaml) accordingly
* record a video (for instance, by using `save_video.py` in the folder `calibration`)
* configure the `VIDEO_DATASET` section of `config.yaml` in order to point to your recorded video.

--- 
## Comparison pySLAM vs ORB-SLAM3

For a comparison of the trajectories estimated by pySLAM and by ORB-SLAM3, see this [trajectory comparison notebook](https://github.com/anathonic/Trajectory-Comparison-ORB-SLAM3-pySLAM/blob/main/trajectories_comparison.ipynb). 

Note that pySLAM pose estimates are saved **online**: At each frame, the current pose estimate is saved. On the other end, ORB-SLAM3 pose estimates are saved at the end of the dataset playback: That means each pose estimate $q$ is refined multiple times by LBA and BA over the multiple window optimizations that cover $q$.  

--- 
## Contributing to pySLAM

If you like pySLAM and would like to contribute to the code base, you can report bugs, leave comments and proposing new features through issues and pull requests on github. Feel free to get in touch at *luigifreda(at)gmail[dot]com*. Thank you!

--- 
## References

Suggested books:
* *[Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)* by Richard Hartley and Andrew Zisserman
* *[An Invitation to 3-D Vision](https://link.springer.com/book/10.1007/978-0-387-21779-6)* by Yi-Ma, Stefano Soatto, Jana Kosecka, S. Shankar Sastry 
* *[Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)*, by Richard Szeliski 
* *[Deep Learning](http://www.deeplearningbook.org/lecture_slides.html)*, by Ian Goodfellow, Yoshua Bengio and Aaron Courville
* *[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)*, By Michael Nielsen 

Suggested material:
* *[Vision Algorithms for Mobile Robotics](http://rpg.ifi.uzh.ch/teaching.html)* by Davide Scaramuzza 
* *[CS 682 Computer Vision](http://cs.gmu.edu/~kosecka/cs682.html)* by Jana Kosecka   
* *[ORB-SLAM: a Versatile and Accurate Monocular SLAM System](http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf)* by R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos
* *[Double Window Optimisation for Constant Time Visual SLAM](http://hauke.strasdat.net/files/strasdat2011iccv.pdf)* by H. Strasdat, A. J. Davison
J.M.M. Montielb, K. Konolige
* *[The Role of Wide Baseline Stereo in the Deep Learning World](https://ducha-aiki.github.io/wide-baseline-stereo-blog/2020/03/27/intro.html)* by Dmytro Mishkin
* *[To Learn or Not to Learn: Visual Localization from Essential Matrices](https://arxiv.org/abs/1908.01293)* by Qunjie Zhou, Torsten Sattler, Marc Pollefeys, Laura Leal-Taixe
* *[Awesome local-global descriptors](https://github.com/shamangary/awesome-local-global-descriptor)* repository 
* *[Introduction to Feature Matching Using Neural Networks](https://learnopencv.com/feature-matching/)*

Moreover, you may want to have a look at the OpenCV [guide](https://docs.opencv.org/3.0-beta/modules/calib3d/doc/camera_calibration_and_3d_reconstruction.html) or [tutorials](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html).  

---
## Credits 

* [Pangolin](https://github.com/stevenlovegrove/Pangolin) 
* [g2opy](https://github.com/uoip/g2opy)
* [ORBSLAM2](https://github.com/raulmur/ORB_SLAM2)
* [SuperPointPretrainedNetwork](https://github.com/MagicLeapResearch/SuperPointPretrainedNetwork)
* [Tfeat](https://github.com/vbalnt/tfeat)
* [Image Matching Benchmark Baselines](https://github.com/vcg-uvic/image-matching-benchmark-baselines)
* [Hardnet](https://github.com/DagnyT/hardnet.git)
* [GeoDesc](https://github.com/lzx551402/geodesc.git)
* [SOSNet](https://github.com/yuruntian/SOSNet.git)
* [L2Net](https://github.com/yuruntian/L2-Net)
* [Log-polar descriptor](https://github.com/cvlab-epfl/log-polar-descriptors)
* [D2-Net](https://github.com/mihaidusmanu/d2-net)
* [DELF](https://github.com/tensorflow/models/blob/master/research/delf/INSTALL_INSTRUCTIONS.md)
* [Contextdesc](https://github.com/lzx551402/contextdesc)
* [LFNet](https://github.com/vcg-uvic/lf-net-release)
* [R2D2](https://github.com/naver/r2d2)
* [BEBLID](https://raw.githubusercontent.com/iago-suarez/BEBLID/master/BEBLID_Boosted_Efficient_Binary_Local_Image_Descriptor.pdf)
* [DISK](https://arxiv.org/abs/2006.13566)
* [Xfeat](https://arxiv.org/abs/2404.19174)
* [LightGlue](https://arxiv.org/abs/2306.13643)
* [Key.Net](https://github.com/axelBarroso/Key.Net)
* [Twitchslam](https://github.com/geohot/twitchslam)
* [MonoVO](https://github.com/uoip/monoVO-python)
* [VPR_Tutorial](https://github.com/stschubert/VPR_Tutorial.git)
* Many thanks to [Anathonic](https://github.com/anathonic) for adding the trajectory-saving feature and for the comparison notebook:  [pySLAM vs ORB-SLAM3](https://github.com/anathonic/Trajectory-Comparison-ORB-SLAM3-pySLAM/blob/main/trajectories_comparison.ipynb).

---
## TODOs

Many improvements and additional features are currently under development: 

- [ ] loop closing
- [ ] relocalization 
- [x] stereo and RGBD support
- [x] map saving/loading 
- [x] modern DL matching algorithms 
- [ ] object detection and semantic segmentation 
- [ ] 3D dense reconstruction 
- [x] unified install procedure (single branch) for all OSs 
- [x] trajectory saving 

