# pySLAM v2.6.0

Author: **[Luigi Freda](https://www.luigifreda.com)**

<!-- TOC -->

- [pySLAM v2.6.0](#pyslam-v260)
  - [Install](#install)
    - [Main requirements](#main-requirements)
    - [Ubuntu](#ubuntu)
    - [MacOS](#macos)
    - [Docker](#docker)
    - [How to install non-free OpenCV modules](#how-to-install-non-free-opencv-modules)
  - [Troubleshooting and performance issues](#troubleshooting-and-performance-issues)
  - [Usage](#usage)
    - [Visual odometry](#visual-odometry)
    - [Full SLAM](#full-slam)
    - [Selecting a dataset and different configuration parameters](#selecting-a-dataset-and-different-configuration-parameters)
    - [Feature tracking](#feature-tracking)
    - [Loop closing](#loop-closing)
      - [Vocabulary management](#vocabulary-management)
      - [Vocabulary-free loop closing](#vocabulary-free-loop-closing)
      - [Double-check your loop detection configuration and verify vocabulary compability](#double-check-your-loop-detection-configuration-and-verify-vocabulary-compability)
        - [Loop detection method based on a pre-trained vocabulary](#loop-detection-method-based-on-a-pre-trained-vocabulary)
        - [Missing vocabulary for the selected front-end descriptor type](#missing-vocabulary-for-the-selected-front-end-descriptor-type)
    - [Volumetric reconstruction](#volumetric-reconstruction)
      - [Dense reconstruction while running SLAM](#dense-reconstruction-while-running-slam)
      - [Reload a saved sparse map and perform dense reconstruction](#reload-a-saved-sparse-map-and-perform-dense-reconstruction)
      - [Reload and check your dense reconstruction](#reload-and-check-your-dense-reconstruction)
      - [Controlling the spatial distribution of keyframe FOV centers](#controlling-the-spatial-distribution-of-keyframe-fov-centers)
    - [Depth prediction](#depth-prediction)
    - [Saving and reloading](#saving-and-reloading)
      - [Save the a map](#save-the-a-map)
      - [Reload a saved map and relocalize in it](#reload-a-saved-map-and-relocalize-in-it)
      - [Trajectory saving](#trajectory-saving)
    - [SLAM GUI](#slam-gui)
    - [Monitor the logs for tracking, local mapping, and loop closing simultaneously](#monitor-the-logs-for-tracking-local-mapping-and-loop-closing-simultaneously)
  - [System overview](#system-overview)
  - [Supported components and models](#supported-components-and-models)
    - [Supported local features](#supported-local-features)
    - [Supported matchers](#supported-matchers)
    - [Supported global descriptors and local descriptor aggregation methods](#supported-global-descriptors-and-local-descriptor-aggregation-methods)
        - [Local descriptor aggregation methods](#local-descriptor-aggregation-methods)
        - [Global descriptors](#global-descriptors)
    - [Supported depth prediction models](#supported-depth-prediction-models)
    - [Supported volumetric mapping methods](#supported-volumetric-mapping-methods)
  - [Datasets](#datasets)
    - [KITTI Datasets](#kitti-datasets)
    - [TUM Datasets](#tum-datasets)
    - [EuRoC Datasets](#euroc-datasets)
    - [Replica Datasets](#replica-datasets)
  - [Camera Settings](#camera-settings)
  - [Comparison pySLAM vs ORB-SLAM3](#comparison-pyslam-vs-orb-slam3)
  - [Contributing to pySLAM](#contributing-to-pyslam)
  - [References](#references)
  - [Credits](#credits)
  - [License](#license)
  - [TODOs](#todos)

<!-- /TOC -->
 
**pySLAM** is a python implementation of a *Visual SLAM* pipeline that supports **monocular**, **stereo** and **RGBD** cameras. It provides the following **features**:
- A wide range of classical and modern **[local features](#supported-local-features)** with a convenient interface for their integration.
- Various loop closing methods, including **[descriptor aggregators](#supported-global-descriptors-and-local-descriptor-aggregation-methods)** such as visual Bag of Words (BoW, iBow), Vector of Locally Aggregated Descriptors (VLAD) and modern **[global descriptors](#supported-global-descriptors-and-local-descriptor-aggregation-methods)** (image-wise descriptors).
- A **[volumetric reconstruction pipeline](#volumetric-reconstruction)** that processes available depth and color images with volumetric integration and provides an output dense reconstruction. This can use **TSDF** with voxel hashing or incremental **Gaussian Splatting**. 
- Integration of **[depth prediction models](#depth-prediction)** within the SLAM pipeline. These include DepthPro, DepthAnythingV2, RAFT-Stereo, CREStereo, etc.  
- A collection of other useful tools for VO and SLAM.

**Main Scripts**
* `main_vo.py` combines the simplest VO ingredients without performing any image point triangulation or windowed bundle adjustment. At each step $k$, `main_vo.py` estimates the current camera pose $C_k$ with respect to the previous one $C_{k-1}$. The inter-frame pose estimation returns $[R_{k-1,k},t_{k-1,k}]$ with $\Vert t_{k-1,k} \Vert=1$. With this very basic approach, you need to use a ground truth in order to recover a correct inter-frame scale $s$ and estimate a valid trajectory by composing $C_k = C_{k-1} [R_{k-1,k}, s t_{k-1,k}]$. This script is a first start to understand the basics of inter-frame feature tracking and camera pose estimation.

* `main_slam.py` adds feature tracking along multiple frames, point triangulation, keyframe management, bundle adjustment, loop closing, dense mapping and depth inference in order to estimate the camera trajectory and build both a sparse and dense map. It's a full SLAM pipeline and includes all the basic and advanced blocks which are necessary to develop a real visual SLAM pipeline.

* `main_feature_matching.py` shows how to use the basic feature tracker capabilities (*feature detector* + *feature descriptor* + *feature matcher*) and allows to test the different available local features. 

* `main_depth_prediction.py` shows how to use the available depth inference models to get depth estimations from input color images.
  
* `main_map_viewer.py` reloads a saved map and visualizes it. Further details on how to save a map [here](#reload-a-saved-map-and-relocalize-in-it).

* `main_map_dense_reconstruction.py` reloads a saved map and uses a configured volumetric integrator to obtain a dense reconstruction (see [here](#volumetric-reconstruction)). 

**System overview**      
[Here](./docs/system_overview.md) you can find a couple of diagram sketches that provide an overview of the main SLAM workflow, system components, and classes relationships/dependencies.

pySLAM can be used as flexible baseline framework to experiment with VO/SLAM techniques, *[local features](#supported-local-features)*, *[descriptor aggregators](#supported-global-descriptors-and-local-descriptor-aggregation-methods)*, *[global descriptors](#supported-global-descriptors-and-local-descriptor-aggregation-methods)*, *[volumetric integration](#volumetric-reconstruction-pipeline)* and *[depth prediction](#depth-prediction)*. It allows to explore, prototype and develop VO/SLAM pipelines. Users should note that pySLAM is a research framework and a work in progress. It is not optimized for real-time performances.   

**Enjoy it!**

<p align="center" style="margin:0">
<img src="./images/feature-matching.png" alt="Feature Matching" height="160" border="0" /> <img src="./images/main-feature-matching.png" alt="Feature matching and Visual Odometry" height="160" border="0" /> <img src="./images/main-vo-rerun.png" alt="Feature matching and Visual Odometry" height="160" border="0" /> 
<img src="./images/STEREO.png" alt="Visual Odometry" height="160" border="0" /> <img src="./images/RGBD2.png" alt="SLAM" height="160" border="0" /> <img src="images/kitti-stereo.png" alt="Stereo SLAM" height="160" border="0" /> 
<img src="./images/loop-detection2.png" alt="Loop detection" height="160" border="0" /> <img src="./images/depth-prediction.png" alt="Depth Prediction" height="160" border="0" /> 
<img src="./images/dense-reconstruction-with-depth-prediction.png" alt="Dense Reconstruction - Stereo Depth Prediction - Kitti" height="160" border="0" /> <img src="./images/dense-reconstruction2.png" alt="Dense Reconstruction" height="160" border="0" /> 
</p>

<p align="center" style="margin:0">
<img src="./images/dense-reconstruction-composition.gif" alt="Dense Reconstruction - Stereo Depth Prediction - Kitti" width="600" border="0" /> 
</p>

<!-- <img src="./images/dense-reconstruction-euroc-width-depth-prediction.png" alt="Dense Reconstruction - Stereo Depth Prediction - Euroc" height="160" border="0" /> <img src="./images/dense-reconstruction-euroc-with-depth-prediction-gsm.png" alt="Dense Reconstruction - Gaussian Splatting" height="160" border="0" /> <img src="./images/dense-reconstruction-tum-gsm.png" alt="Dense Reconstruction - Stereo Depth Prediction - Euroc" height="160" border="0" />  -->

--- 
## Install 

First, clone this repo and its submodules by running 
```bash
$ git clone --recursive https://github.com/luigifreda/pyslam.git
$ cd pyslam 
```

Then, under **Ubuntu** and **MacOs** you can simply run:
```bash
$ ./install_all.sh
```
This install scripts creates a **single python environment** `pyslam` that hosts all the [supported components and models](#supported-components-and-models). If `conda` is available, it automatically uses it, otherwise it installs and uses `venv`. 

Refer to these links for further details about the specific install procedures that are supported.
- **Ubuntu**  [=>](#ubuntu)
- **MacOs** [=>](#macos)  
- **Windows** [=>](https://github.com/luigifreda/pyslam/issues/51)
- **Docker** [=>](#docker)

Once everything is completed you can jump the [usage section](#usage).

### Main requirements

* Python **3.10.12**
* OpenCV >=4.10 (see [below](#how-to-install-non-free-opencv-modules))
* PyTorch >=2.3.1
* Tensorflow >=2.13.1
* Kornia >=0.7.3
* Rerun
* You need **CUDA** in order to run Gaussian splatting and dust3r-based methods. Check you have installed a suitable version of **cuda toolkit**. You can check it with `./cuda_config.sh` 

The internal pySLAM libraries are imported by using a `Config` instance (from [config.py](config.py)) in the main/test scripts. 

If you encounter any issues or performance problems, refer to the [TROUBLESHOOTING](./docs/TROUBLESHOOTING.md) file for assistance.


### Ubuntu 

- With **venv**: Follow the instructions reported [here](./docs/PYTHON-VIRTUAL-ENVS.md).  The procedure has been tested on *Ubuntu 18.04*, *20.04*, *22.04* and *24.04*. 
- With **conda**: Run the procedure described in this other [file](./docs/CONDA.md).

Both procedures will create a new virtual environment `pyslam`.

### MacOS

Follow the instructions in this [file](./docs/MAC.md). The reported procedure was tested under *Sequoia 15.1.1* and *Xcode 16.1*.


### Docker

If you prefer docker or you have an OS that is not supported yet, you can use [rosdocker](https://github.com/luigifreda/rosdocker): 
- with its custom `pyslam` / `pyslam_cuda` docker files and follow the instructions [here](https://github.com/luigifreda/rosdocker#pyslam). 
- with one of the suggested docker images (*ubuntu\*_cuda* or *ubuntu\**), where you can build and run pyslam. 


### How to install non-free OpenCV modules

The provided install scripts will install a recent opencv version (>=**4.10**) with non-free modules enabled (see the provided scripts [install_pip3_packages.sh](./install_pip3_packages.sh) and [install_opencv_python.sh](./install_opencv_python.sh)). To quickly verify your installed opencv version run:
```bash       
$ . pyenv-activate.sh     #  Activate pyslam python virtual environment. This is only needed once in a new terminal.
$ ./scripts/opencv_check.py
```
Otherwise, run the following commands: 
```bash       
$ python3 -c "import cv2; print(cv2.__version__)" # check opencv version               
$ python3 -c "import cv2; detector = cv2.xfeatures2d.SURF_create()"  # check if you have non-free OpenCV module support (no errors imply success)
```

---

## Troubleshooting and performance issues

If you run into issues or errors during the installation process or at run-time, please, check the [docs/TROUBLESHOOTING.md](./docs/TROUBLESHOOTING.md) file. Before submitting a new git issue please read [here](docs/TROUBLESHOOTING.md#submitting-a-git-issue).

--- 
## Usage

Open a new terminal and start experimenting with the scripts. In each new terminal you are supposed to start with this command:
```bash
$ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is only needed once in a new terminal.
```
The file [config.yaml](./config.yaml) can be used as a unique entry-point to configure the system and its global configuration parameters contained in [config_parameters.py](./config_parameters.py). Further information on how to configure pySLAM are provided [here](#selecting-a-dataset-and-different-configuration-parameters).

--- 

### Visual odometry

The basic **Visual Odometry** (VO) can be run with the following commands:
```bash
$ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is only needed once in a new terminal.
$ ./main_vo.py
```
By default, this processes a [KITTI](http://www.cvlibs.net/datasets/kitti/eval_odometry.php) video (available in the folder `data/videos`) by using its corresponding camera calibration file (available in the folder `settings`), and its groundtruth (available in the same `data/videos` folder). If matplotlib windows are used, you can stop `main_vo.py` by focusing/clicking on one of them and pressing the key 'Q'. As explained above, this very *basic* script `main_vo.py` **strictly requires a ground truth**. 
Now, with RGBD datasets, you can also test the **RGBD odometry** with the classes `VisualOdometryRgbd` or `VisualOdometryRgbdTensor` (ground truth is not required here). 

--- 

### Full SLAM

Similarly, you can test the **full SLAM** by running `main_slam.py`:
```bash
$ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is only needed once in a new terminal.
$ ./main_slam.py
```

This will process the same default [KITTI]((http://www.cvlibs.net/datasets/kitti/eval_odometry.php)) video (available in the folder `data/videos`) by using its corresponding camera calibration file (available in the folder `settings`). You can stop it by focusing/clicking on one of the opened windows and pressing the key 'Q' or closing the 3D pangolin GUI. 
<!-- **Note**: Due to information loss in video compression, `main_slam.py` tracking may peform worse with the available KITTI videos than with the original KITTI image sequences. The available videos are intended to be used for a first quick test. Please, download and use the original KITTI image sequences as explained [below](#datasets). -->

--- 

### Selecting a dataset and different configuration parameters

The file [config.yaml](./config.yaml) can be used as a unique entry-point to configure the system and its global configuration parameters contained in [config_parameters.py](./config_parameters.py). 

To process a different **dataset** with both VO and SLAM scripts, you need to update the file [config.yaml](./config.yaml):
* Select your dataset `type` in the section `DATASET` (further details in the section *[Datasets](#datasets)* below for further details). This identifies a corresponding dataset section (e.g. `KITTI_DATASET`, `TUM_DATASET`, etc). 
* Select the `sensor_type` (`mono`, `stereo`, `rgbd`) in the chosen dataset section.  
* Select the camera `settings` file in the dataset section (further details in the section *[Camera Settings](#camera-settings)* below).
* The `groudtruth_file` accordingly (further details in the section *[Datasets](#datasets)* below and check the files `io/ground_truth.py` and `io/convert_groundtruth.py`).

You can use the section `GLOBAL_PARAMETERS` of the file [config.yaml](./config.yaml) to override the parameters in [config_parameters.py](./config_parameters.py). 

---

### Feature tracking

If you just want to test the basic feature tracking capabilities (*feature detector* + *feature descriptor* + *feature matcher*) and get a taste of the different available local features, run
```bash
$ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is only needed once in a new terminal.
$ ./main_feature_matching.py
```

In any of the above scripts, you can choose any detector/descriptor among *ORB*, *SIFT*, *SURF*, *BRISK*, *AKAZE*, *SuperPoint*, etc. (see the section *[Supported Local Features](#supported-local-features)* below for further information). 

Some basic examples are available in the subfolder `test/loopclosing`. In particular, as for feature detection/description, you may want to take a look at [test/cv/test_feature_manager.py](./test/cv/test_feature_manager.py) too.

---

### Loop closing

Many [loop closing methods](#loop-closing) are available, combining different [aggregation methods](#local-descriptor-aggregation-methods) and [global descriptors](#global-descriptors).

While running full SLAM, loop closing is enabled by default and can be disabled by setting `kUseLoopClosing=False` in `config_parameters.py`. Different configuration options `LoopDetectorConfigs` can be found in [loop_closing/loop_detector_configs.py](loop_closing/loop_detector_configs.py): Code comments provide additional useful details.

One can start experimenting with loop closing methods by using the examples in `test/loopclosing`. The example [test/loopclosing/test_loop_detector.py](./test/loopclosing/test_loop_detector.py) is the recommended entry point.


#### Vocabulary management 

`DBoW2`, `DBoW3`, and `VLAD` require **pre-trained vocabularies**. ORB-based vocabularies are automatically downloaded in the `data` folder (see [loop_closing/loop_detector_configs.py](loop_closing/loop_detector_configs.py)).

To create a new vocabulary, follow these steps:

1. **Generate an array of descriptors**: Use the script `test/loopclosing/test_gen_des_array_from_imgs.py` to generate the array of descriptors that will be used to train the new vocabulary. Select your desired descriptor type via the tracker configuration. 

2.  **DBOW vocabulary generation**: Train your target DBOW vocabulary by using the script `test/loopclosing/test_gen_dbow_voc_from_des_array.py`.

3. **VLAD vocabulary generation**: Train your target VLAD "vocabulary" by using the script `test/loopclosing/test_gen_vlad_voc_from_des_array.py`.

Once you have trained the vocabulary, you can add it in [loop_closing/loop_detector_vocabulary.py](loop_closing/loop_detector_vocabulary.py) and correspondingly create a new loop detector configuration in [loop_closing/loop_detector_configs.py](loop_closing/loop_detector_configs.py) that uses it.

#### Vocabulary-free loop closing

Most methods do not require pre-trained vocabularies. Specifically:
- `iBoW` and `OBindex2`: These methods incrementally build bags of binary words and, if needed, convert (front-end) non-binary descriptors into binary ones. 
- Others: Methods like `HDC_DELF`, `SAD`, `AlexNet`, `NetVLAD`, `CosPlace`, and `EigenPlaces` directly extract their specific **global descriptors** and process them using dedicated aggregators, independently from the used front-end descriptors.

As mentioned above, only `DBoW2`, `DBoW3`, and `VLAD` require pre-trained vocabularies.

#### Double-check your loop detection configuration and verify vocabulary compability

##### Loop detection method based on a pre-trained vocabulary

When selecting a **loop detection method based on a pre-trained vocabulary** (such as `DBoW2`, `DBoW3`, and `VLAD`), ensure the following:
1. The back-end and the front-end are using the same descriptor type (this is also automatically checked for consistency) or their descriptor managers are independent (see further details in the configuration options `LoopDetectorConfigs` available in [loop_closing/loop_detector_configs.py](loop_closing/loop_detector_configs.py)).
2. A corresponding pre-trained vocubulary is available. For more details, refer to the [vocabulary management section](#vocabulary-management).

##### Missing vocabulary for the selected front-end descriptor type

If you lack a compatible vocabulary for the selected front-end descriptor type, you can follow one of these options:     
1. Create and load the vocabulary (refer to the [vocabulary management section](#vocabulary-management)).     
2. Choose an `*_INDEPENDENT` loop detector method, which works with an independent local_feature_manager.     
3. Select a vocabular-free loop closing method.      
   
See the file [loop_closing/loop_detector_configs.py](loop_closing/loop_detector_configs.py) for further details.

---

### Volumetric reconstruction

#### Dense reconstruction while running SLAM 

The SLAM back-end hosts a volumetric reconstruction pipeline. This is disabled by default. You can enable it by setting `kUseVolumetricIntegration=True` and selecting your preferred method `kVolumetricIntegrationType` in `config_parameters.py`. At present, two methods are available: `TSDF` and `GAUSSIAN_SPLATTING` (see [dense/volumetric_integrator_factory.py](dense/volumetric_integrator_factory.py)). Note that you need CUDA in order to run `GAUSSIAN_SPLATTING` method.

At present, the volumetric reconstruction pipeline works with:
- RGBD datasets 
- When a [depth estimator](#depth-prediction) is used
  * in the back-end with STEREO datasets (you can't use depth prediction in the back-end with MONOCULAR datasets, further details [here](#depth-prediction))
  * in the front-end (to emulate an RGBD sensor) and a depth prediction/estimation gets available for each processed keyframe. 

If you want a mesh as output then set `kVolumetricIntegrationExtractMesh=True` in `config_parameters.py`.

#### Reload a saved sparse map and perform dense reconstruction 

Use the script `main_map_dense_reconstruction.py` to reload a saved sparse map and to perform dense reconstruction by using its posed keyframes as input. You can select your preferred dense reconstruction method directly in the script. 

- To check what the volumetric integrator is doing, run in another shell `tail -f logs/volumetric_integrator.log` (from repository root folder).
- To save the obtained dense and sparse maps, press the `Save` button on the GUI. 

#### Reload and check your dense reconstruction 

You can check the output pointcloud/mesh by using [CloudCompare](https://www.cloudcompare.org/). 

In the case of a saved Gaussian splatting model, you can visualize it by:
1. Using the [superslat editor](https://playcanvas.com/supersplat/editor) (drag and drop the saved Gaussian splatting `.ply` pointcloud in the editor interface). 
2. Getting into the folder `test/gaussian_splatting` and running:      
    `$ python test_gsm.py --load <gs_checkpoint_path>`      
    The ` <gs_checkpoint_path>` is expected to have the following structure:      
    ```bash
    ├── gs_checkpoint_path
        ├── pointcloud   # folder containing different subfolders, each one with a saved .ply econding the gaussian splatting model at a specific iteration/checkpoint
        ├── last_camera.json
        ├── config.yml
    ```

#### Controlling the spatial distribution of keyframe FOV centers

If you are targeting volumetric reconstruction while running SLAM, you can enable a **keyframe generation policy** designed to manage the spatial distribution of keyframe field-of-view (FOV) centers. The *FOV center of a camera* is defined as the backprojection of its image center, calculated using the median depth of the frame. With this policy, a new keyframe is generated only if its FOV center is farther than a predefined distance from the nearest existing keyframe's FOV center. You can enable this policy by setting the following parameters in the yaml setting:
```yaml
KeyFrame.useFovCentersBasedGeneration: 1  # compute 3D fov centers of camera frames by using median depth and use their distances to control keyframe generation
KeyFrame.maxFovCentersDistance: 0.2       # max distance between fov centers in order to generate a keyframe
```

---

### Depth prediction

The available depth prediction models can be utilized both in the SLAM back-end and front-end. 
- **Back-end**: Depth prediction can be enabled in the [volumetric reconstruction](#volumetric-reconstruction) pipeline by setting the parameter `kVolumetricIntegrationUseDepthEstimator=True` and selecting your preferred `kVolumetricIntegrationDepthEstimatorType` in `config_parameters.py`. 
- **Front-end**: Depth prediction can be enabled in the front-end by setting the parameter `kUseDepthEstimatorInFrontEnd` in `config_parameters.py`. This feature estimates depth images from input color images to emulate a RGBD camera. Please, note this functionality is still *experimental* at present time [WIP].   

**Notes**: 
* In the case of a **monocular SLAM**, do NOT use depth prediction in the back-end volumetric integration: The SLAM (fake) scale will conflict with the absolute metric scale of depth predictions. With monocular datasets, you can enable depth prediction to run in the front-end (to emulated an RGBD sensor).
- The depth inference may be very slow (for instance, with DepthPro it takes ~1s per image on my machine). Therefore, the resulting volumetric reconstruction pipeline may be very slow.

Refer to the file `depth_estimation/depth_estimator_factory.py` for further details. Both stereo and monocular prediction approaches are supported. You can test depth prediction/estimation by using the script `main_depth_prediction.py`.

---

### Saving and reloading

#### Save the a map

When you run the script `main_slam.py` (`main_map_dense_reconstruction.py`):
- You can save the current map state by pressing the button `Save` on the GUI. This saves the current map along with front-end, and backend configurations into the default folder `results/slam_state` (`results/slam_state_dense_reconstruction`). 
- To change the default saving path, open `config.yaml` and update target `folder_path` in the section: 
  ```bash
  SYSTEM_STATE:
    folder_path: results/slam_state   # default folder path (relative to repository root) where the system state is saved or reloaded
  ```

#### Reload a saved map and relocalize in it 

- A saved map can be loaded and visualized in the GUI by running: 
  ```bash
  $ . pyenv-activate.sh   #  Activate pyslam python virtual environment. This is only needed once in a new terminal.
  $ ./main_map_viewer.py  #  Use the --path options to change the input path
  ```
  
- To enable map reloading and relocalization when running `main_slam.py`, open `config.yaml` and set 
  ```bash
  SYSTEM_STATE:
    load_state: True                  # flag to enable SLAM state reloading (map state + loop closing state)
    folder_path: results/slam_state   # default folder path (relative to repository root) where the system state is saved or reloaded
  ```

Note that pressing the `Save` button saves the current map, front-end, and backend configurations. Reloading a saved map overwrites the current system configurations to ensure descriptor compatibility.  


#### Trajectory saving

Estimated trajectories can be saved in three **formats**: *TUM* (The Open Mapping format), *KITTI* (KITTI Odometry format), and *EuRoC* (EuRoC MAV format). pySLAM saves two **types** of trajectory estimates:

- **Online**: In *online* trajectories, each pose estimate depends only on past poses. A pose estimate is saved at the end of each front-end iteration on current frame.
- **Final**: In *final* trajectories, each pose estimate depends on both past and future poses. A pose estimate is refined multiple times by LBA windows that cover it and by PGO and GBA during loop closures.


To enable trajectory saving, open `config.yaml` and search for the `SAVE_TRAJECTORY`: set `save_trajectory: True`, select your `format_type` (`tum`, `kitti`, `euroc`), and the output filename. For instance for a `kitti` format output:   
```bash
SAVE_TRAJECTORY:
  save_trajectory: True
  format_type: kitti             # supported formats: `tum`, `kitti`, `euroc`
  output_folder: results/metrics # relative to pyslam root folder 
  basename: trajectory           # basename of the trajectory saving output
```


---

### SLAM GUI 

Some quick information about the non-trivial GUI buttons of `main_slam.py`: 
- `Step`: Enter in the *Step by step mode*. Press the button `Step` a first time to pause. Then, press it again to make the pipeline process a single new frame.
- `Save`: Save the map into the file `map.json`. You can visualize it back by using the script `/main_map_viewer.py` (as explained above). 
- `Reset`: Reset SLAM system. 
- `Draw Grount Truth`:  If a ground truth dataset is loaded (e.g., from KITTI, TUM, EUROC, or REPLICA), you can visualize it by pressing this button. The ground truth trajectory will be displayed in 3D and progressively aligned (approximately every 30 frames) with the estimated trajectory. The alignment improves as more samples are added to the estimated trajectory. After ~20 frames, if the button is pressed, a window will appear showing the Cartesian alignment errors (ground truth vs. estimated trajectory) along the axes.  

---

### Monitor the logs for tracking, local mapping, and loop closing simultaneously

The logs generated by the modules `local_mapping.py`, `loop_closing.py`, `loop_detecting_process.py`, and `global_bundle_adjustments.py` are collected in the files `local_mapping.log`, `loop_closing.log`, `loop_detecting.log`, and `gba.log`, which are all stored in the folder `logs`. For debugging, you can monitor a parallel flow by running the following command in a separate shell:    
`$ tail -f logs/<log file name>`     
Otherwise, to check all parallel logs with tmux, run:          
`$ ./scripts/launch_tmux_logs.sh`           
To launch slam and check all logs in a single tmux, run:     
`$ ./scripts/launch_tmux_slam.sh`      
Press `CTRL+A` and then `CTRL+Q` to exit from `tmux` environment.

---

## System overview
      
[Here](./docs/system_overview.md) you can find a couple of diagram sketches that provide an overview of the main SLAM workflow, system components, and classes relationships/dependencies. Documentation is a work in progress.

---

## Supported components and models
### Supported local features

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
  
For more information, refer to [local_features/feature_types.py](local_features/feature_types.py) file. Some of the local features consist of a *joint detector-descriptor*. You can start playing with the supported local features by taking a look at `test/cv/test_feature_manager.py` and `main_feature_matching.py`.

In both the scripts `main_vo.py` and `main_slam.py`, you can create your preferred detector-descritor configuration and feed it to the function `feature_tracker_factory()`. Some ready-to-use configurations are already available in the file [local_features/feature_tracker.configs.py](local_features/feature_tracker_configs.py)

The function `feature_tracker_factory()` can be found in the file `local_features/feature_tracker.py`. Take a look at the file `local_features/feature_manager.py` for further details.

**N.B.**: You just need a *single* python environment to be able to work with all the [supported local features](#supported-local-features)!


### Supported matchers 

* *BF*: Brute force matcher on descriptors (with KNN).
* *[FLANN](https://www.semanticscholar.org/paper/Fast-Approximate-Nearest-Neighbors-with-Automatic-Muja-Lowe/35d81066cb1369acf4b6c5117fcbb862be2af350)* 
* *[XFeat](https://arxiv.org/abs/2404.19174)*      
* *[LightGlue](https://arxiv.org/abs/2306.13643)*
* *[LoFTR](https://arxiv.org/abs/2104.00680)*
* *[MASt3R](https://arxiv.org/abs/2406.09756)*
  
See the file `local_features/feature_matcher.py` for further details.


### Supported global descriptors and local descriptor aggregation methods

##### Local descriptor aggregation methods

* Bag of Words (BoW): [DBoW2](https://github.com/dorian3d/DBoW2), [DBoW3](https://github.com/rmsalinas/DBow3).  [[paper](https://doi.org/10.1109/TRO.2012.2197158)]
* Vector of Locally Aggregated Descriptors: [VLAD](https://www.vlfeat.org/api/vlad.html).  [[paper](https://doi.org/10.1109/CVPR.2010.5540039)] 
* Incremental Bags of Binary Words (iBoW) via Online Binary Image Index: [iBoW](https://github.com/emiliofidalgo/ibow-lcd), [OBIndex2](https://github.com/emiliofidalgo/obindex2).  [[paper](https://doi.org/10.1109/LRA.2018.2849609)]
* Hyperdimensional Computing: [HDC](https://www.tu-chemnitz.de/etit/proaut/hdc_desc).  [[paper](https://openaccess.thecvf.com/content/CVPR2021/html/Neubert_Hyperdimensional_Computing_as_a_Framework_for_Systematic_Aggregation_of_Image_CVPR_2021_paper.html)]


**NOTE**: *iBoW* and *OBIndex2* incrementally build a binary image index and do not need a prebuilt vocabulary. In the implemented classes, when needed, the input non-binary local descriptors are transparently transformed into binary descriptors.

##### Global descriptors

Also referred to as *holistic descriptors*:

* [SAD](https://ieeexplore.ieee.org/document/6224623)
* [AlexNet](https://github.com/BVLC/caffe/tree/master/models/bvlc_alexnet)
* [NetVLAD](https://www.di.ens.fr/willow/research/netvlad/)
* [HDC-DELF](https://www.tu-chemnitz.de/etit/proaut/hdc_desc)
* [CosPlace](https://github.com/gmberton/CosPlace)
* [EigenPlaces](https://github.com/gmberton/EigenPlaces)


Different [loop closing methods](#loop-closing) are available. These combines the above aggregation methods and global descriptors.
See the file [loop_closing/loop_detector_configs.py](loop_closing/loop_detector_configs.py) for further details.


### Supported depth prediction models

Both monocular and stereo depth prediction models are available. SGBM algorithm has been included as a classic reference approach. 

* [SGBM](https://ieeexplore.ieee.org/document/4359315): Depth SGBM from OpenCV (Stereo, classic approach)
* [Depth-Pro](https://arxiv.org/abs/2410.02073) (Monocular)
* [DepthAnythingV2](https://arxiv.org/abs/2406.09414) (Monocular)
* [RAFT-Stereo](https://arxiv.org/abs/2109.07547) (Stereo)
* [CREStereo](https://arxiv.org/abs/2203.11483) (Stereo)
* [MASt3R](https://arxiv.org/abs/2406.09756) (Stereo/Monocular)
* [MV-DUSt3R](https://arxiv.org/abs/2412.06974) (Stereo/Monocular)

### Supported volumetric mapping methods

* [TSDF](https://arxiv.org/pdf/2110.00511) with voxel block grid (parallel spatial hashing)
* Incremental 3D Gaussian Splatting. See [here](https://repo-sam.inria.fr/fungraph/3d-gaussian-splatting/) and [MonoGS](https://arxiv.org/abs/2312.06741) for a description of its backend.

--- 
## Datasets

Five different types of datasets are available:

Dataset | type in `config.yaml`
--- | --- 
[KITTI odometry data set (grayscale, 22 GB)](http://www.cvlibs.net/datasets/kitti/eval_odometry.php)  | `type: KITTI_DATASET` 
[TUM dataset](https://vision.in.tum.de/data/datasets/rgbd-dataset/download)                           | `type: TUM_DATASET` 
[EUROC dataset](http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets)          | `type: EUROC_DATASET` 
[REPLICA dataset](https://github.com/facebookresearch/Replica-Dataset)        | `type: REPLICA_DATASET` 
Video file                                                                                            | `type: VIDEO_DATASET` 
Folder of images                                                                                      | `type: FOLDER_DATASET` 

Use the download scripts available in the folder `scripts` to download some of the following datasets.

### KITTI Datasets

pySLAM code expects the following structure in the specified KITTI path folder (specified in the section `KITTI_DATASET` of the file `config.yaml`). : 
```bash
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

2. Select the corresponding calibration settings file (section `KITTI_DATASET: cam_settings:` in the file `config.yaml`)


### TUM Datasets 

pySLAM code expects a file `associations.txt` in each TUM dataset folder (specified in the section `TUM_DATASET:` of the file `config.yaml`). 

1. Download a sequence from http://vision.in.tum.de/data/datasets/rgbd-dataset/download and uncompress it.
2. Associate RGB images and depth images using the python script [associate.py](http://vision.in.tum.de/data/datasets/rgbd-dataset/tools). You can generate your `associations.txt` file by executing:
```bash
$ python associate.py PATH_TO_SEQUENCE/rgb.txt PATH_TO_SEQUENCE/depth.txt > associations.txt
```
3. Select the corresponding calibration settings file (section `TUM_DATASET: cam_settings:` in the file `config.yaml`).


### EuRoC Datasets

1. Download a sequence (ASL format) from http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets (check this direct [link](http://robotics.ethz.ch/~asl-datasets/ijrr_euroc_mav_dataset/))
2. Use the script `io/generate_euroc_groundtruths_as_tum.sh` to generate the TUM-like groundtruth files `path + '/' + name + '/mav0/state_groundtruth_estimate0/data.tum'` that are required by the `EurocGroundTruth` class.
3. Select the corresponding calibration settings file (section `EUROC_DATASET: cam_settings:` in the file `config.yaml`).


### Replica Datasets

1. You can download the zip file containing all the sequences by running:    
   `$ wget https://cvg-data.inf.ethz.ch/nice-slam/data/Replica.zip`    
2. Then, uncompress it and deploy the files as you wish.
3. Select the corresponding calibration settings file (section `REPLICA_DATASET: cam_settings:` in the file `config.yaml`).

--- 
## Camera Settings

The folder `settings` contains the camera settings files which can be used for testing the code. These are the same used in the framework [ORB-SLAM2](https://github.com/raulmur/ORB_SLAM2). You can easily modify one of those files for creating your own new calibration file (for your new datasets).

In order to calibrate your camera, you can use the scripts in the folder `calibration`. In particular: 
1. Use the script `grab_chessboard_images.py` to collect a sequence of images where the chessboard can be detected (set the chessboard size therein, you can use the calibration pattern `calib_pattern.pdf` in the same folder) 
2. Use the script `calibrate.py` to process the collected images and compute the calibration parameters (set the chessboard size therein)

For more information on the calibration process, see this [tutorial](https://learnopencv.com/camera-calibration-using-opencv/) or this other [link](https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html). 

If you want to **use your camera**, you have to:
* Calibrate it and configure [WEBCAM.yaml](./settings/WEBCAM.yaml) accordingly
* Record a video (for instance, by using `save_video.py` in the folder `calibration`)
* Configure the `VIDEO_DATASET` section of `config.yaml` in order to point to your recorded video.

--- 
## Comparison pySLAM vs ORB-SLAM3

For a comparative evaluation, "**online**" trajectory estimated by pySLAM vs "**final**" trajectories estimated by ORB-SLAM3, see this nice [notebook](https://github.com/anathonic/Trajectory-Comparison-ORB-SLAM3-pySLAM/blob/main/trajectories_comparison.ipynb). For further details about "online" and "final" trajectories, see [here](#trajectory-saving) .

Note that pySLAM is able to save both online and final pose estimates. On the other end, ORB-SLAM3 pose estimates are saved at the end of the full dataset playback. See this [section](#trajectory-saving) for further details on how to save trajectories with pySLAM.

--- 
## Contributing to pySLAM

If you like pySLAM and would like to contribute to the code base, you can report bugs, leave comments and proposing new features through issues and pull requests on github. Feel free to get in touch at *luigifreda(at)gmail[dot]com*. Thank you!

--- 
## References

Suggested books:
* *[Multiple View Geometry in Computer Vision](https://www.robots.ox.ac.uk/~vgg/hzbook/)* by Richard Hartley and Andrew Zisserman
* *[An Invitation to 3-D Vision](https://link.springer.com/book/10.1007/978-0-387-21779-6)* by Yi-Ma, Stefano Soatto, Jana Kosecka, S. Shankar Sastry 
* *[State Estimation for Robotics](http://asrl.utias.utoronto.ca/~tdb/)* by Timothy D. Barfoot
* *[Computer Vision: Algorithms and Applications](http://szeliski.org/Book/)*, by Richard Szeliski 
* *[Introduction to Visual SLAM](https://link.springer.com/book/10.1007/978-981-16-4939-4)* by Xiang Gao, Tao Zhang
* *[Deep Learning](http://www.deeplearningbook.org/lecture_slides.html)*, by Ian Goodfellow, Yoshua Bengio and Aaron Courville
* *[Neural Networks and Deep Learning](http://neuralnetworksanddeeplearning.com/index.html)*, By Michael Nielsen 

Suggested material:
* *[Vision Algorithms for Mobile Robotics](http://rpg.ifi.uzh.ch/teaching.html)* by Davide Scaramuzza 
* *[CS 682 Computer Vision](http://cs.gmu.edu/~kosecka/cs682.html)* by Jana Kosecka   
* *[ORB-SLAM: a Versatile and Accurate Monocular SLAM System](http://webdiis.unizar.es/~raulmur/MurMontielTardosTRO15.pdf)* by R. Mur-Artal, J. M. M. Montiel, and J. D. Tardos
* *[Double Window Optimisation for Constant Time Visual SLAM](http://hauke.strasdat.net/files/strasdat2011iccv.pdf)* by H. Strasdat, A. J. Davison, J.M.M. Montiel, K. Konolige
* *[The Role of Wide Baseline Stereo in the Deep Learning World](https://ducha-aiki.github.io/wide-baseline-stereo-blog/2020/03/27/intro.html)* by Dmytro Mishkin
* *[To Learn or Not to Learn: Visual Localization from Essential Matrices](https://arxiv.org/abs/1908.01293)* by Qunjie Zhou, Torsten Sattler, Marc Pollefeys, Laura Leal-Taixe
* *[Awesome local-global descriptors](https://github.com/shamangary/awesome-local-global-descriptor)* repository 
* *[Introduction to Feature Matching Using Neural Networks](https://learnopencv.com/feature-matching/)*
* *[Visual Place Recognition: A Tutorial](https://arxiv.org/pdf/2303.03281)*
* *[Bags of Binary Words for Fast Place Recognition in Image Sequences](http://doriangalvez.com/papers/GalvezTRO12.pdf)*

Moreover, you may want to have a look at the OpenCV [guide](https://docs.opencv.org/4.x/index.html) or [tutorials](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html).  

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
* [DepthAnythingV2](https://github.com/DepthAnything/Depth-Anything-V2)
* [DepthPro](https://github.com/apple/ml-depth-pro)
* [RAFT-Stereo](https://github.com/princeton-vl/RAFT-Stereo)
* [CREStereo](https://github.com/megvii-research/CREStereo) and [CREStereo-Pytorch](https://github.com/ibaiGorordo/CREStereo-Pytorch)
* [MonoGS](https://github.com/muskie82/MonoGS)
* [mast3r](https://github.com/naver/mast3r)
* [mvdust3r](https://github.com/facebookresearch/mvdust3r)
* Many thanks to [Anathonic](https://github.com/anathonic) for adding the trajectory-saving feature and for the comparison notebook: [pySLAM vs ORB-SLAM3](https://github.com/anathonic/Trajectory-Comparison-ORB-SLAM3-pySLAM/blob/main/trajectories_comparison.ipynb).


---
## License 

pySLAM is released under [GPLv3 license](./LICENSE). pySLAM contains some modified libraries, each one coming with its license. Where nothing is specified, a GPLv3 license applies to the software.

If you use pySLAM in your projects, please cite this document:
["pySLAM: An Open-Source, Modular, and Extensible Framework for SLAM"](https://arxiv.org/abs/2502.11955), *Luigi Freda*      
You may find an update version of this document [here](./docs/tex/document.pdf).

---
## TODOs

Many improvements and additional features are currently under development: 

- [x] loop closing
- [x] relocalization 
- [x] stereo and RGBD support
- [x] map saving/loading 
- [x] modern DL matching algorithms 
- [ ] object detection 
- [ ] semantic segmentation 
- [x] 3D dense reconstruction 
- [x] unified install procedure (single branch) for all OSs 
- [x] trajectory saving 
- [x] depth prediction integration
- [ ] ROS support
- [x] gaussian splatting integration
- [x] documentation [WIP]
- [ ] gtsam integration
- [ ] IMU integration
- [ ] LIDAR integration
- [ ] XSt3r-based methods integration [WIP]
