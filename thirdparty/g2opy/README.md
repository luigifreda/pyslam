# g2opy

This is a python binding of graph optimization C++ framework [g2o](https://github.com/RainerKuemmerle/g2o).

> g2o is an open-source C++ framework for optimizing graph-based nonlinear error functions. g2o has been designed to be easily extensible to a wide range of problems and a new problem typically can be specified in a few lines of code. The current implementation provides solutions to several variants of SLAM and BA.  
A wide range of problems in robotics as well as in computer-vision involve the minimization of a non-linear error function that can be represented as a graph. Typical instances are simultaneous localization and mapping (SLAM) or bundle adjustment (BA). The overall goal in these problems is to find the configuration of parameters or state variables that maximally explain a set of measurements affected by Gaussian noise. g2o is an open-source C++ framework for such nonlinear least squares problems. g2o has been designed to be easily extensible to a wide range of problems and a new problem typically can be specified in a few lines of code. The current implementation provides solutions to several variants of SLAM and BA.

Currently, this project doesn't support writing user-defined types in python, but the predefined types are enough to implement the most common algorithms, say **PnP, ICP, Bundle Adjustment and Pose Graph Optimization** in 2d or 3d scenarios. g2o's visualization part is not wrapped, if you want to visualize point clouds or graph, you can give [pangolin](https://github.com/uoip/pangolin) a try, it's a python binding of C++ library [Pangolin](http://github.com/stevenlovegrove/Pangolin).

For convenience, some frequently used Eigen types (Quaternion, Rotation2d, Isometry3d, Isometry2d, AngleAxis) are packed into this library.  
In the contrib folder, I collected some useful 3rd-party C++ code related to g2o, like robust pose graph optimization library [vertigo](http://www.openslam.org/vertigo), stereo sba and smooth estimate propagator from [sptam](https://github.com/lrse/sptam).


## Requirements
* [C++ requirements](#g2oRequirements).   
([pybind11](https://github.com/pybind/pybind11) is also required, but it's built in this repository, you don't need to install) 


## Installation
```
git clone https://github.com/uoip/g2opy.git
cd g2opy
mkdir build
cd build
cmake ..
make -j8
cd ..
python setup.py install
```
Tested under Ubuntu 16.04, Python 3.6+.


## Get Started
The code snippets below show the core parts of BA and Pose Graph Optimization in a SLAM system.
#### Bundle Adjustment
```python
import numpy
import g2o

class BundleAdjustment(g2o.SparseOptimizer):
    def __init__(self, ):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=10):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_pose(self, pose_id, pose, cam, fixed=False):
        sbacam = g2o.SBACam(pose.orientation(), pose.position())
        sbacam.set_cam(cam.fx, cam.fy, cam.cx, cam.cy, cam.baseline)

        v_se3 = g2o.VertexCam()
        v_se3.set_id(pose_id * 2)   # internal id
        v_se3.set_estimate(sbacam)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3) 

    def add_point(self, point_id, point, fixed=False, marginalized=True):
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(point_id * 2 + 1)
        v_p.set_estimate(point)
        v_p.set_marginalized(marginalized)
        v_p.set_fixed(fixed)
        super().add_vertex(v_p)

    def add_edge(self, point_id, pose_id, 
            measurement,
            information=np.identity(2),
            robust_kernel=g2o.RobustKernelHuber(np.sqrt(5.991))):   # 95% CI

        edge = g2o.EdgeProjectP2MC()
        edge.set_vertex(0, self.vertex(point_id * 2 + 1))
        edge.set_vertex(1, self.vertex(pose_id * 2))
        edge.set_measurement(measurement)   # projection
        edge.set_information(information)

        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, pose_id):
        return self.vertex(pose_id * 2).estimate()

    def get_point(self, point_id):
        return self.vertex(point_id * 2 + 1).estimate()
```

#### Pose Graph Optimization
```python
import numpy
import g2o

class PoseGraphOptimization(g2o.SparseOptimizer):
    def __init__(self):
        super().__init__()
        solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(solver)
        super().set_algorithm(solver)

    def optimize(self, max_iterations=20):
        super().initialize_optimization()
        super().optimize(max_iterations)

    def add_vertex(self, id, pose, fixed=False):
        v_se3 = g2o.VertexSE3()
        v_se3.set_id(id)
        v_se3.set_estimate(pose)
        v_se3.set_fixed(fixed)
        super().add_vertex(v_se3)

    def add_edge(self, vertices, measurement, 
            information=np.identity(6),
            robust_kernel=None):

        edge = g2o.EdgeSE3()
        for i, v in enumerate(vertices):
            if isinstance(v, int):
                v = self.vertex(v)
            edge.set_vertex(i, v)

        edge.set_measurement(measurement)  # relative pose
        edge.set_information(information)
        if robust_kernel is not None:
            edge.set_robust_kernel(robust_kernel)
        super().add_edge(edge)

    def get_pose(self, id):
        return self.vertex(id).estimate()
```
For more details, checkout [python examples](python/examples) or project [stereo_ptam](https://github.com/uoip/stereo_ptam).  
Thanks to [pybind11](https://github.com/pybind/pybind11), g2opy works seamlessly between numpy and underlying Eigen.  


## Motivation
This project is my first step towards implementing complete SLAM system in python, and interacting with Deep Learning models.  
Deep Learning is the hottest field in AI nowadays, it has greatly benefited many Robotics/Computer Vision tasks, like
* Reinforcement Learning
* Self-Supervision
* Control
* Object Tracking
* Object Detection
* Semantic Segmentation
* Instance Segmentation
* Place Recognition
* Face Recognition
* 3D Object Detection
* Point Cloud Segmentation
* Human Pose Estimation
* Stereo Matching
* Depth Estimation
* Optical Flow Estimation
* Interest Point Detection
* Correspondence Estimation
* Image Enhancement 
* Style Transfer
* ...

SLAM, as a subfield of Robotics and Computer Vision, is one of the core modules of robots, MAV, autonomous driving, and augmented reality. The combination of SLAM and Deep Learning (and Deep Learning driving computer vision techniques) is very promising, actually, there are increasing work in this direction, e.g. [CNN-SLAM](https://arxiv.org/abs/1704.03489), [SfM-Net](https://arxiv.org/abs/1704.07804), [DeepVO](https://arxiv.org/abs/1709.08429), [DPC-Net](https://arxiv.org/abs/1709.03128), [MapNet](https://arxiv.org/abs/1712.03342), [SuperPoint](https://arxiv.org/abs/1712.07629).   
Deep Learning community has developed many easy-to-use python libraries, like [TensorFlow](https://github.com/tensorflow/tensorflow), [PyTorch](https://github.com/pytorch/pytorch), [Chainer](https://github.com/chainer/chainer), [MXNet](https://mxnet.incubator.apache.org/). These libraries make writing/training DL models easier, and in turn boost the development of the field itself. But in SLAM/Robotics fields, python is still underrated, most of the software stacks are writen for C/C++ users. Lacking of tools makes it inconvenient to interact with the booming Deep Learning comunity and python scientific computing ecosystem.     
Hope this project can slightly relieve the situation.


## TODO
* Installation via pip;
* Solve the found segfault bugs (be easy, they do not appear in the python examples);
* Introduce **Automatic Differentiation**, make writing user-defined types in python possible.


## License
* For g2o's original C++ code, see [License](#g2oLicense).
* The binding code and python example code of this project is licensed under BSD License.  


## Contact
If you have problems related to **binding code/python interface/python examples** of this project, you can report isseus, or email me (qihang@outlook.com).

<br><br><br><br><br><br>





g2o's README:

--------------------------------
--------------------------------
--------------------------------


g2o - General Graph Optimization
================================

Linux: [![Build Status](https://travis-ci.org/RainerKuemmerle/g2o.svg?branch=master)](https://travis-ci.org/RainerKuemmerle/g2o)
Windows: [![Build status](https://ci.appveyor.com/api/projects/status/9w0cpb9krc6t4nt7/branch/master?svg=true)](https://ci.appveyor.com/project/RainerKuemmerle/g2o/branch/master)

g2o is an open-source C++ framework for optimizing graph-based nonlinear error
functions. g2o has been designed to be easily extensible to a wide range of
problems and a new problem typically can be specified in a few lines of code.
The current implementation provides solutions to several variants of SLAM and
BA.

A wide range of problems in robotics as well as in computer-vision involve the
minimization of a non-linear error function that can be represented as a graph.
Typical instances are simultaneous localization and mapping (SLAM) or bundle
adjustment (BA). The overall goal in these problems is to find the
configuration of parameters or state variables that maximally explain a set of
measurements affected by Gaussian noise. g2o is an open-source C++ framework
for such nonlinear least squares problems. g2o has been designed to be easily
extensible to a wide range of problems and a new problem typically can be
specified in a few lines of code. The current implementation provides solutions
to several variants of SLAM and BA. g2o offers a performance comparable to
implementations of state-of-the-art approaches for the specific problems
(02/2011).

### Papers Describing the Approach:
Rainer Kuemmerle, Giorgio Grisetti, Hauke Strasdat,
Kurt Konolige, and Wolfram Burgard
g2o: A General Framework for Graph Optimization
IEEE International Conference on Robotics and Automation (ICRA), 2011
http://ais.informatik.uni-freiburg.de/publications/papers/kuemmerle11icra.pdf

### Documentation
A detailed description of how the library is structured and how to use and extend it can be found in /doc/g2o.pdf
The API documentation can be generated as described in doc/doxygen/readme.txt

### <span id="g2oLicense">License</span>
g2o is licensed under the BSD License. However, some libraries are available
under different license terms. See below.

The following parts are licensed under LGPL3+:
- csparse\_extension

The following parts are licensed under GPL3+:
- g2o_viewer
- g2o_incremental
- slam2d_g2o (example for 2D SLAM with a QGLviewer GUI)

<!-- - [g2o](#g2o)
    - [Requirements](#requirements)
    - [Installation](#installation)
    - [Get Started](#get-started)
            - [Bundle Adjustment](#bundle-adjustment)
            - [Pose Graph Optimization](#pose-graph-optimization)
    - [Motivation](#motivation)
    - [TODO](#todo)
    - [License](#license)
    - [Contact](#contact)
- [g2o - General Graph Optimization](#g2o---general-graph-optimization)
        - [Papers Describing the Approach:](#papers-describing-the-approach)
        - [Documentation](#documentation)
        - [<span id="License">License</span>](#span-idlicenselicensespan)
        - [<span id="Requirements">Requirements</span>](#span-idrequirementsrequirementsspan)
            - [Optional requirements](#optional-requirements)
            - [Mac OS X](#mac-os-x)
        - [Compilation](#compilation)
        - [Cross-Compiling for Android](#cross-compiling-for-android)
        - [Acknowledgments](#acknowledgments)
        - [Contact information](#contact-information) -->

Please note that some features of CHOLMOD (which may be used by g2o, see
libsuitesparse below) are licensed under the GPL. To avoid that your binary has
to be licensed under the GPL, you may have to re-compile CHOLMOD without
including its GPL features. The CHOLMOD library distributed with, for example,
Ubuntu or Debian includes the GPL features. The supernodal factorization is
considered by g2o, if it is available.

Within the folder EXTERNAL we include software not written by us to
guarantee easy compilation.
- csparse: LPGL2.1 (see EXTERNAL/csparse/License.txt)
  csparse is compiled if it is not provided by the system.
- ceres: BSD (see EXTERNAL/ceres/LICENSE)
  Headers to perform Automatic Differentiation
- freeglut: X Consortium (Copyright (c) 1999-2000 Pawel W. Olszta)
  We use a stripped down version for drawing text in OpenGL.

See the doc folder for the full text of the licenses.

g2o is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
licenses for more details.


### <a name="g2oRequirements">Requirements</a>
  * cmake             http://www.cmake.org/
  * Eigen3            http://eigen.tuxfamily.org

  On Ubuntu / Debian these dependencies are resolved by installing the
  following packages.
    - cmake
    - libeigen3-dev

#### Optional requirements
  * suitesparse       http://www.cise.ufl.edu/research/sparse/SuiteSparse/
  * Qt5               http://qt-project.org
  * libQGLViewer      http://www.libqglviewer.com/

  On Ubuntu / Debian these dependencies are resolved by installing the
  following packages.
    - libsuitesparse-dev
    - qtdeclarative5-dev
    - qt5-qmake
    - libqglviewer-dev

#### Mac OS X
If using [Homebrew](http://brew.sh/), then

`brew install homebrew/science/g2o`

will install g2o together with its required dependencies. In this case no manual compilation is necessary.

### Compilation
Our primary development platform is Linux. Experimental support for
Mac OS X, Android and Windows (MinGW or MSVC).
We recommend a so-called out of source build which can be achieved
by the following command sequence.

- `mkdir build`
- `cd build`
- `cmake ../`
- `make`

The binaries will be placed in bin and the libraries in lib which
are both located in the top-level folder.
If you are compiling on Windows, please download Eigen3 and extract it.
Within cmake-gui set the variable G2O\_EIGEN3\_INCLUDE to that directory.

### Cross-Compiling for Android

- `mkdir build`
- `cd build`
- `cmake -DCMAKE_TOOLCHAIN_FILE=../script/android.toolchain.cmake -DANDROID_NDK=<YOUR_PATH_TO_ANDROID_NDK_r10d+> -DCMAKE_BUILD_TYPE=Release -DANDROID_ABI="armeabi-v7a with NEON" -DEIGEN3_INCLUDE_DIR="<YOUR_PATH_TO_EIGEN>" -DEIGEN3_VERSION_OK=ON .. && cmake --build .`


### Acknowledgments
We thank the following contributors for providing patches:
- Simon J. Julier: patches to achieve compatibility with Mac OS X and others.
- Michael A. Eriksen for submitting patches to compile with MSVC.
- Mark Pupilli for submitting patches to compile with MSVC.


### Contact information
Rainer Kuemmerle <kuemmerl@informatik.uni-freiburg.de>   
Giorgio Grisetti <grisetti@dis.uniroma1.it>   
Hauke Strasdat <strasdat@gmail.com>   
Kurt Konolige <konolige@willowgarage.com>   
Wolfram Burgard <burgard@informatik.uni-freiburg.de>   
