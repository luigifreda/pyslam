"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
"""

kCosMinParallax=0.99998  # 0.9998

kMaxReprojectionDistance=15 # pixels 
kMaxOrbDistanceMatch=50
kMaxOrbDistanceSearchByReproj=50
kMatchRatioTest=0.75

kNumFeatures=2000

kLocalWindow=10   
kEveryNumFramesLargeWindowBA=10   # num of frames between two large window BA  
kLargeWindow=20
kUseLargeWindowBA = True        # True: use BA over a large window, False: do not use large window BA   


kChi2Mono = 5.991 # chi-square 2 DOFs, used for reprojection error  (Hartley Zisserman pg 119)

kInitializerDesiredMedianDepth = 20    # after initialization: the initial median depth is computed and forced to this value 
kMinTraslation=0.0001*kInitializerDesiredMedianDepth

kUseMotionModel = True  # True: use kinematic motion model for computing a first guess pose (to be optimized by current pose optimization) without fitting an essential mat 
                        # False: pose and keypoint match inliers are estimated by fitting an essential mat (5 points algorithm), WARNING: this approach comes with some limitations (please, read the comments of the method slam.estimate_pose_ess_mat())

kUseSearchFrameByEpipolarLines=False  # Match frames by using epipolar lines; the current available interframe pose estimate is used for computing the fundamental mat F
kMinDistanceFromEpipole=100           # Used with search by epipolar lines 