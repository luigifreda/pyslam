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

'''
List of shared parameters 
''' 


# num of desired keypoints per frame 
kNumFeatures=2000


# min cos angle for triangulation (min parallax angle)
kCosMinParallax=0.99998  # 0.9998


# initializer 
kInitializerDesiredMedianDepth = 20    # when initializing, the initial median depth is computed and forced to this value 
kMinTraslation=0.0001*kInitializerDesiredMedianDepth


# tracking 
kUseMotionModel = True  # True: use kinematic motion model for computing a first guess pose (to be optimized by current pose optimization) without fitting an essential mat 
                        # False: pose and keypoint match inliers are estimated by fitting an essential mat (5 points algorithm), WARNING: this approach comes with some limitations (please, read the comments of the method slam.estimate_pose_ess_mat())


# search matches by projection 
kMaxReprojectionDistance=15 # [pixels] 
kMatchRatioTest=0.75
#
kMaxOrbDistanceSearchByReproj=50     # ORB
kMaxBriskDistanceSearchByReproj=50   # BRISK     (needs better tuning)
kMaxAkazeDistanceSearchByReproj=80   # AKAZE     (needs better tuning)
kMaxSiftDistanceSearchByReproj=100   # SIFT    (needs better tuning)
kMaxSurfDistanceSearchByReproj=0.05  # SURF   (needs better tuning)
kMaxDescriptorDistanceSearchByReproj=kMaxOrbDistanceSearchByReproj # main (use ORB setting by default)


# search matches for triangulation by using epipolar lines 
kUseSearchFrameByProjection=False  # Match frames by using frame map points projection and epipolar lines; here, the current available interframe pose estimate is used for computing the fundamental mat F
kMinDistanceFromEpipole=100        # [pixels] Used with search by epipolar lines 
#  
kMaxDescriptorDistanceSearchEpipolar=kMaxDescriptorDistanceSearchByReproj 


# BA
kLocalWindow=10                   #  [# frames]   
kEveryNumFramesLargeWindowBA=10   # num of frames between two large window BA  
kLargeWindow=20                   #  [# frames] 
kUseLargeWindowBA = True          # True: perform BA over a large window; False: do not perform large window BA   


kChi2Mono = 5.991 # chi-square 2 DOFs, used for reprojection error  (Hartley Zisserman pg 119)

