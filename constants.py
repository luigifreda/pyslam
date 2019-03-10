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
kUseLargeWindowBA = True          # True: use BA over a large window 
                                  # False: do not use large window BA   
kEveryNumFramesLargeWindowBA=10   # num of frames between two large window BA  
kLargeWindow=20

kChi2Mono = 5.991 # chi-square 2 DOFs  (Hartley Zisserman pg 119)

kUseMotionModel = True  # True: use motion model for computing a first guess pose (to be optimized by current pose optimization)
                        # False: pose is first estimated by using the essential mat (5 points algorithm), WARNING: this approach comes with some limitations (please, read the comments of the method slam.estimate_pose_ess_mat())