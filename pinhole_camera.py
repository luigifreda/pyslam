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

import numpy as np 
import cv2
from geom_helpers import add_ones

# D = [k1, k2, p1, p2, k3]
class Camera: 
    def __init__(self, width, height, fx, fy, cx, cy, D):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.is_distorted = (abs(D[0]) > 0)
        self.D = D # np.array([k1, k2, p1, p2, k3])  distortion coefficients 


class PinholeCamera(Camera):
    def __init__(self, width, height, fx, fy, cx, cy, D):
        super().__init__(width, height, fx, fy, cx, cy, D)
        self.K = np.array([[fx, 0,cx],
                           [ 0,fy,cy],
                           [ 0, 0, 1]])
        self.Kinv = np.array([[1/fx,    0,-cx/fx],
                              [   0, 1/fy,-cy/fy],
                              [   0,    0,    1]])                            

    # project 3D point xc (w.r.t. camera frame) on the image plane [u,v]
    def project(self, xc):
        u = self.fx * xc[0]/xc[0] + self.cx
        v = self.fy * xc[1]/xc[0] + self.cy
        return u, v 

    # in:  pts_c  np array [nx2]
    # out: uvs    np array [nx2]
    def projectPoints(self, pts_c):
        uvs_hom = self.K @ add_ones(pts_c).T
        uvs = uvs_hom/uvs_hom[3]
        return uvs.T[:, 0:2]          

    # unproject 2D point uv (pixels on image plane) on 
    def unproject(self, uv):
        x = (uv[0] - self.cx)/self.fx
        y = (uv[1] - self.cy)/self.fy
        return x,y

    # in:  pts_hom_c  np array [nx3]
    # out: uvs [nx2]
    def unprojectPoints(self, uvs):
        return np.dot(self.Kinv, add_ones(uvs).T).T[:, 0:2]        

    def undistortPoints(self, uv):
        if self.is_distorted is True:
            uv_undistorted = cv2.undistortPoints(np.expand_dims(uv, axis=1), self.K, self.D, None, self.K)      
            return uv_undistorted.ravel().reshape(uv_undistorted.shape[0], 2)
        else:
            return uv 