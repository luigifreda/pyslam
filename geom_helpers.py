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
* along with PYVO. If not, see <http://www.gnu.org/licenses/>.
"""

import os
import numpy as np
import cv2


# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    if len(x.shape) == 1:
        return np.concatenate([x,np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)


# turn w in its skew symmetrix matrix representation 
def skew(w):
    wx,wy,wz = w    
    return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])


def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)    


# DLT with normalized image coordinates (see [HartleyZisserman Sect. 12.2 ])
def triangulate(pose1, pose2, pts1, pts2):    
    ret = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)):
        A = np.zeros((4,4))
        A[0] = p[0][0] * pose1[2] - pose1[0]
        A[1] = p[0][1] * pose1[2] - pose1[1]
        A[2] = p[1][0] * pose2[2] - pose2[0]
        A[3] = p[1][1] * pose2[2] - pose2[1]
        _, _, vt = np.linalg.svd(A)
        ret[i] = vt[3]
    return ret    


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret    


def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]        

# create a generator over an image to extract its sub-blocks
def imgBlocks(img, row_divs, col_divs):
    rows, cols = img.shape[:2]
    #print('img.shape: ', img.shape)
    xs = np.uint32(np.rint(np.linspace(0, cols, num=col_divs+1)))   # num = Number of samples to generate
    ys = np.uint32(np.rint(np.linspace(0, rows, num=row_divs+1)))
    #print('imgBlocks xs: ', xs)
    #print('imgBlocks ys: ', ys)
    ystarts, yends = ys[:-1], ys[1:]
    xstarts, xends = xs[:-1], xs[1:]
    for y1, y2 in zip(ystarts, yends):
        for x1, x2 in zip(xstarts, xends):
            yield img[y1:y2, x1:x2], y1, x1    # return block, row, col