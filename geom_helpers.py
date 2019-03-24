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

import os
import numpy as np
import cv2


def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret    


# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    if len(x.shape) == 1:
        return np.concatenate([x,np.array([1.0])], axis=0)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# turn [[x,y,w]] -> [[x/w,y/w,1]]
def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]        


# turn w in its skew symmetrix matrix representation 
# w in IR^3 -> [0,-wz,wy],
#              [wz,0,-wx],
#              [-wy,wx,0]]
def skew(w):
    wx,wy,wz = w.ravel()    
    return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])


def hamming_distance(a, b):
    r = (1 << np.arange(8))[:,None]
    return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)   

def l2_distance(a, b):
    return np.linalg.norm(a.ravel()-b.ravel())


# DLT with normalized image coordinates (see [HartleyZisserman Sect. 12.2 ])
def triangulate_point(pose1, pose2, pt1, pt2):      
    A = np.zeros((4,4))
    A[0] = pt1[0] * pose1[2] - pose1[0]
    A[1] = pt1[1] * pose1[2] - pose1[1]
    A[2] = pt2[0] * pose2[2] - pose2[0]
    A[3] = pt2[1] * pose2[2] - pose2[1]
    _, _, vt = np.linalg.svd(A)
    return vt[3]


def triangulate_points(pose1, pose2, pts1, pts2, mask = None): 
    if mask is not None: 
        return triangulate_points_with_mask(pose1, pose2, pts1, pts2, mask)
    ret = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)):
        ret[i] = triangulate_point(pose1, pose2, p[0], p[1])
    return ret  


def triangulate_points_with_mask(pose1, pose2, pts1, pts2, mask):      
    ret = np.zeros((pts1.shape[0], 4))
    for i, p in enumerate(zip(pts1, pts2)): 
        if mask[i]:
            ret[i] = triangulate_point(pose1, pose2, p[0], p[1])
    return ret   


# create a generator over an image to extract 'row_divs' x 'col_divs' sub-blocks 
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



## Drawing stuff ##

# draw a list of points with different random colors
def draw_points(img, pts, radius=5): 
    if img.ndim < 3:
        img = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
    for pt in pts:
        color = tuple(np.random.randint(0,255,3).tolist())
        img = cv2.circle(img,tuple(pt),radius,color,-1)
    return img    

# draw corresponding points with the same random color 
def draw_points2(img1, img2, pts1, pts2, radius=5): 
    if img1.ndim < 3:
        img1 = cv2.cvtColor(img1,cv2.COLOR_GRAY2BGR)
    if img2.ndim < 3:        
        img2 = cv2.cvtColor(img2,cv2.COLOR_GRAY2BGR)
    for pt1,pt2 in zip(pts1,pts2):
        color = tuple(np.random.randint(0,255,3).tolist())
        img1 = cv2.circle(img1,tuple(pt1),radius,color,-1)
        img2 = cv2.circle(img2,tuple(pt2),radius,color,-1)
    return img1,img2    

# line_edges is assumed to be a list of 2D img points
def draw_lines(img, line_edges, pts=None, radius=5):
    pt = None 
    for i,l in enumerate(line_edges):
        color = tuple(np.random.randint(0,255,3).tolist())
        x0,y0 = l[0]
        x1,y1 = l[1]
        img = cv2.line(img, (int(x0),int(y0)), (int(x1),int(y1)), color,1)
        if pts is not None: 
            pt = pts[i]        
            img = cv2.circle(img,tuple(pt),radius,color,-1)
    return img


## SIFT stuff ##

# from https://stackoverflow.com/questions/48385672/opencv-python-unpack-sift-octave
def unpackSiftOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @brief Unpack Sift Keypoint
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1/(1<<octave))
    else:
        scale = float(1<<-octave)
    return (octave, layer, scale)

def unpackSiftOctaveSimple(kpt):
    """unpackSIFTOctave(kpt)->octave
    @brief Unpack Sift Keypoint
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    if octave>=128:
        octave |= -128        
    return abs(octave)   # N.B.: abs() is an HACK that needs to be further verified 

def unpackSiftOctaveKps(kps):    
    for kpt in kps: 
        kpt.octave = unpackSiftOctaveSimple(kpt)        