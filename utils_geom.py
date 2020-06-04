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
import math

sign = lambda x: math.copysign(1, x)

# returns the difference between ang1 [deg] and ang2 [deg] in the manifold S1 (unit circle)
# result is the representation of the angle with the smallest absolute value
def s1_diff_deg(angle1,angle2):
    diff = (angle1 - angle2) % 360 # now delta is in [0,360)
    if diff > 180:
        diff -= 360
    return diff 

# returns the positive distance between ang1 [deg] and ang2 [deg] in the manifold S1 (unit circle)
# result is smallest positive angle between ang1 and ang2
def s1_dist_deg(angle1, angle2):
    diff = (angle1 - angle2) % 360 # now delta is in [0,360)
    if diff > 180:
        diff -= 360
    return math.fabs(diff) 

# returns the difference between ang1 [rad] and ang2 [rad] in the manifold S1 (unit circle)
# result is the representation of the angle with the smallest absolute value
k2pi=2*math.pi
def s1_diff_rad(angle1,angle2):
    diff = (angle1 - angle2) % k2pi # now delta is in [0,k2pi)
    if diff > math.pi:
        diff -= k2pi
    return diff 

# returns the positive distance between ang1 [rad] and ang2 [rad] in the manifold S1 (unit circle)
# result is smallest positive angle between ang1 and ang2
def s1_dist_rad(angle1,angle2):
    diff = (angle1 - angle2) % k2pi # now delta is in [0,k2pi)
    if diff > math.pi:
        diff -= k2pi
    return math.fabs(diff) 

            
# [4x4] homogeneous T from [3x3] R and [3x1] t             
def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret   

# [4x4] homogeneous inverse T^-1 from T represented with [3x3] R and [3x1] t  
def inv_poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R.T
    ret[:3, 3] = -R.T @ t
    return ret     

# [4x4] homogeneous inverse T^-1 from [4x4] T     
def inv_T(T):
    ret = np.eye(4)
    R_T = T[:3,:3].T
    t   = T[:3,3]
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ t
    return ret       

def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1.e-10: 
       return v, norm
    return v/norm, norm

def normalize_vector2(v):
    norm = np.linalg.norm(v)
    if norm < 1.e-10: 
       return v
    return v/norm

# turn [[x,y]] -> [[x,y,1]]
def add_ones(x):
    if len(x.shape) == 1:
        return add_ones_1D(x)
    else:
        return np.concatenate([x, np.ones((x.shape[0], 1))], axis=1)

# turn [[x,y]] -> [[x,y,1]]
def add_ones_1D(x):
    #return np.concatenate([x,np.array([1.0])], axis=0)
    return np.array([x[0], x[1], 1])
    #return np.append(x, 1)
    
    
# turn [[x,y,w]]= Kinv*[u,v,1] into [[x/w,y/w,1]]
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
    #r = (1 << np.arange(8))[:,None]
    #return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)   
    return np.count_nonzero(a!=b)

def hamming_distances(a, b):
    return np.count_nonzero(a!=b,axis=1)

def l2_distance(a, b):
    return np.linalg.norm(a.ravel()-b.ravel())

def l2_distances(a,b):
    return np.linalg.norm(a-b, axis=-1, keepdims=True)    


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


def triangulate_normalized_points(pose_1w, pose_2w, kpn_1, kpn_2):
    # P1w = np.dot(K1,  M1w) # K1*[R1w, t1w]
    # P2w = np.dot(K2,  M2w) # K2*[R2w, t2w]
    # since we are working with normalized coordinates x_hat = Kinv*x, one has         
    P1w = pose_1w[:3,:] # [R1w, t1w]
    P2w = pose_2w[:3,:] # [R2w, t2w]

    point_4d_hom = cv2.triangulatePoints(P1w, P2w, kpn_1.T, kpn_2.T)
    good_pts_mask = np.where(point_4d_hom[3]!= 0)[0]
    point_4d = point_4d_hom / point_4d_hom[3] 
    
    if __debug__:
        if False: 
            point_reproj = P1w @ point_4d;
            point_reproj = point_reproj / point_reproj[2] - add_ones(kpn_1).T
            err = np.sum(point_reproj**2)
            print('reproj err: ', err)     

    #return point_4d.T
    points_3d = point_4d[:3, :].T
    return points_3d, good_pts_mask  


# compute the fundamental mat F12 and the infinite homography H21 [Hartley Zisserman pag 339]
def computeF12(f1, f2):
    R1w = f1.Rcw
    t1w = f1.tcw 
    R2w = f2.Rcw
    t2w = f2.tcw

    R12 = R1w @ R2w.T
    t12 = -R1w @ (R2w.T @ t2w) + t1w
    
    t12x = skew(t12)
    K1Tinv = f1.camera.Kinv.T
    R21 = R12.T
    H21 = (f2.camera.K @ R21) @ f1.camera.Kinv  # infinite homography from f1 to f2 [Hartley Zisserman pag 339]
    F12 = ( (K1Tinv @ t12x) @ R12 ) @ f2.camera.Kinv
    return F12, H21  


def check_dist_epipolar_line(kp1,kp2,F12,sigma2_kp2):
    # Epipolar line in second image l = kp1' * F12 = [a b c]
    l = np.dot(F12.T,np.array([kp1[0],kp1[1],1]))
    num = l[0]*kp2[0] + l[1]*kp2[1] + l[2]  # kp1' * F12 * kp2
    den = l[0]*l[0] + l[1]*l[1]   # a*a+b*b

    if(den==0):
    #if(den < 1e-20):
        return False

    dist_sqr = num*num/den              # squared (minimum) distance of kp2 from the epipolar line l
    return dist_sqr < 3.84 * sigma2_kp2 # value of inverse cumulative chi-square for 1 DOF (Hartley Zisserman pag 567)



# fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
# input: kpn_ref and kpn_cur are two arrays of [Nx2] normalized coordinates of matched keypoints 
# out: a) Trc: homogeneous transformation matrix containing Rrc, trc  ('cur' frame with respect to 'ref' frame)    pr = Trc * pc 
#      b) mask_match: array of N elements, every element of which is set to 0 for outliers and to 1 for the other points (computed only in the RANSAC and LMedS methods)
# N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
# N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
# - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie a ruled quadric 
# - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
# N.B.3: the five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
# N.B.4: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return a correct rotation 
# N.B.5: the OpenCV findEssentialMat function uses the five-point algorithm solver by D. Nister => hence it should work well in the degenerate planar cases
def estimate_pose_ess_mat(kpn_ref, kpn_cur, method=cv2.RANSAC, prob=0.999, threshold=0.0003):	
    # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above )     
    E, mask_match = cv2.findEssentialMat(kpn_cur, kpn_ref, focal=1, pp=(0., 0.), method=method, prob=prob, threshold=threshold)                         
    _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0., 0.))   
    return poseRt(R,t.T), mask_match  # Trc, mask_mat         


# z rotation, input in radians      
def yaw_matrix(yaw):
    return np.array([
    [math.cos(yaw), -math.sin(yaw), 0],
    [math.sin(yaw),  math.cos(yaw), 0],
    [            0,              0, 1]
    ])    
    
# y rotation, input in radians      
def pitch_matrix(pitch):
    return np.array([
    [ math.cos(pitch), 0, math.sin(pitch)],
    [              0,  1,               0],
    [-math.sin(pitch), 0, math.cos(pitch)]
    ]) 
    
# x rotation, input in radians          
def roll_matrix(roll):
    return np.array([
    [1,              0,               0],
    [0, math.cos(roll), -math.sin(roll)],
    [0, math.sin(roll),  math.cos(roll)]
    ])    


# we compute H = K*(R-t*n'/d)*Kinv  with n=(0,0,1)' and d=1  <-- Hartley-Zisserman pag 327  
# => the plane containing `img` is on the optical axis (i.e. along z=(0,0,1)) at a distance d=1, i.e. the plane is Z=1
# we set fx = fy = img.shape[1] = img.width  
# =>  on the plane Z=1 (in front of the camera) we have that 1 meter correspond to fx=img.width pixels  
# =>  tx=0.5 implies a shift of half image width
# input in meters and radians     
def homography_matrix(img,roll,pitch,yaw,tx=0,ty=0,tz=0):
    d=1
    Rwc = (yaw_matrix(yaw) @ pitch_matrix(pitch) @ roll_matrix(roll))
    Rcw = Rwc.T
    fx = fy = img.shape[1]  # => on the plane Z=1 we have that 1 meter correspond to fx = img.width pixels
    (h, w) = img.shape[:2]
    cx, cy = w/2, h/2        
    K = np.array([[fx,  0, cx],
                  [  0, fy, cy],
                  [  0,  0,  1]])
    Kinv = np.array([[1/fx,    0,-cx/fx],
                     [    0, 1/fy,-cy/fy],
                     [    0,    0,    1]])
    # now we compute t*n' where t=(tx,ty,tz)' and n=(0,0,1)'
    t_n = np.array([[ 0, 0, tx],
                    [ 0, 0, ty],
                    [ 0, 0, tz]])/d
    H = K @ (Rcw - t_n) @ Kinv   
    return H  

    



