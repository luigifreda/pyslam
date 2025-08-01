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

from .utils_sys import Printer
from .utils_geom_lie import so3_exp, so3_log, is_so3
from scipy.spatial.transform import Rotation 

from numba import jit


sign = lambda x: math.copysign(1, x)

# returns the difference between ang1 [deg] and ang2 [deg] in the manifold S1 (unit circle)
# result is the representation of the angle with the smallest absolute value
@jit(nopython=True)
def s1_diff_deg(angle1,angle2):
    diff = (angle1 - angle2) % 360.0 # now delta is in [0,360)
    if diff > 180.0:
        diff -= 360.0
    return diff 

# returns the positive distance between ang1 [deg] and ang2 [deg] in the manifold S1 (unit circle)
# result is smallest positive angle between ang1 and ang2
@jit(nopython=True)
def s1_dist_deg(angle1, angle2):
    diff = (angle1 - angle2) % 360.0 # now delta is in [0,360)
    if diff > 180.0:
        diff -= 360.0
    return math.fabs(diff) 

# returns the difference between ang1 [rad] and ang2 [rad] in the manifold S1 (unit circle)
# result is the representation of the angle with the smallest absolute value
k2pi=2.0*math.pi
@jit(nopython=True)
def s1_diff_rad(angle1,angle2):
    diff = (angle1 - angle2) % k2pi # now delta is in [0,k2pi)
    if diff > math.pi:
        diff -= k2pi
    return diff 

# returns the positive distance between ang1 [rad] and ang2 [rad] in the manifold S1 (unit circle)
# result is smallest positive angle between ang1 and ang2
@jit(nopython=True)
def s1_dist_rad(angle1,angle2):
    diff = (angle1 - angle2) % k2pi # now delta is in [0,k2pi)
    if diff > math.pi:
        diff -= k2pi
    return math.fabs(diff) 

            
# [4x4] homogeneous T from [3x3] R and [3x1] t          
@jit(nopython=True)   
def poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R
    ret[:3, 3] = t
    return ret   

# [4x4] homogeneous inverse T^-1 in SE(3) from T represented with [3x3] R and [3x1] t  
@jit(nopython=True)
def inv_poseRt(R, t):
    ret = np.eye(4)
    ret[:3, :3] = R.T
    ret[:3, 3] = -R.T @ np.ascontiguousarray(t)
    return ret     

# [4x4] homogeneous inverse T^-1 in SE(3)from [4x4] T  
@jit(nopython=True)
def inv_T(T):
    ret = np.eye(4)
    R_T = T[:3,:3].T
    t   = T[:3,3]
    ret[:3, :3] = R_T
    ret[:3, 3] = -R_T @ np.ascontiguousarray(t)
    return ret       


class Sim3Pose:
    def __init__(self, R: np.ndarray=np.eye(3, dtype=float), t: np.ndarray=np.zeros(3, dtype=float), s: float=1.0):
        self.R = R
        self.t = t.reshape(3,1)
        self.s = s
        assert s > 0
        self._T = None
        self._inv_T = None

    def __repr__(self):
        return f"Sim3Pose({self.R}, {self.t}, {self.s})"
    
    def from_matrix(self, T):
        if isinstance(T, np.ndarray):
            R = T[:3, :3]
            # Compute scale as the average norm of the rows of the rotation matrix
            #self.s = np.mean([np.linalg.norm(R[i, :]) for i in range(3)])  
            row_norms = np.linalg.norm(R, axis=1)
            self.s = row_norms.mean()
            self.R = R/self.s
            self.t = T[:3, 3].reshape(3,1)
        else:
            raise ValueError(f"Input T is not a numpy array. T={T}")
        return self
    
    def from_se3_matrix(self, T):
        if isinstance(T, np.ndarray):
            self.s = 1.0
            self.R = T[:3, :3]
            self.t = T[:3, 3].reshape(3,1)
        else:
            raise ValueError(f"Input T is not a numpy array. T={T}")
        return self    
    
    # corresponding homogeneous transformation matrix (4x4)
    def matrix(self):
        if self._T is None:
            self._T = poseRt(self.R*self.s, self.t.ravel())
        return self._T
    
    def inverse(self):
        return Sim3Pose(self.R.T, -1.0/self.s * self.R.T @ self.t, 1.0/self.s)
        
    # corresponding homogeneous inverse transformation matrix (4x4)
    def inverse_matrix(self):
        if self._inv_T is None:
            self._inv_T = np.eye(4)
            sR_inv = 1.0/self.s * self.R.T
            self._inv_T[:3, :3] = sR_inv
            self._inv_T[:3, 3] = -sR_inv @ self.t.ravel()
        return self._inv_T

    def to_se3_matrix(self):
        return poseRt(self.R, self.t.squeeze()/self.s) # [R t/s;0 1]
        
    def copy(self):
        return Sim3Pose(self.R.copy(), self.t.copy(), self.s)
        
    # map a 3D point
    def map(self, p3d):
        return (self.s * self.R @ p3d.reshape(3,1) + self.t)
    
    # map a set of 3D points [Nx3]
    def map_points(self, points):
        return (self.s * self.R @ points.T + self.t).T
        
    # Define the @ operator
    def __matmul__(self, other):
        result = None
        if isinstance(other, Sim3Pose):
            # Perform matrix multiplication within the class
            s_res = self.s * other.s
            R_res = self.R @ other.R
            t_res = self.s * self.R @ other.t + self.t
            result = Sim3Pose(R_res, t_res, s_res)
        elif isinstance(other, np.ndarray):
            if other.shape == (4,4):
                # Perform matrix multiplication with numpy (4x4) matrix      
                R_other = other[:3, :3]  # before scaling
                s_other = np.mean([np.linalg.norm(R_other[i, :]) for i in range(3)])  
                R_other = R_other/s_other
                t_other = other[:3, 3].reshape(3,1)
                s_res = self.s * s_other
                R_res = self.R @ R_other
                t_res = self.s * self.R @ t_other + self.t
                result = Sim3Pose(R_res, t_res, s_res)
            # elif (other.ndim == 1 and other.shape[0] == 3) or \
            #      (other.ndim == 2 and other.shape in [(3, 1), (1, 3)]):
            #     # Perform matrix multiplication with numpy (3x1) vector
            #     result = self.s * self.R @ other + self.t
            else: 
                raise TypeError(f"Unsupported operand type(s) for @: '{type(self)}' and '{type(other)}' with shape {other.shape}")
        else:
            raise TypeError("Unsupported operand type(s) for @: '{}' and '{}'".format(type(self), type(other)))
        return result
    
    def __str__(self):
        return f"Sim3Pose(R={self.R}, t={self.t}, s={self.s})"


@jit(nopython=True)
def normalize_vector(v):
    norm = np.linalg.norm(v)
    if norm < 1.e-10: 
       return v, norm
    return v/norm, norm

@jit(nopython=True)
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
@jit(nopython=True)
def add_ones_1D(x):
    #return np.concatenate([x,np.array([1.0])], axis=0)
    return np.array([x[0], x[1], 1])
    #return np.append(x, 1)
    
@jit(nopython=True)
def add_ones_numba(uvs):
    N = uvs.shape[0]
    out = np.ones((N, 3), dtype=uvs.dtype)
    out[:, 0:2] = uvs
    return out
    
# turn [[x,y,w]]= Kinv*[u,v,1] into [[x/w,y/w,1]]
@jit(nopython=True)
def normalize(Kinv, pts):
    return np.dot(Kinv, add_ones(pts).T).T[:, 0:2]        


# turn w in its skew symmetrix matrix representation 
# w in IR^3 -> [0,-wz,wy],
#              [wz,0,-wx],
#              [-wy,wx,0]]
@jit(nopython=True)
def skew(w):
    wx,wy,wz = w.ravel()    
    return np.array([[0,-wz,wy],[wz,0,-wx],[-wy,wx,0]])


@jit(nopython=True)
def hamming_distance(a, b):
    #r = (1 << np.arange(8))[:,None]
    #return np.count_nonzero((np.bitwise_xor(a,b) & r) != 0)   
    return np.count_nonzero(a!=b)

def hamming_distances(a, b):
    return np.count_nonzero(a!=b,axis=1)


@jit(nopython=True)
def l2_distance(a, b):
    return np.linalg.norm(a.ravel()-b.ravel())

def l2_distances(a,b):
    return np.linalg.norm(a-b, axis=-1, keepdims=True)    


# z rotation, input in radians      
@jit(nopython=True)
def yaw_matrix(yaw):
    return np.array([
    [math.cos(yaw), -math.sin(yaw), 0.0],
    [math.sin(yaw),  math.cos(yaw), 0.0],
    [          0.0,            0.0, 1.0]
    ])    
    
# y rotation, input in radians      
@jit(nopython=True)
def pitch_matrix(pitch):
    return np.array([
    [ math.cos(pitch), 0.0, math.sin(pitch)],
    [             0.0, 1.0,             0.0],
    [-math.sin(pitch), 0.0, math.cos(pitch)]
    ]) 
    
# x rotation, input in radians          
@jit(nopython=True)
def roll_matrix(roll):
    return np.array([
    [1.0,            0.0,             0.0],
    [0.0, math.cos(roll), -math.sin(roll)],
    [0.0, math.sin(roll),  math.cos(roll)]
    ])    
    
    
@jit(nopython=True)
def rotation_matrix_from_yaw_pitch_roll(yaw_degs, pitch_degs, roll_degs):
    # Convert angles from degrees to radians
    yaw = np.radians(yaw_degs)
    pitch = np.radians(pitch_degs)
    roll = np.radians(roll_degs)
    # Rotation matrix for Roll (X-axis rotation)
    Rx = np.array([
        [1.0,          0.0,           0.0],
        [0.0, np.cos(roll), -np.sin(roll)],
        [0.0, np.sin(roll),  np.cos(roll)]
    ])
    # Rotation matrix for Pitch (Y-axis rotation)
    Ry = np.array([
        [ np.cos(pitch), 0.0, np.sin(pitch)],
        [           0.0, 1.0,           0.0],
        [-np.sin(pitch), 0.0, np.cos(pitch)]
    ])
    # Rotation matrix for Yaw (Z-axis rotation)
    Rz = np.array([
        [np.cos(yaw), -np.sin(yaw), 0.0],
        [np.sin(yaw),  np.cos(yaw), 0.0],
        [0.0,          0.0,         1.0]
    ])
    # Final rotation matrix: R = Rz * Ry * Rx
    R = Rz @ Ry @ Rx
    return R

# rotation matrix to rpy
def rpy_from_rotation_matrix(R):
    return Rotation.from_matrix(R).as_euler('xyz', degrees=False)

def euler_from_rotation(R, order='xyz'):
    return Rotation.from_matrix(R).as_euler(order, degrees=False)



@jit(nopython=True)
def rodrigues_rotation_matrix(axis, angle):
    """
    Compute the rotation matrix from axis and angle using Rodrigues' formula.
    
    Parameters:
    - axis: unit vector (3,) — axis of rotation
    - angle: float — rotation angle in radians
    
    Returns:
    - R: (3, 3) rotation matrix
    """
    K = np.array([
        [0.0, -axis[2], axis[1]],
        [axis[2], 0.0, -axis[0]],
        [-axis[1], axis[0], 0.0]
    ])
    I = np.eye(3)
    R = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    return R

@jit(nopython=True)
def clip_scalar(x, x_min, x_max):
    if x < x_min:
        return x_min
    elif x > x_max:
        return x_max
    else:
        return x
    
# from z-vector to rotation matrix
# input: vector is a 3x1 vector
@jit(nopython=True)
def get_rotation_from_z_vector(vector):
    vector_norm = np.linalg.norm(vector)
    if vector_norm < 1e-8:
        return np.eye(3)

    # Normalize the input vector
    v = vector / vector_norm

    # Reference direction (z-axis)
    z = np.array([0.0, 0.0, 1.0])

    # Compute rotation axis and angle
    axis = np.cross(z, v)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-8:
        # Already aligned or opposite
        if np.dot(z, v) > 0:
            return np.eye(3)
        else:
            # 180-degree rotation: flip around X or Y
            return np.array([
                [-1.0,  0.0,  0.0],
                [ 0.0, -1.0,  0.0],
                [ 0.0,  0.0,  1.0]
            ])

    axis = axis / axis_norm
    angle = np.arccos(clip_scalar(np.dot(z, v), -1.0, 1.0))
    return rodrigues_rotation_matrix(axis, angle)

    
# Checks if a matrix is a valid rotation matrix
@jit(nopython=True)
def is_rotation_matrix(R):
    Rt = R.T
    should_be_identity = Rt @ R
    I = np.eye(R.shape[0])
    norm_diff = np.linalg.norm(should_be_identity - I)

    det = np.linalg.det(R)
    return norm_diff < 1e-8 and abs(det - 1.0) < 1e-6

# Computes the closest orthogonal matrix to a given matrix.
def closest_orthogonal_matrix(A):
  # Singular Value Decomposition
  U, _, Vt = np.linalg.svd(A)
  R = np.dot(U, Vt)
  return R

# Computes the closest rotation matrix to a given matrix.
def closest_rotation_matrix(A):
  Q = closest_orthogonal_matrix(A)
  detQ = np.linalg.det(Q)
  if detQ < 0:
    Q[:, -1] *= -1
  return Q



# from quaternion vector to rotation matrix
# input: qvec = [qx, qy, qz, qw]
@jit(nopython=True)
def qvec2rotmat(qvec):
    qx, qy, qz, qw = qvec
    return np.array([
        [
            1.0 - 2.0 * (qy**2 + qz**2),
            2.0 * (qx * qy - qw * qz),
            2.0 * (qx * qz + qw * qy),
        ],
        [
            2.0 * (qx * qy + qw * qz),
            1.0 - 2.0 * (qx**2 + qz**2),
            2.0 * (qy * qz - qw * qx),
        ],
        [
            2.0 * (qx * qz - qw * qy),
            2.0 * (qy * qz + qw * qx),
            1.0 - 2.0 * (qx**2 + qy**2),
        ],
    ])

# from rotation matrix to quaternion vector
# input: R is a 3x3 rotation matrix
# output: qvec = [qx, qy, qz, qw]
# implements the eigenvalue decomposition approach using the matrix K (the Shepperd method).
def rotmat2qvec(R):
    Rxx, Ryx, Rzx, Rxy, Ryy, Rzy, Rxz, Ryz, Rzz = R.ravel()
    K = np.array([
        [Rxx - Ryy - Rzz, Ryx + Rxy, Rzx + Rxz, Ryz - Rzy],
        [Ryx + Rxy, Ryy - Rxx - Rzz, Rzy + Ryz, Rzx - Rxz],
        [Rzx + Rxz, Rzy + Ryz, Rzz - Rxx - Ryy, Rxy - Ryx],
        [Ryz - Rzy, Rzx - Rxz, Rxy - Ryx, Rxx + Ryy + Rzz],
    ]) / 3.0
    eigvals, eigvecs = np.linalg.eigh(K)
    qvec = eigvecs[:, np.argmax(eigvals)]  # Eigenvector corresponding to largest eigenvalue
    if qvec[3] < 0:  # Ensure consistent quaternion sign
        qvec *= -1
    return qvec[[0, 1, 2, 3]]  # Return as [qx, qy, qz, qw]


# input: x,y,z,qx,qy,qz,qw
# output: 4x4 transformation matrix
def xyzq2Tmat(x,y,z,qx,qy,qz,qw):
    R = qvec2rotmat([qx,qy,qz,qw])
    return np.array([[R[0,0],R[0,1],R[0,2],x],[R[1,0],R[1,1],R[1,2],y],[R[2,0],R[2,1],R[2,2],z],[0,0,0,1]])


# we compute H = K*(R-t*n'/d)*Kinv  with n=(0,0,1)' and d=1  <-- Hartley-Zisserman pag 327  
# => the plane containing `img` is on the optical axis (i.e. along z=(0,0,1)) at a distance d=1, i.e. the plane is Z=1
# we set fx = fy = img.shape[1] = img.width  
# =>  on the plane Z=1 (in front of the camera) we have that 1 meter correspond to fx=img.width pixels  
# =>  tx=0.5 implies a shift of half image width
# input in meters and radians     
@jit(nopython=True)
def homography_matrix(img,roll,pitch,yaw,tx=0,ty=0,tz=0):
    d=1
    Rwc = (yaw_matrix(yaw) @ pitch_matrix(pitch) @ roll_matrix(roll))
    Rcw = Rwc.T
    fx = fy = img.shape[1]  # => on the plane Z=1 we have that 1 meter correspond to fx = img.width pixels
    (h, w) = img.shape[:2]
    cx, cy = w/2, h/2        
    K = np.array([[  fx,  0.0,   cx],
                  [ 0.0,   fy,   cy],
                  [ 0.0,  0.0,  1.0]])
    Kinv = np.array([[1.0/fx,    0.0, -cx/fx],
                     [   0.0, 1.0/fy, -cy/fy],
                     [   0.0,    0.0,    1.0]])
    # now we compute t*n' where t=(tx,ty,tz)' and n=(0,0,1)'
    t_n = np.array([[ 0.0, 0.0, tx],
                    [ 0.0, 0.0, ty],
                    [ 0.0, 0.0, tz]])/d
    H = K @ (Rcw - t_n) @ Kinv   
    return H  