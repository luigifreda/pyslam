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
from utils_sys import Printer


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


class Sim3Pose:
    def __init__(self, R: np.ndarray=np.eye(3, dtype=float), t: np.ndarray=np.zeros(3, dtype=float), s: float=1.0):
        self.R = R
        self.t = t.reshape(3,1)
        self.s = s
        
        self._T = None
        self._inv_T = None

    def __repr__(self):
        return f"Sim3Pose({self.R}, {self.t}, {self.s})"
    
    def from_matrix(self, T):
        if isinstance(T, np.ndarray):
            R = T[:3, :3]
            # Compute scale as the average norm of the rows of the rotation matrix
            self.s = np.mean([np.linalg.norm(R[i, :]) for i in range(3)])     
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
            self._T = poseRt(self.R*self.s, self.t)
        return self._T
    
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
    
    def inverse(self):
        return Sim3Pose(self.R.T, -1.0/self.s * self.R.T @ self.t, 1.0/self.s)
    
    def copy(self):
        return Sim3Pose(self.R.copy(), self.t.copy(), self.s)
        
    def map(self, p3d):
        return (self.s * self.R @ p3d.reshape(3,1) + self.t)
        
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
                s_other = np.trace(other[:3, :3])/3.0
                R_other = other[:3, :3]/s_other
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


# from quaternion vector to rotation matrix
# input: qvec = [qx, qy, qz, qw]
def qvec2rotmat(qvec):
    qx, qy, qz, qw = qvec
    return np.array([
        [
            1 - 2 * (qy**2 + qz**2),
            2 * (qx * qy - qw * qz),
            2 * (qx * qz + qw * qy),
        ],
        [
            2 * (qx * qy + qw * qz),
            1 - 2 * (qx**2 + qz**2),
            2 * (qy * qz - qw * qx),
        ],
        [
            2 * (qx * qz - qw * qy),
            2 * (qy * qz + qw * qx),
            1 - 2 * (qx**2 + qy**2),
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

    
# Checks if a matrix is a valid rotation matrix
def is_rotation_matrix(R):
  Rt = np.transpose(R)
  should_be_identity = np.dot(Rt, R)
  identity = np.identity(len(R))
  n = np.linalg.norm(should_be_identity - identity)
  return n < 1e-8 and np.allclose(np.linalg.det(R), 1.0)

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


class AlignmentGroundTruthData:
    def __init__(self, timestamps_associations=[], filter_t_w_i=[], gt_t_w_i=[], T_gt_est=None, error=-1.0, is_aligned=False):
        self.timestamps_associations = timestamps_associations
        self.filter_t_w_i = filter_t_w_i
        self.gt_t_w_i = gt_t_w_i
        self.T_gt_est = T_gt_est
        self.error = error
        self.is_aligned = is_aligned
        
    def copyTo(self, other):
        other.timestamps_associations = self.timestamps_associations
        other.filter_t_w_i = self.filter_t_w_i
        other.gt_t_w_i = self.gt_t_w_i
        other.T_gt_est = self.T_gt_est
        other.error = self.error
        other.is_aligned = self.is_aligned


# align filter trajectory with ground truth trajectory by computing the SE(3) transformation between the two
# - filter_timestamps [Nx1]
# - filter_t_w_i [Nx3]
# - gt_timestamps [Nx1]
# - gt_t_w_i [Nx3]
# - find_scale allows to compute the full Sim(3) transformation in case the scale is unknown
def align_trajs_with_svd(filter_timestamps, filter_t_w_i, gt_timestamps, gt_t_w_i, align_gt=True, compute_align_error=True, find_scale=False, verbose=False):
    est_associations = []
    gt_associations = []
    timestamps_associations = []

    if verbose:
        print(f'filter_timestamps: {filter_timestamps.shape}')
        print(f'filter_t_w_i: {filter_t_w_i.shape}')
        print(f'gt_timestamps: {gt_timestamps.shape}')
        print(f'gt_t_w_i: {gt_t_w_i.shape}')        
        print(f'filter_timestamps: {filter_timestamps}')
        print(f'gt_timestamps: {gt_timestamps}')

    for i in range(len(filter_t_w_i)):
        timestamp = filter_timestamps[i]

        # Find the index in gt_timestamps where gt_timestamps[j] > timestamp
        j = 0
        while j < len(gt_timestamps) and gt_timestamps[j] <= timestamp:
            j += 1
        j -= 1
        assert j>=0, f'j {j}'
        
        if j >= len(gt_timestamps) - 1:
            continue

        dt = timestamp - gt_timestamps[j]
        dt_gt = gt_timestamps[j + 1] - gt_timestamps[j]

        assert dt >= 0, f"dt {dt}"
        assert dt_gt > 0, f"dt_gt {dt_gt}"

        # Skip if the interval between gt is larger than 100ms
        # if dt_gt > 1.1e8:
        #     continue

        ratio = dt / dt_gt

        assert 0 <= ratio < 1

        #t_gt = (1 - ratio) * gt_timestamps[j] + ratio * gt_timestamps[j + 1]
        gt = (1 - ratio) * gt_t_w_i[j] + ratio * gt_t_w_i[j + 1]

        timestamps_associations.append(timestamp)
        gt_associations.append(gt)
        est_associations.append(filter_t_w_i[i])

    num_samples = len(est_associations)
    if verbose: 
        print(f'num associations: {num_samples}')

    gt = np.zeros((3, num_samples))
    est = np.zeros((3, num_samples))

    for i in range(num_samples):
        gt[:, i] = gt_associations[i]
        est[:, i] = est_associations[i]

    mean_gt = np.mean(gt, axis=1)
    mean_est = np.mean(est, axis=1)

    gt -= mean_gt[:, None]
    est -= mean_est[:, None]

    cov = np.dot(gt, est.T)
    if find_scale:
        # apply Kabsch–Umeyama algorithm
        cov /= gt.shape[0]
        variance_gt = np.mean(np.linalg.norm(gt,axis=1)**2) 

    try: 
        U, D, Vt = np.linalg.svd(cov)
    except: 
        Printer.red('[align_trajs_with_svd] SVD failed!!!\n')
        return np.eye(4), -1, AlignmentGroundTruthData()

    c = 1
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    if find_scale: 
        # apply Kabsch–Umeyama algorithm
        c = variance_gt/np.trace(np.diag(D) @ S)

    rot_gt_est = np.dot(U, np.dot(S, Vt))
    trans = mean_gt - c * np.dot(rot_gt_est, mean_est)

    T_gt_est = np.eye(4)
    T_gt_est[:3, :3] = c * rot_gt_est
    T_gt_est[:3, 3] = trans

    #T_est_gt = np.linalg.inv(T_gt_est)
    T_est_gt = np.eye(4)  # Identity matrix initialization
    R_gt_est = T_gt_est[:3, :3]
    t_gt_est = T_gt_est[:3, 3]
    if find_scale:
        # Compute scale as the average norm of the rows of the rotation matrix
        s = np.mean([np.linalg.norm(R_gt_est[i, :]) for i in range(3)])
        R = R_gt_est / s
        sR_inv = (1.0 / s) * R.T
        T_est_gt[:3, :3] = sR_inv
        T_est_gt[:3, 3] = -sR_inv @ t_gt_est.ravel()
    else:
        T_est_gt[:3, :3] = R_gt_est.T
        T_est_gt[:3, 3] = -R_gt_est.T @ t_gt_est.ravel()  
        
    # Update gt_t_w_i with transformation
    if align_gt:
        for i in range(len(gt_t_w_i)):
            gt_t_w_i[i] = np.dot(T_est_gt[:3, :3], gt_t_w_i[i]) + T_est_gt[:3, 3]

    # Compute error
    error = 0
    if compute_align_error:
        for i in range(len(est_associations)):
            res = (np.dot(T_gt_est[:3, :3], est_associations[i]) + T_gt_est[:3, 3]) - gt_associations[i]
            error += np.dot(res.T, res)

        error /= len(est_associations)
        error = np.sqrt(error)

    aligned_gt_data = AlignmentGroundTruthData(timestamps_associations, est_associations, gt_associations, T_gt_est, error, is_aligned=align_gt)

    return T_gt_est, error, aligned_gt_data