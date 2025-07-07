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
from .utils_geom import *


def set_rotations_from_translations(t_wi):
    num_samples = len(t_wi)
    R_wi = [np.eye(3) for i in range(num_samples)] 
    for i in range(1, num_samples):
        R_wi[i] = get_rotation_from_z_vector(t_wi[i] - t_wi[i-1])
    if num_samples>1:
        R_wi[0] = R_wi[1]
    return R_wi


# Align corresponding 3D points in SE(3)/Sime(3) using SVD, we assume the points have been already associated, i.e. gt_points[i] corresponds to est_points[i].
# Here, we use the algorith Kabsch–Umeyama to compute the optimal rotation and translation between the two sets of points.
# Reference: "Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991
# https://web.stanford.edu/class/cs273/refs/umeyama.pdf
# input: gt_points [Nx3] and est_points [Nx3] are the corresponding 3D points
def align_3d_points_with_svd(gt_points, est_points, find_scale=True):
    assert len(gt_points) == len(est_points), "The number of points must be the same"
    is_ok = False 
    
    # Next, align the two trajectories on the basis of their associations
    gt = np.array(gt_points).T     # 3xN
    est = np.array(est_points).T   # 3xN

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
        Printer.red('[align_3d_points_with_svd] SVD failed!!!\n')
        return np.eye(4), is_ok

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

    T_est_gt = np.eye(4)  # Identity matrix initialization
    R_gt_est = T_gt_est[:3, :3]
    t_gt_est = T_gt_est[:3, 3]
    if find_scale:
        # Compute scale as the average norm of the rows of the rotation matrix
        s = c # np.mean([np.linalg.norm(R_gt_est[i, :]) for i in range(3)])
        R = rot_gt_est # R_gt_est / s
        sR_inv = (1.0 / s) * R.T
        T_est_gt[:3, :3] = sR_inv
        T_est_gt[:3, 3] = -sR_inv @ t_gt_est.ravel()
    else:
        T_est_gt[:3, :3] = R_gt_est.T
        T_est_gt[:3, 3] = -R_gt_est.T @ t_gt_est.ravel() 
    
    is_ok = True
    return T_gt_est, T_est_gt, is_ok  


# Find associations between 3D trajectories timestamps in filter and gt. For each filter timestamp, find the closest gt timestamps
# and interpolate the 3D trajectory between them.
# Inputs: filter_timestamps: List of filter timestamps assumed to be in seconds
#         filter_t_wi: List of 3D positions, filter trajectory
#         gt_timestamps: List of gt timestamps assumed to be in seconds
#         gt_t_wi: List of 3D positions, ground truth trajectory
#         max_align_dt: maximum time difference between filter and gt timestamps in seconds
def find_trajectories_associations(filter_timestamps, filter_t_wi, gt_timestamps, gt_t_wi, max_align_dt=1e-1, verbose=True):
    max_dt = 0

    filter_associations = []
    gt_associations = []
    timestamps_associations = []
    
    # First, find associations between timestamps in filter and gt
    for i in range(len(filter_t_wi)):
        timestamp = filter_timestamps[i]

        # Find the index in gt_timestamps where gt_timestamps[j] > timestamp
        j = 0
        while j < len(gt_timestamps) and gt_timestamps[j] <= timestamp:
            j += 1
        j -= 1
        #assert j>=0, f'j {j}'
        if j < 0:
            print(f'find_trajectories_associations: got j={j} at sample i={i}')
            continue        
        
        if j >= len(gt_timestamps) - 1:
            continue

        dt = timestamp - gt_timestamps[j]
        dt_gt = gt_timestamps[j + 1] - gt_timestamps[j]
        
        abs_dt = abs(dt)

        assert dt >= 0, f"dt {dt}"
        assert dt_gt > 0, f"dt_gt {dt_gt}"

        # Skip if the interval between gt is larger than max delta
        if abs_dt > max_align_dt:
            continue

        max_dt = max(max_dt, abs_dt)
        
        ratio = dt / dt_gt

        assert 0 <= ratio < 1, f"ratio {ratio}"

        #gt_t_interpolated = (1 - ratio) * gt_timestamps[j] + ratio * gt_timestamps[j + 1]  # equal to timestamp
        gt_t_wi_interpolated = (1 - ratio) * gt_t_wi[j] + ratio * gt_t_wi[j + 1]

        timestamps_associations.append(timestamp)
        gt_associations.append(gt_t_wi_interpolated)
        filter_associations.append(filter_t_wi[i])
    
    if verbose:
        print(f'find_trajectories_associations: max trajectory align dt: {max_dt}')
    
    return np.array(timestamps_associations), np.array(filter_associations), np.array(gt_associations)


# Associate each filter pose to a gt pose on the basis of their timestamps. Interpolate the gt poses where needed.
# We assume the gt poses are in SE(3).


# Find associations between poses in filter and gt. For each filter timestamp, find the closest gt timestamps
# and interpolate the pose between them.
# Inputs: filter_timestamps: List of filter timestamps assumed to be in seconds
#         filter_T_wi: List of filter poses in SE(3), each one a [4x4] transformation matrix
#         gt_timestamps: List of gt timestamps assumed to be in seconds
#         gt_T_wi: List of ground truth poses, each one a [4x4] transformation matrix
#         max_align_dt: maximum time difference between filter and gt timestamps in seconds
def find_poses_associations(filter_timestamps, filter_T_wi, gt_timestamps, gt_T_wi, max_align_dt=1e-1, verbose=True):
    max_dt = 0

    filter_associations = []
    gt_associations = []
    timestamps_associations = []
    
    # First, find associations between timestamps in filter and gt
    for i in range(len(filter_T_wi)):
        timestamp = filter_timestamps[i]

        # Find the index in gt_timestamps where gt_timestamps[j] > timestamp
        j = 0
        while j < len(gt_timestamps) and gt_timestamps[j] <= timestamp:
            j += 1
        j -= 1
        #assert j>=0, f'j {j}'
        if j < 0:
            print(f'find_trajectories_associations: got j={j} at sample i={i}')
            continue        
        
        if j >= len(gt_timestamps) - 1:
            continue

        dt = timestamp - gt_timestamps[j]
        dt_gt = gt_timestamps[j + 1] - gt_timestamps[j]
        
        abs_dt = abs(dt)

        assert dt >= 0, f"dt {dt}"
        assert dt_gt > 0, f"dt_gt {dt_gt}"

        # Skip if the interval between gt is larger than max delta
        if abs_dt > max_align_dt:
            continue

        max_dt = max(max_dt, abs_dt)
        
        ratio = dt / dt_gt

        assert 0 <= ratio < 1, f"ratio {ratio}"

        gt_t_wi_j = gt_T_wi[j][:3, 3]
        gt_R_wi_j = gt_T_wi[j][:3, :3]
        gt_t_wi_jp1 = gt_T_wi[j + 1][:3, 3]
        gt_R_wi_jp1 = gt_T_wi[j + 1][:3, :3]
        
        #t_gt_interpolated = (1 - ratio) * gt_timestamps[j] + ratio * gt_timestamps[j + 1] # equal to timestamp
        t_wi_gt_interpolated = (1 - ratio) * gt_t_wi_j + ratio * gt_t_wi_jp1  # gt_t_wi_j + ratio * (gt_t_wi_jp1-gt_t_wi_j)
        delta_R = gt_R_wi_jp1 @ gt_R_wi_j.T
        if not is_so3(delta_R):
            delta_R = closest_rotation_matrix(delta_R)
        R_wi_gt_interpolated = gt_R_wi_j @ so3_exp(ratio * so3_log(delta_R))

        timestamps_associations.append(timestamp)
        gt_associations.append(poseRt(R_wi_gt_interpolated, t_wi_gt_interpolated))
        filter_associations.append(filter_T_wi[i])
    
    if verbose:
        print(f'find_trajectories_associations: max trajectory align dt: {max_dt}')
    
    return np.array(timestamps_associations), np.array(filter_associations), np.array(gt_associations)


# Data representing the alignment of an estimated trajectory with ground truth
class TrajectoryAlignementData:
    def __init__(self, timestamps_associations=[], estimated_t_wi=[], gt_t_wi=[], T_gt_est=None, T_est_gt=None, rms_error=-1.0, is_est_aligned=False, max_error=-1.0):
        # Here we store associated data between estimated and ground truth trajectories
        # timestamps_associations [Nx1] assumed to be in seconds
        # estimated_t_wi [Nx3]      =>  estimated_t_wi[i] corresponds to gt_t_wi[i] at timestamps_associations[i]
        # gt_t_wi [Nx3]             =>  gt_t_wi[i] corresponds to estimated_t_wi[i] at timestamps_associations[i]
        # T_gt_est [4x4]           
        # T_est_gt [4x4]
        self.timestamps_associations = timestamps_associations
        self.estimated_t_wi = estimated_t_wi
        self.gt_t_wi = gt_t_wi
        self.T_gt_est = T_gt_est
        self.T_est_gt = T_est_gt
        self.rms_error = rms_error           # average alignment error
        self.max_error = max_error   # max alignement error 
        self.is_est_aligned = is_est_aligned  # is estimated traj aligned?
        self.num_associations = len(timestamps_associations)


# Align filter trajectory with ground truth trajectory by computing the SE(3)/Sim(3) transformation between the two trajectories.
# First, find associations between filter timestamps and gt timestamps.
# Next, align the two trajectories with SVD on the basis of their associations.
# - filter_timestamps: List of filter timestamps assumed to be in seconds
# - filter_t_wi: List of filter 3D positions
# - gt_timestamps: List of gt timestamps assumed to be in seconds
# - gt_t_wi: List of gt 3D positions
# - max_align_dt: maximum time difference between filter and gt timestamps in seconds
# - find_scale allows to compute the full Sim(3) transformation in case the scale is unknown
def align_trajectories_with_svd(filter_timestamps, filter_t_wi, gt_timestamps, gt_t_wi, \
                         compute_align_error=True, find_scale=False, max_align_dt=1e-1, \
                         verbose=False, outlier_rejection=False):
    if verbose:
        print('align_trajectories:')
        print(f'\tfilter_timestamps: {filter_timestamps.shape}')
        print(f'\tfilter_t_wi: {filter_t_wi.shape}')
        print(f'\tgt_timestamps: {gt_timestamps.shape}')
        print(f'\tgt_t_wi: {gt_t_wi.shape}')        
        #print(f'\tfilter_timestamps: {filter_timestamps}')
        #print(f'\tgt_timestamps: {gt_timestamps}')

    # First, find associations between timestamps in filter and gt    
    timestamps_associations, filter_associations, gt_associations = find_trajectories_associations(filter_timestamps, filter_t_wi, gt_timestamps, gt_t_wi, max_align_dt=max_align_dt, verbose=verbose)

    num_samples = len(filter_associations)
    if verbose: 
        print(f'align_trajectories: num associations: {num_samples}')

    # Next, align the two trajectories on the basis of their associations    
    T_gt_est, T_est_gt, is_ok = align_3d_points_with_svd(gt_associations, filter_associations, find_scale=find_scale)
    if not is_ok:
        return np.eye(4), -1, TrajectoryAlignementData()
    
    # second pass with naive outlier rejection
    if outlier_rejection:
        errors = ((T_gt_est[:3, :3] @ np.array(filter_associations).T).T + T_gt_est[:3, 3]) - np.array(gt_associations)
        distances = np.linalg.norm(errors, axis=1)
        median_distance = np.median(distances)
        sigma_mad = 1.4826 * median_distance
        mask = distances < 3 * sigma_mad 
        if np.sum(mask) != len(filter_associations):
            if verbose:
                print(f'align_trajectories: second pass: #num associations: {np.sum(mask)}, median_error: {median_error}, sigma_mad: {sigma_mad}')
            T_gt_est, T_est_gt, is_ok  = align_3d_points_with_svd(gt_associations[mask], filter_associations[mask], find_scale=find_scale)
            if not is_ok:
                return np.eye(4), -1, TrajectoryAlignementData()
            
    # Compute error
    rms_error = 0
    max_error = float("-inf")
    if compute_align_error:
        errors = ((T_gt_est[:3, :3] @ np.array(filter_associations).T).T + T_gt_est[:3, 3]) - np.array(gt_associations)
        distances = np.linalg.norm(errors, axis=1)
        squared_errors = np.power(distances,2)  
        rms_error = np.sqrt(np.mean(squared_errors))    
        max_error = np.sqrt(np.max(squared_errors))   
        median_error = np.sqrt(np.median(squared_errors))
        if verbose:
            print(f'align_trajectories: RMS error: {rms_error}, max_error: {max_error}, median_error: {median_error}')     

    aligned_gt_data = TrajectoryAlignementData(timestamps_associations, filter_associations, gt_associations, T_gt_est, T_est_gt, rms_error, max_error=max_error)
        
    return T_gt_est, rms_error, aligned_gt_data
