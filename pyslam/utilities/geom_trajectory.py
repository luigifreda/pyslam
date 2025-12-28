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

import time
import numpy as np
import torch.multiprocessing as mp

from .geometry import *
from .logging import Printer
from .multi_processing import MultiprocessingManager
from .data_management import empty_queue, get_last_item_from_queue

import sim3solver
import trajectory_tools

from numba import njit


@njit(cache=True)
def set_rotations_from_translations(t_wi):
    num_samples = t_wi.shape[0]
    R_wi = np.zeros((num_samples, 3, 3))
    for i in range(1, num_samples):
        R_wi[i] = get_rotation_from_z_vector(t_wi[i] - t_wi[i - 1])
    if num_samples > 1:
        R_wi[0] = R_wi[1]
    return R_wi


def compute_alignment_errors(T_gt_est, filter_associations, gt_associations):
    """
    Compute the alignment errors between the ground truth and aligned estimated trajectories.
    Inputs:
        T_gt_est: [4x4] transformation matrix from the estimated trajectory to the ground truth trajectory
        filter_associations: [Nx3] array of filter associations
        gt_associations: [Nx3] array of ground truth associations
    Outputs:
        rms_error: [float] root mean square error between the estimated and ground truth trajectories
        max_error: [float] maximum error between the estimated and ground truth trajectories
        median_error: [float] median error between the estimated and ground truth trajectories
    """
    # Vectorized computation - no numba needed as numpy operations are already optimized
    R = T_gt_est[:3, :3]
    t = T_gt_est[:3, 3]
    F = filter_associations.T  # 3xN
    errors = gt_associations - ((R @ F).T + t)  # Nx3
    distances = np.linalg.norm(errors, axis=1)
    squared_errors = distances**2
    rms_error = np.sqrt(np.mean(squared_errors))
    max_error = np.sqrt(np.max(squared_errors))
    median_error = np.sqrt(np.median(squared_errors))
    return rms_error, max_error, median_error, errors


# Align corresponding 3D points in SE(3)/Sime(3) using SVD, we assume the points have been already associated, i.e. gt_points[i] corresponds to est_points[i].
# Here, we use the algorith Kabsch–Umeyama to compute the optimal rotation and translation between the two sets of points.
# Reference: "Least-squares estimation of transformation parameters between two point patterns", Shinji Umeyama, PAMI 1991
# https://web.stanford.edu/class/cs273/refs/umeyama.pdf
# input: gt_points [Nx3] and est_points [Nx3] are the corresponding 3D points
def align_3d_points_with_svd(gt_points, est_points, find_scale=True):
    assert len(gt_points) == len(est_points), "The number of points must be the same"
    is_ok = False

    # Next, align the two trajectories on the basis of their associations
    gt = np.array(gt_points).T  # 3xN
    est = np.array(est_points).T  # 3xN

    mean_gt = np.mean(gt, axis=1)
    mean_est = np.mean(est, axis=1)

    gt -= mean_gt[:, None]
    est -= mean_est[:, None]

    cov = np.dot(gt, est.T)
    if find_scale:
        # apply Kabsch–Umeyama algorithm
        cov /= gt.shape[0]
        variance_gt = np.mean(np.linalg.norm(gt, axis=1) ** 2)

    try:
        U, D, Vt = np.linalg.svd(cov)
    except:
        Printer.red("[align_3d_points_with_svd] SVD failed!!!\n")
        return np.eye(4), is_ok

    c = 1
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    if find_scale:
        # apply Kabsch–Umeyama algorithm
        c = variance_gt / np.trace(np.diag(D) @ S)

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
        s = c  # np.mean([np.linalg.norm(R_gt_est[i, :]) for i in range(3)])
        R = rot_gt_est  # R_gt_est / s
        sR_inv = (1.0 / s) * R.T
        T_est_gt[:3, :3] = sR_inv
        T_est_gt[:3, 3] = -sR_inv @ t_gt_est.ravel()
    else:
        T_est_gt[:3, :3] = R_gt_est.T
        T_est_gt[:3, 3] = -R_gt_est.T @ t_gt_est.ravel()

    is_ok = True
    return T_gt_est, T_est_gt, is_ok


def align_3d_points_with_svd_cpp(gt_points, est_points, find_scale=True):
    return trajectory_tools.align_3d_points_with_svd(gt_points, est_points, find_scale)


# Find associations between 3D trajectories timestamps in filter and gt. For each filter timestamp, find the closest gt timestamps
# and interpolate the 3D trajectory between them.
# Inputs: filter_timestamps: List of filter timestamps assumed to be in seconds
#         filter_t_wi: List of 3D positions, filter trajectory
#         gt_timestamps: List of gt timestamps assumed to be in seconds
#         gt_t_wi: List of 3D positions, ground truth trajectory
#         max_align_dt: maximum time difference between filter and gt timestamps in seconds
def find_trajectories_associations(
    filter_timestamps, filter_t_wi, gt_timestamps, gt_t_wi, max_align_dt=1e-1, verbose=True
):
    max_dt = 0
    filter_associations = []
    gt_associations = []
    timestamps_associations = []

    # First, find associations between timestamps in filter and gt
    for i, timestamp in enumerate(filter_timestamps):

        # Find the index in gt_timestamps where gt_timestamps[j] > timestamp
        # j = 0
        # while j < len(gt_timestamps) and gt_timestamps[j] <= timestamp:
        #     j += 1
        # j -= 1

        # Find right index in sorted gt_timestamps using binary search
        # j = bisect.bisect_right(gt_timestamps, timestamp) - 1
        j = np.searchsorted(gt_timestamps, timestamp, side="right") - 1

        if j < 0 or j >= len(gt_timestamps) - 1:
            continue  # out of bounds

        dt = timestamp - gt_timestamps[j]
        dt_gt = gt_timestamps[j + 1] - gt_timestamps[j]
        abs_dt = abs(dt)

        # assert dt >= 0, f"dt {dt}"
        # assert dt_gt > 0, f"dt_gt {dt_gt}"

        if dt < 0 or dt_gt <= 0:
            continue  # skip inconsistent data

        # Skip if the interval between gt is larger than max delta
        if abs_dt > max_align_dt:
            continue

        max_dt = max(max_dt, abs_dt)
        ratio = dt / dt_gt

        # assert 0 <= ratio < 1, f"ratio {ratio}"

        gt_t_wi_interpolated = (1 - ratio) * gt_t_wi[j] + ratio * gt_t_wi[j + 1]

        timestamps_associations.append(timestamp)
        gt_associations.append(gt_t_wi_interpolated)
        filter_associations.append(filter_t_wi[i])

    if verbose:
        print(f"find_trajectories_associations: max trajectory align dt: {max_dt}")

    return (
        np.array(timestamps_associations),
        np.array(filter_associations),
        np.array(gt_associations),
    )


def find_trajectories_associations_cpp(
    filter_timestamps, filter_t_wi, gt_timestamps, gt_t_wi, max_align_dt=1e-1, verbose=True
):
    return trajectory_tools.find_trajectories_associations(
        filter_timestamps, filter_t_wi, gt_timestamps, gt_t_wi, max_align_dt, verbose
    )


# Associate each filter pose to a gt pose on the basis of their timestamps. Interpolate the gt poses where needed.
# We assume the gt poses are in SE(3).
# Find associations between poses in filter and gt. For each filter timestamp, find the closest gt timestamps
# and interpolate the pose between them.
# Inputs: filter_timestamps: List of filter timestamps assumed to be in seconds
#         filter_T_wi: List of filter poses in SE(3), each one a [4x4] transformation matrix
#         gt_timestamps: List of gt timestamps assumed to be in seconds
#         gt_T_wi: List of ground truth poses, each one a [4x4] transformation matrix
#         max_align_dt: maximum time difference between filter and gt timestamps in seconds
def find_poses_associations(
    filter_timestamps, filter_T_wi, gt_timestamps, gt_T_wi, max_align_dt=1e-1, verbose=True
):
    max_dt = 0
    filter_associations = []
    gt_associations = []
    timestamps_associations = []

    # First, find associations between timestamps in filter and gt
    for i, timestamp in enumerate(filter_timestamps):

        # Find the index in gt_timestamps where gt_timestamps[j] > timestamp
        # j = 0
        # while j < len(gt_timestamps) and gt_timestamps[j] <= timestamp:
        #     j += 1
        # j -= 1

        # Find right index in sorted gt_timestamps using binary search
        # j = bisect.bisect_right(gt_timestamps, timestamp) - 1
        j = np.searchsorted(gt_timestamps, timestamp, side="right") - 1

        if j < 0 or j >= len(gt_timestamps) - 1:
            continue  # out of bounds

        dt = timestamp - gt_timestamps[j]
        dt_gt = gt_timestamps[j + 1] - gt_timestamps[j]
        abs_dt = abs(dt)

        # assert dt >= 0, f"dt {dt}"
        # assert dt_gt > 0, f"dt_gt {dt_gt}"

        if dt < 0 or dt_gt <= 0:
            continue  # skip inconsistent data

        # Skip if the interval between gt is larger than max delta
        if abs_dt > max_align_dt:
            continue

        max_dt = max(max_dt, abs_dt)
        ratio = dt / dt_gt

        # assert 0 <= ratio < 1, f"ratio {ratio}"

        gt_t_wi_j = gt_T_wi[j][:3, 3]
        gt_R_wi_j = gt_T_wi[j][:3, :3]
        gt_t_wi_jp1 = gt_T_wi[j + 1][:3, 3]
        gt_R_wi_jp1 = gt_T_wi[j + 1][:3, :3]

        t_wi_gt_interpolated = (
            1 - ratio
        ) * gt_t_wi_j + ratio * gt_t_wi_jp1  # gt_t_wi_j + ratio * (gt_t_wi_jp1-gt_t_wi_j)
        delta_R = gt_R_wi_jp1 @ gt_R_wi_j.T
        if not is_so3(delta_R):
            delta_R = closest_rotation_matrix(delta_R)
        R_wi_gt_interpolated = gt_R_wi_j @ so3_exp(ratio * so3_log(delta_R))

        timestamps_associations.append(timestamp)
        gt_associations.append(poseRt(R_wi_gt_interpolated, t_wi_gt_interpolated))
        filter_associations.append(filter_T_wi[i])

    if verbose:
        print(f"find_poses_associations: max trajectory align dt: {max_dt}")

    return (
        np.array(timestamps_associations),
        np.array(filter_associations),
        np.array(gt_associations),
    )


# Data representing the alignment of an estimated trajectory with ground truth
class TrajectoryAlignementData:
    def __init__(
        self,
        timestamps_associations=None,
        estimated_t_wi=None,
        gt_t_wi=None,
        T_gt_est=None,
        T_est_gt=None,
        rms_error=-1.0,
        is_est_aligned=False,
        max_error=-1.0,
        errors=None,
    ):
        # Here we store associated data between estimated and ground truth trajectories
        # timestamps_associations [Nx1] assumed to be in seconds
        # estimated_t_wi [Nx3]      =>  estimated_t_wi[i] corresponds to gt_t_wi[i] at timestamps_associations[i]
        # gt_t_wi [Nx3]             =>  gt_t_wi[i] corresponds to estimated_t_wi[i] at timestamps_associations[i]
        # T_gt_est [4x4]
        # T_est_gt [4x4]
        self.timestamps_associations = (
            timestamps_associations if timestamps_associations is not None else []
        )
        self.estimated_t_wi = estimated_t_wi if estimated_t_wi is not None else []
        self.gt_t_wi = gt_t_wi if gt_t_wi is not None else []
        self.T_gt_est = T_gt_est
        self.T_est_gt = T_est_gt
        self.rms_error = rms_error  # average alignment error
        self.max_error = max_error  # max alignement error
        self.is_est_aligned = is_est_aligned  # is estimated traj aligned?
        self.errors = (
            errors if errors is not None else []
        )  # vector of 3d errors between gt trajectory and aligned estimated trajectory
        self.num_associations = (
            len(timestamps_associations) if timestamps_associations is not None else 0
        )
        #
        self.gt_trajectory_aligned = None
        self.gt_trajectory_aligned_associated = None
        self.estimated_trajectory_aligned = None


# Align filter trajectory with ground truth trajectory by computing the SE(3)/Sim(3) transformation between the two trajectories.
# First, find associations between filter timestamps and gt timestamps.
# Next, align the two trajectories with SVD on the basis of their associations.
# - filter_timestamps: List of filter timestamps assumed to be in seconds
# - filter_t_wi: List of filter 3D positions
# - gt_timestamps: List of gt timestamps assumed to be in seconds
# - gt_t_wi: List of gt 3D positions
# - max_align_dt: maximum time difference between filter and gt timestamps in seconds
# - find_scale allows to compute the full Sim(3) transformation in case the scale is unknown
def align_trajectories_with_svd(
    filter_timestamps,
    filter_t_wi,
    gt_timestamps,
    gt_t_wi,
    compute_align_error=True,
    find_scale=False,
    max_align_dt=1e-1,
    verbose=False,
    outlier_rejection=False,
):
    if verbose:
        print("align_trajectories:")
        print(f"\tfilter_timestamps: {filter_timestamps.shape}")
        print(f"\tfilter_t_wi: {filter_t_wi.shape}")
        print(f"\tgt_timestamps: {gt_timestamps.shape}")
        print(f"\tgt_t_wi: {gt_t_wi.shape}")
        # print(f'\tfilter_timestamps: {filter_timestamps}')
        # print(f'\tgt_timestamps: {gt_timestamps}')

    # First, find associations between timestamps in filter and gt
    timestamps_associations, filter_associations, gt_associations = (
        find_trajectories_associations_cpp(
            filter_timestamps,
            filter_t_wi,
            gt_timestamps,
            gt_t_wi,
            max_align_dt=max_align_dt,
            verbose=verbose,
        )
    )

    num_samples = len(filter_associations)
    if verbose:
        print(f"align_trajectories: num associations: {num_samples}")

    # Next, align the two trajectories on the basis of their associations
    T_gt_est, T_est_gt, is_ok = align_3d_points_with_svd_cpp(
        gt_associations, filter_associations, find_scale=find_scale
    )
    if not is_ok:
        return np.eye(4), -1, TrajectoryAlignementData()

    # second pass with naive outlier rejection
    if outlier_rejection:
        errors = np.array(gt_associations) - (
            (T_gt_est[:3, :3] @ np.array(filter_associations).T).T + T_gt_est[:3, 3]
        )
        distances = np.linalg.norm(errors, axis=1)
        median_distance = np.median(distances)
        sigma_mad = 1.4826 * median_distance
        mask = distances < 3 * sigma_mad
        if np.sum(mask) != len(filter_associations):
            if verbose:
                print(
                    f"align_trajectories: second pass: #num associations: {np.sum(mask)}, median_error: {median_error}, sigma_mad: {sigma_mad}"
                )
            T_gt_est, T_est_gt, is_ok = align_3d_points_with_svd_cpp(
                gt_associations[mask], filter_associations[mask], find_scale=find_scale
            )
            if not is_ok:
                return np.eye(4), -1, TrajectoryAlignementData()

    # Compute error
    rms_error = 0
    max_error = float("-inf")
    errors = None
    if compute_align_error:
        rms_error, max_error, median_error, errors = compute_alignment_errors(
            T_gt_est,
            np.array(filter_associations, dtype=np.float64),
            np.array(gt_associations, dtype=np.float64),
        )
        if verbose:
            print(
                f"align_trajectories: RMS error: {rms_error}, max_error: {max_error}, median_error: {median_error}"
            )

    aligned_gt_data = TrajectoryAlignementData(
        timestamps_associations,
        filter_associations,
        gt_associations,
        T_gt_est,
        T_est_gt,
        rms_error,
        max_error=max_error,
        errors=errors,
    )

    return T_gt_est, rms_error, aligned_gt_data


def align_trajectories_with_ransac(
    filter_timestamps,
    filter_t_wi,
    gt_timestamps,
    gt_t_wi,
    compute_align_error=True,
    find_scale=False,
    max_align_dt=1e-1,
    num_ransac_iterations=5,
    verbose=True,
):
    if verbose:
        print("align_trajectories:")
        print(f"\tfilter_timestamps: {filter_timestamps.shape}")
        print(f"\tfilter_t_wi: {filter_t_wi.shape}")
        print(f"\tgt_timestamps: {gt_timestamps.shape}")
        print(f"\tgt_t_wi: {gt_t_wi.shape}")
        # print(f'\tfilter_timestamps: {filter_timestamps}')
        # print(f'\tgt_timestamps: {gt_timestamps}')

    # First, find associations between timestamps in filter and gt
    timestamps_associations, filter_associations, gt_associations = (
        find_trajectories_associations_cpp(
            filter_timestamps,
            filter_t_wi,
            gt_timestamps,
            gt_t_wi,
            max_align_dt=max_align_dt,
            verbose=verbose,
        )
    )

    num_samples = len(filter_associations)
    if verbose:
        print(f"align_trajectories: num associations: {num_samples}")

    # Next, align the two trajectories on the basis of their associations
    # T_gt_est, T_est_gt, is_ok = align_3d_points_with_svd(gt_associations, filter_associations, find_scale=find_scale)

    solver_input_data = sim3solver.Sim3PointRegistrationSolverInput()
    solver_input_data.sigma2 = 0.05
    solver_input_data.fix_scale = not find_scale
    solver_input_data.points_3d_w1 = filter_associations  # points_3d_w1
    solver_input_data.points_3d_w2 = gt_associations  # points_3d_w2

    # Create Sim3PointRegistrationSolver object with the input data
    solver = sim3solver.Sim3PointRegistrationSolver(solver_input_data)
    # Set RANSAC parameters (using defaults here)
    solver.set_ransac_parameters(0.99, 20, 300)
    transformation, bNoMore, vbInliers, nInliers, bConverged = solver.iterate(num_ransac_iterations)

    if not bConverged:
        return np.eye(4), -1, TrajectoryAlignementData()

    R12 = solver.get_estimated_rotation()
    t12 = solver.get_estimated_translation()
    scale12 = solver.get_estimated_scale()

    T_est_gt = np.eye(4)
    T_est_gt[:3, :3] = scale12 * R12
    T_est_gt[:3, 3] = t12

    T_gt_est = np.eye(4)
    R21 = R12.T
    T_gt_est[:3, :3] = R21 / scale12
    T_gt_est[:3, 3] = -R21 @ t12 / scale12

    # Compute error
    error = 0
    max_error = float("-inf")
    if compute_align_error:
        # if align_est_associations:
        #     filter_associations = (T_gt_est[:3, :3] @ np.array(filter_associations).T).T + T_gt_est[:3, 3]
        #     residuals = filter_associations - np.array(gt_associations)
        # else:
        residuals = (
            (T_gt_est[:3, :3] @ np.array(filter_associations).T).T + T_gt_est[:3, 3]
        ) - np.array(gt_associations)
        squared_errors = np.sum(residuals**2, axis=1)
        error = np.sqrt(np.mean(squared_errors))
        max_error = np.sqrt(np.max(squared_errors))
        median_error = np.sqrt(np.median(squared_errors))
        if verbose:
            print(
                f"align_trajectories: error: {error}, max_error: {max_error}, median_error: {median_error}"
            )

    aligned_gt_data = TrajectoryAlignementData(
        timestamps_associations,
        filter_associations,
        gt_associations,
        T_gt_est,
        T_est_gt,
        error,
        max_error=max_error,
    )

    return T_gt_est, error, aligned_gt_data


class TrajectoryAlignerProcessBatch(mp.Process):
    def __init__(
        self,
        input_queue,
        output_queue,
        is_running_flag,
        gt_trajectory,
        gt_timestamps,
        find_scale=False,
        compute_align_error=False,
    ):
        """
        Batch trajectory alignment process.
        Aligns a trajectory estimated by the filter with a ground truth trajectory.
        The alignment is computed from scratch by computing the SE(3)/Sim(3) transformation
        between the two trajectories.
        Args:
            input_queue: Queue for receiving (pose_timestamps, estimated_trajectory) tuples
            output_queue: Queue for sending (T_gt_est, error, alignment_gt_data) tuples
            is_running_flag: Shared flag to control process execution
            gt_trajectory: Ground truth trajectory positions [Nx3]
            gt_timestamps: Ground truth timestamps [N]
            find_scale: Whether to estimate scale (Sim(3) vs SE(3))
            compute_align_error: Whether to compute alignment error
        """
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.is_running_flag = is_running_flag

        # Copy (contiguous) GT data to avoid issues with process boundaries
        self.gt_trajectory = np.array(gt_trajectory, copy=True, order="C")
        self.gt_timestamps = np.array(gt_timestamps, copy=True, order="C")
        self.find_scale = find_scale
        self.compute_align_error = compute_align_error

        self.daemon = True

    def quit(self):
        print("TrajectoryAlignerProcessBatch: quitting...")
        self.is_running_flag.value = 0
        timeout = 5 if mp.get_start_method() != "spawn" else 10
        self.join(timeout=timeout)
        if self.is_alive():
            print(
                f"Warning: TrajectoryAlignerProcessBatch process did not terminate in time, forcing kill."
            )
            self.terminate()
            # Wait a bit more after terminate
            self.join(timeout=1.0)
        print("TrajectoryAlignerProcessBatch: closed")

    def run(self):
        self.is_running_flag.value = 1
        self._last_alignment_time = time.time()
        self._last_num_poses = 0
        self._last_frame_id = 0

        while self.is_running_flag.value == 1:
            try:
                data = get_last_item_from_queue(self.input_queue)
                # Check if data is a valid tuple/list with 2 elements
                if data is not None:
                    try:
                        if isinstance(data, (tuple, list)) and len(data) == 2:
                            pose_timestamps, estimated_trajectory = data
                            self._process_alignment(pose_timestamps, estimated_trajectory)
                        else:
                            time.sleep(0.1)
                    except (ValueError, TypeError) as e:
                        # Invalid data format - skip this item
                        Printer.red(f"TrajectoryAlignerProcessBatch: Invalid data format: {e}")
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
            except (EOFError, OSError, ValueError) as e:
                # Queue closed or invalid state - exit cleanly
                break
            except Exception as e:
                Printer.red(f"TrajectoryAlignerProcessBatch: Error: {e}")
                # Continue for other errors, but check flag
                if self.is_running_flag.value == 0:
                    break
                time.sleep(0.1)
        try:
            empty_queue(self.input_queue)
        except Exception as e:
            print(f"TrajectoryAlignerProcessBatch: Error emptying input queue: {e}")
        print("TrajectoryAlignerProcessBatch: run: closed")

    def _process_alignment(self, pose_timestamps, estimated_trajectory):
        align_trajectories_fun = align_trajectories_with_svd
        # align_trajectories_fun = align_trajectories_with_ransac   # WIP (just a test)
        try:
            pose_timestamps = np.ascontiguousarray(pose_timestamps)
            estimated_trajectory = np.ascontiguousarray(estimated_trajectory)
            if False:
                print(
                    f"pose_timestamps: {pose_timestamps.shape}, contiguous: {pose_timestamps.flags.c_contiguous}"
                )
                print(
                    f"estimated_trajectory: {estimated_trajectory.shape}, contiguous: {estimated_trajectory.flags.c_contiguous}"
                )
                print(
                    f"self.gt_timestamps: {self.gt_timestamps.shape}, contiguous: {self.gt_timestamps.flags.c_contiguous}"
                )
                print(
                    f"self.gt_trajectory: {self.gt_trajectory.shape}, contiguous: {self.gt_trajectory.flags.c_contiguous}"
                )
                print(
                    f"self.gt_timestamps: {self.gt_timestamps.shape}, contiguous: {self.gt_timestamps.flags.c_contiguous}"
                )
                print(
                    f"self.gt_trajectory: {self.gt_trajectory.shape}, contiguous: {self.gt_trajectory.flags.c_contiguous}"
                )
            T_gt_est, error, alignment_gt_data = align_trajectories_fun(
                pose_timestamps,
                estimated_trajectory,
                self.gt_timestamps,
                self.gt_trajectory,
                compute_align_error=self.compute_align_error,
                find_scale=self.find_scale,
            )

            T_est_gt = alignment_gt_data.T_est_gt
            # compute all gt data aligned to the estimated trajectory
            alignment_gt_data.gt_trajectory_aligned = (
                T_est_gt[:3, :3] @ np.array(self.gt_trajectory).T
            ).T + T_est_gt[:3, 3]
            # compute only the associated gt samples aligned to the estimated samples
            alignment_gt_data.gt_trajectory_aligned_associated = (
                T_est_gt[:3, :3] @ np.array(alignment_gt_data.gt_t_wi).T
            ).T + T_est_gt[:3, 3]
            # compute the estimated trajectory aligned to the gt trajectory
            alignment_gt_data.estimated_trajectory_aligned = (
                T_gt_est[:3, :3] @ np.array(alignment_gt_data.estimated_t_wi).T
            ).T + T_gt_est[:3, 3]

            self.output_queue.put((T_gt_est, error, alignment_gt_data))
            print(f"TrajectoryAlignerProcessBatch: Aligned, RMS error: {error:.4f}")
        except Exception as e:
            Printer.red(f"TrajectoryAlignerProcessBatch: Alignment failed: {e}")


class TrajectoryAlignerProcessIncremental(mp.Process):
    def __init__(
        self,
        input_queue,
        output_queue,
        is_running_flag,
        gt_trajectory,
        gt_timestamps,
        find_scale=False,
        compute_align_error=False,
        max_align_dt=1e-1,
    ):
        """
        Incremental trajectory alignment process using IncrementalTrajectoryAligner.
        Aligns a trajectory estimated by the filter with a ground truth trajectory.
        The alignment is computed incrementally by updating the associations and
        recomputing the transformation.
        Args:
            input_queue: Queue for receiving (pose_timestamps, estimated_trajectory) tuples
            output_queue: Queue for sending (T_gt_est, error, alignment_gt_data) tuples
            is_running_flag: Shared flag to control process execution
            gt_trajectory: Ground truth trajectory positions [Nx3]
            gt_timestamps: Ground truth timestamps [N]
            find_scale: Whether to estimate scale (Sim(3) vs SE(3))
            compute_align_error: Whether to compute alignment error
            max_align_dt: Maximum time difference for association (seconds)
        """
        super().__init__()
        self.input_queue = input_queue
        self.output_queue = output_queue
        self.is_running_flag = is_running_flag

        # Copy (contiguous) GT data to avoid issues with process boundaries
        self.gt_trajectory = np.array(gt_trajectory, copy=True, order="C")
        self.gt_timestamps = np.array(gt_timestamps, copy=True, order="C")
        self.find_scale = find_scale
        self.compute_align_error = compute_align_error
        self.max_align_dt = max_align_dt

        # Initialize incremental aligner
        # Convert GT data to lists for C++ module
        gt_timestamps_list = self.gt_timestamps.tolist()
        gt_t_wi_list = [self.gt_trajectory[i] for i in range(len(self.gt_trajectory))]

        # Set up alignment options
        opts = trajectory_tools.AlignmentOptions()
        opts.max_align_dt = max_align_dt
        opts.find_scale = find_scale
        opts.verbose = False

        # Initialize incremental aligner
        self.incremental_aligner = trajectory_tools.IncrementalTrajectoryAligner(
            gt_timestamps_list, gt_t_wi_list, opts
        )

        self.daemon = True

    def quit(self):
        print("TrajectoryAlignerProcessIncremental: quitting...")
        self.is_running_flag.value = 0
        # Put a sentinel value in the queue to wake up the process if it's blocked
        try:
            if hasattr(self, "input_queue") and self.input_queue:
                # Put None to signal exit - this will wake up get_last_item_from_queue
                try:
                    self.input_queue.put_nowait(None)
                except:
                    pass  # Queue might be full or closed, that's okay
        except Exception as e:
            print(f"TrajectoryAlignerProcessIncremental: Error signaling queue: {e}")
        timeout = 5 if mp.get_start_method() != "spawn" else 10
        self.join(timeout=timeout)
        if self.is_alive():
            print(
                f"Warning: TrajectoryAlignerProcessIncremental process did not terminate in time, forcing kill."
            )
            self.terminate()
            # Wait a bit more after terminate
            self.join(timeout=1.0)
        print("TrajectoryAlignerProcessIncremental: closed")

    def run(self):
        self.is_running_flag.value = 1
        self._last_alignment_time = time.time()
        self._last_num_poses = 0
        self._last_frame_id = 0

        while self.is_running_flag.value == 1:
            try:
                data = get_last_item_from_queue(self.input_queue)
                # Check if we received a shutdown signal (None)
                if data is None:
                    break
                # Check if data is a valid tuple/list with 2 elements
                if data is not None:
                    try:
                        if isinstance(data, (tuple, list)) and len(data) == 2:
                            pose_timestamps, estimated_trajectory = data
                            self._process_alignment(pose_timestamps, estimated_trajectory)
                        else:
                            time.sleep(0.1)
                    except (ValueError, TypeError) as e:
                        # Invalid data format - skip this item
                        Printer.red(
                            f"TrajectoryAlignerProcessIncremental: Invalid data format: {e}"
                        )
                        time.sleep(0.1)
                else:
                    time.sleep(0.1)
            except (EOFError, OSError, ValueError) as e:
                # Queue closed or invalid state - exit cleanly
                break
            except Exception as e:
                Printer.red(f"TrajectoryAlignerProcessIncremental: Error: {e}")
                # Continue for other errors, but check flag
                if self.is_running_flag.value == 0:
                    break
                time.sleep(0.1)

        try:
            empty_queue(self.input_queue)
        except Exception as e:
            print(f"TrajectoryAlignerProcessIncremental: Error emptying input queue: {e}")
        print("TrajectoryAlignerProcessIncremental: run: closed")

    def _process_alignment(self, pose_timestamps, estimated_trajectory):
        try:
            pose_timestamps = np.ascontiguousarray(pose_timestamps)
            estimated_trajectory = np.ascontiguousarray(estimated_trajectory)

            # Convert to lists for C++ module
            pose_timestamps_list = pose_timestamps.tolist()
            estimated_trajectory_list = [
                estimated_trajectory[i] for i in range(len(estimated_trajectory))
            ]

            # Update trajectory with full current trajectory
            self.incremental_aligner.update_trajectory(
                pose_timestamps_list, estimated_trajectory_list
            )

            # Get result
            inc_result = self.incremental_aligner.result()

            if not inc_result.valid:
                raise RuntimeError("Incremental alignment result is not valid")

            T_gt_est = inc_result.T_gt_est
            n_pairs = inc_result.n_pairs

            # Compute error if requested
            error = 0.0
            max_error = 0.0
            timestamps_associations = []
            estimated_t_wi = []
            gt_t_wi = []

            # Get only associated pairs (positions with valid GT associations)
            # This ensures we only compute errors for positions that were actually used in alignment
            est_timestamps, est_positions, gt_interp_positions = (
                self.incremental_aligner.get_associated_pairs()
            )

            if self.compute_align_error:
                # Compute RMS error from associated pairs using the same function as batch version
                if len(est_positions) > 0 and n_pairs > 0:
                    # Convert to numpy arrays for error computation
                    est_positions_array = np.array(est_positions, dtype=np.float64)
                    gt_interp_positions_array = np.array(gt_interp_positions, dtype=np.float64)

                    # Use the same error computation function as batch version
                    error, max_error, median_error, errors = compute_alignment_errors(
                        T_gt_est, est_positions_array, gt_interp_positions_array
                    )

                    # Store associations for alignment data
                    for i in range(len(est_positions)):
                        timestamps_associations.append(float(est_timestamps[i]))
                        estimated_t_wi.append(est_positions[i])
                        gt_t_wi.append(gt_interp_positions[i])
            else:
                # Even if not computing error, we can still get associations
                for i in range(len(est_positions)):
                    timestamps_associations.append(float(est_timestamps[i]))
                    estimated_t_wi.append(np.array(est_positions[i]).tolist())
                    gt_t_wi.append(np.array(gt_interp_positions[i]).tolist())

            # Create alignment data structure compatible with batch version
            alignment_gt_data = TrajectoryAlignementData(
                timestamps_associations=timestamps_associations,
                estimated_t_wi=estimated_t_wi,
                gt_t_wi=gt_t_wi,
                T_gt_est=T_gt_est,
                T_est_gt=inc_result.T_est_gt,
                rms_error=error,
                max_error=max_error,
            )

            T_est_gt = alignment_gt_data.T_est_gt
            # compute all gt data aligned to the estimated trajectory
            alignment_gt_data.gt_trajectory_aligned = (
                T_est_gt[:3, :3] @ np.array(self.gt_trajectory).T
            ).T + T_est_gt[:3, 3]
            # compute only the associated gt samples aligned to the estimated samples
            alignment_gt_data.gt_trajectory_aligned_associated = (
                T_est_gt[:3, :3] @ np.array(alignment_gt_data.gt_t_wi).T
            ).T + T_est_gt[:3, 3]
            # compute the estimated trajectory aligned to the gt trajectory
            alignment_gt_data.estimated_trajectory_aligned = (
                T_gt_est[:3, :3] @ np.array(alignment_gt_data.estimated_t_wi).T
            ).T + T_gt_est[:3, 3]

            self.output_queue.put((T_gt_est, error, alignment_gt_data))
            print(
                f"TrajectoryAlignerProcessIncremental: Aligned, RMS error: {error:.4f}, pairs: {n_pairs}"
            )
        except Exception as e:
            Printer.red(f"TrajectoryAlignerProcessIncremental: Alignment failed: {e}")
