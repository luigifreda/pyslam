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
import open3d as o3d

# from .camera import CameraUtils
from pyslam.slam import CameraUtils

from pyslam.utilities.serialization import SerializableEnum

import sim3solver
import pnpsolver


# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .camera import CameraUtils


# NOTE: This file collects a set of pose estimators that estimate the interframe pose between two frames starting from 2D-2D/2D-3D/3D-3D correspondences.
#      Some of following classes were inspired by the repository  https://github.com/nianticlabs/map-free-reloc


class PoseEstimatorType(SerializableEnum):
    ESSENTIAL_MATRIX_2D_2D = 0  # 2D2D
    ESSENTIAL_MATRIX_METRIC_SIMPLE = (
        1  # Start with 2D2D and then extract metric translation from depth information by averaging
    )
    ESSENTIAL_MATRIX_METRIC = 2  # Start with 2D2D and then extract metric translation from depth information by using a RANSAC-like loop
    PNP = 3  # 2D3D
    PNP_WITH_SIGMAS = 4  # 2D3D (C++ implementation)
    MLPNP_WITH_SIGMAS = 5  # 2D3D (C++ implementation)
    PROCUSTES = 6  # 3D3D
    SIM3_3D3D = 7  # 3D3D


def pose_estimator_factory(pose_estimator_type, K1, K2=None):
    if pose_estimator_type == PoseEstimatorType.ESSENTIAL_MATRIX_2D_2D:
        return EssentialMatrixPoseEstimator2d2d(K1, K2)
    elif pose_estimator_type == PoseEstimatorType.ESSENTIAL_MATRIX_METRIC_SIMPLE:
        return EssentialMatrixMetricSimplePoseEstimator(K1, K2)
    elif pose_estimator_type == PoseEstimatorType.ESSENTIAL_MATRIX_METRIC:
        return EssentialMatrixMetricPoseEstimator(K1, K2)
    elif pose_estimator_type == PoseEstimatorType.PNP:
        return PnPPoseEstimator(K1, K2)
    elif pose_estimator_type == PoseEstimatorType.PNP_WITH_SIGMAS:
        return PnPWithSigmasPoseEstimator(K1, K2)
    elif pose_estimator_type == PoseEstimatorType.MLPNP_WITH_SIGMAS:
        return MlPnPWithSigmasPoseEstimator(K1, K2)
    elif pose_estimator_type == PoseEstimatorType.PROCUSTES:
        return ProcrustesPoseEstimator(K1, K2)
    elif pose_estimator_type == PoseEstimatorType.SIM3_3D3D:
        return Sim3PoseEstimator(K1, K2)
    else:
        raise ValueError("Unknown pose estimator type")


# Class for hosting a "multi-modal" input for the pose estimators
class PoseEstimatorInput:
    def __init__(
        self,
        kpts1=None,
        kpts2=None,
        sigmas2_1=None,
        sigmas2_2=None,
        depth1=None,
        depth2=None,
        pts1=None,
        pts2=None,
        K1=None,
        K2=None,
        fix_scale=True,
    ):
        self.kpts1 = kpts1  # 2D keypoints in image 1, numpy array of shape (N, 2)
        self.kpts2 = kpts2  # 2D keypoints in image 2, numpy array of shape (N, 2)
        self.sigmas2_1 = sigmas2_1  # scalar sigmas squared of 2D keypoints in image 1, numpy array of shape (N,)  [used in the reprojection error evaluation]
        self.sigmas2_2 = sigmas2_2  # scalar sigmas squared of 2D keypoints in image 2, numpy array of shape (N,)  [used in the reprojection error evaluation]
        self.depth1 = depth1  # depth map image 1
        self.depth2 = depth2  # depth map image 2
        self.pts1 = pts1  # 3D points w.r.t frame 1, numpy array of shape (N, 3)
        self.pts2 = pts2  # 3D points w.r.t frame 2, numpy array of shape (N, 3)
        self.K1 = K1  # camera matrix of the first camera, numpy array of shape (3, 3)
        self.K2 = (
            K2 if K2 is not None else K1
        )  # camera matrix of the second camera, numpy array of shape (3, 3)
        self.fix_scale = fix_scale  # used by the sim3 pose estimators

    @staticmethod
    def from_dict(data: dict):
        output = PoseEstimatorInput()
        if "kpts1" in data:
            output.kpts1 = data["kpts1"]
        if "kpts2" in data:
            output.kpts2 = data["kpts2"]
        if "sigmas2_1" in data:
            output.sigmas2_1 = data["sigmas2_1"]
        if "sigmas2_2" in data:
            output.sigmas2_2 = data["sigmas2_2"]
        if "depth1" in data:
            output.depth1 = data["depth1"]
        if "depth2" in data:
            output.depth2 = data["depth2"]
        if "pts1" in data:
            output.pts1 = data["pts1"]
        if "pts2" in data:
            output.pts2 = data["pts2"]
        if "K1" in data:
            output.K1 = data["K1"]
        if "K2" in data:
            output.K2 = data["K2"]
        if "fix_scale" in data:
            output.fix_scale = data["fix_scale"]
        return output

    def to_dict(self):
        output = {}
        if self.kpts1 is not None:
            output["kpts1"] = self.kpts1
        if self.kpts2 is not None:
            output["kpts2"] = self.kpts2
        if self.sigmas2_1 is not None:
            output["sigmas2_1"] = self.sigmas2_1
        if self.sigmas2_2 is not None:
            output["sigmas2_2"] = self.sigmas2_2
        if self.depth1 is not None:
            output["depth1"] = self.depth1
        if self.depth2 is not None:
            output["depth2"] = self.depth2
        if self.pts1 is not None:
            output["pts1"] = self.pts1
        if self.pts2 is not None:
            output["pts2"] = self.pts2
        if self.K1 is not None:
            output["K1"] = self.K1
        if self.K2 is not None:
            output["K2"] = self.K2
        output["fix_scale"] = self.fix_scale
        return output


# Base class for all pose estimators returning a pose in SE(3) or Sim(3)
class PoseEstimator:
    def __init__(self, K1=None, K2=None, pose_estimator_type=None):
        self.K1 = K1
        self.K2 = K2 if K2 is not None else K1
        self.pose_estimator_type = pose_estimator_type

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        raise NotImplementedError


# Estimate pose by using the essential matrix estimation from a set of 2D-2D correspondences.
# Fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
# out: [Rrc, trc]   (with respect to 'ref' frame)
# N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
# N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
# - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie on a ruled quadric
# - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
# N.B.3: The five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
# N.B.4: As it is reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return the rotation matrix
class EssentialMatrixPoseEstimator2d2d(PoseEstimator):
    def __init__(self, K1, K2=None, pose_estimator_type=PoseEstimatorType.ESSENTIAL_MATRIX_2D_2D):
        super().__init__(K1, K2, pose_estimator_type)
        self.ransac_pixel_threshold = 2.0  # pixels
        self.ransac_confidence = 0.999

    # get an estimate of [R21,t21] up to scale from a set of 2D-2D correspondences
    def estimate2d2d(self, kpts1, kpts2):
        R = np.full((3, 3), np.nan)
        t = np.full((3, 1), np.nan)
        if len(kpts1) < 5:
            return R, t, 0

        # Normalize keypoints
        kpts1 = (kpts1 - self.K1[[0, 1], [2, 2]]) / self.K1[
            [0, 1], [0, 1]
        ]  # (p1 - [cx1,cy1])/[fx1,fy1]
        kpts2 = (kpts2 - self.K2[[0, 1], [2, 2]]) / self.K2[
            [0, 1], [0, 1]
        ]  # (p2 - [cx2,cy2])/[fx2,fy2]

        # Transform ransac pixel threshold into normalized-coordinates pixel threshold
        ransac_thr = self.ransac_pixel_threshold / np.mean(
            [self.K1[0, 0], self.K1[1, 1], self.K2[0, 0], self.K2[1, 1]]
        )  # np.mean(fx1,fy1,fx2,fy2)

        # Compute pose with OpenCV
        # ransac_method = cv2.RANSAC
        ransac_method = cv2.USAC_MAGSAC
        E, mask = cv2.findEssentialMat(
            kpts1,
            kpts2,
            np.eye(3),
            threshold=ransac_thr,
            prob=self.ransac_confidence,
            method=ransac_method,
        )
        self.mask = mask
        if E is None:
            return R, t, 0

        # Recover pose from E
        best_num_inliers = 0
        ret = R, t, 0
        for _E in np.split(E, len(E) / 3):
            num_inliers, R, t, _ = cv2.recoverPose(_E, kpts1, kpts2, np.eye(3), 1e9, mask=mask)
            if num_inliers > best_num_inliers:
                best_num_inliers = num_inliers
                ret = (R, t[:, 0], num_inliers)
        return ret

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        if isinstance(data, dict):
            data = PoseEstimatorInput.from_dict(data)
        if not isinstance(data, PoseEstimatorInput):
            raise TypeError
        if data.K1 is not None:
            self.K1 = data.K1
            self.K2 = data.K2 if data.K2 is not None else data.K1
        assert data.kpts1 is not None and data.kpts2 is not None
        return self.estimate2d2d(data.kpts1, data.kpts2)


# Use EssentialMatrixPoseEstimator2d2d as a base and use depth values at inliers to obtain the metric translation vector.
# Procedure:
# 1. Back-project 2D inlier correspondences (from the Essential matrix) into 3D using depth maps.
# 2. Form 3D-3D correspondences between the two views.
# 3. Compute the optimal scale for the translation vector from each 3D-3D correspondence.
# 4. Aggregate the scale estimates by averaging them to obtain a robust metric translation estimate.
class EssentialMatrixMetricSimplePoseEstimator(EssentialMatrixPoseEstimator2d2d):
    def __init__(
        self, K1, K2=None, pose_estimator_type=PoseEstimatorType.ESSENTIAL_MATRIX_METRIC_SIMPLE
    ):
        super().__init__(K1, K2, pose_estimator_type)

    # get an estimate of [R21,t21] with scale from a set of 2D-2D correspondences and the depth maps
    def estimate2d2d_with_depth(self, kpts1, kpts2, depth1, depth2):
        # Get pose up to scale
        R, t, inliers = super().estimate2d2d(kpts1, kpts2)
        if inliers == 0:
            return R, t, inliers

        # Get essential matrix inlier mask from super class and get inliers depths
        mask = self.mask.ravel() == 1
        assert mask.shape[0] == kpts1.shape[0]
        inliers_kpts1 = kpts1[mask]
        inliers_kpts2 = kpts2[mask]
        inliers_kpts1_int = inliers_kpts1.astype(np.int32)
        inliers_kpts2_int = inliers_kpts2.astype(np.int32)
        depth_inliers_1 = depth1[inliers_kpts1_int[:, 1], inliers_kpts1_int[:, 0]]
        depth_inliers_2 = depth2[inliers_kpts2_int[:, 1], inliers_kpts2_int[:, 0]]

        # Check for valid depths
        valid = (depth_inliers_1 > 0) * (depth_inliers_2 > 0)
        inliers = valid.sum()
        if inliers < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers

        # Backproject
        xyz1 = CameraUtils.backproject_3d(inliers_kpts1[valid], depth_inliers_1[valid], self.K1)
        xyz2 = CameraUtils.backproject_3d(inliers_kpts2[valid], depth_inliers_2[valid], self.K2)

        xyz1 = xyz1.reshape(-1, 3)
        xyz2 = xyz2.reshape(-1, 3)

        # Get average point for each camera
        pmean1 = np.mean(xyz1, axis=0)
        pmean2 = np.mean(xyz2, axis=0)

        # Now, we want to minimize J(scale) = sum_i ||R21 * p1_i + scale * t21 - p2_i||^2
        # This entails dJ/dscale = 0 => scale = t21.T @ (pmean2 - R21 * pmean1)/ (t21.T @ t21)
        diff = (pmean2 - R @ pmean1).ravel()
        # print(f't norm: {np.linalg.norm(t)}')
        scale = np.dot(diff, t.ravel())  # t = t21 and we assume t21.norm()==1
        t_metric = scale * t
        # print(f'pmean diff = {diff}, t = {t}, scale = {scale}, final diff = {np.linalg.norm(t_metric - diff)}')
        t_metric = t_metric.reshape(3, 1)

        return R, t_metric, inliers

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        if isinstance(data, dict):
            data = PoseEstimatorInput.from_dict(data)
        if not isinstance(data, PoseEstimatorInput):
            raise TypeError
        if data.K1 is not None:
            self.K1 = data.K1
            self.K2 = data.K2 if data.K2 is not None else data.K1
        assert (
            data.kpts1 is not None
            and data.kpts2 is not None
            and data.depth1 is not None
            and data.depth2 is not None
        )
        return self.estimate2d2d_with_depth(data.kpts1, data.kpts2, data.depth1, data.depth2)


# Use EssentialMatrixPoseEstimator2d2d as a base and use depth values at inliers to obtain the metric translation vector.
# Procedure:
# 1. Back-project 2D inlier correspondences (from the Essential matrix) into 3D using depth maps.
# 2. Form 3D-3D correspondences between the two views.
# 3. Get the individual scales for each 3D-3D correspondence.
# 4. Use a simple RANSAC-like loop to get the most-voted scale.
class EssentialMatrixMetricPoseEstimator(EssentialMatrixPoseEstimator2d2d):
    def __init__(self, K1, K2=None, pose_estimator_type=PoseEstimatorType.ESSENTIAL_MATRIX_METRIC):
        super().__init__(K1, K2, pose_estimator_type)
        self.ransac_scale_threshold = 0.1

    # get an estimate of [R21,t21] with scale from a set of 2D-2D correspondences and the depth maps
    def estimate2d2d_with_depth(self, kpts1, kpts2, depth1, depth2):

        # Get pose up to scale
        R, t, inliers = super().estimate2d2d(kpts1, kpts2)
        if inliers == 0:
            return R, t, inliers

        # Get essential matrix inlier mask from super class and get inliers depths
        mask = self.mask.ravel() == 1
        inliers_kpts1 = kpts1[mask]
        inliers_kpts2 = kpts2[mask]
        inliers_kpts1_int = inliers_kpts1.astype(np.int32)
        inliers_kpts2_int = inliers_kpts2.astype(np.int32)
        depth_inliers_1 = depth1[inliers_kpts1_int[:, 1], inliers_kpts1_int[:, 0]]
        depth_inliers_2 = depth2[inliers_kpts2_int[:, 1], inliers_kpts2_int[:, 0]]

        # Check for valid depths
        valid = (depth_inliers_1 > 0) * (depth_inliers_2 > 0)
        if valid.sum() < 1:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = 0
            return R, t, inliers

        # Backproject
        xyz1 = CameraUtils.backproject_3d(inliers_kpts1[valid], depth_inliers_1[valid], self.K1)
        xyz2 = CameraUtils.backproject_3d(inliers_kpts2[valid], depth_inliers_2[valid], self.K2)

        xyz1 = xyz1.reshape(-1, 3)
        xyz2 = xyz2.reshape(-1, 3)

        # Rotate xyz1 to xyz2 frame (so that axes are parallel)
        xyz1_2 = (R @ xyz1.T).T
        diff = xyz2 - xyz1_2

        # Get individual scales (for each 3D-3D correspondence)
        scales = np.dot(diff.reshape(-1, 3), t.reshape(3, 1))
        # print(f'scales = {scales}')
        # scales = np.abs(scales)  # [N, 1]

        # Get the most-voted scale by using a simple RANSAC-like loop
        best_scale = None
        best_num_inliers = 0
        for scale in scales:
            num_inliers = (np.abs(scales - scale) < self.ransac_scale_threshold).sum().item()
            if num_inliers > best_num_inliers:
                best_scale = scale
                best_num_inliers = num_inliers

        # Output results
        t_metric = best_scale * t
        t_metric = t_metric.reshape(3, 1)

        return R, t_metric, best_num_inliers

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        if isinstance(data, dict):
            data = PoseEstimatorInput.from_dict(data)
        if not isinstance(data, PoseEstimatorInput):
            raise TypeError
        if data.K1 is not None:
            self.K1 = data.K1
            self.K2 = data.K2 if data.K2 is not None else data.K1
        assert (
            data.kpts1 is not None
            and data.kpts2 is not None
            and data.depth1 is not None
            and data.depth2 is not None
        )
        return self.estimate2d2d_with_depth(data.kpts1, data.kpts2, data.depth1, data.depth2)


# Estimate relative metric pose by using Perspective-n-Point algorithm from a set of 2D-3D correspondences
class PnPPoseEstimator(PoseEstimator):
    def __init__(self, K1, K2=None, pose_estimator_type=PoseEstimatorType.PNP):
        super().__init__(K1, K2, pose_estimator_type)
        # PnP RANSAC parameters
        self.ransac_num_iterations = 1000
        self.reprojection_inlier_threshold = 3.0  # pixels
        self.ransac_confidence = 0.9999

    # get an estimate of [R21,t21] with scale from a set of 2D-3D correspondences
    def estimate2d3d(self, kpts1, kpts2, depth1):
        kpts1_int = np.int32(kpts1)
        if len(kpts1_int) < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # Get depth at correspondence points
        depth_pts1 = depth1[kpts1_int[:, 1], kpts1_int[:, 0]]
        min_depth1 = max(0.0, np.min(depth_pts1))

        # Remove invalid pts (depth == 0)
        valid = depth_pts1 > min_depth1
        if valid.sum() < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        kpts1 = kpts1[valid]
        kpts2 = kpts2[valid]
        depth_pts1 = depth_pts1[valid]

        # Backproject points to 3D in each sensors' local coordinates
        xyz_1 = CameraUtils.backproject_3d(kpts1, depth_pts1, self.K1)

        # Get relative pose using PnP + RANSAC
        success, rvec, tvec, inliers = cv2.solvePnPRansac(
            xyz_1,
            kpts2,
            self.K2,
            None,
            iterationsCount=self.ransac_num_iterations,
            reprojectionError=self.reprojection_inlier_threshold,
            confidence=self.ransac_confidence,
            flags=cv2.SOLVEPNP_P3P,
        )

        # Refine with iterative PnP using inliers only
        if success and len(inliers) >= 6:
            success, rvec, tvec, _ = cv2.solvePnPGeneric(
                xyz_1[inliers],
                kpts2[inliers],
                self.K2,
                None,
                useExtrinsicGuess=True,
                rvec=rvec,
                tvec=tvec,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            rvec = rvec[0]
            tvec = tvec[0]

        # avoid degenerate solutions
        if success:
            if np.linalg.norm(tvec) > 1000:
                success = False

        if success:
            R, _ = cv2.Rodrigues(rvec)
            t = tvec.reshape(3, 1)
        else:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = []

        return R, t, len(inliers)

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        if isinstance(data, dict):
            data = PoseEstimatorInput.from_dict(data)
        if not isinstance(data, PoseEstimatorInput):
            raise TypeError
        if data.K1 is not None:
            self.K1 = data.K1
            self.K2 = data.K2 if data.K2 is not None else data.K1
        assert data.kpts1 is not None and data.kpts2 is not None and data.depth1 is not None
        return self.estimate2d3d(data.kpts1, data.kpts2, data.depth1)


# Estimate relative metric pose by using Perspective-n-Point algorithm from a set of 2D-3D correspondences
class PnPWithSigmasPoseEstimator(PoseEstimator):
    def __init__(self, K1, K2=None, pose_estimator_type=PoseEstimatorType.PNP):
        super().__init__(K1, K2, pose_estimator_type)

    # get an estimate of [R21,t21] with scale from a set of 2D-3D correspondences
    def estimate2d3d(self, kpts1, kpts2, depth1, sigmas2_2=None):
        kpts1_int = np.int32(kpts1)
        if len(kpts1_int) < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # Get depth at correspondence points
        depth_pts1 = depth1[kpts1_int[:, 1], kpts1_int[:, 0]]
        min_depth1 = max(0.0, np.min(depth_pts1))

        # Remove invalid pts (depth == 0)
        valid = depth_pts1 > min_depth1
        if valid.sum() < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        kpts1 = kpts1[valid]
        kpts2 = kpts2[valid]
        depth_pts1 = depth_pts1[valid]

        # Backproject points to 3D in each sensors' local coordinates
        xyz_1 = CameraUtils.backproject_3d(kpts1, depth_pts1, self.K1)

        num_points = xyz_1.shape[0]

        solver_input = pnpsolver.PnPsolverInput()
        solver_input.points_2d = kpts2.tolist()
        solver_input.points_3d = xyz_1.tolist()
        if sigmas2_2 is not None:
            solver_input.sigmas2 = sigmas2_2
        else:
            solver_input.sigmas2 = [1.0 for _ in range(num_points)]
        solver_input.fx = self.K2[0, 0]
        solver_input.fy = self.K2[1, 1]
        solver_input.cx = self.K2[0, 2]
        solver_input.cy = self.K2[1, 2]
        solver = pnpsolver.PnPsolver(solver_input)

        # Run the PnP solver
        ok, transformation, no_more, inliers, n_inliers = solver.iterate(5)
        inliers = np.array(inliers).astype(bool)
        R12 = transformation[:3, :3]
        t12 = transformation[:3, 3]
        R12 = R12.reshape(3, 3)
        t12 = t12.reshape(3, 1)

        if ok:
            R21 = R12.T
            t21 = -R21 @ t12
            R = R12
            t = t12
        else:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = []

        return R, t, len(inliers)

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        if isinstance(data, dict):
            data = PoseEstimatorInput.from_dict(data)
        if not isinstance(data, PoseEstimatorInput):
            raise TypeError
        if data.K1 is not None:
            self.K1 = data.K1
            self.K2 = data.K2 if data.K2 is not None else data.K1
        assert data.kpts1 is not None and data.kpts2 is not None and data.depth1 is not None
        num_points = data.kpts1.shape[0]
        sigmas2_2 = [1.0 for _ in range(num_points)] if data.sigmas2_2 is None else data.sigmas2_2
        return self.estimate2d3d(data.kpts1, data.kpts2, data.depth1, sigmas2_2)


# Estimate relative metric pose by using Perspective-n-Point algorithm from a set of 2D-3D correspondences
class MlPnPWithSigmasPoseEstimator(PoseEstimator):
    def __init__(self, K1, K2=None, pose_estimator_type=PoseEstimatorType.PNP):
        super().__init__(K1, K2, pose_estimator_type)
        # PnP RANSAC parameters
        self.ransac_num_iterations = 1000
        self.reprojection_inlier_threshold = 3.0  # pixels
        self.ransac_confidence = 0.9999

    # get an estimate of [R21,t21] with scale from a set of 2D-3D correspondences
    def estimate2d3d(self, kpts1, kpts2, depth1, sigmas2_2=None):
        kpts1_int = np.int32(kpts1)
        if len(kpts1_int) < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # Get depth at correspondence points
        depth_pts1 = depth1[kpts1_int[:, 1], kpts1_int[:, 0]]
        min_depth1 = max(0.0, np.min(depth_pts1))

        # Remove invalid pts (depth == 0)
        valid = depth_pts1 > min_depth1
        if valid.sum() < 4:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        kpts1 = kpts1[valid]
        kpts2 = kpts2[valid]
        depth_pts1 = depth_pts1[valid]

        # Backproject points to 3D in each sensors' local coordinates
        xyz_1 = CameraUtils.backproject_3d(kpts1, depth_pts1, self.K1)

        num_points = xyz_1.shape[0]

        solver_input = pnpsolver.PnPsolverInput()
        solver_input.points_2d = kpts2.tolist()
        solver_input.points_3d = xyz_1.tolist()
        if sigmas2_2 is not None:
            solver_input.sigmas2 = sigmas2_2
        else:
            solver_input.sigmas2 = [1.0 for _ in range(num_points)]
        solver_input.fx = self.K2[0, 0]
        solver_input.fy = self.K2[1, 1]
        solver_input.cx = self.K2[0, 2]
        solver_input.cy = self.K2[1, 2]
        solver = pnpsolver.MLPnPsolver(solver_input)

        # Run the PnP solver
        ok, transformation, no_more, inliers, n_inliers = solver.iterate(5)
        inliers = np.array(inliers).astype(bool)
        R12 = transformation[:3, :3]
        t12 = transformation[:3, 3]
        R12 = R12.reshape(3, 3)
        t12 = t12.reshape(3, 1)

        if ok:
            R21 = R12.T
            t21 = -R21 @ t12
            R = R12
            t = t12
        else:
            R = np.full((3, 3), np.nan)
            t = np.full((3, 1), np.nan)
            inliers = []

        return R, t, len(inliers)

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        if isinstance(data, dict):
            data = PoseEstimatorInput.from_dict(data)
        if not isinstance(data, PoseEstimatorInput):
            raise TypeError
        if data.K1 is not None:
            self.K1 = data.K1
            self.K2 = data.K2 if data.K2 is not None else data.K1
        assert data.kpts1 is not None and data.kpts2 is not None and data.depth1 is not None
        num_points = data.kpts1.shape[0]
        sigmas2_2 = [1.0 for _ in range(num_points)] if data.sigmas2_2 is None else data.sigmas2_2
        return self.estimate2d3d(data.kpts1, data.kpts2, data.depth1, sigmas2_2)


# Estimate relative metric pose by using Procrustes algorithm from a set of 3D-3D correspondences
# ICP is used to refine the estimate.
class ProcrustesPoseEstimator(PoseEstimator):
    def __init__(self, K1, K2=None, pose_estimator_type=PoseEstimatorType.PROCUSTES):
        super().__init__(K1, K2, pose_estimator_type)
        # Procrustes RANSAC parameters
        self.ransac_max_correspondence_distance = 0.05  # meters
        self.refine_with_icp = True
        self.icp_relative_fitness = 1e-4
        self.icp_relative_rmse = 1e-4
        self.icp_max_iterations = 30

    # get an estimate of [R21,t21] with scale from a set of 3D-3D correspondences
    def estimate3d3d(self, kpts1, kpts2, depth1, depth2):
        kpts1_int = np.int32(kpts1)
        kpts2_int = np.int32(kpts2)

        if len(kpts1) < 3:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0

        # Get depth at correspondence points
        depth_pts1 = depth1[kpts1_int[:, 1], kpts1_int[:, 0]]
        depth_pts2 = depth2[kpts2_int[:, 1], kpts2_int[:, 0]]
        depth_1_min = max(0.0, np.min(depth_pts1))
        depth_2_min = max(0.0, np.min(depth_pts2))

        # Remove invalid pts (depth == 0)
        valid = (depth_pts1 > depth_1_min) * (depth_pts2 > depth_2_min)
        if valid.sum() < 3:
            return np.full((3, 3), np.nan), np.full((3, 1), np.nan), 0
        kpts1 = kpts1[valid]
        kpts2 = kpts2[valid]
        depth_pts1 = depth_pts1[valid]
        depth_pts2 = depth_pts2[valid]

        # Backproject points to 3D in each sensors' local coordinates
        xyz_1 = CameraUtils.backproject_3d(kpts1, depth_pts1, self.K1)
        xyz_2 = CameraUtils.backproject_3d(kpts2, depth_pts2, self.K2)

        # Create open3d point cloud objects and correspondences idxs
        pcl_1 = o3d.geometry.PointCloud()
        pcl_1.points = o3d.utility.Vector3dVector(xyz_1)
        pcl_2 = o3d.geometry.PointCloud()
        pcl_2.points = o3d.utility.Vector3dVector(xyz_2)
        corr_idx = np.arange(kpts1.shape[0])
        # Create a correspondence matrix that matches the keypoints between the two point clouds
        corr_idx = np.tile(
            corr_idx.reshape(-1, 1), (1, 2)
        )  # Duplicates the corr_idx array along a new axis, creating a 2D array where each row contains the same index twice.
        corr_idx = o3d.utility.Vector2iVector(corr_idx)

        # Obtain relative pose using procrustes
        ransac_criteria = o3d.pipelines.registration.RANSACConvergenceCriteria()
        res = o3d.pipelines.registration.registration_ransac_based_on_correspondence(
            pcl_1,
            pcl_2,
            corr_idx,
            self.ransac_max_correspondence_distance,
            criteria=ransac_criteria,
        )

        # Refine with ICP using all the depth map information from both images.
        # We exploit the first pose estimate we obtained from the procrustes algorithm as a starting guess for ICP.
        if self.refine_with_icp:
            # First, backproject both (whole) point clouds
            vv1, uu1 = np.mgrid[0 : depth1.shape[0], 0 : depth1.shape[1]]
            uv1_coords = np.concatenate([uu1.reshape(-1, 1), vv1.reshape(-1, 1)], axis=1).astype(
                np.float32
            )
            uv1_coords += np.array([0.5, 0.5], dtype=np.float32)

            # First, backproject both (whole) point clouds
            if depth2.shape[0] != depth1.shape[0] or depth2.shape[1] != depth1.shape[1]:
                vv2, uu2 = np.mgrid[0 : depth2.shape[0], 0 : depth2.shape[1]]
                uv2_coords = np.concatenate(
                    [uu2.reshape(-1, 1), vv2.reshape(-1, 1)], axis=1
                ).astype(np.float32)
                uv2_coords += np.array([0.5, 0.5], dtype=np.float32)
            else:
                uv2_coords = uv1_coords

            valid = depth1.reshape(-1) > 0
            xyz_1 = CameraUtils.backproject_3d(
                uv1_coords[valid], depth1.reshape(-1)[valid], self.K1
            )
            # print(f'ICP xyz_1 shape: {xyz_1.shape}')

            valid = depth2.reshape(-1) > 0
            xyz_2 = CameraUtils.backproject_3d(
                uv2_coords[valid], depth2.reshape(-1)[valid], self.K2
            )
            # print(f'ICP xyz_2 shape: {xyz_2.shape}')

            pcl_1 = o3d.geometry.PointCloud()
            pcl_1.points = o3d.utility.Vector3dVector(xyz_1)
            pcl_2 = o3d.geometry.PointCloud()
            pcl_2.points = o3d.utility.Vector3dVector(xyz_2)

            icp_criteria = o3d.pipelines.registration.ICPConvergenceCriteria(
                relative_fitness=self.icp_relative_fitness,
                relative_rmse=self.icp_relative_rmse,
                max_iteration=self.icp_max_iterations,
            )

            res = o3d.pipelines.registration.registration_icp(
                pcl_1,
                pcl_2,
                self.ransac_max_correspondence_distance,
                init=res.transformation,
                criteria=icp_criteria,
            )

        inliers = int(res.fitness * np.asarray(pcl_2.points).shape[0])

        R = res.transformation[:3, :3]
        t = res.transformation[:3, -1].reshape(3, 1)

        return R, t, inliers

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        if isinstance(data, dict):
            data = PoseEstimatorInput.from_dict(data)
        if not isinstance(data, PoseEstimatorInput):
            raise TypeError
        if data.K1 is not None:
            self.K1 = data.K1
            self.K2 = data.K2 if data.K2 is not None else data.K1
        assert (
            data.kpts1 is not None
            and data.kpts2 is not None
            and data.depth1 is not None
            and data.depth2 is not None
        )
        return self.estimate3d3d(data.kpts1, data.kpts2, data.depth1, data.depth2)


# Estimate relative metric pose in Sim3 space by using 3D-3D correspondences
class Sim3PoseEstimator(PoseEstimator):
    def __init__(self, K1, K2=None, pose_estimator_type=PoseEstimatorType.PROCUSTES):
        super().__init__(K1, K2, pose_estimator_type)
        self.ransac_probability = 0.99
        self.min_num_inliers = 20
        self.ransac_max_num_iterations = 300

    # get an estimate of [R21,t21] with scale from a set of 3D-3D correspondences
    def estimate3d3d(self, pts1, pts2, fix_scale):
        solver_input_data = sim3solver.Sim3SolverInput2()
        assert pts1.shape[0] == pts2.shape[0]
        num_points = pts1.shape[0]
        solver_input_data.sigmas2_1 = np.full(num_points, 1.0, dtype=np.float32)
        solver_input_data.sigmas2_2 = solver_input_data.sigmas2_1
        solver_input_data.fix_scale = fix_scale
        solver_input_data.points_3d_c1 = pts1
        solver_input_data.points_3d_c2 = pts2
        # Create Sim3Solver object with the input data
        solver = sim3solver.Sim3Solver(solver_input_data)
        # Set RANSAC parameters (using defaults here)
        solver.set_ransac_parameters(
            self.ransac_probability, self.min_num_inliers, self.ransac_max_num_iterations
        )
        # Prepare variables for iterative solving
        # vbInliers = [False] * len(solver_input_data2.points_3d_c1)
        # nInliers = 0
        # bConverged = False
        # Test the first iteration (e.g., 10 iterations)
        transformation, bNoMore, vbInliers, nInliers, bConverged = solver.iterate(5)
        if False:
            print("Estimated transformation after 10 iterations:")
            print(transformation)
        registration_error3d = solver.compute_3d_registration_error()
        R12 = solver.get_estimated_rotation()
        t12 = solver.get_estimated_translation()
        scale12 = solver.get_estimated_scale()
        print(
            f"Sim3PoseEstimator: #inliers {nInliers}, #points {num_points}, bConverged {bConverged}, reg error 3d: {registration_error3d}"
        )

        R12 = R12.reshape(3, 3)
        t12 = t12.reshape(3, 1)

        R21 = R12.T
        t21 = -R21 @ t12
        R = R21
        t = t21
        inliers = nInliers

        return R, t, inliers

    # Estimate pose [R21,t21] from data.
    # Data PoseEstimatorInput object or dict containing the same information.
    def estimate(self, data):
        if isinstance(data, dict):
            data = PoseEstimatorInput.from_dict(data)
        if not isinstance(data, PoseEstimatorInput):
            raise TypeError
        if data.K1 is not None:
            self.K1 = data.K1
            self.K2 = data.K2 if data.K2 is not None else data.K1
        assert (data.pts1 is not None and data.pts2 is not None) or (
            data.kpts1 is not None
            and data.kpts2 is not None
            and data.depth1 is not None
            and data.depth2 is not None
        )
        fix_scale = data.fix_scale
        if data.pts1 is not None and data.pts2 is not None:
            return self.estimate3d3d(data.pts1, data.pts2, fix_scale)
        else:
            kpts1_int = np.int32(data.kpts1)
            kpts2_int = np.int32(data.kpts2)
            depths1 = data.depth1[kpts1_int[:, 1], kpts1_int[:, 0]]
            depths2 = data.depth2[kpts2_int[:, 1], kpts2_int[:, 0]]
            pts1 = CameraUtils.backproject_3d(data.kpts1, depths1, self.K1)
            pts2 = CameraUtils.backproject_3d(data.kpts2, depths2, self.K2)
            return self.estimate3d3d(pts1, pts2, fix_scale)
