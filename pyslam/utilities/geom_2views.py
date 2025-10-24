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

from .geometry import skew, poseRt

from numba import njit


@njit(cache=True)
def computeF12_numba(R1w, t1w, R2w, t2w, K1inv, K2, K2inv):
    R12 = R1w @ R2w.T
    t12 = -R1w @ (R2w.T @ t2w) + t1w

    t12x = skew(t12)
    R21 = R12.T
    H21 = (K2 @ R21) @ K1inv
    F12 = ((K1inv.T @ t12x) @ R12) @ K2inv
    return F12, H21


# compute the fundamental mat F12 and the infinite homography H21 [Hartley Zisserman pag 339]
# from two frames
def computeF12_(f1, f2):
    f1_Tcw = f1.Tcw()
    f2_Tcw = f2.Tcw()
    R1w = f1_Tcw[:3, :3]
    t1w = f1_Tcw[:3, 3]
    R2w = f2_Tcw[:3, :3]
    t2w = f2_Tcw[:3, 3]

    R12 = R1w @ R2w.T
    t12 = -R1w @ (R2w.T @ t2w) + t1w

    t12x = skew(t12)
    K1Tinv = f1.camera.Kinv.T
    R21 = R12.T
    H21 = (
        f2.camera.K @ R21
    ) @ f1.camera.Kinv  # infinite homography from f1 to f2 [Hartley Zisserman pag 339]
    F12 = ((K1Tinv @ t12x) @ R12) @ f2.camera.Kinv
    return F12, H21


def computeF12(f1, f2):
    f1_Tcw = f1.Tcw()
    f2_Tcw = f2.Tcw()
    R1w = np.ascontiguousarray(f1_Tcw[:3, :3])
    t1w = np.ascontiguousarray(f1_Tcw[:3, 3])
    R2w = np.ascontiguousarray(f2_Tcw[:3, :3])
    t2w = np.ascontiguousarray(f2_Tcw[:3, 3])
    return computeF12_numba(R1w, t1w, R2w, t2w, f1.camera.Kinv, f2.camera.K, f2.camera.Kinv)


@njit(cache=True)
def check_dist_epipolar_line(kp1, kp2, F12, sigma2_kp2):
    # Epipolar line in second image l = kp1' * F12 = [a b c]
    l = np.dot(F12.T, np.array([kp1[0], kp1[1], 1]))
    num = l[0] * kp2[0] + l[1] * kp2[1] + l[2]  # kp1' * F12 * kp2
    den = l[0] * l[0] + l[1] * l[1]  # a*a+b*b

    if den == 0:
        # if(den < 1e-20):
        return False

    dist_sqr = num * num / den  # squared (minimum) distance of kp2 from the epipolar line l
    return (
        dist_sqr < 3.84 * sigma2_kp2
    )  # value of inverse cumulative chi-square for 1 DOF (Hartley Zisserman pag 567)


# Fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
# Input: kpn_ref and kpn_cur are two arrays of [Nx2] normalized coordinates of matched keypoints
# out: a) Trc: homogeneous transformation matrix containing Rrc, trc  ('cur' frame with respect to 'ref' frame)    pr = Trc * pc
#      b) mask_match: array of N elements, every element of which is set to 0 for outliers and to 1 for the other points (computed only in the RANSAC and LMedS methods)
# N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
# N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
# - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie a ruled quadric
# - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
# N.B.3: the five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
# N.B.4: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return a correct rotation matrix.
# N.B.5: the OpenCV findEssentialMat function uses the five-point algorithm solver by D. Nister => hence it should work well in the degenerate planar cases
def estimate_pose_ess_mat(kpn_ref, kpn_cur, method=cv2.RANSAC, prob=0.999, threshold=0.0003):
    # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above )
    E, mask_match = cv2.findEssentialMat(
        kpn_cur, kpn_ref, focal=1, pp=(0.0, 0.0), method=method, prob=prob, threshold=threshold
    )
    _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0.0, 0.0))
    return poseRt(R, t.T), mask_match  # Trc, mask_mat
