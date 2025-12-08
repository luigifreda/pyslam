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


# DLT with normalized image coordinates (see [HartleyZisserman Sect. 12.2 ])
def triangulate_point(pose1, pose2, pt1, pt2):
    A = np.zeros((4, 4))
    A[0] = pt1[0] * pose1[2] - pose1[0]
    A[1] = pt1[1] * pose1[2] - pose1[1]
    A[2] = pt2[0] * pose2[2] - pose2[0]
    A[3] = pt2[1] * pose2[2] - pose2[1]
    _, _, vt = np.linalg.svd(A)
    return vt[3]


def triangulate_points(pose1, pose2, pts1, pts2, mask=None):
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
    P1w = pose_1w[:3, :]  # [R1w, t1w]
    P2w = pose_2w[:3, :]  # [R2w, t2w]

    point_4d_hom = cv2.triangulatePoints(P1w, P2w, kpn_1.T, kpn_2.T)
    good_pts_mask = np.where(point_4d_hom[3] != 0)[0]
    point_4d = point_4d_hom / point_4d_hom[3]

    if __debug__:
        if False:
            point_reproj = P1w @ point_4d
            point_reproj = point_reproj / point_reproj[2] - add_ones(kpn_1).T
            err = np.sum(point_reproj**2)
            print("reproj err: ", err)

    # return point_4d.T
    points_3d = point_4d[:3, :].T
    return points_3d, good_pts_mask
