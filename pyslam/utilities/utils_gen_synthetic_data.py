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


# generate random point in camera image
# output: [Nx2] (N = number of points)
def generate_random_points_2d(width, height, num_points):
    points_2d = np.random.uniform(low=[0, 0], high=[width, height], size=(num_points, 2)).astype(
        np.float64
    )
    return points_2d


# back project 2d image points with given camera matrix by using a random depths in the range [minD, maxD]
# output: points_3d [Nx3], z [Nx1] (N = number of points)
def backproject_points(K, points_2d, minD, maxD):
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]

    z = np.random.uniform(minD, maxD, size=points_2d.shape[0]).astype(np.float64)
    x = (points_2d[:, 0] - cx) * z / fx
    y = (points_2d[:, 1] - cy) * z / fy

    points_3d = np.vstack((x, y, z)).T
    return points_3d.astype(np.float64), z


# project 3d points with given camera matrix and pose and return the 2d image points, the visibility mask
# input: K [3x3], Tcw [4x4], points_3d_w [Nx3]
# output: points_2d [Nx2], points_3d_c [Nx3], mask [Nx1]
def project_points(K, Tcw, points_3d_w, width, height, depth_eps=1e-6):
    num_points = points_3d_w.shape[0]
    points_2d = np.zeros((num_points, 2), dtype=np.float64)
    mask = np.full(num_points, False, dtype=bool)
    fx = K[0, 0]
    fy = K[1, 1]
    cx = K[0, 2]
    cy = K[1, 2]
    if Tcw is None:
        points_3d_c = points_3d_w
    else:
        points_3d_c = (Tcw[:3, :3] @ points_3d_w.T + Tcw[:3, 3].reshape(3, 1)).T
    for i in range(num_points):
        point = points_3d_c[i, :]
        if point[2] <= depth_eps:
            continue
        x = point[0] / point[2] * fx + cx
        y = point[1] / point[2] * fy + cy
        points_2d[i, :] = np.array([x, y])
        if x >= 0 and x < width and y >= 0 and y < height:
            mask[i] = True
    return points_2d, points_3d_c, mask
