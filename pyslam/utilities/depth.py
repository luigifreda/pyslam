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


# create a scaled image of uint8 from a image of floats
def img_from_depth(img_flt, img_max=None, img_min=None, eps=1e-9):
    assert img_flt.dtype in [np.float32, np.float64, np.float16, np.double, np.single]
    img_max = np.max(img_flt) if img_max is None else img_max
    img_min = np.min(img_flt) if img_min is None else img_min
    if img_max is not None or img_min is not None:
        img_flt = np.clip(img_flt, img_min, img_max)
    img_range = img_max - img_min
    if img_range < eps:
        img_range = 1
    img = (img_flt - img_min) / img_range * 255
    return img.astype(np.uint8)


class PointCloud:
    def __init__(self, points=None, colors=None, semantics=None, object_ids=None):
        self.points = points  # array Nx3
        self.colors = colors  # array Nx3
        self.semantics = semantics  # array Nx1 (or NxD where D is the semantic dimension)
        self.object_ids = object_ids  # array Nx1    (object IDs or projected instance IDs)


def depth2pointcloud(
    depth,
    image,
    fx,
    fy,
    cx,
    cy,
    max_depth=np.inf,
    min_depth=0.0,
    semantic_image=None,
    object_ids_image=None,  # object IDs or projected instance IDs
):
    """
    Convert depth image and color image to point cloud.
    If semantic_image is provided, it will be used to extract semantics.
    If instance_ids_image/object_ids_image is provided, it will be used to extract instance_ids/object_ids.
    Otherwise, semantics and instance_ids will be set to None.
    """
    # mask for valid depth values
    valid = (depth > min_depth) & (depth < max_depth)
    # use boolean indexing directly (faster than np.where)
    z = depth[valid]
    # pre-compute inverse focal lengths to avoid repeated divisions
    inv_fx = 1.0 / fx
    inv_fy = 1.0 / fy
    # get row and column indices for valid pixels
    rows, cols = np.where(valid)
    x = (cols - cx) * z * inv_fx
    y = (rows - cy) * z * inv_fy
    points = np.column_stack([x, y, z])  # [N, 3] in camera coordinates
    # colors corresponding to valid depth values
    colors = image[valid] / 255.0
    if semantic_image is not None and semantic_image.size > 0:
        semantics = semantic_image[valid]
    else:
        semantics = None
    if object_ids_image is not None and object_ids_image.size > 0:
        object_ids = object_ids_image[valid]
    else:
        object_ids = None
    return PointCloud(points, colors, semantics, object_ids)


def depth2pointcloud_v2(depth, image, fx, fy, cx, cy):
    width, height = depth.shape[0], depth.shape[1]
    # Generate mesh grid and calculate point cloud coordinates
    x, y = np.meshgrid(np.arange(width), np.arange(height))
    x = (x - cx) / fx
    y = (y - cy) / fy
    z = np.array(depth)
    points = np.stack((np.multiply(x, z), np.multiply(y, z), z), axis=-1).reshape(-1, 3)
    colors = np.array(image).reshape(-1, 3) / 255.0
    return PointCloud(points, colors)


# Remove depth values where the depth difference is too large.
# In other words, it removes the ghost/shadow points appearing on edge discontinuties.
# If delta_depth=None, it uses the median absolute deviation (MAD) of the depth differences.
def filter_shadow_points(depth, delta_depth=None, delta_x=2, delta_y=2, fill_value=-1):
    depth_out = depth.copy()

    # Initialize mask with False
    mask = np.zeros_like(depth, dtype=bool)

    # Check depth differences in all directions
    delta_values = []
    if delta_y > 0:
        delta_depth_y = np.abs(depth[delta_y:, :] - depth[:-delta_y, :])
        if delta_depth is None:
            delta_values.append(delta_depth_y.flatten())
    if delta_x > 0:
        delta_depth_x = np.abs(depth[:, delta_x:] - depth[:, :-delta_x])
        if delta_depth is None:
            delta_values.append(delta_depth_x.flatten())
    if delta_depth is None:
        delta_values = np.concatenate(delta_values)
        delta_values = delta_values[delta_values > 0]

        mad = np.median(
            delta_values
        )  # MAD, approximating median(deltas)=0 in MAD=median(deltas - median(deltas))
        sigma_depth = 1.4826 * mad
        delta_depth = (
            3 * sigma_depth
        )  # + mad  # the final "+ mad" is for adding back a bias (the median itself) to the depth threshold since the delta distribution is not centered at zero
        # print(f'filter_shadow_points: delta_depth={delta_depth}, mad: {mad}')

    # Update mask
    if delta_y > 0:
        delta_depth_y_is_big = delta_depth_y > delta_depth
        mask[delta_y:, :] |= delta_depth_y_is_big
        mask[:-delta_y, :] |= delta_depth_y_is_big

    if delta_x > 0:
        delta_depth_x_is_big = delta_depth_x > delta_depth
        mask[:, delta_x:] |= delta_depth_x_is_big
        mask[:, :-delta_x] |= delta_depth_x_is_big

    # Set invalid depth values to fill_value
    depth_out[mask] = fill_value

    return depth_out


def point_cloud_to_depth(points, K, w, h):
    """
    Generate a depth image from a point cloud.

    Parameters:
      points: (N, 3) numpy array of points in the camera coordinate system.
      K: (3, 3) intrinsic calibration matrix.
      w: target image width.
      h: target image height.

    Returns:
      depth_img: (h, w) numpy array with depth values. Pixels with no point are set to 0.
    """
    fx, fy = K[0, 0], K[1, 1]
    cx, cy = K[0, 2], K[1, 2]

    # Only consider points with positive depth
    valid = points[:, 2] > 0
    valid_points = points[valid]

    X, Y, Z = valid_points[:, 0], valid_points[:, 1], valid_points[:, 2]
    # Using round to get pixel coordinates
    u = np.round((X / Z) * fx + cx).astype(np.int32)
    v = np.round((Y / Z) * fy + cy).astype(np.int32)

    mask = (u >= 0) & (u < w) & (v >= 0) & (v < h)
    u = u[mask]
    v = v[mask]
    Z = Z[mask]

    depth_img = np.full((h, w), np.inf, dtype=np.float32)
    idx = v * w + u
    order = np.argsort(Z)
    idx = idx[order]
    Z = Z[order]
    u_sorted = u[order]
    v_sorted = v[order]

    _, unique_indices = np.unique(idx, return_index=True)
    unique_u = u_sorted[unique_indices]
    unique_v = v_sorted[unique_indices]
    unique_Z = Z[unique_indices]

    depth_img[unique_v, unique_u] = unique_Z
    depth_img[np.isinf(depth_img)] = 0
    return depth_img
