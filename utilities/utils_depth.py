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
    assert(img_flt.dtype in [np.float32, np.float64, np.float16, np.double, np.single])
    img_max = np.max(img_flt) if img_max is None else img_max
    img_min = np.min(img_flt) if img_min is None else img_min
    if img_max is not None or img is not None:
        img_flt = np.clip(img_flt, img_min, img_max) 
    img_range = img_max - img_min
    if img_range < eps:
        img_range = 1
    img = (img_flt-img_min)/img_range * 255   
    return img.astype(np.uint8) 



class PointCloud:
    def __init__(self, points, colors):
        self.points = points
        self.colors = colors
        
def depth2pointcloud(depth, image, fx, fy, cx, cy, max_depth, min_depth=0.0):
    # mask for valid depth values
    valid = (depth > min_depth) & (depth < max_depth)
    # indices of valid depth values
    rows, cols = np.where(valid)
    z = depth[rows, cols]
    x = (cols - cx) * z / fx
    y = (rows - cy) * z / fy
    points = np.stack([x, y, z], axis=-1)
    # colors corresponding to valid depth values
    colors = image[rows, cols] / 255.0
    return PointCloud(points, colors)

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
        
        mad = np.median(delta_values) # MAD, approximating median(deltas)=0 in MAD=median(deltas - median(deltas))
        sigma_depth = 1.4826 * mad
        delta_depth = 3 * sigma_depth # + mad  # the final "+ mad" is for adding back a bias (the median itself) to the depth threshold since the delta distribution is not centered at zero
        #print(f'filter_shadow_points: delta_depth={delta_depth}, mad: {mad}')
            
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
