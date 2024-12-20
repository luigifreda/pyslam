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
import time 
import math 
import numpy as np
import cv2
from enum import Enum

from scipy.spatial import cKDTree
#from pykdtree.kdtree import KDTree # slower!

from utils_sys import Printer, import_from, is_opencv_version_greater_equal
from utils_geom import add_ones, s1_diff_deg, s1_dist_deg, l2_distances

ORBextractor = import_from('orbslam2_features', 'ORBextractor')  
      
kPySlamUtilsAvailable= True      
try:
    import pyslam_utils
except:
    kPySlamUtilsAvailable = False 
    Printer.orange('WARNING: cannot import pyslam_utils')
      
from config_parameters import Parameters 



# convert matrix of pts into list of cv2 keypoints
def convert_pts_to_keypoints(pts, size=1): 
    kps = []
    if pts is not None: 
        if pts.ndim > 2:
            # convert matrix [Nx1x2] of pts into list of keypoints
            if is_opencv_version_greater_equal(4,5,3):  
                kps = [ cv2.KeyPoint(p[0][0], p[0][1], size=size) for p in pts ]          
            else:
                kps = [ cv2.KeyPoint(p[0][0], p[0][1], _size=size) for p in pts ]
        else: 
            # convert matrix [Nx2] of pts into list of keypoints  
            if is_opencv_version_greater_equal(4,5,3):  
                kps = [ cv2.KeyPoint(p[0], p[1], size=size) for p in pts ]                      
            else:
                kps = [ cv2.KeyPoint(p[0], p[1], _size=size) for p in pts ]     
    return kps         


# from https://stackoverflow.com/questions/48385672/opencv-python-unpack-sift-octave
# from https://gist.github.com/lxc-xx/7088609 (SIFT implementation)
# from https://stackoverflow.com/questions/17015995/opencv-sift-descriptor-keypoint-radius
# from https://github.com/vlfeat/vlfeat/blob/38a03e12daf50ee98633de06120834d0d1d87e23/vl/sift.c#L1948  (vlfeat SIFT implementation)
# see also https://www.vlfeat.org/api/sift.html (documentation of vlfeat SIFT implementation)
# N.B.: the opencv SIFT implementation uses a negative first octave (int firstOctave = -1) to work with an higher resolution image (scale=2.0, double size)
def unpackSiftOctave(kpt):
    """unpackSIFTOctave(kpt)->(octave,layer,scale)
    @brief Unpack Sift Keypoint
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = int(_octave)&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1.0/(1<<octave))
    else:
        scale = float(1<<(-octave))
    #print('sift octave: ', octave,' layer: ', layer, ' scale: ', scale, 'size: ', kpt.size)
    return (octave, layer, scale)
    
def unpackSiftOctavePlusOne(kpt):
    """unpackSiftOctavePlusOne(kpt)->octave+1 (-1 is upsampled layer, 0 is input image layer, 1 is the first layer and so on... )
    @brief Unpack Sift Keypoint
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = _octave&0xFF
    if octave>=128:
        octave |= -128        
    return octave+1    

# return a virtual 'intra'-level with a virtual scale factor =  2^(1/3) where 3 is the number of intra-layers per octave    
def unpackSiftOctaveIntra(kpt):
    """unpackSiftOctaveVirtual(kpt)-> (octave+1)*3 + layer
    @brief Unpack Sift Keypoint; return a virtual 'intra' level with a virtual scale factor =  2^(1/3) where 3 is the number of intra-layers per octave
    @param kpt: cv2.KeyPoint (of SIFT)
    """
    _octave = kpt.octave
    octave = int(_octave)&0xFF
    layer  = (_octave>>8)&0xFF
    if octave>=128:
        octave |= -128
    if octave>=0:
        scale = float(1.0/(1<<octave))
    else:
        scale = float(1<<(-octave))
    return (octave+1)*3 + layer   # return a virtual intra-level with a virtual scale factor =  2^(1/3) where 3 is the number of layers per octave

class UnpackOctaveMethod(Enum):
    DEFAULT       = 1
    PLUS_ONE      = 2
    INTRAL_LAYERS = 3    

def unpackSiftOctaveKps(kps, method=UnpackOctaveMethod.DEFAULT):  
    if method == UnpackOctaveMethod.DEFAULT:
        for kpt in kps: 
            kpt.octave,_,_ = unpackSiftOctave(kpt)
    elif method == UnpackOctaveMethod.PLUS_ONE:
        for kpt in kps: 
            kpt.octave = unpackSiftOctavePlusOne(kpt)   
    elif method == UnpackOctaveMethod.INTRAL_LAYERS:           
        for kpt in kps: 
            kpt.octave = unpackSiftOctaveIntra(kpt)            
        
        
# robust estimatation of descriptor distance standard deviation by using MAD (Median Absolute Deviation)      
# N.B: you can use the thresholding condition: 
#      descriptor_distance < factor * sigma_mad
# https://en.wikipedia.org/wiki/Median_absolute_deviation 
def descriptor_sigma_mad(des1, des2, descriptor_distances=l2_distances):
    #dists = np.full(des1.shape[0], 0., dtype=np.float32) 
    # for i in range(des1.shape[0]):
    #     dists[i] = descriptor_distance(des1[i],des2[i])
    dists = descriptor_distances(des1,des2)
    dists_median = np.median(dists)     # MAD, approximating dists_median=0 
    sigma_mad = 1.4826 * dists_median
    return sigma_mad, dists        

# robust estimation of descriptor distance standard deviation by using MAD (Median Absolute Deviation)  
# N.B: you can use the thresholding condition:
#  (descriptor_distance < dists_median) or (descriptor_distance - dists_median < factor * sigma_mad)
# https://en.wikipedia.org/wiki/Median_absolute_deviation
def descriptor_sigma_mad_v2(des1, des2, descriptor_distances=l2_distances):
    #dists = np.full(des1.shape[0], 0., dtype=np.float32) 
    # for i in range(des1.shape[0]):
    #     dists[i] = descriptor_distance(des1[i],des2[i])
    dists = descriptor_distances(des1,des2)
    dists_median = np.median(dists)
    ads = np.fabs(dists - dists_median) # absolute deviations from median 
    sigma_mad = 1.4826 * np.median(ads) 
    return sigma_mad, dists_median, dists       


# keep the first 'self.num_features' best features
def sat_num_features(kps, des=None, num_features=Parameters.kNumFeatures):    
    if len(kps) > num_features:
        # keep the features with the best response 
        if des is None: 
            kps = sorted(kps, key=lambda x:x.response, reverse=True)[:num_features]     
        else:            
            # sort by score to keep highest score features 
            # print('sat with des')
            order = np.argsort([kp.response for kp in kps])[::-1][:num_features]      # [::-1] is for reverse order 
            kps = np.array(kps)[order]    
            des = np.array(des)[order]             
    return kps, des 

     
# kdtree-based non-maximum suppression of keypoints 
# adapted and optimized from https://stackoverflow.com/questions/9210431/well-distributed-features-using-opencv/50267891
def kdt_nms(kps, des=None, num_features=Parameters.kNumFeatures, r=Parameters.kKdtNmsRadius, k_max=9):
    """ Use kd-tree to perform local non-maximum suppression of key-points
    kps - key points obtained by one of openCVs 2d features detectors (SIFT, SURF, AKAZE etc..)
    r - the radius of points to query for removal
    k_max - maximum points retreived in single query (not used)
    """
    
    if len(kps)==0:
        return kps, des
    if des is not None:
        assert(len(des)==len(kps))
    
    # sort by score to give priority to highest score features 
    order = np.argsort([kp.response for kp in kps])[::-1]  # [::-1] is for reverse order     
    kps = np.array(kps)[order] 

    # create kd-tree for quick NN queries 
    data_pts = np.array([kp.pt for kp in kps],dtype=np.float32)
        
    kd_tree = cKDTree(data_pts)

    # perform NMS using kd-tree, by querying points by score order, 
    # and removing neighbors from future queries
    N = len(kps)
    idxs_removed = set() 
    
    #time_start = time.time()     
    kd_idxs = kd_tree.query_ball_point(data_pts,r) 
    #print('elapsed: ', time.time()-time_start)     
     
    for i in range(N):
        if i in idxs_removed:
            continue
        #idxs_removed.update([j for j in kd_idxs[i] if j >i])
        for j in kd_idxs[i]: 
            if j>i:
                idxs_removed.add(j)  
    idxs_remaining = [i for i in range(N) if i not in idxs_removed] 
    
    kps_out = kps[idxs_remaining]
    des_out = None
    if des is not None:
        #print('des.shape:',des.shape)
        des = des[order]    
        des_out = des[idxs_remaining]
    if len(kps_out) > num_features:
        kps_out = kps_out[:num_features]
        if des_out is not None:
            des_out = des_out[:num_features]                                
    return kps_out, des_out


# adapted from https://github.com/BAILOOL/ANMS-Codes
def ssc_nms(kps, des, cols, rows, num_ret_points=Parameters.kNumFeatures, tolerance=0.1):

    if len(kps)==0:
        return kps, des
        
    exp1 = rows + cols + 2 * num_ret_points
    exp2 = 4 * cols + 4 * num_ret_points + 4 * rows * num_ret_points + rows * rows + cols * cols - \
           2 * rows * cols + 4 * rows * cols * num_ret_points
    exp3 = math.sqrt(exp2)
    exp4 = num_ret_points - 1

    sol1 = -round(float(exp1 + exp3) / exp4)  # first solution
    sol2 = -round(float(exp1 - exp3) / exp4)  # second solution

    high = sol1 if (sol1 > sol2) else sol2  # binary search range initialization with positive solution
    low = math.floor(math.sqrt(len(kps) / num_ret_points))

    prev_width = -1 
    result_list = []
    result = []
    complete = False
    k = num_ret_points
    k_min = round(k - (k * tolerance))
    k_max = round(k + (k * tolerance))

    while not complete:
        width = low + (high - low) / 2
        if width == prev_width or low > high:  # needed to reassure the same radius is not repeated again
            result_list = result  # return the keypoints from the previous iteration
            break

        c = width / 2  # initializing Grid
        num_cell_cols = int(math.floor(cols / c))
        num_cell_rows = int(math.floor(rows / c))
        covered_vec = [[False for _ in range(num_cell_cols + 1)] for _ in range(num_cell_cols + 1)]
        result = []

        for i in range(len(kps)):
            row = int(math.floor(kps[i].pt[1] / c))  # get position of the cell current point is located at
            col = int(math.floor(kps[i].pt[0] / c))
            if not covered_vec[row][col]:  # if the cell is not covered
                result.append(i)
                # get range that current radius is covering
                row_min = int((row - math.floor(width / c)) if ((row - math.floor(width / c)) >= 0) else 0)
                row_max = int(
                    (row + math.floor(width / c)) if (
                            (row + math.floor(width / c)) <= num_cell_rows) else num_cell_rows)
                col_min = int((col - math.floor(width / c)) if ((col - math.floor(width / c)) >= 0) else 0)
                col_max = int(
                    (col + math.floor(width / c)) if (
                            (col + math.floor(width / c)) <= num_cell_cols) else num_cell_cols)
                for rowToCov in range(row_min, row_max + 1):
                    for colToCov in range(col_min, col_max + 1):
                        if not covered_vec[rowToCov][colToCov]:
                            # cover cells within the square bounding box with width w
                            covered_vec[rowToCov][colToCov] = True

        if k_min <= len(result) <= k_max:  # solution found
            result_list = result
            complete = True
        elif len(result) < k_min:
            high = width - 1  # update binary search range
        else:
            low = width + 1
        prev_width = width

    #for i in range(len(result_list)):
    #    kps_out.append(kps[result_list[i]])
    des_out = None
    if des is not None:
        des_out = des[result_list]
    kps_out = kps[result_list]                
    return kps_out, des_out


# Distribute keypoints by using a octree (as a matter of fact, a quadtree)
# Interface (pybind11) to fast C++ code from ORBSLAM2
def octree_nms(frame, kps, num_features):
    minX=0
    maxX=frame.shape[1]
    minY=0
    maxY=frame.shape[0]
    kps_tuples = [ (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps]
    kps_tuples = ORBextractor.DistributeOctTree(kps_tuples,minX,maxX,minY,maxY,num_features,0)
    kps = [ cv2.KeyPoint(*kp) for kp in kps_tuples]     
    return kps


# adapted from https://github.com/magicleap/SuperPointPretrainedNetwork/blob/master/demo_superpoint.py, similar to octree_nms 
def grid_nms(kps, des, H, W, num_features, dist_thresh=4):
    """
    Run a fast approximate Non-Max-Suppression on arrays of keypoints and descriptors 
  
    Algo summary: Create a grid sized HxW. Assign each corner location a 1, rest
    are zeros. Iterate through all the 1's and convert them either to -1 or 0.
    Suppress points by setting nearby values to 0.
  
    Grid Value Legend:
    -1 : Kept.
     0 : Empty or suppressed.
     1 : To be processed (converted to either kept or supressed).
  
    NOTE: The NMS first rounds points to integers, so NMS distance might not
    be exactly dist_thresh. It also assumes points are within image boundaries.
  
    Inputs
      kps - array of N cv2.KeyPoint.
      des - array of N descriptors (numpy array NxD, where D is the dimension of the descriptor)
      H - Image height.
      W - Image width.
      dist_thresh - Distance to suppress, measured as an infinty norm distance.
    Returns
      nmsed_corners - 3xN numpy matrix with surviving corners.
      nmsed_inds - N length numpy vector with surviving corner indices.
    """
    in_corners = np.array([(kp.pt[0],kp.pt[1],kp.response) for kp in kps]).T  #  3xN [x_i,y_i,conf_i]^T
    grid = np.zeros((H, W)).astype(int) # Track NMS data.
    inds = np.zeros((H, W)).astype(int) # Store indices of points.
    # Sort by confidence and round to nearest int.
    inds1 = np.argsort(-in_corners[2,:])
    corners = in_corners[:,inds1]
    rcorners = corners[:2,:].round().astype(int) # Rounded corners.
    # Check for edge case of 0 or 1 corners.
    if rcorners.shape[1] == 0:
      return np.zeros((3,0)).astype(int), np.zeros(0).astype(int)
    if rcorners.shape[1] == 1:
      out = np.vstack((rcorners, in_corners[2])).reshape(3,1)
      return out, np.zeros((1)).astype(int)
    # Initialize the grid.
    for i, rc in enumerate(rcorners.T):
      grid[rcorners[1,i], rcorners[0,i]] = 1
      inds[rcorners[1,i], rcorners[0,i]] = i
    # Pad the border of the grid, so that we can NMS points near the border.
    pad = dist_thresh
    grid = np.pad(grid, ((pad,pad), (pad,pad)), mode='constant')
    # Iterate through points, highest to lowest conf, suppress neighborhood.
    count = 0
    for i, rc in enumerate(rcorners.T):
      # Account for top and left padding.
      pt = (rc[0]+pad, rc[1]+pad)
      if grid[pt[1], pt[0]] == 1: # If not yet suppressed.
        grid[pt[1]-pad:pt[1]+pad+1, pt[0]-pad:pt[0]+pad+1] = 0
        grid[pt[1], pt[0]] = -1
        count += 1
    # Get all surviving -1's and return sorted array of remaining corners.
    keepy, keepx = np.where(grid==-1)
    keepy, keepx = keepy - pad, keepx - pad
    inds_keep = inds[keepy, keepx]
    out = corners[:, inds_keep]
    values = out[-1, :]
    inds2 = np.argsort(-values)
    #out = out[:, inds2]
    out_inds = inds1[inds_keep[inds2]]
    kps_out = np.array(kps)[out_inds][:num_features]
    if des is not None: 
        des_out = des[out_inds][:num_features]
    else: 
        des_out = None 
    return kps_out, des_out, out_inds


# apply Non-Maxima Suppression to an image that represents a score map 
def nms_from_map(score_map, size):
    kernel = np.ones((size,size),np.uint8)
    score_map = score_map * (score_map == cv2.dilate(score_map, kernel))  # cv2.dilate is a maximum filter 
    return score_map

# return the indexes of the best N points from an image that represents a score map 
def get_best_score_idxs(score_map, num_points=1000, threshold=-1):
    if threshold == -1:
        flatten = score_map.flatten()
        order_array = np.sort(flatten)[::-1]
        threshold = order_array[num_points-1]
        if threshold <= 0.0:
            indexes = np.argwhere(order_array > 0.0)
            if len(indexes) == 0:
                threshold = 0.0
            else:
                threshold = order_array[indexes[len(indexes)-1]]
    indexes = np.argwhere(score_map >= threshold)
    return indexes[:num_points]

# return the point coordinates of the best N points from an image that represents a score map 
def get_best_points_coordinates(score_map, num_points=1000, threshold=-1):
    indexes = get_best_score_idxs(score_map, num_points=num_points, threshold=threshold)
    coords = []
    for idx in indexes:
        score = score_map[idx[0], idx[1]]
        tmp = [idx[1], idx[0], score]
        coords.append(tmp)
    return np.asarray(coords)


# Compute homography reprojection error
def compute_hom_reprojection_error(H, kps1, kps2, mask=None):
    if mask is not None: 
        mask_idxs = (mask.ravel() == 1) 
        kps1 = kps1[mask_idxs]
        kps2 = kps2[mask_idxs]  
    kps1_reproj = H @ add_ones(kps1).T
    kps1_reproj = kps1_reproj[:2]/kps1_reproj[2]
    error_vecs = kps1_reproj.T - kps2
    return np.mean(np.sum(error_vecs*error_vecs,axis=1))
    
# compute ZNCC between the mached keypoints of img1 and img2
# - window_size is the size of the window used to compute the ZNCC
# - kps1 and kps2 are the matched keypoints and expected to be of size [N x 2] 
def compute_ZNCC_between_matched_keypoints(img1, img2, kps1, kps2, window_size=5):
    def extract_windows(img, kps, window_size):
        half_window = window_size // 2
        windows = []
        for kp in kps:
            x = int(kp[0])
            y = int(kp[1])
            if (x - half_window >= 0 and x + half_window < img.shape[1] and
                y - half_window >= 0 and y + half_window < img.shape[0]):
                window = img[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
                windows.append(window)
            else:
                window = np.zeros((window_size, window_size), dtype=img.dtype)
                windows.append(window)
        return np.array(windows)

    def normalize_windows(windows):
        means = np.mean(windows, axis=(1, 2), keepdims=True)
        stds = np.std(windows, axis=(1, 2), keepdims=True)
        stds = np.where(stds == 0, 1, stds)
        normalized_windows = (windows - means) / stds
        return normalized_windows

    def compute_cross_correlation(windows1, windows2):
        correlations = []
        for w1, w2 in zip(windows1, windows2):
            correlation = np.sum(w1 * w2) / (w1.size)
            correlations.append(correlation)
        return np.array(correlations)
        
    windows1 = extract_windows(img1, kps1, window_size=window_size)
    windows2 = extract_windows(img2, kps2, window_size=window_size)
    normalized_windows1 = normalize_windows(windows1)
    normalized_windows2 = normalize_windows(windows2)
    correlations = compute_cross_correlation(normalized_windows1, normalized_windows2)
    return correlations


# compute NSAD between the mached keypoints of img1 and img2
# - window_size is the size of the window used to compute the ZNCC
# - kps1 and kps2 are the matched keypoints and expected to be of size [N x 2] 
def compute_NSAD_between_matched_keypoints(img1, img2, kps1, kps2, window_size=5):
    def extract_windows(img, kps, window_size):
        half_window = window_size // 2
        windows = []
        for kp in kps:
            x = int(kp[0])
            y = int(kp[1])
            if (x - half_window >= 0 and x + half_window < img.shape[1] and
                y - half_window >= 0 and y + half_window < img.shape[0]):
                window = img[y - half_window:y + half_window + 1, x - half_window:x + half_window + 1]
            else:
                window = np.zeros((window_size, window_size), dtype=img.dtype)
            windows.append(window)
        #print(f'img shape: {img.shape}')
        #print(f'num windows: {len(windows)}')
        #print(f'num kps: {len(kps)}')
        return np.array(windows)

    def normalize_windows(windows):
        means = np.mean(windows, axis=(1, 2), keepdims=True)
        normalized_windows = (windows - means)
        return normalized_windows

    def compute_sum_of_abs_differences(windows1, windows2):
        sads = []
        for w1, w2 in zip(windows1, windows2):
            sad = np.sum(np.abs(w1 - w2))/(w1.size)
            sads.append(sad)
        return np.array(sads)
        
    windows1 = extract_windows(img1, kps1, window_size=window_size)
    windows2 = extract_windows(img2, kps2, window_size=window_size)
    normalized_windows1 = normalize_windows(windows1)
    normalized_windows2 = normalize_windows(windows2)
    sads = compute_sum_of_abs_differences(normalized_windows1, normalized_windows2)
    return sads



# Perform subpixel matching by correlation using the base resolution images.
# input: 
#   kps_left: keypoints of the left image
#   kps_right: keypoints of the right image
#   min_disparity: minimum disparity value
#   max_disparity: maximum disparity value
#   bf: baseline * focal_length of the stereo camera
#   image_left: left image
#   image_right: right image
# output: 
#   disparities: disparity values of the left image
#   us_right: u values of the right image
#   valid_idxs: valid indexes of the left image
def stereo_match_subpixel_correlation(kps_left, kps_right, 
                                      min_disparity, max_disparity, 
                                      bf, image_left, image_right):
    
    # we avoid converting images to float and instead use int16
    # as it is more efficient
    processing_type = np.int16
    
    num_keypoints = len(kps_left)
    us_right = np.full(num_keypoints, -1, dtype=np.float32)
    disparities = np.full(num_keypoints, -1, dtype=np.float32)
    dist_idx = []

    w = 5  # Window size
    l = 5  # Half window size for disparity search

    # Process all keypoints
    for i, kp_left in enumerate(kps_left):
        u_r0 = round(kps_right[i][0])  # x-coordinate of the right keypoint
        u_l = round(kp_left[0])
        v_l = round(kp_left[1])

        # Check if the patch exceeds image bounds
        if (v_l - w < 0 or v_l + w >= image_left.shape[0] or 
            u_l - w < 0 or u_l + w >= image_left.shape[1]):
            continue

        # Extract left patch and normalize
        left_patch = image_left[v_l - w:v_l + w + 1, u_l - w:u_l + w + 1].astype(processing_type)
        left_patch -= left_patch[w, w]

        best_dist = np.inf
        best_inc_r = 0
        dists = np.zeros(2 * l + 1, dtype=np.float32)

        for inc_r in range(-l, l + 1):
            start_u_r = u_r0 + inc_r - w
            end_u_r = u_r0 + inc_r + w + 1

            if start_u_r < 0 or end_u_r >= image_right.shape[1]:
                continue

            # Extract right patch and normalize
            right_patch = image_right[v_l - w:v_l + w + 1, start_u_r:end_u_r].astype(processing_type)
            right_patch -= right_patch[w, w]

            # Compute distance
            dist = np.sum(np.abs(left_patch - right_patch))
            dists[l + inc_r] = dist

            if dist < best_dist:
                best_dist = dist
                best_inc_r = inc_r

        if best_inc_r in [-l, l]:
            continue

        # Sub-pixel match with parabola fitting
        dist1, dist2, dist3 = dists[l + best_inc_r - 1], dists[l + best_inc_r], dists[l + best_inc_r + 1]
        delta_r = (dist1 - dist3) / (2.0 * (dist1 + dist3 - 2.0 * dist2))

        if not (-1 <= delta_r <= 1):
            continue

        # Re-scaled coordinate
        best_u_r = u_r0 + best_inc_r + delta_r
        disparity = kp_left[0] - best_u_r

        if min_disparity <= disparity < max_disparity:
            if disparity <= 0:
                disparity = 0.01
                best_u_r = kp_left[0] - 0.01

            disparities[i] = disparity
            us_right[i] = best_u_r
            dist_idx.append((best_dist, i))

    # Sort by distance and apply threshold
    dist_idx.sort()
    if dist_idx:
        median_dist = dist_idx[len(dist_idx) // 2][0]
        threshold_dist = 1.5 * 1.4826 * median_dist

        # Filter out invalid matches
        valid_idxs = [idx for dist, idx in dist_idx if dist < threshold_dist]
        invalid_idxs = [idx for dist, idx in dist_idx if dist >= threshold_dist]

        us_right[invalid_idxs] = -1
        disparities[invalid_idxs] = -1
    else:
        valid_idxs = []

    return disparities, us_right, np.array(valid_idxs, dtype=np.int32)


# extract/rectify patches around openCV keypoints, and returns patches tensor
# out: patches as a numpy array of size (len(kps), 1, patch_size, patch_size)
def extract_patches_tensor(img, kps, patch_size=32, mag_factor=1.0, warp_flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS):
    patches = np.ndarray((len(kps), 1, patch_size, patch_size), dtype=np.float32)
    half_patch_size=0.5*patch_size
    for i,kp in enumerate(kps):
        x,y = kp.pt
        s = kp.size
        a = kp.angle

        scale = mag_factor * s/patch_size
        a_rad = a * math.pi/180.0
        cos = math.cos(a_rad) if a_rad >=0 else 1.0 
        sin = math.sin(a_rad) if a_rad >=0 else 0.0 
        scale_cos = scale*cos
        scale_sin = scale*sin 

        M = np.matrix([
            [+scale_cos, -scale_sin, (-scale_cos + scale_sin) * half_patch_size + x],
            [+scale_sin, +scale_cos, (-scale_sin - scale_cos) * half_patch_size + y]])

        patch = cv2.warpAffine(img, M, (patch_size, patch_size), flags=warp_flags)
        patches[i,0,:,:] = cv2.resize(patch,(patch_size,patch_size))
    return patches  
            
            
# extract/rectify patches around openCV keypoints, and returns patches array   
# out: `patches` as an array of len(kps) element of size (patch_size, patch_size)        
# N.B.: you can obtain a numpy array of size (len(kps), patch_size, patch_size) by wrapping: 
#       patches = np.asarray(patches)
def extract_patches_array(img, kps, patch_size=32, mag_factor=1.0, warp_flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS):
    patches = []
    half_patch_size=0.5*patch_size
    for kp in kps:
        x,y = kp.pt
        s = kp.size
        a = kp.angle

        scale = mag_factor * s/patch_size
        a_rad = a * math.pi/180.0
        cos = math.cos(a_rad) if a_rad >=0 else 1.0 
        sin = math.sin(a_rad) if a_rad >=0 else 0.0 
        scale_cos = scale*cos
        scale_sin = scale*sin 

        M = np.matrix([
            [+scale_cos, -scale_sin, (-scale_cos + scale_sin) * half_patch_size + x],
            [+scale_sin, +scale_cos, (-scale_sin - scale_cos) * half_patch_size + y]])

        patch = cv2.warpAffine(img, M, (patch_size, patch_size), flags=warp_flags)
        patches.append(patch)
    return patches 


# extract/rectify patches around openCV keypoints, and returns patches array            
# out: `patches` as an array of len(kps) element of size (patch_size, patch_size)  
# N.B.: you can obtain a numpy array of size (len(kps), patch_size, patch_size) by wrapping: 
#       patches = np.asarray(patches) 
def extract_patches_array_cpp(img, kps, patch_size=32, mag_factor=1.0, warp_flags=cv2.WARP_INVERSE_MAP + cv2.INTER_CUBIC + cv2.WARP_FILL_OUTLIERS):
    if kPySlamUtilsAvailable:
        kps_tuples = [ (kp.pt[0], kp.pt[1], kp.size, kp.angle, kp.response, kp.octave) for kp in kps]
        return pyslam_utils.extract_patches(image=img, kps=kps_tuples, patch_size=patch_size, use_orientation=True, scale_factor=mag_factor, warp_flags=warp_flags)
    else:
        print('using python version extract_patches_array()')
        return extract_patches_array(img=img, kps=kps, patch_size=patch_size, mag_factor=mag_factor, warp_flags=warp_flags)

# Transform float descriptors [NxD] into a binary descriptors [Nx(D-1)].
# Use the approach of LBP, BRIEF, CENSUS: compare a descriptor float component to its adjacent values 
# to get a binary descriptor: [...,d[i-1]>d[i],d[i]<d[i+1],...]
# In the end, this is equivalent to [...,d[i]>d[i-1],d[i+1]>d[i],...] 
def transform_float_to_binary_descriptor(float_des):
    binary_des = (float_des[:, :-1] > float_des[:, 1:]).astype(np.uint8)
    return binary_des


# Represents an image grid with specified cell size equal to (width/num_div_x,height/num_div_y)
# This can be used to count the num of features that are present in each cell
class ImageGrid:
    def __init__(self, width, height, num_div_x=4, num_div_y=4):
        self.width = width
        self.height = height
        self.offset_x = 0  
        self.offset_y = 0
        self.cell_x = int(width / num_div_x)
        self.cell_y = int(height / num_div_y)
        assert (self.cell_x > 0 and self.cell_y > 0)
        self.num_cells_x = num_div_x
        self.num_cells_y = num_div_y
        self.offset_x = (width - num_div_x * self.cell_x)/2
        self.offset_y = (height - num_div_y * self.cell_y)/2
        #print(f'cell_x: {self.cell_x}, cell_y: {self.cell_y}, num_cells_x: {self.num_cells_x}, num_cells_y: {self.num_cells_y}')
        self.point_map = np.zeros((self.num_cells_y, self.num_cells_x), dtype=int)

    def add_point(self, pt):
        cell_x_index = int((pt[0]-self.offset_x) // self.cell_x)
        cell_y_index = int((pt[1]-self.offset_y) // self.cell_y)
        if 0 <= cell_x_index < self.num_cells_x and 0 <= cell_y_index < self.num_cells_y:
            self.point_map[cell_y_index, cell_x_index] += 1

    # points is an array of shape (N, 2)
    def add_points(self, points):
        cell_x_indices = ((points[:, 0]-self.offset_x) // self.cell_x).astype(int)
        cell_y_indices = ((points[:, 1]-self.offset_y) // self.cell_y).astype(int)
        cell_x_indices = np.clip(cell_x_indices, 0, self.num_cells_x-1)
        cell_y_indices = np.clip(cell_y_indices, 0, self.num_cells_y-1)
        np.add.at(self.point_map, (cell_y_indices, cell_x_indices), 1)

    # check if each cell has at least 'num_min_points' points
    def is_each_cell_covered(self, num_min_points=1):
        return np.all(self.point_map.ravel() >= num_min_points)
    
    def num_cells_uncovered(self, num_min_points=1):
        return np.sum(self.point_map.ravel() < num_min_points)
    
    def get_grid_img(self, num_min_points=1):
        width = self.num_cells_x * self.cell_x
        height = self.num_cells_y * self.cell_y
        img = np.zeros((height, width), dtype=np.uint8)
        for y in range(self.num_cells_y):
            for x in range(self.num_cells_x):
                if self.point_map[y,x] >= num_min_points:
                    img[y*self.cell_y:(y+1)*self.cell_y, x*self.cell_x:(x+1)*self.cell_x] = 255
        return img

