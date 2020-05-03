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

import sys 
import math 
import numpy as np
import cv2 

from frame import Frame 
from utils_geom import skew, add_ones, normalize_vector, computeF12, check_dist_epipolar_line
from utils_draw import draw_lines, draw_points 
from utils import Printer, getchar
from parameters import Parameters  
from timer import Timer


# map an epipolar line to its end points, given an image with 'cols' columns, the epipole 'e' and the point xinf 
# "e" and "xinf" work as search limits [Hartley Zisserman pag 340]
# edge points are on the rectangular boundary of the frame 
def epiline_to_end_points(line, e, xinf, cols): 
    c = cols 
    l = line # [a,b,c] such that ax + by + c = 0, abs(b) > 0 otherwise we have a pependicular epipolar line
    u0,v0 = map(int, [0, -l[2]/l[1] ])
    u1,v1 = map(int, [c, -(l[2]+l[0]*c)/l[1] ])

    delta = xinf - e
    length = np.linalg.norm(delta.ravel())
    delta /= length # normalize delta 
    step = min(Parameters.kMinDistanceFromEpipole, length)  # keep a minimum distance from epipole if possible 
    e = e + delta * step 

    xmin = xinf.ravel() 
    xmax = e.ravel()
    swap = False 

    if __debug__:
        if False: 
            check_xmin = l[0]*xmin[0] + l[1]*xmin[1] + l[2]
            check_xmax = l[0]*xmax[0] + l[1]*xmax[1] + l[2]
            print("check_xmin: ", check_xmin)
            print("check_xmax: ", check_xmax)

    # swap the two variables so that xmin lies on the left of xmax
    if xmin[0] > xmax[0]:
        xmin, xmax = xmax, xmin 
        swap = True   

    if not math.isnan(xmin[0]) and not math.isnan(xmin[1]) and xmin[0] > u0 and xmin[0] < cols: # xmin is on the right of (u0,v0) => replace (u0,v0)
        u0 = xmin[0]
        v0 = xmin[1]

    if not math.isnan(xmax[0]) and not math.isnan(xmax[1]) and xmax[0] < u1 and xmax[0] > 0: # xmax is on the left of (u1,v1) => replace (u1,v1)
        u1 = xmax[0]
        v1 = xmax[1]        

    # swap in order to force the search starting from xinf 
    if swap is True:
        u0, v0, u1, v1 = u1, v1, u0, v0                

    return [ np.array([u0,v0]), np.array([u1,v1]) ]


# find a keypoint match on 'f' along line 'line' for the input 'descriptor'
# 'e' is the epipole 
# the found keypoint match must not already have a point match
# overlap_ratio must be in [0,1]
def find_matches_along_line(f, e, line, descriptor, radius = Parameters.kMaxReprojectionDistanceFrame, overlap_ratio = 0.5):
    
    max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistanceSearchEpipolar
    
    #print('line: ', line[0], ", ", line[1])    
    step = radius*(1-overlap_ratio)
    delta = line[1].ravel() - line[0].ravel()
    length = np.linalg.norm(delta)
    n = max( int(math.ceil(length/step)), 1) 
    delta = delta/n  
    #print('delta: ', delta)

    best_dist = math.inf 
    best_dist2 = math.inf
    best_k_idx = -1

    for i in range(n+1):
        x = line[0].ravel() + i*delta 
        for k_idx in f.kd.query_ball_point(x, radius):
            # if no point associated 
            if f.get_point_match(k_idx) is None: 
                descriptor_dist = Frame.descriptor_distance(f.des[k_idx], descriptor)                
                if descriptor_dist < max_descriptor_distance:  
                    if True: 
                        return k_idx, descriptor_dist  # stop at the first match 
                    if descriptor_dist < best_dist:
                        best_dist2 = best_dist
                        best_dist = descriptor_dist
                        best_k_idx = k_idx     
                    else: 
                        if descriptor_dist < best_dist2:  
                            best_dist2 = descriptor_dist   
                    if False: 
                        if best_dist2 < max_descriptor_distance:  # stop once we have a second to best match                        
                            break 
    # N.B.: the search "segment line" can have a large width => it is better to use the match distance ratio test                                           
    if best_dist < best_dist2 * Parameters.kMatchRatioTestEpipolarLine: 
       return best_k_idx, best_dist
    else:   
       return -1, 0   
    #return best_k_idx                      
            
     

# search keypoint matches (for triangulations) between f1 and f2
# search for matches between unmatched keypoints (without a corresponding map point)
# in input we have already some pose estimates for f1 and f2
# N.B.: experimental version, just for testing 
def search_frame_for_triangulation_test(f1, f2, img2, img1 = None):   

    idxs2_out = []
    idxs1_out = []
    lines_out = [] 
    num_found_matches = 0
    img2_epi = None     

    if __debug__:
        timer = Timer()
        timer.start()

    O1w = f1.Ow
    O2w = f2.Ow 
    # compute epipoles
    e1,_ = f1.project_point(O2w)  # in first frame  
    e2,_ = f2.project_point(O1w)  # in second frame  
    #print('e1: ', e1)
    #print('e2: ', e2)    
    
    baseline = np.linalg.norm(O1w-O2w) 
        
    # if the translation is too small we cannot triangulate 
    # if False: 
    #     if baseline < Parameters.kMinTraslation:  # we assume the Inializer has been used for building the first map 
    #         Printer.red("search for triangulation: impossible with almost zero translation!")
    #         return idxs1_out, idxs2_out, num_found_matches, img2_epi # EXIT
    # else:
    medianDepthF2 = f2.compute_points_median_depth()
    ratioBaselineDepth = baseline/medianDepthF2
    if ratioBaselineDepth < Parameters.kMinRatioBaselineDepth:  
        Printer.red("search for triangulation: impossible with too low ratioBaselineDepth!")
        return idxs1_out, idxs2_out, num_found_matches, img2_epi # EXIT           

    # compute the fundamental matrix between the two frames by using their estimated poses 
    F12, H21 = computeF12(f1, f2)

    idxs1 = []
    for i, p in enumerate(f1.get_points()): 
        if p is None:  # we consider just unmatched keypoints 
            kp = f1.kpsu[i]
            scale_factor = Frame.feature_manager.scale_factors[f1.octaves[i]]
            # discard points which are too close to the epipole 
            if np.linalg.norm(kp-e1) < Parameters.kMinDistanceFromEpipole * scale_factor:             
                continue    
            idxs1.append(i)      
    kpsu1 = f1.kpsu[idxs1]
    
    if __debug__:
        print('search_frame_for_triangulation - timer1: ', timer.elapsed())  
        timer.start()        

    # compute epipolar lines in second image 
    lines2 = cv2.computeCorrespondEpilines(kpsu1.reshape(-1,1,2), 2, F12)  
    lines2 = lines2.reshape(-1,3)
    
    xs2_inf = np.dot(H21, add_ones(kpsu1).T).T # x2inf = H21 * x1  where x2inf is corresponding point according to infinite homography [Hartley Zisserman pag 339]
    xs2_inf = xs2_inf[:, 0:2] / xs2_inf[:, 2:]  
    line_edges = [ epiline_to_end_points(line, e2, x2_inf, f2.width) for line, x2_inf in zip(lines2, xs2_inf) ] 
    #print("line_edges: ", line_edges)

    if __debug__:
        print('search_frame_for_triangulation - timer3: ', timer.elapsed())
        assert(len(line_edges) == len(idxs1))    
        timer.start()

    len_des2 = len(f2.des)
    flag_match = np.full(len_des2, False, dtype=bool)
    dist_match = np.zeros(len_des2)
    index_match = np.full(len_des2, 0, dtype=int)
    for i, idx in enumerate(idxs1):   # N.B.: a point in f1 can be matched to more than one point in f2, we avoid this by caching matches with f2 points
        f2_idx, dist = find_matches_along_line(f2, e2, line_edges[i], f1.des[idx])
        if f2_idx > -1:
            if not flag_match[f2_idx]: # new match
                flag_match[f2_idx] = True 
                dist_match[f2_idx] = dist 
                idxs2_out.append(f2_idx)
                idxs1_out.append(idx)
                index_match[f2_idx] = len(idxs2_out)-1                
                assert(f2.get_point_match(f2_idx) is None)            
                assert(f1.get_point_match(idx) is None)
                if __debug__:
                    lines_out.append(line_edges[i])
            else: # already matched
                if dist < dist_match[f2_idx]:  # update in case of a smaller distance 
                    dist_match[f2_idx] = dist 
                    index = index_match[f2_idx]  
                    idxs1_out[index] = idx    
                    if __debug__:
                        lines_out[index] = line_edges[i]                              
 
    num_found_matches = len(idxs1_out)
    assert(len(idxs1_out) == len(idxs2_out)) 

    if __debug__:   
        #print("num found matches: ", num_found_matches)
        if True: 
            kpsu2 = f2.kpsu[idxs2_out]
            img2_epi = draw_lines(img2.copy(), lines_out, kpsu2)
            #cv2.imshow("epipolar lines",img2_epi)
            #cv2.waitKey(0)

    if __debug__:
        print('search_frame_for_triangulation - timer4: ', timer.elapsed())

    return idxs1_out, idxs2_out, num_found_matches, img2_epi

