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
from geom_helpers import skew, add_ones, hamming_distance, draw_lines, draw_points
from helpers import Printer, getchar
import parameters  
from timer import Timer

from mplot_figure import MPlotFigure

# search by projection matches between {map points of f_ref} and {keypoints of f_cur}
def search_frame_by_projection(f_ref, f_cur):
    found_pts_count = 0
    idx_ref = []
    idx_cur = [] 
    for i, p in enumerate(f_ref.points):
        if p is None or p.is_bad is True:
            continue 
        # project point on f_cur
        proj = f_cur.project_map_point(p)
        # check if point is visible 
        is_visible  = (proj[0] > 0) & (proj[0] < f_cur.W) & \
                      (proj[1] > 0) & (proj[1] < f_cur.H)
        if is_visible is False:
            continue 
        for idx_c in f_cur.kd.query_ball_point(proj, parameters.kMaxReprojectionDistance):
            # if no point associated 
            if f_cur.points[idx_c] is None:                 
                orb_dist = p.orb_distance(f_cur.des[idx_c])
                #print('b_dist : ', orb_dist)
                if orb_dist < parameters.kMaxOrbDistanceSearchByReproj:
                    p.add_observation(f_cur, idx_c)
                    found_pts_count += 1
                    idx_ref.append(i)
                    idx_cur.append(idx_c)
                    break      
    return idx_ref, idx_cur, found_pts_count     


# search by projection matches between {input map points} and {unmatched keypoints of frame f_cur}
def search_by_projection(points, f_cur):
    found_pts_count = 0
    if len(points) > 0:

        # project the points on frame f_cur
        projs = f_cur.project_map_points(points)

        # check if points are visible 
        visible_pts = (projs[:, 0] > 0) & (projs[:, 0] < f_cur.W) & \
                        (projs[:, 1] > 0) & (projs[:, 1] < f_cur.H)

        for i, p in enumerate(points):
            if not visible_pts[i] or p.is_bad is True:
                # point not visible in frame or is bad 
                continue
            if f_cur in p.frames:
                # we already matched this map point to this frame
                continue
            for m_idx in f_cur.kd.query_ball_point(projs[i], parameters.kMaxReprojectionDistance):
                # if no point associated 
                if f_cur.points[m_idx] is None: 
                    orb_dist = p.orb_distance(f_cur.des[m_idx])
                    #print('b_dist : ', orb_dist)
                    if orb_dist < parameters.kMaxOrbDistanceSearchByReproj:
                        p.add_observation(f_cur, m_idx)
                        found_pts_count += 1
                        break      
    return found_pts_count     


# search by projection matches between {map points of last frames} and {unmatched keypoints of f_cur}
def search_local_frames_by_projection(map, f_cur, local_window = parameters.kLocalWindow):
    # take the points in the last N frame 
    points = []
    point_id_set = set()
    frames = map.frames[-local_window:]
    f_points = [p for f in frames for p in f.points if (p is not None)]
    for p in f_points: 
        if p.id not in point_id_set:
            points.append(p)
            point_id_set.add(p.id)
    print('searching %d map points' % len(points))
    return search_by_projection(points, f_cur)  


# search by projection matches between {all map points} and {unmatched keypoints of f_cur}
def search_map_by_projection(map, f_cur):
    return search_by_projection(map.points, f_cur)      

# compute the fundamental mat F12 and the infinite homography H21 [Hartley Zisserman pag 339]
def computeF12(f1, f2):
    R1w = f1.Rcw
    t1w = f1.tcw 
    R2w = f2.Rcw
    t2w = f2.tcw

    R12 = R1w @ R2w.T
    t12 = -R1w @ (R2w.T @ t2w) + t1w
    
    t12x = skew(t12)
    K1Tinv = f1.Kinv.T
    R21 = R12.T
    H21 = (f2.K @ R21) @ f1.Kinv  # infinite homography from f1 to f2 [Hartley Zisserman pag 339]
    F12 = ( (K1Tinv @ t12x) @ R12 ) @ f2.Kinv
    return F12, H21  

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
    delta /= length 
    step = min(parameters.kMinDistanceFromEpipole, length)  # keep a minimum distance from epipole if possible 
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

    # swap in order to force the search starting from the epipole 
    if swap is True:
        u0, v0, u1, v1 = u1, v1, u0, v0                

    return [ np.array([u0,v0]), np.array([u1,v1]) ]


# find a match on f along line for the input descriptor 
# e is the epipole 
def find_matches_along_line(f, e, line, descriptor, radius = 5, overlap_ratio = 0.2):
    #print('line: ', line[0], ", ", line[1])    
    step = radius*(1-overlap_ratio)
    delta = line[1].ravel() - line[0].ravel()
    length = np.linalg.norm(delta)
    n = int(math.ceil(length/step))
    delta = delta/n  
    #print('delta: ', delta)
    for i in range(n+1):
        x = line[0].ravel() + i*delta 
        for k_idx in f.kd.query_ball_point(x, radius):
            # if no point associated 
            if f.points[k_idx] is None: 
                orb_dist = hamming_distance(f.des[k_idx], descriptor)                
                if orb_dist < parameters.kMaxOrbDistanceSearchByReproj:  # TODO: refine! here, we just take the first match 
                    return k_idx    
    return -1                    
     

# search for triangulations between f1 and f2
# find matches between unmatched keypoints 
# we have pose estimates for both frames
def search_frame_for_triangulation(f1, f2, img2, img1 = None):   

    if __debug__:
        timer = Timer()
        timer.start()

    f1.update_camera_pose()
    f2.update_camera_pose() 

    O1w = f1.Ow
    O2w = f2.Ow 
    # compute epipoles
    e1 = f1.project_point(O2w).ravel()  # in first frame  
    e2 = f2.project_point(O1w).ravel()  # in second frame  

    #print('e1: ', e1)
    #print('e2: ', e2)    

    # if the translation is too small we cannot triangulate 
    if np.linalg.norm(O1w-O2w) < parameters.kMinTraslation:  # we assume the Inializer has been used for building the first map 
        Printer.red("search for triangulation: impossible with zero translation!")
        return  # EXIT

    # compute the fundamental matrix between the two frames 
    F12, H21 = computeF12(f1, f2)

    idxs1 = []
    for i, p in enumerate(f1.points): 
        if p is None:  # we consider just unmatched keypoints 
            kp = f1.kpsu[i]
            scale_factor = Frame.detector.scale_factors[f1.octaves[i]]
            if np.linalg.norm(kp-e1) < parameters.kMinDistanceFromEpipole * scale_factor:             
                continue    
            idxs1.append(i)      
    kpsu1 = f1.kpsu[idxs1]

    # compute epipolar lines in second image 
    lines2 = cv2.computeCorrespondEpilines(kpsu1.reshape(-1,1,2), 2, F12)  
    lines2 = lines2.reshape(-1,3)

    xs2_inf = np.dot(H21, add_ones(kpsu1).T).T # x2inf = H21 * x1  where x2inf is corresponding point according to infinite homography [Hartley Zisserman pag 339]
    xs2_inf = xs2_inf[:, 0:2] / xs2_inf[:, 2:]  
    line_edges = [ epiline_to_end_points(line, e2, x2_inf, f2.W) for line, x2_inf in zip(lines2, xs2_inf) ] 
    #print("line_edges: ", line_edges)

    if __debug__:
        print('search_frame_for_triangulation - timer1: ', timer.elapsed())
        assert(len(line_edges) == len(idxs1))    
        timer.start()

    idxs2_out = []
    idxs1_out = []
    lines_out = [] 
    for i, idx in enumerate(idxs1): 
        f2_idx = find_matches_along_line(f2, e2, line_edges[i], f1.des[idx])
        if f2_idx>-1:
            idxs2_out.append(f2_idx)
            idxs1_out.append(idxs1[i])
            assert(f2.points[f2_idx] is None)            
            assert(f1.points[idxs1[i]] is None)
            if __debug__:
                lines_out.append(line_edges[i])
 
    num_found_matches = len(idxs1_out)
    assert(len(idxs1_out) == len(idxs2_out)) 

    img2_epi = None 
    if __debug__:   
        #print("num found matches: ", num_found_matches)
        if True: 
            kpsu2 = f2.kpsu[idxs2_out]
            img2_epi = draw_lines(img2.copy(), lines_out, kpsu2)
            #cv2.imshow("epipolar lines",img2_epi)
            #cv2.waitKey(0)

    if __debug__:
        print('search_frame_for_triangulation - timer2: ', timer.elapsed())

    return idxs1_out, idxs2_out, num_found_matches, img2_epi



