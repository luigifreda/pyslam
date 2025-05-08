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

from frame import Frame, FeatureTrackerShared, are_map_points_visible, are_map_points_visible_in_frame
from keyframe import KeyFrame
from map_point import predict_detection_levels

from utils_geom import Sim3Pose
from utils_geom_2views import computeF12, check_dist_epipolar_line
from utils_sys import Printer
from config_parameters import Parameters  
from timer import Timer
from rotation_histogram import RotationHistogram


kMinDistanceFromEpipole = Parameters.kMinDistanceFromEpipole
kMinDistanceFromEpipole2 = kMinDistanceFromEpipole*kMinDistanceFromEpipole
kCheckFeaturesOrientation = Parameters.kCheckFeaturesOrientation 


# propagate map point matches from f_ref to f_cur (access frames from tracking thread, no need to lock)
def propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur,
                                max_descriptor_distance=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance
        
    idx_ref_out = []
    idx_cur_out = []
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features
        
    # populate f_cur with map points by propagating map point matches of f_ref; 
    # to this aim, we use map points observed in f_ref and keypoint matches between f_ref and f_cur  
    num_matched_map_pts = 0
    for i, idx in enumerate(idxs_ref): # iterate over keypoint matches 
        p_ref = f_ref.points[idx]
        if p_ref is None: # we don't have a map point P for i-th matched keypoint in f_ref
            continue 
        if f_ref.outliers[idx] or p_ref.is_bad: # do not consider pose optimization outliers or bad points 
            continue  
        idx_cur = idxs_cur[i]
        p_cur = f_cur.points[idx_cur]
        if p_cur is not None: # and p_cur.num_observations > 0: # if we already matched p_cur => no need to propagate anything  
            continue
        des_distance = p_ref.min_des_distance(f_cur.des[idx_cur])
        if des_distance > max_descriptor_distance: 
            continue 
        if p_ref.add_frame_view(f_cur, idx_cur): # => P is matched to the i-th matched keypoint in f_cur
            num_matched_map_pts += 1
            idx_ref_out.append(idx)
            idx_cur_out.append(idx_cur)
            
            if check_orientation:
                index_match = len(idx_cur_out)-1
                rot = f_ref.angles[idx]-f_cur.angles[idx_cur]
                rot_histo.push(rot, index_match)
            
    if check_orientation:            
        valid_match_idxs = rot_histo.get_valid_idxs()     
        print('checking orientation consistency - valid matches % :', len(valid_match_idxs)/max(1,len(idxs_cur))*100,'% of ', len(idxs_cur),'matches')
        #print('rotation histogram: ', rot_histo)
        idx_ref_out = np.array(idx_ref_out)[valid_match_idxs]
        idx_cur_out = np.array(idx_cur_out)[valid_match_idxs]
        num_matched_map_pts = len(valid_match_idxs)            
                            
    return num_matched_map_pts, idx_ref_out, idx_cur_out  

  
# search by projection matches between {map points of f_ref} and {keypoints of f_cur},  (access frames from tracking thread, no need to lock)
def search_frame_by_projection(f_ref: Frame, 
                               f_cur: Frame,
                               max_reproj_distance=Parameters.kMaxReprojectionDistanceFrame,
                               max_descriptor_distance=None,
                               ratio_test=Parameters.kMatchRatioTestMap,
                               is_monocular=True,
                               already_matched_ref_idxs=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance    

    found_pts_count = 0
    idxs_ref = []
    idxs_cur = [] 
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features    
    
    check_forward_backward = not is_monocular
    
    trc = None
    forward = False
    backward = False
    if check_forward_backward:
        Tcw = f_cur.pose
        Rcw = Tcw[:3,:3]
        tcw = Tcw[:3,3]
        twc = -Rcw.T.dot(tcw)
        
        Trw = f_ref.pose
        Rrw = Trw[:3,:3]
        trw = Trw[:3,3]
        trc = Rrw.T.dot(twc)+trw
        forward = trc[2] > f_cur.camera.b
        backward = trc[2] < -f_cur.camera.b
          
    # get all matched points of f_ref which are non-outlier 
    if isinstance(f_ref.points, np.ndarray):
        matched_ref_idxs = np.flatnonzero( (f_ref.points!=None) & (f_ref.outliers==False)) 
    else:
        matched_ref_idxs = [i for i, p in enumerate(f_ref.points) if p is not None and not p.is_outlier]
    
    # if we have some already matched points in reference frame, remove them from the list
    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)
    
    matched_ref_points = f_ref.points[matched_ref_idxs]

    # project f_ref points on frame f_cur
    projs, depths = f_cur.project_map_points(matched_ref_points, f_cur.camera.is_stereo())
    # check if points lie on the image frame 
    is_visible = f_cur.are_in_image(projs, depths)

    # # check if points are visible 
    # is_visible, projs, depths, dists = f_cur.are_visible(matched_ref_points)
        
    kp_ref_octaves = f_ref.octaves[matched_ref_idxs]       
    kp_ref_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[kp_ref_octaves]              
    radiuses = max_reproj_distance * kp_ref_scale_factors     
    kd_cur_idxs = f_cur.kd.query_ball_point(projs[:,:2], radiuses)   
    
    do_check_stereo_reproj_err = f_cur.kps_ur is not None 
                     
    for ref_idx,p,j in zip(matched_ref_idxs, matched_ref_points, range(len(matched_ref_points))):
    
        if not is_visible[j]:
            continue 
        
        kp_ref_octave = f_ref.octaves[ref_idx]  
        
        best_dist = math.inf 
        #best_dist2 = math.inf
        #best_level = -1 
        #best_level2 = -1                   
        best_k_idx = -1   
        best_ref_idx = -1          
                  
        kd_cur_idxs_j = kd_cur_idxs[j]                
        if do_check_stereo_reproj_err:
            check_stereo = f_cur.kps_ur[kd_cur_idxs_j]>0 
            kp_cur_octaves = f_cur.octaves[kd_cur_idxs_j]       
            kp_cur_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[kp_cur_octaves]  
            errs_ur = np.fabs(projs[j,2] - f_cur.kps_ur[kd_cur_idxs_j]) 
            ok_errs_ur = np.where(check_stereo, errs_ur < max_reproj_distance * kp_cur_scale_factors,True)  
                           
        for h, kd_idx in enumerate(kd_cur_idxs[j]):                   
            
            p_f_cur = f_cur.points[kd_idx]
            if  p_f_cur is not None:
                if p_f_cur.num_observations > 0: # we already matched p_f_cur => discard it 
                    continue          
    
            p_f_cur_octave = f_cur.octaves[kd_idx]
            
            # check if point is in the same octave as the reference point
            if check_forward_backward:
                if backward and p_f_cur_octave > kp_ref_octave:
                    continue
                elif forward and p_f_cur_octave < kp_ref_octave:
                    continue
                elif p_f_cur_octave < (kp_ref_octave-1) or p_f_cur_octave > (kp_ref_octave+1):
                    continue
            else:
                if p_f_cur_octave < (kp_ref_octave-1) or p_f_cur_octave > (kp_ref_octave+1):
                    continue
                                       
            if do_check_stereo_reproj_err:
                if not ok_errs_ur[h]:
                    continue
                    
            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
            if descriptor_dist < best_dist:                
                best_dist = descriptor_dist
                best_k_idx = kd_idx
                best_ref_idx = ref_idx     
                
            # if descriptor_dist < best_dist:                                      
            #     best_dist2 = best_dist
            #     best_level2 = best_level
            #     best_dist = descriptor_dist
            #     best_level = f_cur.octaves[kd_idx]
            #     best_k_idx = kd_idx  
            #     best_ref_idx = i                      
            # else: 
            #     if descriptor_dist < best_dist2:  
            #         best_dist2 = descriptor_dist
            #         best_level2 = f_cur.octaves[kd_idx]                       
                                
        #if best_k_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance: 
            # apply match distance ratio test only if the best and second are in the same scale level 
            #if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test): 
            #    continue                        
            #print('b_dist : ', best_dist)            
            if p.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1
                idxs_ref.append(best_ref_idx)
                idxs_cur.append(best_k_idx)   
                
                if check_orientation:                  
                    index_match = len(idxs_cur)-1
                    rot = f_ref.angles[best_ref_idx]-f_cur.angles[best_k_idx]
                    rot_histo.push(rot,index_match)
                
            #print('best des distance: ', best_dist, ", max dist: ", max_descriptor_distance)                                        
            #des_dists.append(best_dist)
            
    if check_orientation:            
        valid_match_idxs = rot_histo.get_valid_idxs()     
        print('checking orientation consistency - valid matches % :', len(valid_match_idxs)/max(1,len(idxs_cur))*100,'% of ', len(idxs_cur),'matches')
        #print('rotation histogram: ', rot_histo)
        idxs_ref = np.array(idxs_ref)[valid_match_idxs]
        idxs_cur = np.array(idxs_cur)[valid_match_idxs]
        found_pts_count = len(valid_match_idxs)
    
    return np.array(idxs_ref), np.array(idxs_cur), found_pts_count
    #return idxs_ref, idxs_cur, found_pts_count 


# Search by projection between {keyframe map points} and {current frame keypoints}
def search_keyframe_by_projection(kf_ref: KeyFrame,
                                  f_cur: Frame,
                                  max_reproj_distance=None,
                                  max_descriptor_distance=None,
                                  ratio_test=Parameters.kMatchRatioTestMap,
                                  already_matched_ref_idxs=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance
        
    assert kf_ref.is_keyframe, '[search_keyframe_by_projection] kf_ref must be a KeyFrame'

    found_pts_count = 0
    idxs_ref = []
    idxs_cur = []

    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features

    Tcw = f_cur.pose
    Rcw = Tcw[:3, :3]
    tcw = Tcw[:3, 3]
    Ow = -Rcw.T @ tcw  # camera center in world coords

    ref_mps = kf_ref.get_matched_points()

    if len(ref_mps) == 0:
        return np.array([]), np.array([]), 0

    # Get valid map points (non-bad, non-outliers)
    matched_ref_idxs = [i for i, p in enumerate(ref_mps) if p is not None and not p.is_bad]

    # Remove already matched points if given
    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)

    matched_ref_points = [ref_mps[i] for i in matched_ref_idxs]
    if len(matched_ref_points) == 0:
        return np.array([]), np.array([]), 0

    points_w = np.array([p for p in matched_ref_points])

    # Project points
    # projs, depths = f_cur.project_map_points(points_w, f_cur.camera.is_stereo())
    # is_visible = f_cur.are_visible(projs, depths)
    is_visible, projs, depths, dists = f_cur.are_visible(matched_ref_points)

    # Predict detection levels
    #dists = np.linalg.norm(points_w - Ow, axis=1)
    #predicted_levels = np.array([p.predict_scale(dist, f_cur) for p, dist in zip(matched_ref_points, dists)])
    predicted_levels = predict_detection_levels(matched_ref_points, dists)     
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors
    kd_cur_idxs = f_cur.kd.query_ball_point(projs[:, :2], radiuses)

    for j, (ref_idx, mp) in enumerate(zip(matched_ref_idxs, matched_ref_points)):
        if not is_visible[j]:
            continue

        predicted_level = predicted_levels[j]
        kd_indices = kd_cur_idxs[j]

        best_dist = math.inf
        best_dist2 = math.inf
        best_level = -1
        best_level2 = -1
        best_k_idx = -1

        for idx2 in kd_indices:
            if f_cur.points[idx2] is not None:
                continue  # already matched

            kp_level = f_cur.octaves[idx2]
            if (kp_level < predicted_level - 1) or (kp_level > predicted_level + 1):
                continue

            descriptor_dist = mp.min_des_distance(f_cur.des[idx2])

            if descriptor_dist < best_dist:
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = idx2
            elif descriptor_dist < best_dist2:
                best_dist2 = descriptor_dist
                best_level2 = kp_level

        if best_dist < max_descriptor_distance:
            if (best_level == best_level2) and (best_dist > best_dist2 * ratio_test):
                continue

            if mp.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1
                idxs_ref.append(ref_idx)
                idxs_cur.append(best_k_idx)

                if check_orientation:
                    rot = kf_ref.angles[ref_idx] - f_cur.angles[best_k_idx]
                    rot_histo.push(rot, len(idxs_cur) - 1)

    if check_orientation:
        valid_match_idxs = rot_histo.get_valid_idxs()
        print('checking orientation consistency - valid matches %:', len(valid_match_idxs) / max(1, len(idxs_cur)) * 100, '% of', len(idxs_cur), 'matches')

        idxs_ref = np.array(idxs_ref)[valid_match_idxs]
        idxs_cur = np.array(idxs_cur)[valid_match_idxs]
        found_pts_count = len(valid_match_idxs)

    return np.array(idxs_ref), np.array(idxs_cur), found_pts_count


# search by projection matches between {input map points} and {unmatched keypoints of frame f_cur}, (access frame from tracking thread, no need to lock)
def search_map_by_projection(points, 
                             f_cur: Frame, 
                             max_reproj_distance=Parameters.kMaxReprojectionDistanceMap, 
                             max_descriptor_distance=None,
                             ratio_test=Parameters.kMatchRatioTestMap,
                             far_points_threshold=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance  
           
    found_pts_count = 0
    found_pts_fidxs = []   # idx of matched points in current frame 
    
    #reproj_dists = []
    
    if len(points) == 0:
        return 0,0,0
            
    # check if points are visible 
    visible_pts, projs, depths, dists = f_cur.are_visible(points)
    
    predicted_levels = predict_detection_levels(points, dists) 
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    kd_cur_idxs = f_cur.kd.query_ball_point(projs, radiuses)
    
    # trick to filter out far points if required => we mark them as not visible
    if far_points_threshold is not None: 
        #print(f'search_map_by_projection: using far points threshold: {far_points_threshold}')
        visible_pts = np.logical_and(visible_pts, depths < far_points_threshold)

    for i, p in enumerate(points):
        if not visible_pts[i] or p.is_bad:     # point not visible in frame or is bad 
            continue        
        if p.last_frame_id_seen == f_cur.id:   # we already matched this map point to current frame or it was outlier 
            continue
        
        p.increase_visible()
          
        predicted_level = predicted_levels[i]         
                       
        best_dist = math.inf 
        best_dist2 = math.inf
        best_level = -1 
        best_level2 = -1               
        best_k_idx = -1  

        # find closest keypoints of f_cur         
        for kd_idx in kd_cur_idxs[i]:
     
            p_f = f_cur.points[kd_idx]
            # check there is not already a match               
            if  p_f is not None:
                if p_f.num_observations > 0:
                    continue 
                
            # check detection level     
            kp_level = f_cur.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):
                continue                
                
            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
  
            if descriptor_dist < best_dist:                                      
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = kd_idx    
            else: 
                if descriptor_dist < best_dist2:  
                    best_dist2 = descriptor_dist
                    best_level2 = kp_level                                        
                                                       
        #if best_k_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance:            
            # apply match distance ratio test only if the best and second are in the same scale level 
            if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test): 
                continue 
            #print('best des distance: ', best_dist, ", max dist: ", Parameters.kMaxDescriptorDistance)                    
            if p.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1  
                found_pts_fidxs.append(best_k_idx)  
            
            #reproj_dists.append(np.linalg.norm(projs[i] - f_cur.kpsu[best_k_idx]))   
            
    # if len(reproj_dists) > 1:        
    #     reproj_dist_sigma = 1.4826 * np.median(reproj_dists)
                       
    return found_pts_count, found_pts_fidxs   


# search by projection matches between {map points of last frames} and {unmatched keypoints of f_cur}, (access frame from tracking thread, no need to lock)
def search_local_frames_by_projection(map, f_cur, local_window=Parameters.kLocalBAWindow, max_descriptor_distance=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance
        
    # take the points in the last N frame 
    frames = map.keyframes[-local_window:]
    f_points = set([p for f in frames for p in f.get_points() if (p is not None)])
    print('searching %d map points' % len(f_points))
    return search_map_by_projection(list(f_points), f_cur, max_descriptor_distance=max_descriptor_distance)  


# search by projection matches between {all map points} and {unmatched keypoints of f_cur}
def search_all_map_by_projection(map, f_cur, max_descriptor_distance=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance  
           
    return search_map_by_projection(map.get_points(), f_cur, max_descriptor_distance=max_descriptor_distance)      



# search by projection more matches between {input map points} and {unmatched keypoints of frame f_cur}
# in: 
#   points: input map points
#   f_cur: current frame
#   f_cur_matched_points: matched points in current frame  (f_cur_matched_points[i] is the i-th map point matched on f_cur or None)   
#   Scw: suggested se3 or sim3 transformation
# The suggested transformation Scw (in se3 or sim3) is used in the search (instead of using the current frame pose)
def search_more_map_points_by_projection(points: set, 
                                    f_cur: Frame,
                                    f_cur_matched_points: list,  # f_cur_matched_points[i] is the i-th map point matched in f_cur or None
                                    Scw,
                                    max_reproj_distance=Parameters.kMaxReprojectionDistanceMap, 
                                    max_descriptor_distance=None,
                                    print_fun=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance # more conservative check
        
    found_pts_count = 0
    if len(points) == 0:
        return found_pts_count, f_cur_matched_points
            
    assert(len(f_cur.points) == len(f_cur_matched_points))
    
    # extract from sim3 Scw=[s*Rcw, tcw; 0, 1] the corresponding se3 transformation Tcw=[Rcw, tcw/s]
    if isinstance(Scw, np.ndarray):
        sRcw = Scw[:3,:3]
        scw = math.sqrt(np.dot(sRcw[0,:3], sRcw[0,:3]))
        Rcw = sRcw / scw
        tcw = Scw[:3,3]/scw
    elif isinstance(Scw, Sim3Pose):
        scw = Scw.s
        Rcw = Scw.R
        tcw = Scw.t/ scw
    else: 
        raise TypeError("Unsupported type '{}' for Scw".format(type(Scw)))
    
    if not isinstance(points, set):
        points = set(points)
    target_points = points.difference([p for p in f_cur_matched_points if p is not None])    
    target_points = list(target_points)
    
    if len(target_points) == 0:
        if print_fun is not None:
            print_fun('search_more_map_points_by_projection: no target points available after difference')
        return found_pts_count, f_cur_matched_points
    
    # check if points are visible     
    visible_pts, projs, depths, dists = are_map_points_visible_in_frame(target_points, f_cur, Rcw, tcw)
    
    if print_fun is not None:
        print_fun(f'search_more_map_points_by_projection: #visible points: {len(visible_pts)}')
    
    predicted_levels = predict_detection_levels(target_points, dists) 
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    kd_cur_idxs = f_cur.kd.query_ball_point(projs, radiuses)
    
    # num_failures_vis_or_bad = 0
    # num_failures_pf_is_not_none = 0
    # num_failures_kp_level = 0
    # num_failures_max_des_distance = 0
                               
    for i, p in enumerate(target_points):
        if not visible_pts[i] or p.is_bad:     # point not visible in frame or is bad 
            # num_failures_vis_or_bad +=1
            continue        
  
        predicted_level = predicted_levels[i]         
                       
        best_dist = math.inf 
        best_k_idx = -1  
 
        # find closest keypoints of f_cur        
        for kd_idx in kd_cur_idxs[i]:
     
            p_f = f_cur_matched_points[kd_idx]
            # check there is not already a match in f_cur_matched_points              
            if  p_f is not None:
                # num_failures_pf_is_not_none +=1
                continue 
                
            # check detection level     
            kp_level = f_cur.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):
                # if print_fun is not None:                
                #     print_fun(f'search_more_map_points_by_projection: bad kp level: {kp_level},  predicted_level: {predicted_level}')
                # num_failures_kp_level += 1
                continue                
                
            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
  
            if descriptor_dist < best_dist:                                      
                best_dist = descriptor_dist
                best_k_idx = kd_idx                               
                                                       
        if best_dist < max_descriptor_distance:
            f_cur_matched_points[best_k_idx] = p
            found_pts_count += 1
        # else:
        #     if print_fun is not None:
        #         print_fun(f'search_more_map_points_by_projection: bad best_des_distance: {best_dist}, max_descriptor_distance: {max_descriptor_distance}')
        #     num_failures_max_des_distance += 1
            
    # if print_fun is not None:
    #     print_fun(f'search_more_map_points_by_projection: num_failures_vis_or_bad: {num_failures_vis_or_bad}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_pf_is_not_none: {num_failures_pf_is_not_none}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_kp_level: {num_failures_kp_level}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_max_des_distance: {num_failures_max_des_distance}')
    
    return found_pts_count, f_cur_matched_points
            

# search keypoint matches (for triangulations) between f1 and f2
# search for matches between unmatched keypoints (without a corresponding map point)
# in input we have already some pose estimates for f1 and f2
def search_frame_for_triangulation(kf1, kf2, idxs1=None, idxs2=None, 
                                   max_descriptor_distance=None,
                                   is_monocular=True):
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance # more conservative check
           
    idxs2_out = []
    idxs1_out = []
    num_found_matches = 0
    img2_epi = None     

    if __debug__:
        timer = Timer()
        timer.start()

    O1w = kf1.Ow
    O2w = kf2.Ow
    # compute epipoles
    e1,_ = kf1.project_point(O2w)  # in first frame 
    e2,_ = kf2.project_point(O1w)  # in second frame  
    #print('e1: ', e1)
    #print('e2: ', e2)    
    
    baseline = np.linalg.norm(O1w-O2w) 
    #print(f'search_frame_for_triangulation: baseline: {baseline}, camera.b: {kf2.camera.b}')

    # if the translation is too small we cannot triangulate 
    if not is_monocular:  # we assume the Inializer has been used for building the first map 
        if baseline < kf2.camera.b:
            return idxs1_out, idxs2_out, num_found_matches, img2_epi # EXIT
    else:    
        medianDepth = kf2.compute_points_median_depth()
        if medianDepth == -1:
            Printer.orange("search for triangulation: f2 with no points")        
            medianDepth = kf1.compute_points_median_depth()        
        ratioBaselineDepth = baseline/medianDepth
        if ratioBaselineDepth < Parameters.kMinRatioBaselineDepth:  
            Printer.orange("search for triangulation: impossible with too low ratioBaselineDepth!")
            return idxs1_out, idxs2_out, num_found_matches, img2_epi # EXIT        

    # compute the fundamental matrix between the two frames by using their estimated poses 
    F12, H21 = computeF12(kf1, kf2)

    if idxs1 is None or idxs2 is None:
        timerMatch = Timer()
        timerMatch.start()
        matching_result = FeatureTrackerShared.feature_matcher.match(kf1.img, kf2.img, kf1.des, kf2.des)  
        idxs1, idxs2 = matching_result.idxs1, matching_result.idxs2      
        if __debug__:        
            print('search_frame_for_triangulation - matching - timer: ', timerMatch.elapsed())        
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features     
        
        
    level_sigmas2 = FeatureTrackerShared.feature_manager.level_sigmas2
    scale_factors = FeatureTrackerShared.feature_manager.scale_factors
    
    # check epipolar constraints 
    for i1,i2 in zip(idxs1,idxs2):
        if kf1.get_point_match(i1) is not None or kf2.get_point_match(i2) is not None: # we are searching for keypoint matches where both keypoints do not have a corresponding map point 
            #print('existing point on match')
            continue 
        
        descriptor_dist = FeatureTrackerShared.descriptor_distance(kf1.des[i1], kf2.des[i2])
        if descriptor_dist > max_descriptor_distance:
            continue     
        
        kp1 = kf1.kpsu[i1]
        #kp1_scale_factor = scale_factors[kf1.octaves[i1]]
        #kp1_size = f1.sizes[i1]
        # discard points which are too close to the epipole            
        #if np.linalg.norm(kp1-e1) < Parameters.kMinDistanceFromEpipole * kp1_scale_factor:                 
        #if np.linalg.norm(kp1-e1) - kp1_size < Parameters.kMinDistanceFromEpipole:  # N.B.: this is too much conservative => it filters too much                                       
        #    continue   
        
        kp2 = kf2.kpsu[i2]
        kp2_scale_factor = scale_factors[kf2.octaves[i2]]        
        # kp2_size = f2.sizes[i2]        
        # discard points which are too close to the epipole            
        delta = kp2-e2        
        #if np.linalg.norm(delta) < Parameters.kMinDistanceFromEpipole * kp2_scale_factor:   
        if np.inner(delta,delta) < kMinDistanceFromEpipole2 * kp2_scale_factor:   # OR.            
        # #if np.linalg.norm(delta) - kp2_size < Parameters.kMinDistanceFromEpipole:  # N.B.: this is too much conservative => it filters too much                                                  
             continue           
        
        # check epipolar constraint         
        sigma2_kp2 = level_sigmas2[kf2.octaves[i2]]
        if check_dist_epipolar_line(kp1,kp2,F12,sigma2_kp2):
            idxs1_out.append(i1)
            idxs2_out.append(i2)
            
            if check_orientation:
                index_match = len(idxs1_out)-1
                rot = kf1.angles[i1]-kf2.angles[i2]
                rot_histo.push(rot,index_match)            
        #else:
        #    print('discarding point match non respecting epipolar constraint')
         
    if check_orientation:            
        valid_match_idxs = rot_histo.get_valid_idxs()     
        #print('checking orientation consistency - valid matches % :', len(valid_match_idxs)/max(1,len(idxs1_out))*100,'% of ', len(idxs1_out),'matches')
        #print('rotation histogram: ', rot_histo)
        idxs1_out = np.array(idxs1_out)[valid_match_idxs]
        idxs2_out = np.array(idxs2_out)[valid_match_idxs]
                 
    num_found_matches = len(idxs1_out)
             
    if __debug__:
        print('search_frame_for_triangulation - timer: ', timer.elapsed())

    return idxs1_out, idxs2_out, num_found_matches, img2_epi


# search by projection matches between {input map points} and {keyframe points} and fuse them if they are close enough
def search_and_fuse(points, keyframe: KeyFrame, 
                    max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                    max_descriptor_distance = None,
                    ratio_test=Parameters.kMatchRatioTestMap):    
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance # more conservative check
    
    fused_pts_count = 0
    if len(points) == 0:
        Printer.red('search_and_fuse - no points')        
        return fused_pts_count
        
    # get all matched points of keyframe 
    if isinstance(points, np.ndarray):
        good_pts_idxs = np.flatnonzero(points!=None) 
    else:
        good_pts_idxs = [i for i, p in enumerate(points) if p is not None]
    good_pts = points[good_pts_idxs] 
     
    if len(good_pts_idxs) == 0:
        Printer.red('search_and_fuse - no matched points')
        return fused_pts_count
    
    # check if points are visible 
    good_pts_visible, good_projs, good_depths, good_dists = keyframe.are_visible(good_pts, keyframe.camera.is_stereo())
    
    if np.sum(good_pts_visible) == 0:
        Printer.red('search_and_fuse - no visible points')
        return fused_pts_count   
    
    predicted_levels = predict_detection_levels(good_pts, good_dists) 
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    
    kd_idxs = keyframe.kd.query_ball_point(good_projs[:,:2], radiuses)    
    
    do_check_stereo_reproj_err = keyframe.kps_ur is not None    

    #for i, p in enumerate(points):
    for i,p,j in zip(good_pts_idxs,good_pts,range(len(good_pts))):            
                
        if not good_pts_visible[j] or p.is_bad:     # point not visible in frame or point is bad 
            #print('p[%d] visible: %d, bad: %d' % (i, int(good_pts_visible[j]), int(p.is_bad))) 
            continue  
                  
        if p.is_in_keyframe(keyframe):    # we already matched this map point to this keyframe
            #print('p[%d] already in keyframe' % (i)) 
            continue
                   
        predicted_level = predicted_levels[j]     
            
        best_dist = math.inf 
        best_dist2 = math.inf
        best_level = -1 
        best_level2 = -1               
        best_kd_idx = -1        
            
        # find closest keypoints of frame        
        proj = good_projs[j]

        kd_idxs_j = kd_idxs[j]                
        if do_check_stereo_reproj_err:
            check_stereo = keyframe.kps_ur[kd_idxs_j]>0 
            errs_ur = proj[2] - keyframe.kps_ur[kd_idxs_j] # proj_ur - kp_ur
            errs_ur2 = errs_ur*errs_ur
            
        inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2
                        
        for h, kd_idx in enumerate(kd_idxs_j):             
                
            # check detection level     
            kp_level = keyframe.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):   
                #print('p[%d] wrong predicted level **********************************' % (i))                       
                continue
        
            # check the reprojection error     
            kp = keyframe.kpsu[kd_idx]
            invSigma2 = inv_level_sigmas2[kp_level]            
                                    
            err = proj[:2] - kp
            chi2 = np.dot(err,err)*invSigma2     
            if do_check_stereo_reproj_err and check_stereo[h]:
                chi2 += errs_ur2[h]*invSigma2 
                if chi2 > Parameters.kChi2Stereo: # chi-square 3 DOFs  (Hartley Zisserman pg 119)
                    #print('p[%d] big reproj err %f **********************************' % (i,chi2))
                    continue
            else:           
                if chi2 > Parameters.kChi2Mono: # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                    #print('p[%d] big reproj err %f **********************************' % (i,chi2))
                    continue                  
                            
            descriptor_dist = p.min_des_distance(keyframe.des[kd_idx])
            #print('p[%d] descriptor_dist %f **********************************' % (i,descriptor_dist))            
            
            #if descriptor_dist < max_descriptor_distance and descriptor_dist < best_dist:     
            if descriptor_dist < best_dist:                                      
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_kd_idx = kd_idx   
            elif descriptor_dist < best_dist2:  # N.O.
                best_dist2 = descriptor_dist       
                best_level2 = kp_level                                   
                                                            
        #if best_kd_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance:         
            # apply match distance ratio test only if the best and second are in the same scale level 
            if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test):  # N.O.
                #print('p[%d] best_dist > best_dist2 * ratio_test **********************************' % (i))
                continue                
            p_keyframe = keyframe.get_point_match(best_kd_idx)
            # if there is already a map point replace it otherwise add a new point
            if p_keyframe is not None:
                # if not p_keyframe.is_bad:
                #     if p_keyframe.num_observations > p.num_observations:
                #         p.replace_with(p_keyframe)
                #     else:
                #         p_keyframe.replace_with(p)        
                p_keyframe_is_bad, p_keyframe_is_good_with_better_num_obs = p_keyframe.is_bad_and_is_good_with_min_obs(p.num_observations)
                if not p_keyframe_is_bad:
                    if p_keyframe_is_good_with_better_num_obs:
                        p.replace_with(p_keyframe)
                    else:
                        p_keyframe.replace_with(p)
            else:
                p.add_observation(keyframe, best_kd_idx) 
                #p.update_info()    # done outside!
            fused_pts_count += 1                  
    return fused_pts_count     



# search by projection matches between {input map points} and {keyframe points} and fuse them if they are close enough
# use suggested Scw to project
def search_and_fuse_for_loop_correction(keyframe: KeyFrame, 
                                        Scw, 
                                        points, 
                                        replace_points,
                                        max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceFuse,
                                        max_descriptor_distance = None):    
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance # more conservative check
    
    assert(len(points) == len(replace_points))
    
    fused_pts_count = 0
    if len(points) == 0:
        Printer.red('search_and_fuse - no points')        
        return replace_points
        
    # get all matched points of keyframe 
    good_pts_idxs = np.flatnonzero(points!=None) 
    good_pts = points[good_pts_idxs] 
     
    if len(good_pts_idxs) == 0:
        Printer.red('search_and_fuse - no matched points')
        return replace_points
    
    # extract from sim3 Scw=[s*Rcw, tcw; 0, 1] the corresponding se3 transformation Tcw=[Rcw, tcw/s]
    if isinstance(Scw, np.ndarray):
        sRcw = Scw[:3,:3]
        scw = math.sqrt(np.dot(sRcw[0,:3], sRcw[0,:3]))
        Rcw = sRcw / scw
        tcw = Scw[:3,3]/scw
    elif isinstance(Scw, Sim3Pose):
        scw = Scw.s
        Rcw = Scw.R
        tcw = Scw.t/ scw
    else: 
        raise TypeError("Unsupported type '{}' for Scw".format(type(Scw)))
    
    # check if points are visible     
    good_pts_visible, good_projs, good_depths, good_dists = are_map_points_visible_in_frame(good_pts, keyframe, Rcw, tcw)
        
    if np.sum(good_pts_visible) == 0:
        Printer.red('search_and_fuse - no visible points')
        return replace_points
    
    predicted_levels = predict_detection_levels(good_pts, good_dists) 
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    
    kd_idxs = keyframe.kd.query_ball_point(good_projs[:,:2], radiuses)    

    for idx,p,j in zip(good_pts_idxs,good_pts,range(len(good_pts))):            
                
        if not good_pts_visible[j] or p.is_bad:     # point not visible in frame or point is bad 
            #print('p[%d] visible: %d, bad: %d' % (i, int(good_pts_visible[j]), int(p.is_bad))) 
            continue  
                  
        if p.is_in_keyframe(keyframe):    # we already matched this map point to this keyframe
            #print('p[%d] already in keyframe' % (i)) 
            continue
                     
        predicted_level = predicted_levels[j]
        
        best_dist = math.inf      
        best_kd_idx = -1        
            
        # find closest keypoints of frame        
        proj = good_projs[j]

        kd_idxs_j = kd_idxs[j]                
            
        inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2
                        
        for h, kd_idx in enumerate(kd_idxs_j):             
                
            # check detection level     
            kp_level = keyframe.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):   
                #print('p[%d] wrong predicted level **********************************' % (i))                       
                continue
                                    
            descriptor_dist = p.min_des_distance(keyframe.des[kd_idx])
            #print('p[%d] descriptor_dist %f **********************************' % (i,descriptor_dist))            
            
            if descriptor_dist < best_dist:                                      
                best_dist = descriptor_dist
                best_kd_idx = kd_idx   
                              
                                                            
        if best_dist < max_descriptor_distance:                      
            p_keyframe = keyframe.get_point_match(best_kd_idx)
            # if there is already a map point replace it 
            if p_keyframe is not None:
                if not p_keyframe.is_bad:
                    replace_points[idx] = p_keyframe
            else:
                p.add_observation(keyframe, best_kd_idx) 
                #p.update_info()    # done outside!
            fused_pts_count += 1       
                       
    return replace_points     


# search new matches between unmatched map points of kf1 and kf2 by using a know sim3 transformation (guided matching)
# in:
#   kf1, kf2: keyframes
#   idxs1, idxs2:  kf1.points(idxs1[i]) is matched with kf2.points(idxs2[i])  
#   s12, R12, t12: sim3 transformation that guides the matching
# out: 
#   new_matches12: where kf2.points(new_matches12[i]) is matched to i-th map point in kf1 (includes the input matches) if new_matches12[i]>0
#   new_matches21: where kf1.points(new_matches21[i]) is matched to i-th map point in kf2 (includes the input matches) if new_matches21[i]>0
def search_by_sim3(kf1: KeyFrame, kf2: KeyFrame, 
                   idxs1, idxs2, 
                   s12, R12, t12, 
                   max_reproj_distance=Parameters.kMaxReprojectionDistanceSim3, 
                   max_descriptor_distance=None,
                   print_fun=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance
        
    assert(len(idxs1) == len(idxs2))        
    # Sim3 transformations between cameras
    sR12 = s12 * R12
    sR21 = (1.0 / s12) * R12.T
    t21 = -sR21 @ t12

    map_points1 = kf1.get_points() # get all map points of kf1
    n1 = len(map_points1)
    new_matches12 = np.full(n1, -1, dtype=np.int32) # kf2.points(new_matches12[i]) is matched to i-th map point in kf1 if new_matches12[i]>0 (from 1 to 2)
    good_points1 = np.array([True if mp is not None and not mp.is_bad else False for mp in map_points1])
    
    map_points2 = kf2.get_points() # get all map points of kf2
    n2 = len(map_points2)
    new_matches21 = np.full(n2, -1, dtype=np.int32) # kf1.points(new_matches21[i]) is matched to i-th map point in kf2 if new_matches21[i]>0 (from 2 to 1)
    good_points2 = np.array([True if mp is not None and not mp.is_bad else False for mp in map_points2])
        
    for idx1, idx2 in zip(idxs1, idxs2):
        # Integrate the matches we already have as input into the output
        if good_points1[idx1] and good_points2[idx2]:
            new_matches12[idx1] = idx2
            new_matches21[idx2] = idx1        
        
    # if print_fun is not None: 
    #     print_fun(f'search_by_sim3: starting num mp matches: {np.sum(new_matches12!=-1)}')
        
    # Find unmatched map points
    unmatched_idxs1 = [idx for idx in range(n1) if good_points1[idx] and new_matches12[idx]<0]
    unmatched_map_points1 = map_points1[unmatched_idxs1]
    unmatched_idxs2 = [idx for idx in range(n2) if good_points2[idx] and new_matches21[idx]<0]
    unmatched_map_points2 = map_points2[unmatched_idxs2]
        
    # if print_fun is not None:
    #     print_fun(f'search_by_sim3: found: {len(unmatched_idxs1)} unmatched map points of kf1 {kf1.id}, {len(unmatched_idxs2)} unmatched map points of kf2 {kf2.id}')

    scale_factors = FeatureTrackerShared.feature_manager.scale_factors
    
    # check which unmatched points of kf1 are visible on kf2 
    visible_flags_21, projs_21, depths_21, dists_21 = \
        are_map_points_visible(kf1, kf2, unmatched_map_points1, sR21, t21)
        
    num_visible_21 = np.sum(visible_flags_21)
    # if print_fun is not None:    
    #     print_fun(f'search_by_sim3: {num_visible_21} map points of kf1 {kf1.id} are visible on kf2 {kf2.id}')
                
    if num_visible_21 > 0: 
        predicted_levels = predict_detection_levels(unmatched_map_points1, dists_21) 
        kp_scale_factors = scale_factors[predicted_levels]              
        radiuses = max_reproj_distance * kp_scale_factors     
        kd2_idxs = kf2.kd.query_ball_point(projs_21[:,:2], radiuses)   # search NN kps on kf2
        
        for i1, mp1 in enumerate(unmatched_map_points1):
            kd2_idxs_i = kd2_idxs[i1]                  
            predicted_level = predicted_levels[i1]                
                            
            best_dist = float('inf')
            best_idx = -1                                      
            for kd2_idx in kd2_idxs_i:             
                # check detection level     
                kp_level = kf2.octaves[kd2_idx]    
                if (kp_level<predicted_level-1) or (kp_level>predicted_level):                       
                    continue                      
                
                dist = mp1.min_des_distance(kf2.des[kd2_idx])

                if dist < best_dist:
                    best_dist = dist
                    best_idx = kd2_idx

            if best_dist <= max_descriptor_distance:
                if new_matches21[best_idx]==-1:
                    new_matches12[unmatched_idxs1[i1]] = best_idx
                 
    # check which unmatched points of kf2 are visible on kf1 
    visible_flags_12, projs_12, depths_12, dists_12 = \
        are_map_points_visible(kf2, kf1, unmatched_map_points2, sR12, t12)
        
    num_visible_12 = np.sum(visible_flags_12)
    # if print_fun is not None:         
    #     print_fun(f'search_by_sim3: {num_visible_12} map points of kf2 {kf2.id} are visible on kf1 {kf1.id}')        
          
    if num_visible_12 > 0:
        predicted_levels = predict_detection_levels(unmatched_map_points2, dists_12) 
        kp_scale_factors = scale_factors[predicted_levels]              
        radiuses = max_reproj_distance * kp_scale_factors     
        kd1_idxs = kf1.kd.query_ball_point(projs_12[:,:2], radiuses)   # search NN kps on kf1
        
        for i2, mp2 in enumerate(unmatched_map_points2):
            kd1_idxs_i = kd1_idxs[i2]                  
            predicted_level = predicted_levels[i2]                
                            
            best_dist = float('inf')
            best_idx = -1                                      
            for kd1_idx in kd1_idxs_i:             
                # check detection level     
                kp_level = kf1.octaves[kd1_idx]    
                if (kp_level<predicted_level-1) or (kp_level>predicted_level):                       
                    continue                      
                
                dist = mp2.min_des_distance(kf1.des[kd1_idx])

                if dist < best_dist:
                    best_dist = dist
                    best_idx = kd1_idx

            if best_dist <= max_descriptor_distance:
                if new_matches12[best_idx] == -1:
                    new_matches21[unmatched_idxs2[i2]] = best_idx

    # if print_fun is not None:    
    #     print_fun(f'search_by_sim3: new matches before check: 1->2: {np.sum(new_matches12!=-1)}, 2->1: {np.sum(new_matches21!=-1)}')

    # Check agreement
    num_matches_found = 0
    for i1 in range(n1):
        idx2 = new_matches12[i1] # index of kf2 point that matches with i1-th kf1 point
        if idx2 >= 0:
            idx1 = new_matches21[idx2] # index of kf1 point that matches with idx2-th kf2 point
            if idx1 != i1: # reset if mismatch
                new_matches12[i1] = -1
                new_matches21[idx2] = -1
            else: 
                num_matches_found += 1
 
    # if print_fun is not None:    
    #     print_fun(f'search_by_sim3: num matches found after final check: {num_matches_found}')
    #     print_fun(f'search_by_sim3: new matches after check: 1->2: {np.sum(new_matches12!=-1)}, 2->1: {np.sum(new_matches21!=-1)}')
        
    return num_matches_found, new_matches12, new_matches21