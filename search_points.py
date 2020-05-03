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
from map_point import predict_detection_levels

from utils_geom import skew, add_ones, normalize_vector, computeF12, check_dist_epipolar_line
from utils_draw import draw_lines, draw_points 
from utils import Printer, getchar
from parameters import Parameters  
from timer import Timer
from rotation_histogram import RotationHistogram


kMinDistanceFromEpipole = Parameters.kMinDistanceFromEpipole
kMinDistanceFromEpipole2 = kMinDistanceFromEpipole*kMinDistanceFromEpipole
kCheckFeaturesOrientation = Parameters.kCheckFeaturesOrientation 


# propagate map point matches from f_ref to f_cur (access frames from tracking thread, no need to lock)
def propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur,
                                max_descriptor_distance=Parameters.kMaxDescriptorDistance):
    idx_ref_out = []
    idx_cur_out = []
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and Frame.oriented_features
        
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
                rot_histo.push(rot,index_match)
            
    if check_orientation:            
        valid_match_idxs = rot_histo.get_valid_idxs()     
        print('checking orientation consistency - valid matches % :', len(valid_match_idxs)/max(1,len(idxs_cur))*100,'% of ', len(idxs_cur),'matches')
        #print('rotation histogram: ', rot_histo)
        idx_ref_out = np.array(idx_ref_out)[valid_match_idxs]
        idx_cur_out = np.array(idx_cur_out)[valid_match_idxs]
        num_matched_map_pts = len(valid_match_idxs)            
                            
    return num_matched_map_pts, idx_ref_out, idx_cur_out  

  
# search by projection matches between {map points of f_ref} and {keypoints of f_cur},  (access frames from tracking thread, no need to lock)
def search_frame_by_projection(f_ref, f_cur,
                               max_reproj_distance=Parameters.kMaxReprojectionDistanceFrame,
                               max_descriptor_distance=Parameters.kMaxDescriptorDistance,
                               ratio_test=Parameters.kMatchRatioTestMap):
    found_pts_count = 0
    idxs_ref = []
    idxs_cur = [] 
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and Frame.oriented_features    
    
    #des_dists = []
    
    # get all matched points of f_ref which are non-outlier 
    matched_ref_idxs = np.flatnonzero( (f_ref.points!=None) & (f_ref.outliers==False)) 
    matched_ref_points = f_ref.points[matched_ref_idxs]
    
    if True: 
        # project f_ref points on frame f_cur
        projs, depths = f_cur.project_map_points(matched_ref_points)
        # check if points lie on the image frame 
        is_visible = f_cur.are_in_image(projs, depths)
    else: 
        # check if points are visible 
        is_visible, projs, depths, dists = f_cur.are_visible(matched_ref_points)
        
    kp_ref_octaves = f_ref.octaves[matched_ref_idxs]       
    kp_ref_scale_factors = Frame.feature_manager.scale_factors[kp_ref_octaves]              
    radiuses = max_reproj_distance * kp_ref_scale_factors     
    kd_idxs = f_cur.kd.query_ball_point(projs, radiuses)        
                                  
    for i,p,j in zip(matched_ref_idxs, matched_ref_points, range(len(matched_ref_points))):
    
        if not is_visible[j]:
            continue 
        
        kp_ref_octave = f_ref.octaves[i]
        #kp_ref_scale_factor = Frame.feature_manager.scale_factors[kp_ref_octave]
        #radius = max_reproj_distance * kp_ref_scale_factor        
        
        best_dist = math.inf 
        #best_dist2 = math.inf
        best_level = -1 
        #best_level2 = -1                   
        best_k_idx = -1   
        best_ref_idx = -1          
                
        #for kd_idx in f_cur.kd.query_ball_point(projs[j], radius):            
        for kd_idx in kd_idxs[j]:                   
            
            p_f_cur = f_cur.points[kd_idx]
            if  p_f_cur is not None:
                if p_f_cur.num_observations > 0: # we already matched p_f_cur => discard it 
                    continue          
    
            p_f_cur_octave = f_cur.octaves[kd_idx]
            if p_f_cur_octave < (kp_ref_octave-1) or p_f_cur_octave > (kp_ref_octave+1):
                continue                           
            
            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])
            #if descriptor_dist < max_descriptor_distance and descriptor_dist < best_dist:
            if descriptor_dist < best_dist:                
                best_dist = descriptor_dist
                best_k_idx = kd_idx
                best_ref_idx = i     
                
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


# search by projection matches between {input map points} and {unmatched keypoints of frame f_cur}, (access frame from tracking thread, no need to lock)
def search_map_by_projection(points, f_cur, 
                             max_reproj_distance=Parameters.kMaxReprojectionDistanceMap, 
                             max_descriptor_distance=Parameters.kMaxDescriptorDistance,
                             ratio_test=Parameters.kMatchRatioTestMap):
    Ow = f_cur.Ow 
    
    found_pts_count = 0
    found_pts_fidxs = []   # idx of matched points in current frame 
    
    #reproj_dists = []
    
    if len(points) == 0:
        return 0 
            
    # check if points are visible 
    visible_pts, projs, depths, dists = f_cur.are_visible(points)
    
    predicted_levels = predict_detection_levels(points, dists) 
    kp_scale_factors = Frame.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    kd_idxs = f_cur.kd.query_ball_point(projs, radiuses)
                           
    for i, p in enumerate(points):
        if not visible_pts[i] or p.is_bad:     # point not visible in frame or is bad 
            continue        
        if p.last_frame_id_seen == f_cur.id:   # we already matched this map point to current frame or it was outlier 
            continue
        
        p.increase_visible()
        
        # predicted_level = p.predict_detection_level(dists[i])     
        predicted_level = predicted_levels[i]        
        # kp_scale_factor = Frame.feature_manager.scale_factors[predicted_level]              
        # radius = max_reproj_distance * kp_scale_factor     
                       
        best_dist = math.inf 
        best_dist2 = math.inf
        best_level = -1 
        best_level2 = -1               
        best_k_idx = -1  

        # find closest keypoints of f_cur        
        #for kd_idx in f_cur.kd.query_ball_point(projs[i], radius):
        #for kd_idx in f_cur.kd.query_ball_point(proj, radius):  
        for kd_idx in kd_idxs[i]:
     
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
    # else:
    reproj_dist_sigma = max_descriptor_distance
                       
    return found_pts_count, reproj_dist_sigma, found_pts_fidxs   


# search by projection matches between {map points of last frames} and {unmatched keypoints of f_cur}, (access frame from tracking thread, no need to lock)
def search_local_frames_by_projection(map, f_cur, local_window = Parameters.kLocalBAWindow):
    # take the points in the last N frame 
    points = []
    frames = map.keyframes[-local_window:]
    f_points = set([p for f in frames for p in f.get_points() if (p is not None)])
    print('searching %d map points' % len(points))
    return search_map_by_projection(points, f_cur)  


# search by projection matches between {all map points} and {unmatched keypoints of f_cur}
def search_all_map_by_projection(map, f_cur):
    return search_map_by_projection(map.get_points(), f_cur)      


# search keypoint matches (for triangulations) between f1 and f2
# search for matches between unmatched keypoints (without a corresponding map point)
# in input we have already some pose estimates for f1 and f2
def search_frame_for_triangulation(kf1, kf2, idxs1=None, idxs2=None, 
                                   max_descriptor_distance=0.5*Parameters.kMaxDescriptorDistance):   
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

    # if the translation is too small we cannot triangulate 
    # if baseline < Parameters.kMinTraslation:  # we assume the Inializer has been used for building the first map 
    #     Printer.red("search for triangulation: impossible with almost zero translation!")
    #     return idxs1_out, idxs2_out, num_found_matches, img2_epi # EXIT
    # else:    
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
        idxs1, idxs2 = Frame.feature_matcher.match(kf1.des, kf2.des)        
        print('search_frame_for_triangulation - matching - timer: ', timerMatch.elapsed())        
    
    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and Frame.oriented_features     
        
    # check epipolar constraints 
    for i1,i2 in zip(idxs1,idxs2):
        if kf1.get_point_match(i1) is not None or kf2.get_point_match(i2) is not None: # we are searching for keypoint matches where both keypoints do not have a corresponding map point 
            #print('existing point on match')
            continue 
        
        descriptor_dist = Frame.descriptor_distance(kf1.des[i1], kf2.des[i2])
        if descriptor_dist > max_descriptor_distance:
            continue     
        
        kp1 = kf1.kpsu[i1]
        #kp1_scale_factor = Frame.feature_manager.scale_factors[kf1.octaves[i1]]
        #kp1_size = f1.sizes[i1]
        # discard points which are too close to the epipole            
        #if np.linalg.norm(kp1-e1) < Parameters.kMinDistanceFromEpipole * kp1_scale_factor:                 
        #if np.linalg.norm(kp1-e1) - kp1_size < Parameters.kMinDistanceFromEpipole:  # N.B.: this is too much conservative => it filters too much                                       
        #    continue   
        
        kp2 = kf2.kpsu[i2]
        kp2_scale_factor = Frame.feature_manager.scale_factors[kf2.octaves[i2]]        
        # kp2_size = f2.sizes[i2]        
        # discard points which are too close to the epipole            
        delta = kp2-e2        
        #if np.linalg.norm(delta) < Parameters.kMinDistanceFromEpipole * kp2_scale_factor:   
        if np.inner(delta,delta) < kMinDistanceFromEpipole2 * kp2_scale_factor:   # OR.            
        # #if np.linalg.norm(delta) - kp2_size < Parameters.kMinDistanceFromEpipole:  # N.B.: this is too much conservative => it filters too much                                                  
             continue           
        
        # check epipolar constraint         
        sigma2_kp2 = Frame.feature_manager.level_sigmas2[kf2.octaves[i2]]
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


# search by projection matches between {input map points} and {unmatched keypoints of frame}
def search_and_fuse(points, keyframe, 
                    max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                    max_descriptor_distance = 0.5*Parameters.kMaxDescriptorDistance,
                    ratio_test=Parameters.kMatchRatioTestMap):
    #max_descriptor_distance = 0.5 * Parameters.kMaxDescriptorDistance 
    
    fused_pts_count = 0
    Ow = keyframe.Ow 
    if len(points) == 0:
        Printer.red('search_and_fuse - no points')        
        return 
        
    # get all matched points of keyframe 
    good_pts_idxs = np.flatnonzero(points!=None) 
    good_pts = points[good_pts_idxs] 
     
    if len(good_pts_idxs) == 0:
        Printer.red('search_and_fuse - no matched points')
        return
    
    # check if points are visible 
    good_pts_visible, good_projs, good_depths, good_dists = keyframe.are_visible(good_pts)
    
    if len(good_pts_visible) == 0:
        Printer.red('search_and_fuse - no visible points')
        return    
    
    predicted_levels = predict_detection_levels(good_pts, good_dists) 
    kp_scale_factors = Frame.feature_manager.scale_factors[predicted_levels]              
    radiuses = max_reproj_distance * kp_scale_factors     
    kd_idxs = keyframe.kd.query_ball_point(good_projs, radiuses)    

    #for i, p in enumerate(points):
    for i,p,j in zip(good_pts_idxs,good_pts,range(len(good_pts))):            
                
        if not good_pts_visible[j] or p.is_bad:     # point not visible in frame or point is bad 
            #print('p[%d] visible: %d, bad: %d' % (i, int(good_pts_visible[j]), int(p.is_bad))) 
            continue  
                  
        if p.is_in_keyframe(keyframe):    # we already matched this map point to this keyframe
            #print('p[%d] already in keyframe' % (i)) 
            continue
                
        # predicted_level = p.predict_detection_level(good_dists[j])         
        # kp_scale_factor = Frame.feature_manager.scale_factors[predicted_level]              
        # radius = max_reproj_distance * kp_scale_factor     
        predicted_level = predicted_levels[j]
        
        #print('p[%d] radius: %f' % (i,radius))         
            
        best_dist = math.inf 
        best_dist2 = math.inf
        best_level = -1 
        best_level2 = -1               
        best_kd_idx = -1        
            
        # find closest keypoints of frame        
        proj = good_projs[j]
        #for kd_idx in keyframe.kd.query_ball_point(proj, radius):  
        for kd_idx in kd_idxs[j]:             
                
            # check detection level     
            kp_level = keyframe.octaves[kd_idx]    
            if (kp_level<predicted_level-1) or (kp_level>predicted_level):   
                #print('p[%d] wrong predicted level **********************************' % (i))                       
                continue
        
            # check the reprojection error     
            kp = keyframe.kpsu[kd_idx]
            invSigma2 = Frame.feature_manager.inv_level_sigmas2[kp_level]                
            err = proj - kp       
            chi2 = np.inner(err,err)*invSigma2           
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
            else: 
                if descriptor_dist < best_dist2:  # N.O.
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
                if not p_keyframe.is_bad:
                    if p_keyframe.num_observations > p.num_observations:
                        p.replace_with(p_keyframe)
                    else:
                        p_keyframe.replace_with(p)                  
            else:
                p.add_observation(keyframe, best_kd_idx) 
                #p.update_info()    # done outside!
            fused_pts_count += 1                  
    return fused_pts_count     