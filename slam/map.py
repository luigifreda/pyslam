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

import time
import numpy as np
import math 
import cv2

import ujson as json

from collections import Counter, deque

from ordered_set import OrderedSet # from https://pypi.org/project/ordered-set/

from threading import RLock, Thread

from utils_geom import poseRt, add_ones, add_ones_1D
from config_parameters import Parameters 
from frame import Frame, FrameShared, FrameBase
from keyframe import KeyFrame
from map_point import MapPoint, MapPointBase

from utils_sys import Printer

import traceback

import g2o
import optimizer_g2o 


kVerbose = True 
kMaxLenFrameDeque = 20


if not kVerbose:
    def print(*args, **kwargs):
        pass 
    
             
class ReloadedSessionMapInfo:
    def __init__(self, num_keyframes, num_points, max_point_id, max_frame_id, max_keyframe_id):
        self.num_keyframes = num_keyframes
        self.num_points = num_points
        self.max_point_id = max_point_id
        self.max_frame_id = max_frame_id
        self.max_keyframe_id = max_keyframe_id

                 
class Map(object):
    def __init__(self):
        self._lock = RLock()   
        self._update_lock = RLock()  
        
        self.frames = deque(maxlen=kMaxLenFrameDeque)  # deque with max length, it is thread-safe 
        self.keyframes = OrderedSet()  
        self.points = set()
        
        self.keyframe_origins = OrderedSet()   # first keyframe(s) where the map is rooted
        
        self.keyframes_map = {} # map: frame id -> keyframe  (for fast retrieving keyframe from img_id/frame_id)
        
        self.max_point_id = 0     # 0 is the first point id
        self.max_frame_id = 0     # 0 is the first frame id        
        self.max_keyframe_id = 0  # 0 is the first keyframe id (kid)
        
        self.reloaded_session_map_info = None  # type: ReloadedSessionMapInfo

        # local map 
        #self.local_map = LocalWindowMap(map=self)
        self.local_map = LocalCovisibilityMap(map=self)
        
        self.viewer_scale = -1
        
    def is_reloaded(self):
        return self.reloaded_session_map_info is not None
        
    def reset(self):
        print('Map: reset...')        
        with self._lock:
            with self._update_lock:
                self.frames.clear()
                self.keyframes.clear()
                self.points.clear()

                self.keyframe_origins.clear()  
                    
                self.keyframes_map.clear()
                
                self.local_map.reset()
                
    def reset_session(self):
        print('Map: reset_session...')        
        with self._lock:
            with self._update_lock:
                if self.reloaded_session_map_info is None:
                    self.reset()
                else:
                    # First, collect keyframes to remove
                    keyframes_to_remove = [kf for kf in self.keyframes if kf.kid >= self.reloaded_session_map_info.max_keyframe_id]
                    for kf in keyframes_to_remove:
                        kf.set_bad()
                        self.keyframes.discard(kf)  # Discard instead of remove to avoid KeyError
                        self.keyframe_origins.discard(kf)  # Safe discard
                        self.keyframes_map.pop(kf.id, None)  # Use pop() to avoid KeyError
                    
                    # Similarly for points
                    points_to_remove = [p for p in self.points if p.id >= self.reloaded_session_map_info.max_point_id]
                    for p in points_to_remove:
                        p.set_bad()
                        self.points.discard(p)  # Safe discard
                    
                    # Similarly for frames
                    frames_to_remove = [f for f in self.frames if f.id >= self.reloaded_session_map_info.max_frame_id]
                    for f in frames_to_remove:
                        self.frames.remove(f)  # Since deque is not a set, use remove here

                    # Reset the session of the local map
                    self.local_map.reset_session(keyframes_to_remove, points_to_remove)


    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the RLock from the state (don't pickle it)
        if '_lock' in state:
            del state['_lock']
        if '_update_lock' in state:
            del state['_update_lock']            
        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        # Recreate the RLock after unpickling
        self._lock = RLock()
        self._update_lock = RLock()
                
                
    @property    
    def lock(self):  
        return self._lock 
    
    @property    
    def update_lock(self):  
        return self._update_lock     
    
    def get_points(self): 
        with self._lock:       
            return self.points.copy()   
        
    def num_points(self):
        with self._lock: 
            return len(self.points)                
     
    def get_frame(self,idx): 
        with self._lock:
            try:       
                return self.frames[idx]
            except:
                return None 
                
    def get_frames(self): 
        with self._lock:       
            return self.frames.copy()       
        
    def num_frames(self):
        with self._lock: 
            return len(self.frames)           
                
    def get_keyframes(self): 
        with self._lock:       
            return self.keyframes.copy()     
                
    def get_last_keyframe(self): 
        with self._lock:         
            return self.keyframes[-1]    
                
    # get the last N=local_window map keyframes          
    def get_last_keyframes(self, local_window=Parameters.kLocalBAWindow): 
        with self._lock:         
            return OrderedSet(self.keyframes.copy()[-local_window:])                     
    
    # return the total number of keyframes
    def num_keyframes(self):
        with self._lock: 
            return len(self.keyframes)             
    
    # return the number of keyframes of this session
    def num_keyframes_session(self):
        with self._lock: 
            if self.reloaded_session_map_info is not None:
                return len(self.keyframes) - self.reloaded_session_map_info.num_keyframes
            else:
                return len(self.keyframes)
    
    def delete(self):
        with self._lock:          
            for f in self.frames:
                f.reset_points()
            for kf in self.keyframes:
                kf.reset_points()
                

    def add_point(self, point):
        with self._lock:           
            ret = self.max_point_id # override original id    
            point.id = ret 
            point.map = self 
            self.max_point_id += 1
            #self.points.append(point)
            self.points.add(point)
            return ret


    def remove_point(self, point):
        with self._lock:    
            try:                    
                self.points.remove(point)
            except:
                pass 
            point.delete()        


    def add_frame(self, frame, ovverride_id=False):
        with self._lock:          
            ret = frame.id
            if ovverride_id: 
                ret = self.max_frame_id
                frame.id = ret # override original id    
                self.max_frame_id += 1
            else: 
                self.max_frame_id = max(self.max_frame_id, frame.id+1)       
            self.frames.append(frame)     
            return ret
        
    def remove_frame(self, frame): 
        with self._lock:                           
            try: 
                self.frames.remove(frame) 
            except: 
                pass 
    
    def add_keyframe(self, keyframe):
        with self._lock:          
            assert(keyframe.is_keyframe)
            ret = self.max_keyframe_id
            keyframe.kid = ret # override original keyframe kid    
            keyframe.is_keyframe = True 
            keyframe.map = self 
            self.keyframes.add(keyframe)
            self.keyframes_map[keyframe.id] = keyframe          
            self.max_keyframe_id += 1                                  
            return ret    
    
    def remove_keyframe(self, keyframe): 
        with self._lock:
            assert(keyframe.is_keyframe)                               
            try: 
                self.keyframes.remove(keyframe)
                del self.keyframes_map[keyframe.id] 
            except: 
                pass     

    def draw_feature_trails(self, img):
        if len(self.frames) > 0:
            img_draw = self.frames[-1].draw_all_feature_trails(img)
            return img_draw
        return img


    # add new points to the map from 3D point estimations, frames and pairwise matches
    # points3d is [Nx3]
    def add_points(self, points3d, mask_pts3d, kf1: KeyFrame, kf2: KeyFrame, idxs1, idxs2, img1, do_check=True, cos_max_parallax=Parameters.kCosMaxParallax):
        with self._lock:             
            assert(kf1.is_keyframe and  kf2.is_keyframe) # kf1 and kf2 must be keyframes 
            assert(points3d.shape[0] == len(idxs1))
            assert(len(idxs2) == len(idxs1))    
            
            idxs1 = np.array(idxs1)
            idxs2 = np.array(idxs2)
            
            added_map_points = []
            out_mask_pts3d = np.full(points3d.shape[0], False, dtype=bool)
            if mask_pts3d is None:
                mask_pts3d = np.full(points3d.shape[0], True, dtype=bool)
                        
            if do_check:
                                                        
                # project points
                uvs1, proj_depths1 = kf1.project_points(points3d)    
                bad_depths1 = proj_depths1 <= 0  
                uvs2, proj_depths2 = kf2.project_points(points3d)    
                bad_depths2 = proj_depths2 <= 0                   
                
                is_stereo1 = np.zeros(len(idxs1), dtype=bool) if kf1.kps_ur is None else kf1.kps_ur[idxs1]>0
                is_mono1 = np.logical_not(is_stereo1)
                is_stereo2 = np.zeros(len(idxs2), dtype=bool) if kf2.kps_ur is None else kf2.kps_ur[idxs2]>0
                is_mono2 = np.logical_not(is_stereo2)
                
                # compute back-projected rays (unit vectors) 
                rays1 = np.dot(kf1.Rwc, add_ones(kf1.kpsn[idxs1]).T).T
                norm_rays1 = np.linalg.norm(rays1, axis=-1, keepdims=True)                  
                rays1 /= norm_rays1                      
                rays2 = np.dot(kf2.Rwc, add_ones(kf2.kpsn[idxs2]).T).T     
                norm_rays2 = np.linalg.norm(rays2, axis=-1, keepdims=True)  
                rays2 /= norm_rays2 
                
                # compute dot products of rays                              
                cos_parallaxs = np.sum(rays1 * rays2, axis=1)  
                
                # if we have depths check if we can use depths in case of bad parallax
                if kf1.depths is not None and kf2.depths is not None:
                    # NOTE: 2.0 is certainly higher than any cos_parallax value
                    cos_parallax_stereo1 = np.where(is_stereo1, np.cos(2.*np.arctan2(kf1.camera.b/2,kf1.depths[idxs1])),2.0) if kf1.depths is not None else [2.0]*len(idxs1)
                    cos_parallax_stereo2 = np.where(is_stereo2, np.cos(2.*np.arctan2(kf2.camera.b/2,kf2.depths[idxs2])),2.0) if kf2.depths is not None else [2.0]*len(idxs2)
                    cos_parallax_stereo = np.minimum(cos_parallax_stereo1, cos_parallax_stereo2)
                    
                    # check if we can recover bad-parallx points from stereo/rgbd data
                    try_recover3d_from_stereo = np.logical_or(cos_parallaxs<0, np.logical_or(cos_parallaxs>cos_parallax_stereo, cos_parallaxs>cos_max_parallax))
                    recover3d_from_stereo1 = np.logical_and(try_recover3d_from_stereo, 
                                                            np.logical_and(is_stereo1,cos_parallax_stereo1<cos_parallax_stereo2))
                    recover3d_from_stereo2 = np.logical_and(np.logical_and(try_recover3d_from_stereo,np.logical_not(recover3d_from_stereo1)),
                                                            np.logical_and(is_stereo2,cos_parallax_stereo2<cos_parallax_stereo1)) 
                    recovered3d_from_stereo = np.logical_or(recover3d_from_stereo1, recover3d_from_stereo2)    
                
                    if np.any(recover3d_from_stereo1):
                        points3d[recover3d_from_stereo1,:],_ = kf1.unproject_points_3d(idxs1[recover3d_from_stereo1], transform_in_world=True)
                    if np.any(recover3d_from_stereo2):
                        points3d[recover3d_from_stereo2,:],_ = kf2.unproject_points_3d(idxs2[recover3d_from_stereo2], transform_in_world=True)
                else: 
                    recovered3d_from_stereo = np.zeros(len(idxs1), dtype=bool)
                                                        
                # we don't have bad parallax where we recovered from stereo
                bad_cos_parallaxs = np.logical_and(np.logical_or(cos_parallaxs < 0, cos_parallaxs > cos_max_parallax), np.logical_not(recovered3d_from_stereo))           
                
                # compute reprojection errors and check chi2
                bad_chis2_1 = None
                bad_chis2_2 = None           
                  
                # compute mono reproj errors on kf1
                errs1_mono_vec = uvs1 - kf1.kpsu[idxs1]
                errs1 = np.where(is_mono1[:, np.newaxis], errs1_mono_vec, np.zeros(2))   # mono errors 
                errs1_sqr = np.sum(errs1 * errs1, axis=1)  # squared reprojection errors 
                kps1_levels = kf1.octaves[idxs1]
                invSigmas2_1 = FrameShared.feature_manager.inv_level_sigmas2[kps1_levels] 
                chis2_1_mono = errs1_sqr * invSigmas2_1         # chi square 
                                
                # stereo reprojection error
                #     u   = fx*x*invz+cx
                #     u_r = u - camera.bf*invz
                #     v1   = fy*y*invz+cy
                #     errX   = u - kp.pt.x
                #     errY   = v - kp.pt.y
                #     errX_r = u_r - kp_ur
                # compute stereo reproj errors on kf1
                if kf1.kps_ur is not None:
                    kp1_ur = kf1.kps_ur[idxs1] if kf1.kps_ur is not None else [-1]*len(idxs1) # kp right coords if available 
                    depths1 = kf1.depths[idxs1] 
                    safe_depths1 = np.where(depths1 == 0, np.inf, depths1) # to prevent division by zero 
                    errs1_stereo_vec = np.concatenate((errs1_mono_vec, (uvs1[:,0] - kf1.camera.bf/safe_depths1 - kp1_ur)[:, np.newaxis]), axis=1)    # stereo errors                     
                    errs1_stereo = np.where(is_stereo1[:, np.newaxis], errs1_stereo_vec, np.zeros(3)) 
                    errs1_stereo_sqr = np.sum(errs1_stereo * errs1_stereo, axis=1)  # squared reprojection errors    
                    chis2_1_stereo = errs1_stereo_sqr * invSigmas2_1         # chi square            
                    bad_chis2_1 = np.logical_or(chis2_1_mono > Parameters.kChi2Mono, chis2_1_stereo > Parameters.kChi2Stereo)                                                  
                else: 
                    bad_chis2_1 = chis2_1_mono > Parameters.kChi2Mono
                              
                # compute mono reproj errors on kf1                                                                 
                errs2_mono_vec = uvs2 - kf2.kpsu[idxs2] # mono errors               
                errs2 = np.where(is_mono2[:, np.newaxis], errs2_mono_vec, np.zeros(2))   # mono errors
                errs2_sqr = np.sum(errs2 * errs2, axis=1) # squared reprojection errors        
                kps2_levels = kf2.octaves[idxs2]
                invSigmas2_2 = FrameShared.feature_manager.inv_level_sigmas2[kps2_levels] 
                chis2_2_mono = errs2_sqr * invSigmas2_2        # chi square 
                
                if kf2.kps_ur is not None:
                    kp2_ur = kf2.kps_ur[idxs2] if kf2.kps_ur is not None else [-1]*len(idxs2) # kp right coords if available
                    depths2 = kf2.depths[idxs2] 
                    safe_depths2 = np.where(depths2 == 0, np.inf, depths2) # to prevent division by zero                     
                    errs2_stereo_vec = np.concatenate((errs2_mono_vec, (uvs2[:,0] - kf2.camera.bf/safe_depths2 - kp2_ur)[:, np.newaxis]), axis=1)    # stereo errors
                    errs2_stereo = np.where(is_stereo2[:, np.newaxis], errs2_stereo_vec, np.zeros(3)) 
                    errs2_stereo_sqr = np.sum(errs2_stereo * errs2_stereo, axis=1)  # squared reprojection errors
                    chis2_2_stereo = errs2_stereo_sqr * invSigmas2_2         # chi square
                    bad_chis2_2 = np.logical_or(chis2_2_mono > Parameters.kChi2Mono, chis2_2_stereo > Parameters.kChi2Stereo)
                else: 
                    bad_chis2_2 = chis2_2_mono > Parameters.kChi2Mono  # chi-square 2 DOFs  (Hartley Zisserman pg 119)                      
                
                # scale consistency check
                ratio_scale_consistency = Parameters.kScaleConsistencyFactor * FrameShared.feature_manager.scale_factor 
                scale_factors_x_depths1 =  FrameShared.feature_manager.scale_factors[kps1_levels] * proj_depths1
                scale_factors_x_depths1_x_ratio_scale_consistency = scale_factors_x_depths1*ratio_scale_consistency                             
                scale_factors_x_depths2 =  FrameShared.feature_manager.scale_factors[kps2_levels] * proj_depths2   
                scale_factors_x_depths2_x_ratio_scale_consistency = scale_factors_x_depths2*ratio_scale_consistency      
                bad_scale_consistency = np.logical_or( (scale_factors_x_depths1 > scale_factors_x_depths2_x_ratio_scale_consistency), 
                                                       (scale_factors_x_depths2 > scale_factors_x_depths1_x_ratio_scale_consistency) )   
                
                # combine all checks 
                bad_points = bad_cos_parallaxs | bad_depths1 | bad_depths2 | bad_chis2_1 | bad_chis2_2 | bad_scale_consistency    
                if False: # for debugging      
                    print(f'[add_points] bad_points = {np.sum(bad_points)} of {len(idxs1)}')
                    print(f'\t bad_depths1 = {np.sum(bad_depths1)}')
                    print(f'\t bad_depths2 = {np.sum(bad_depths2)}')      
                    print(f'\t bad_chis2_1 = {np.sum(bad_chis2_1)}')
                    print(f'\t bad_chis2_2 = {np.sum(bad_chis2_2)}')
                    print(f'\t bad_scale_consistency = {np.sum(bad_scale_consistency)}')                   
            
                # end if do_check
                
            # get color patches
            img_coords = np.rint(kf1.kps[idxs1]).astype(np.intp) # image keypoints coordinates 
            # build img patches coordinates 
            delta = Parameters.kSparseImageColorPatchDelta    
            patch_extension = 1 + 2*delta   # patch_extension x patch_extension
            img_pts_start = img_coords - delta           
            img_pts_end   = img_coords + delta
            img_ranges = np.linspace(img_pts_start,img_pts_end,patch_extension,dtype=np.intp)[:,:].T      
            def img_range_elem(ranges,i):      
                return ranges[:,i]                                                  
            
            for i, p in enumerate(points3d):
                if not mask_pts3d[i]:
                    #print('p[%d] not good' % i)
                    continue
                    
                idx1_i = idxs1[i]
                idx2_i = idxs2[i]
                                        
                # perform different required checks before adding the point 
                if do_check:
                    if bad_points[i]:
                        continue                                  

                # get the color of the point  
                try:
                    #color = img1[int(round(kf1.kps[idx1_i, 1])), int(round(kf1.kps[idx1_i, 0]))]
                    #img_pt = np.rint(kf1.kps[idx1_i]).astype(np.int)
                    # color at the point 
                    #color = img1[img_pt[1],img_pt[0]]   
                                     
                    # buils color patch 
                    #color_patch = img1[img_pt[1]-delta:img_pt[1]+delta,img_pt[0]-delta:img_pt[0]+delta]
                    #color = color_patch.mean(axis=0).mean(axis=0)  # compute the mean color in the patch     
                                                                                           
                    # average color in a (1+2*delta) x (1+2*delta) patch  
                    #pt_start = img_pts_start[i]
                    #pt_end   = img_pts_end[i]        
                    #color_patch = img1[pt_start[1]:pt_end[1],pt_start[0]:pt_end[0]] 
                    
                    # average color in a (1+2*delta) x (1+2*delta) patch 
                    img_range = img_range_elem(img_ranges,i) 
                    color_patch = img1[img_range[1][:,np.newaxis],img_range[0]]         
                    #print('color_patch.shape:',color_patch.shape)    
                                                                                  
                    color = cv2.mean(color_patch)[:3]  # compute the mean color in the patch  
                                                                                                        
                except IndexError:
                    Printer.orange('color out of range')
                    color = (255, 0, 0)
                    
                # add the point to this map                 
                mp = MapPoint(p[0:3], color, kf2, idx2_i) 
                self.add_point(mp) # add point to this map 
                mp.add_observation(kf1, idx1_i)                    
                mp.add_observation(kf2, idx2_i)                   
                mp.update_info()
                out_mask_pts3d[i] = True 
                added_map_points.append(mp)
            return len(added_map_points), out_mask_pts3d, added_map_points


    # add new points to the map from 3D point stereo-back-projection
    # points3d is [Nx3]
    def add_stereo_points(self, points3d, mask_pts3d, f: Frame, kf: KeyFrame, idxs, img):
        with self._lock:             
            assert(kf.is_keyframe) 
            
            if mask_pts3d is None:
                mask_pts3d = np.full(points3d.shape[0], True, dtype=bool)
                            
            img_coords = np.rint(kf.kps[idxs]).astype(np.intp) # image keypoints coordinates 
            # build img patches coordinates 
            delta = Parameters.kSparseImageColorPatchDelta    
            patch_extension = 1 + 2*delta   # patch_extension x patch_extension
            img_pts_start = img_coords - delta           
            img_pts_end   = img_coords + delta
            img_ranges = np.linspace(img_pts_start,img_pts_end,patch_extension,dtype=np.intp)[:,:].T      
            def img_range_elem(ranges,i):      
                return ranges[:,i]                                                  
            
            num_added_points = 0
            for i, p in enumerate(points3d):
                if not mask_pts3d[i]:
                    #print('p[%d] not good' % i)
                    continue
                    
                # get the color of the point  
                try:
                    img_range = img_range_elem(img_ranges,i) 
                    color_patch = img[img_range[1][:,np.newaxis],img_range[0]]         
                    #print('color_patch.shape:',color_patch.shape)    
                                                                                  
                    color = cv2.mean(color_patch)[:3]  # compute the mean color in the patch                                                                                      
                except IndexError:
                    Printer.orange('color out of range')
                    color = (255, 0, 0)
                    
                # add the point to this map                 
                mp = MapPoint(p[0:3], color, kf, idxs[i]) 
                
                # we need to add the point both the originary frame and the newly created keyframe
                f.points[idxs[i]] = mp # add point to the frame
                self.add_point(mp) # add point to this map 
                mp.add_observation(kf, idxs[i])                                  
                mp.update_info()
                num_added_points +=1
            return num_added_points

    # remove points which have a big reprojection error 
    def remove_points_with_big_reproj_err(self, points): 
        with self._lock:             
            with self.update_lock: 
                #print('map points: ', sorted([p.id for p in self.points]))
                #print('points: ', sorted([p.id for p in points]))           
                culled_pt_count = 0
                for p in points:
                    # compute reprojection error
                    chi2s = []
                    for f, idx in p.observations():
                        uv = f.kpsu[idx]
                        proj,z = f.project_map_point(p)
                        invSigma2 = FrameShared.feature_manager.inv_level_sigmas2[f.octaves[idx]]
                        err = (proj-uv)
                        chi2s.append(np.inner(err,err)*invSigma2)
                    # cull
                    #mean_chi2 = np.mean(chi2s)
                    if np.mean(chi2s) > Parameters.kChi2Mono:  # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                        culled_pt_count += 1
                        #print('removing point: ',p.id, 'from frames: ', [f.id for f in p.keyframes])
                        self.remove_point(p)
                Printer.blue("# culled map points: ", culled_pt_count)        



    def compute_mean_reproj_error(self, points=None): 
        chi2 = 0
        num_obs = 0
        with self._lock:             
            with self.update_lock:
                if points is None:
                    points = self.points                
                for p in points:
                    # compute reprojection error
                    for f, idx in p.observations():
                        uv = f.kpsu[idx]
                        proj,_ = f.project_map_point(p)
                        invSigma2 = FrameShared.feature_manager.inv_level_sigmas2[f.octaves[idx]]
                        err = (proj-uv)
                        chi2 += np.inner(err,err)*invSigma2
                        num_obs += 1
        return chi2/max(num_obs,1)


    # BA considering all keyframes: 
    # - local keyframes are adjusted, 
    # - other keyframes are fixed
    # - all points are adjusted  
    def optimize(self, local_window=Parameters.kLargeBAWindow, verbose=False, rounds=10, use_robust_kernel=False, do_cull_points = False, abort_flag=g2o.Flag()):            
        err = optimizer_g2o.bundle_adjustment(self.get_keyframes(), self.get_points(), local_window=local_window, \
                                              rounds=rounds, loop_kf_id=0, use_robust_kernel=use_robust_kernel, abort_flag=abort_flag, verbose=verbose)        
        if do_cull_points: 
            self.remove_points_with_big_reproj_err(self.get_points())
        return err


    # local BA: only local keyframes and local points are adjusted
    def locally_optimize(self, kf_ref, verbose = False, rounds=10, abort_flag=g2o.Flag(), mp_abort_flag=None):   
        from local_mapping import print 
        try:
            keyframes, points, ref_keyframes = self.local_map.update(kf_ref)
            print('local optimization window: ', sorted([kf.id for kf in keyframes]))        
            print('                     refs: ', sorted([kf.id for kf in ref_keyframes]))
            print('                   #points: ', len(points))               
            #print('                   points: ', sorted([p.id for p in points]))        
            #err = optimizer_g2o.optimize(frames, points, None, False, verbose, rounds)  
            # NOTE: Why do we want to use parallel multi-processing instead of multi-threading for local BA?
            #       Unfortunately, the GIL does use a SINGLE CPU-core under multi-threading. On the other hand, multi-processing allows to distribute computation over multiple CPU-cores.
            ba_function = optimizer_g2o.local_bundle_adjustment_parallel if Parameters.kUseParallelProcessLBA \
                     else optimizer_g2o.local_bundle_adjustment
            err, ratio_bad_observations = ba_function(keyframes, points, ref_keyframes, False, verbose, rounds, abort_flag=abort_flag, mp_abort_flag=mp_abort_flag, map_lock=self.update_lock)
            Printer.green('local optimization - perc bad observations: %.2f %%' % (ratio_bad_observations*100) )
            return err
        except Exception as e:
            print(f'locally_optimize: EXCEPTION: {e} !!!')
            traceback_details = traceback.format_exc()
            print(f'\t traceback details: {traceback_details}')                        
            return -1 

    def to_json(self, out_json={}):
        with self._lock:          
            with self.update_lock:  
                # static stuff
                out_json['FrameBase._id'] = FrameBase._id
                out_json['MapPointBase._id'] = MapPointBase._id
                
                # non-static stuff
                out_json['frames'] = [f.to_json() for f in self.frames]
                out_json['keyframes'] = [kf.to_json() for kf in self.keyframes if not kf.is_bad]
                out_json['points'] = [p.to_json() for p in self.points if not p.is_bad]
                out_json['keyframe_origins'] = [kf.to_json() for kf in self.keyframe_origins]
                        
                out_json['max_frame_id'] = self.max_frame_id
                out_json['max_point_id'] = self.max_point_id
                out_json['max_keyframe_id'] = self.max_keyframe_id
                
                out_json['viewer_scale'] = self.viewer_scale
        return out_json 
    
    # NOTE: keep this updated according to new data structure changes
    def serialize(self):
        ret_json  = self.to_json()
        return json.dumps(ret_json)

    # NOTE: keep this updated according to new data structure changes
    def from_json(self, loaded_json):
        # static stuff        
        FrameBase._id = loaded_json['FrameBase._id']
        MapPointBase._id = loaded_json['MapPointBase._id']
        
        with self._lock:          
            with self.update_lock:          
                # non-static stuff 
                self.frames = [KeyFrame.from_json(f) if bool(f['is_keyframe']) else Frame.from_json(f) for f in loaded_json['frames']]
                self.keyframes = [KeyFrame.from_json(kf) for kf in loaded_json['keyframes']]
                self.points = [MapPoint.from_json(p) for p in loaded_json['points']]
                                
                self.max_frame_id = loaded_json['max_frame_id']
                self.max_point_id = loaded_json['max_point_id']
                self.max_keyframe_id = loaded_json['max_keyframe_id']
                
                self.viewer_scale = loaded_json['viewer_scale']
                        
                # now replace ids with actual objects in the map assets
                for f in self.frames: 
                    f.replace_ids_with_objects(self.points, self.frames, self.keyframes)
                for kf in self.keyframes: 
                    kf.replace_ids_with_objects(self.points, self.frames, self.keyframes)
                    kf.map = self # set the map
                for p in self.points: 
                    p.replace_ids_with_objects(self.points, self.frames, self.keyframes)
                    p.map = self # set the map        
                    
                # reconstruct the keyframes_map
                self.keyframes_map = {}
                for kf in self.keyframes:
                    self.keyframes_map[kf.id] = kf   
                                
                # recover keyframe origins from keyframe map                
                self.keyframe_origins = [self.keyframes_map[kfjson['id']] for kfjson in loaded_json['keyframe_origins']]            

                self.frames = deque(self.frames, maxlen=kMaxLenFrameDeque) 
                self.keyframes = OrderedSet(self.keyframes)  
                self.points = set(self.points)
                self.keyframe_origins = OrderedSet(self.keyframe_origins)
                
                self.reloaded_session_map_info = ReloadedSessionMapInfo(len(self.keyframes), len(self.points), self.max_point_id, self.max_frame_id, self.max_keyframe_id)
        
     
    def deserialize(self, s):
        ret = json.loads(s)
        self.from_json(ret)
                
    def save(self, filename):
        with open(filename, 'w') as f:
            f.write(self.serialize())
        Printer.green('\t ...map saved to: ', filename)
            
    def load(self, filename):
        with open(filename, 'rb') as f:
            self.deserialize(f.read())
        Printer.green('\t ...map loaded from: ', filename)


# Local map base class 
class LocalMapBase(object):
    def __init__(self, map=None):
        self._lock = RLock()          
        self.map = map     # type: Map
        self.keyframes     = OrderedSet() # collection of local keyframes 
        self.points        = set() # points visible in 'keyframes'  
        self.ref_keyframes = set() # collection of 'covisible' keyframes not in self.keyframes that see at least one point in self.points   


    def reset(self):
        with self._lock:
            self.keyframes.clear()
            self.points.clear()
            self.ref_keyframes.clear()
            
    def reset_session(self, keyframes_to_remove=None, points_to_remove=None):     
        with self._lock:
                if keyframes_to_remove is None and points_to_remove is None:
                    self.reset()
                else:
                    for kf in keyframes_to_remove:
                            self.keyframes.discard(kf)
                            self.ref_keyframes.discard(kf)
                    for p in points_to_remove:
                            self.points.discard(p)
         
    @property    
    def lock(self):  
        return self._lock 
    
    def is_empty(self):
        with self._lock:           
            return len(self.keyframes)==0 
    
    def get_points(self): 
        with self._lock:       
            return self.points.copy()  
        
    def num_points(self):
        with self._lock: 
            return len(self.points)                
                
    def get_keyframes(self): 
        with self._lock:       
            return self.keyframes.copy()       
    
    def num_keyframes(self):
        with self._lock: 
            return len(self.keyframes)  
            
    # given some input local keyframes, get all the viewed points and all the reference keyframes (that see the viewed points but are not in the local keyframes)
    def update_from_keyframes(self, local_keyframes):
        local_keyframes = set([kf for kf in local_keyframes if not kf.is_bad])   # remove possible bad keyframes                         
        ref_keyframes = set()   # reference keyframes: keyframes not in local_keyframes that see points observed in local_keyframes      

        good_points = set([p for kf in local_keyframes for p in kf.get_matched_good_points()])  # all good points in local_keyframes (only one instance per point)
        for p in good_points:     
            # get the keyframes viewing p but not in local_keyframes      
            for kf_viewing_p in p.keyframes():          
                if (not kf_viewing_p.is_bad) and (not kf_viewing_p in local_keyframes):
                    ref_keyframes.add(kf_viewing_p)      
            # debugging stuff            
            # if not any([f in local_frames for f in p.keyframes()]):
            #     Printer.red('point %d without a viewing keyframe in input frames!!' %(p.id))
            #     Printer.red('         keyframes: ',p.observations_string())   
            #     for f in local_frames:
            #         if p in f.get_points(): 
            #             Printer.red('point {} in keyframe {}-{} '.format(p.id,f.id,list(np.where(f.get_points() is p)[0])))            
            #     assert(False)                         

        with self.lock: 
            #local_keyframes = sorted(local_keyframes, key=lambda x:x.id)              
            #ref_keyframes = sorted(ref_keyframes, key=lambda x:x.id)              
            self.keyframes = local_keyframes
            self.points = good_points 
            self.ref_keyframes = ref_keyframes                                                  
        return local_keyframes, good_points, ref_keyframes   
         

    # from a given input frame compute: 
    # - the reference keyframe (the keyframe that sees most map points of the frame)
    # - the local keyframes 
    # - the local points  
    def get_frame_covisibles(self, frame: Frame):
        points = frame.get_matched_good_points()
        #keyframes = self.get_local_keyframes()
        #assert len(points) > 0
        if len(points) == 0:
            Printer.red('LocalMapBase: get_frame_covisibles - frame without points')
        
        # for all map points in frame check in which other keyframes are they seen
        # increase counter for those keyframes            
        viewing_keyframes = [kf for p in points for kf in p.keyframes() if not kf.is_bad]# if kf in keyframes]
        viewing_keyframes = Counter(viewing_keyframes)
        if len(viewing_keyframes) == 0:
            Printer.red('LocalMapBase: get_frame_covisibles - no viewing keyframes')
            return None, None, None     
        
        kf_ref = viewing_keyframes.most_common(1)[0][0]          
        #local_keyframes = viewing_keyframes.keys()
    
        # include also some not-already-included keyframes that are neighbors to already-included keyframes
        for kf in list(viewing_keyframes.keys()):
            second_neighbors = kf.get_best_covisible_keyframes(Parameters.kNumBestCovisibilityKeyFrames)
            viewing_keyframes.update(second_neighbors)
            children = kf.get_children()
            viewing_keyframes.update(children)        
            if len(viewing_keyframes) >= Parameters.kMaxNumOfKeyframesInLocalMap:
                break                 
        
        local_keyframes_counts = viewing_keyframes.most_common(Parameters.kMaxNumOfKeyframesInLocalMap)           
        local_points = set()
        local_keyframes = []
        for kf,c in local_keyframes_counts:
            local_points.update(kf.get_matched_points())
            local_keyframes.append(kf)
        return kf_ref, local_keyframes, local_points  


# Local window map (last N keyframes) 
class LocalWindowMap(LocalMapBase):
    def __init__(self, map=None, local_window=Parameters.kLocalBAWindow):
        super().__init__(map)
        self.local_window = local_window  # length of the local window
              
    def update_keyframes(self, kf_ref=None): 
        with self._lock:         
            # get the last N=local_window keyframes    
            self.keyframes = self.map.get_last_keyframes(self.local_window)
            return self.keyframes
        
    def get_best_neighbors(self, kf_ref=None, N=Parameters.kLocalMappingNumNeighborKeyFrames):      
        return self.map.get_last_keyframes(N)        
    
    # update the local keyframes, the viewed points and the reference keyframes (that see the viewed points but are not in the local keyframes)
    def update(self, kf_ref=None):
        self.update_keyframes(kf_ref)
        return self.update_from_keyframes(self.keyframes)
         
         
# Local map from covisibility graph
class LocalCovisibilityMap(LocalMapBase):
    def __init__(self, map=None):
        super().__init__(map)
                
    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the RLock from the state (don't pickle it)
        if '_lock' in state:
            del state['_lock']       
        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        # Recreate the RLock after unpickling
        self._lock = RLock()
                        
    def update_keyframes(self, kf_ref): 
        with self._lock:         
            assert(kf_ref is not None)
            self.keyframes = OrderedSet()
            self.keyframes.add(kf_ref)
            neighbor_kfs = [kf for kf in kf_ref.get_covisible_keyframes() if not kf.is_bad]
            self.keyframes.update(neighbor_kfs)
            return self.keyframes
        
    def get_best_neighbors(self, kf_ref, N=Parameters.kLocalMappingNumNeighborKeyFrames): 
        return kf_ref.get_best_covisible_keyframes(N)               
    
    # update the local keyframes, the viewed points and the reference keyframes (that see the viewed points but are not in the local keyframes)
    def update(self, kf_ref):
        self.update_keyframes(kf_ref)
        return self.update_from_keyframes(self.keyframes)         
  
