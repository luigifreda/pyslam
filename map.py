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
import json
import math 
import cv2

from collections import Counter, deque

from ordered_set import OrderedSet # from https://pypi.org/project/ordered-set/

from threading import RLock, Thread

from utils_geom import poseRt, add_ones, add_ones_1D
from parameters import Parameters 
from frame import Frame
from keyframe import KeyFrame
from map_point import MapPoint

from utils import Printer

import g2o
import optimizer_g2o 


kVerbose = True 
kMaxLenFrameDeque = 20


if not kVerbose:
    def print(*args, **kwargs):
        pass 
         
        
class Map(object):
    def __init__(self):
        self._lock = RLock()   
        self._update_lock = RLock()  
        
        self.frames = deque(maxlen=kMaxLenFrameDeque)  # deque with max length, it is thread-safe 
        self.keyframes = OrderedSet()  
        self.points = set()
        
        self.max_frame_id = 0     # 0 is the first frame id
        self.max_point_id = 0     # 0 is the first point id
        self.max_keyframe_id = 0  # 0 is the first keyframe id

        # local map 
        #self.local_map = LocalWindowMap(map=self)
        self.local_map = LocalCovisibilityMap(map=self)

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
            return self.frames[idx] 
                
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
                
    def num_keyframes(self):
        with self._lock: 
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
            self.max_keyframe_id += 1                                  
            return ret    
    
    def remove_keyframe(self, keyframe): 
        with self._lock:
            assert(keyframe.is_keyframe)                               
            try: 
                self.keyframes.remove(keyframe) 
            except: 
                pass     
    
    def num_keyframes(self):      
        return self.max_keyframe_id


    def draw_feature_trails(self, img):
        if len(self.frames) > 0:
            img_draw = self.frames[-1].draw_all_feature_trails(img)
            return img_draw
        return img


    # add new points to the map from 3D point estimations, frames and pairwise matches
    # points3d is [Nx3]
    def add_points(self, points3d, mask_pts3d, kf1, kf2, idxs1, idxs2, img1, do_check=True, cos_max_parallax=Parameters.kCosMaxParallax):
        with self._lock:             
            assert(kf1.is_keyframe and  kf2.is_keyframe) # kf1 and kf2 must be keyframes 
            assert(points3d.shape[0] == len(idxs1))
            assert(len(idxs2) == len(idxs1))    
            
            added_points = []
            out_mask_pts3d = np.full(points3d.shape[0], False, dtype=bool)
            if mask_pts3d is None:
                mask_pts3d = np.full(points3d.shape[0], True, dtype=bool)
            
            ratio_scale_consistency = Parameters.kScaleConsistencyFactor * Frame.feature_manager.scale_factor 
            
            if do_check:            
                # project points
                uvs1, depths1 = kf1.project_points(points3d)    
                bad_depths1 =  depths1 <= 0  
                uvs2, depths2 = kf2.project_points(points3d)    
                bad_depths2 =  depths2 <= 0                   
                
                # compute back-projected rays (unit vectors) 
                rays1 = np.dot(kf1.Rwc, add_ones(kf1.kpsn[idxs1]).T).T
                norm_rays1 = np.linalg.norm(rays1, axis=-1, keepdims=True)                  
                rays1 /= norm_rays1                      
                rays2 = np.dot(kf2.Rwc, add_ones(kf2.kpsn[idxs2]).T).T     
                norm_rays2 = np.linalg.norm(rays2, axis=-1, keepdims=True)  
                rays2 /= norm_rays2 
                
                # compute dot products of rays                              
                cos_parallaxs = np.sum(rays1 * rays2, axis=1)  
                bad_cos_parallaxs = np.logical_or(cos_parallaxs < 0, cos_parallaxs > cos_max_parallax)           
                
                # compute reprojection errors              
                errs1 = uvs1 - kf1.kpsu[idxs1]    
                errs1_sqr = np.sum(errs1 * errs1, axis=1)  # squared reprojection errors 
                kps1_levels = kf1.octaves[idxs1]
                invSigmas2_1 = Frame.feature_manager.inv_level_sigmas2[kps1_levels] 
                chis2_1 = errs1_sqr * invSigmas2_1         # chi square 
                bad_chis2_1 = chis2_1 > Parameters.kChi2Mono   # chi-square 2 DOFs  (Hartley Zisserman pg 119)                   
                                                                 
                errs2 = uvs2 - kf2.kpsu[idxs2]             
                errs2_sqr = np.sum(errs2 * errs2, axis=1) # squared reprojection errors        
                kps2_levels = kf2.octaves[idxs2]
                invSigmas2_2 = Frame.feature_manager.inv_level_sigmas2[kps2_levels] 
                chis2_2 = errs2_sqr * invSigmas2_2        # chi square 
                bad_chis2_2 = chis2_2 > Parameters.kChi2Mono  # chi-square 2 DOFs  (Hartley Zisserman pg 119)                      
                
                # scale consistency 
                scale_factors_x_depths1 =  Frame.feature_manager.scale_factors[kps1_levels] * depths1
                scale_factors_x_depths1_x_ratio_scale_consistency = scale_factors_x_depths1*ratio_scale_consistency                             
                scale_factors_x_depths2 =  Frame.feature_manager.scale_factors[kps2_levels] * depths2   
                scale_factors_x_depths2_x_ratio_scale_consistency = scale_factors_x_depths2*ratio_scale_consistency      
                bad_scale_consistency = np.logical_or( (scale_factors_x_depths1 > scale_factors_x_depths2_x_ratio_scale_consistency), 
                                                       (scale_factors_x_depths2 > scale_factors_x_depths1_x_ratio_scale_consistency) )   
                
                # combine all checks 
                bad_points = bad_cos_parallaxs | bad_depths1 | bad_depths2 | bad_chis2_1 | bad_chis2_2 | bad_scale_consistency                                                           
            
            img_coords = np.rint(kf1.kps[idxs1]).astype(np.intp) # image keypoints coordinates 
            # build img patches coordinates 
            delta = Parameters.kColorPatchDelta    
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
                                        
                    # check parallax is large enough (this is going to filter out all points when the inter-frame motion is almost zero)           
                    # ray1 = np.dot(kf1.Rwc, add_ones_1D(kf1.kpsn[idx1_i]))
                    # ray2 = np.dot(kf2.Rwc, add_ones_1D(kf2.kpsn[idx2_i]))
                    # cos_parallax = ray1.dot(ray2) / (np.linalg.norm(ray1) * np.linalg.norm(ray2))
                    # if cos_parallax < 0 or cos_parallax > cos_max_parallax:
                    #     #print('p[',i,']: ',p,' not enough parallax: ', cos_parallaxs[i]) 
                    #     continue

                    # check points are visible on f1
                    # uv1, depth1 = kf1.project_point(p)
                    # is_visible1  = kf1.is_in_image(uv1, depth1) # N.B.: is_in_image() check is redundant since we check the reproj errror 
                    # if not is_visible1:
                    #     continue   
                    
                    # check points are visible on f2                       
                    # uv2, depth2 = kf2.project_point(p) 
                    # is_visible2  = kf2.is_in_image(uv2, depth2)  # N.B.: is_in_image() check is redundant since we check the reproj errror 
                    # if not is_visible2:
                    #     continue                                      
                        
                    # check reprojection error on f1
                    #kp1_level = kf1.octaves[idx1_i]
                    #invSigma2_1 = Frame.feature_manager.inv_level_sigmas2[kp1_level]                
                    # err1 = uvs1[i] - kf1.kpsu[idx1_i]       
                    # chi2_1 = np.inner(err1,err1)*invSigma2_1 
                    # if chi2_1 > Parameters.kChi2Mono: # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                    #     continue                                 
                    
                    # check reprojection error on f2     
                    # kp2_level = kf2.octaves[idx2_i]                          
                    # invSigma2_2 = Frame.feature_manager.inv_level_sigmas2[kp2_level]                 
                    # err2 = uvs2[i] - kf2.kpsu[idx2_i]         
                    # chi2_2 = np.inner(err2,err2)*invSigma2_2                             
                    # if chi2_2 > Parameters.kChi2Mono: # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                    #     continue               
                    
                    #check scale consistency 
                    # scale_factor_x_depth1 =  Frame.feature_manager.scale_factors[kps1_levels[i]] * depths1[i]
                    # scale_factor_x_depth2 =  Frame.feature_manager.scale_factors[kps2_levels[i]] * depths2[i]
                    # if (scale_factor_x_depth1 > scale_factor_x_depth2*ratio_scale_consistency) or \
                    #    (scale_factor_x_depth2 > scale_factor_x_depth1*ratio_scale_consistency):
                    #     continue                                    

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
                added_points.append(mp)
            return len(added_points), out_mask_pts3d, added_points


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
                        proj,_ = f.project_map_point(p)
                        invSigma2 = Frame.feature_manager.inv_level_sigmas2[f.octaves[idx]]
                        err = (proj-uv)
                        chi2s.append(np.inner(err,err)*invSigma2)
                    # cull
                    mean_chi2 = np.mean(chi2s)
                    if np.mean(chi2s) > Parameters.kChi2Mono:  # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                        culled_pt_count += 1
                        #print('removing point: ',p.id, 'from frames: ', [f.id for f in p.keyframes])
                        self.remove_point(p)
                Printer.blue("# culled map points: ", culled_pt_count)        


    # BA considering all keyframes: 
    # - local keyframes are adjusted, 
    # - other keyframes are fixed
    # - all points are adjusted  
    def optimize(self, local_window=Parameters.kLargeBAWindow, verbose=False, rounds=10, use_robust_kernel=False, do_cull_points = False, abort_flag=g2o.Flag()):            
        err = optimizer_g2o.bundle_adjustment(self.get_keyframes(), self.get_points(), local_window = local_window, verbose = verbose, rounds = rounds, use_robust_kernel=False, abort_flag=abort_flag)        
        if do_cull_points: 
            self.remove_points_with_big_reproj_err(self.get_points())
        return err


    # local BA: only local keyframes and local points are adjusted
    def locally_optimize(self, kf_ref, verbose = False, rounds=10, abort_flag=g2o.Flag()):    
        keyframes, points, ref_keyframes = self.local_map.update(kf_ref)
        print('local optimization window: ', sorted([kf.id for kf in keyframes]))        
        print('                     refs: ', sorted([kf.id for kf in ref_keyframes]))
        print('                   #points: ', len(points))               
        #print('                   points: ', sorted([p.id for p in points]))        
        #err = optimizer_g2o.optimize(frames, points, None, False, verbose, rounds)  
        err, ratio_bad_observations = optimizer_g2o.local_bundle_adjustment(keyframes, points, ref_keyframes, False, verbose, rounds, abort_flag=abort_flag, map_lock=self.update_lock)
        Printer.green('local optimization - perc bad observations: %.2f %%' % (ratio_bad_observations*100) )              
        return err 


    # FIXME: to be updated according to new data structure changes
    def serialize(self):
        ret = {}
        # ret['points'] = [{'id': p.id, 'pt': p.pt.tolist(
        # ), 'color': p.color.tolist()} for p in self.points]
        # ret['keyframes'] = []
        # for f in self.keyframes:
        #     ret['frames'].append({
        #         'id': f.id, 'K': f.K.tolist(), 'pose': f.pose.tolist(), 'H': f.H, 'W': f.W,
        #         'kpus': f.kpus.tolist(), 'des': f.des.tolist(),
        #         'pts': [p.id if p is not None else -1 for p in f.pts]
        #         })
        # ret['max_frame_id'] = self.max_frame_id
        # ret['max_point_id'] = self.max_point_id
        return json.dumps(ret)
        pass 


    # FIXME: to be updated according to new data structure changes
    def deserialize(self, s):
        ret = json.loads(s)
        # self.max_frame_id = ret['max_frame_id']
        # self.max_point_id = ret['max_point_id']
        # self.points = []
        # self.keyframes = []

        # pids = {}
        # for p in ret['points']:
        #     pp = MapPoint(p['pt'], p['color'], p['id'])
        #     self.add_point(pt) # add point to this map 
        #     self.points.append(pp)
        #     pids[p['id']] = pp

        # for f in ret['keyframes']:
        #     ff = Frame(None, f['K'], f['pose'], f['id'])
        #     self.add_frame_view(ff)
        #     ff.W, ff.H = f['W'], f['H']
        #     ff.kpsu = np.array(f['kpus'])
        #     ff.des = np.array(f['des'])
        #     ff.points = [None] * len(ff.kpsu)
        #     for i, p in enumerate(f['pts']):
        #         if p != -1:
        #             ff.points[i] = pids[p]
        #     self.keyframes.append(ff)
        pass




# Local map base class 
class LocalMapBase(object):
    def __init__(self, map=None):
        self._lock = RLock()          
        self.map = map   
        self.keyframes     = OrderedSet() # collection of local keyframes 
        self.points        = set() # points visible in 'keyframes'  
        self.ref_keyframes = set() # collection of 'covisible' keyframes not in self.keyframes that see at least one point in self.points   

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
    def get_frame_covisibles(self, frame):
        points = frame.get_matched_good_points()
        #keyframes = self.get_local_keyframes()
        #assert len(points) > 0
        if len(points) == 0:
            Printer.red('get_frame_covisibles - frame with not points')
        
        # for all map points in frame check in which other keyframes are they seen
        # increase counter for those keyframes            
        viewing_keyframes = [kf for p in points for kf in p.keyframes() if not kf.is_bad]# if kf in keyframes]
        viewing_keyframes = Counter(viewing_keyframes)
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
        self.local_window = local_window
              
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
  