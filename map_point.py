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

import math 
import time
import numpy as np
from threading import RLock, Thread

from utils_geom import poseRt, add_ones, normalize_vector, normalize_vector2
from frame import Frame
from utils import Printer

from parameters import Parameters


class MapPointBase(object):
    _id = 0                 # shared point counter 
    _id_lock = RLock()      # shared lock for id   
    def __init__(self, id=None):
        if id is not None: 
            self.id = id 
        else: 
            with MapPointBase._id_lock:
                self.id = MapPointBase._id
                MapPointBase._id += 1           
        
        self._lock_pos = RLock() 
        self._lock_features = RLock()         
        
        self.map = None  # this is used by the object for automatically removing itself from the map when it becomes bas (see below)         
        
        self._observations = dict() # keyframe observations (used by mapping methods)
                                    # for kf, kidx in self._observations.items(): kf.points[kidx] = this point
        
        self._frame_views = dict()  # frame observations (used for drawing the tracking keypoint trails, frame by frame)
                                    # for f, idx in self._frame_views.items(): f.points[idx] = this point
                
        self._is_bad = False        # a map point becomes bad when its num_observations < 2 (cannot be considered for bundle ajustment or other related operations)
        self._num_observations = 0   # number of keyframe observations 
        self.num_times_visible = 1  # number of times the point is visible in the camera 
        self.num_times_found = 1    # number of times the point was actually matched and not rejected as outlier by the pose optimization in Tracking.track_local_map()
        self.last_frame_id_seen =-1 # last frame id in which this point was seen    
        
        #self.is_replaced = False    # is True when the point was replaced by another point 
        self.replacement = None     # replacing point  

    def __hash__(self):
        return self.id

    def __eq__(self, rhs):
        return (isinstance(rhs, MapPointBase) and  self.id == rhs.id)
    
    def __lt__(self, rhs):
        return self.id < rhs.id
    
    def __le__(self, rhs):
        return self.id <= rhs.id
    
    def observations_string(self):
        obs = sorted([(kf.id, kidx, kf.get_point_match(kidx)!=None) for kf,kidx in self.observations()],key=lambda x:x[0])
        return 'observations: ' + str(obs)
    
    def frame_views_string(self):
        obs = sorted([(f.id, idx, f.get_point_match(idx)!=None) for f,idx in self.frame_views()],key=lambda x:x[0])
        return 'views: ' + str(obs)    
    
    def __str__(self):
        #return str(self.__class__) + ": " + str(self.__dict__)
        return 'MapPoint ' + str(self.id)  + ' { ' + self.observations_string() + ', ' + self.frame_views_string() + ' }'   

    # return a copy of the dictionary’s list of (key, value) pairs
    def observations(self):
        with self._lock_features:
            return list(self._observations.items())   # https://www.python.org/dev/peps/pep-0469/
        
    # return an iterator of the dictionary’s list of (key, value) pairs
    # NOT thread-safe 
    def observations_iter(self):
        return iter(self._observations.items())       # https://www.python.org/dev/peps/pep-0469/

    # return a copy of the dictionary’s list of keys
    def keyframes(self):
        with self._lock_features:
            return list(self._observations.keys())
        
    # return an iterator of the dictionary’s list of keys
    # NOT thread-safe         
    def keyframes_iter(self):
            return iter(self._observations.keys())      
        
    def is_in_keyframe(self, keyframe):
        assert(keyframe.is_keyframe)
        with self._lock_features:
            return (keyframe in self._observations)      
        
    def get_observation_idx(self, keyframe):   
        assert(keyframe.is_keyframe)
        with self._lock_features:
            return self._observations[keyframe]                 

    def add_observation(self, keyframe, idx):         
        assert(keyframe.is_keyframe)   
        with self._lock_features:            
            if keyframe not in self._observations:
                keyframe.set_point_match(self, idx) # add point association in keyframe 
                self._observations[keyframe] = idx
                self._num_observations += 1      
                return True 
            #elif self._observations[keyframe] != idx:     # if the keyframe is already there but it is incoherent then fix it!
            #    self._observations[keyframe] = idx   
            #    return True              
            else: 
                return False                   

    def remove_observation(self, keyframe, idx=None):        
        assert(keyframe.is_keyframe)          
        with self._lock_features:                                
            # remove point association      
            if idx is not None:  
                if __debug__:   
                    assert(self == keyframe.get_point_match(idx))                
                keyframe.remove_point_match(idx)       
                if __debug__:
                    assert(not self in keyframe.points)   # checking there are no multiple instances                  
            else: 
                keyframe.remove_point(self)                
            try:
                del self._observations[keyframe]
                self._num_observations = max(0, self._num_observations-1)
                self._is_bad = (self._num_observations <= 2)  
                if self.kf_ref is keyframe: 
                    self.kf_ref = list(self._observations.keys())[0]
                # if bad remove it from map 
                if self._is_bad and self.map is not None:                    
                    self.map.remove_point(self)                      
            except KeyError:
                pass                
                            
    # return a copy of the dictionary’s list of (key, value) pairs
    def frame_views(self):
        with self._lock_features:
            return list(self._frame_views.items())
        
    # return an iterator of the dictionary’s list of (key, value) pairs
    # NOT thread-safe         
    def frame_views_iter(self):
            return iter(self._frame_views.items())    
        
    # return a copy of the dictionary’s list of keys
    def frames(self):
        with self._lock_features:
            return list(self._frame_views.keys())
        
    # return an iterator of the dictionary’s list of keys
    # NOT thread-safe         
    def frames_iter(self):
            return iter(self._frame_views.keys())            
                      
    def is_in_frame(self, frame):
        with self._lock_features:
            return (frame in self._frame_views)      
                                             
    def add_frame_view(self, frame, idx):
        assert(not frame.is_keyframe)
        with self._lock_features:
            if frame not in self._frame_views:  # do not allow a point to be matched to diffent keypoints 
                frame.set_point_match(self, idx)            
                self._frame_views[frame] = idx
                return True 
            #elif self._frame_views[keyframe] != idx:     # if the frame is already there but it is incoherent then fix it!
            #   self._frame_views[keyframe] = idx                 
            #   return True    
            else:
                return False                 

    def remove_frame_view(self, frame, idx=None): 
        assert(not frame.is_keyframe)        
        with self._lock_features:
            # remove point from frame     
            if idx is not None:
                if __debug__: 
                    assert(self == frame.get_point_match(idx))                     
                frame.remove_point_match(idx)   
                if __debug__:
                    assert(not self in frame.get_points())   # checking there are no multiple instances 
            else: 
                frame.remove_point(self)  # remove all match instances                                    
            try:
                del self._frame_views[frame]
            except KeyError:
                pass    
            
    @property
    def is_bad(self):
        with self._lock_features:
            with self._lock_pos:    
                return self._is_bad                

    @property
    def num_observations(self):
        with self._lock_features:
            return self._num_observations
            
    def increase_visible(self, num_times=1):
        with self._lock_features:
            self.num_times_visible += num_times        
            
    def increase_found(self, num_times=1):
        with self._lock_features:
            self.num_times_found += num_times      
            
    def get_found_ratio(self):
        with self._lock_features:       
            return self.num_times_found/self.num_times_visible                              


# A Point is a 3-D point in the world
# Each Point is observed in multiple Frames
class MapPoint(MapPointBase):
    global_lock = RLock()      # shared global lock for blocking point position update     
    def __init__(self, position, color, keyframe=None, idxf=None, id=None):
        super().__init__(id)
        self._pt = np.array(position)

        self.color = color
            
        self.des = None  # best descriptor (continuously updated)
        self._min_distance, self._max_distance = 0, float('inf')  # depth infos 
        self.normal = np.array([0,0,1])   # just a default 3D vector 
                      
        self.kf_ref = keyframe      
        self.first_kid = -1     # first observation keyframe id 
        #self.idxf_ref = idxf            
        if keyframe is not None:
            self.first_kid = keyframe.kid 
            self.des = keyframe.des[idxf]  
            # update normal and depth infos 
            po = (self._pt - self.kf_ref.Ow)
            self.normal, dist = normalize_vector(po)
            level = keyframe.octaves[idxf]
            level_scale_factor =  Frame.feature_manager.scale_factors[level]
            self._max_distance = dist * level_scale_factor
            self._min_distance = self._max_distance / Frame.feature_manager.scale_factors[Frame.feature_manager.num_levels-1]     
  
        self.num_observations_on_last_update_des = 1       # must be 1!    
        self.num_observations_on_last_update_normals = 1   # must be 1!       
        
    @property  
    def pt(self):
        with self._lock_pos:   
            return self._pt  
        
    def homogeneous(self):
        with self._lock_pos:         
            #return add_ones(self._pt)
            return np.concatenate([self._pt,np.array([1.0])], axis=0)        
         
    def update_position(self, position):
        with self.global_lock:           
            with self._lock_pos:   
                self._pt = position 
                                          
    @property
    def max_distance(self):
        with self._lock_pos:           
            #return Frame.feature_manager.scale_factor * self._max_distance  # give it one level of margin (can be too much with scale factor = 2)
            return Parameters.kMaxDistanceToleranceFactor * self._max_distance  
    
    @property
    def min_distance(self):
        with self._lock_pos:           
            #return Frame.feature_manager.inv_scale_factor * self._min_distance  # give it one level of margin (can be too much with scale factor = 2)   
            return Parameters.kMinDistanceToleranceFactor * self._min_distance     
        
    def get_reference_keyframe(self):
        with self._lock_features:
            return kf_ref            
                                 
    # return array of corresponding descriptors 
    def descriptors(self):
        with self._lock_features:        
            return [kf.des[idx] for kf,idx in self._observations.items()]

    # minimum distance between input descriptor and map point corresponding descriptors 
    def min_des_distance(self, descriptor):
        with self._lock_features:             
            #return min([Frame.descriptor_distance(d, descriptor) for d in self.descriptors()])
            return Frame.descriptor_distance(self.des, descriptor)
    
    def delete(self):                
        with self._lock_features:
            with self._lock_pos:
                #if __debug__:          
                #    Printer.red('deleting ', self, ' is_replaced: ', self.replacement != None)                
                self._is_bad = True 
                self._num_observations = 0                   
                observations = list(self._observations.items()) 
                #frame_views = list(self._frame_views.items())
                self._observations.clear()        
                #self._frame_views.clear()                             
        for kf,idx in observations:
            kf.remove_point_match(idx)         
        #for f,idx in _frame_views:
        #    f.remove_point_match(idx)                                              
        del self  # delete if self is the last reference 
                
    def set_bad(self):
        with self._lock_features:
            with self._lock_pos:
                #if __debug__:          
                #    Printer.red('setting bad ', self, ' is_replaced: ', self.replacement != None)                
                self._is_bad = True 
                self._num_observations = 0                   
                observations = list(self._observations.items()) 
                self._observations.clear()                   
        for kf,idx in observations:
            kf.remove_point_match(idx)
        if self.map is not None:                    
            self.map.remove_point(self)     
                            
    def get_replacement(self): 
        with self._lock_features:
            with self._lock_pos:
                return self.replacement        
            
    def get_normal(self): 
        with self._lock_pos:
            return self.normal                    
                            
    # replace this point with map point p 
    def replace_with(self, p):         
        if p.id == self.id: 
            return 
        #if __debug__:               
        #    Printer.orange('replacing ', self, ' with ', p)               
        observations, num_times_visible, num_times_found = None, 0, 0 
        with self._lock_features:
            with self._lock_pos:   
                observations = list(self._observations.items()) 
                self._observations.clear()     
                num_times_visible = self.num_times_visible
                num_times_found = self.num_times_found      
                self._is_bad = True  
                self._num_observations = 0  
                #self.is_replaced = True    # tell the delete() method not to remove observations and frame views   
                self.replacement = p                 
            
        # replace point observations in keyframes
        for kf, kidx in observations: # we have kf.get_point_match(kidx) = self 
            # if p.is_in_keyframe(kf): 
            #     # point p is already in kf => just remove this point match from kf 
            #     # (do NOT remove the observation otherwise self._num_observations is decreased in the replacement)
            #     kf.remove_point_match(kidx)
            # else: 
            #     # point p is not in kf => add new observation in p
            #     kf.replace_point_match(p,kidx)                 
            #     p.add_observation(kf,kidx)
            if p.add_observation(kf,kidx): 
                # point p was NOT in kf => added new observation in p
                kf.replace_point_match(p,kidx)                  
            else:                
                # point p is already in kf => just remove this point match from kf 
                # (do NOT remove the observation otherwise self._num_observations is decreased in the replacement)
                kf.remove_point_match(kidx)
                #if p.get_observation_idx(kf) != kidx:
                #    kf.remove_point_match(kidx)  
                #else:
                #    kf.replace_point_match(p,kidx)      
                                                                               
        p.increase_visible(num_times_visible)
        p.increase_found(num_times_found)        
        #p.update_info()    
        p.update_best_descriptor(force=True)    
    
        # replace point observations in frames (done by frame.check_replaced_map_points())
        # for f, idx in frame_views:   # we have f.get_point_match(idx) = self
        #     if p.is_in_frame(f): 
        #         if not f.is_keyframe: # if not already managed above in keyframes
        #             # point p is already in f => just remove this point match from f
        #             f.remove_point_match(idx)
        #     else: 
        #         # point p is not in f => add new frame view in p        
        #         f.replace_point_match(p,idx)         
        #         p.add_frame_view(f,idx)                                                               
                     
        self.map.remove_point(self)     
        #if __debug__: 
        #    Printer.green('after replacement ', p)                     
        

    # update normal and depth representations     
    def update_normal_and_depth(self, frame=None, idxf=None,force=False):
        skip = False  
        with self._lock_features:
            with self._lock_pos:   
                if self._is_bad:
                    return                 
                if self._num_observations > self.num_observations_on_last_update_normals or force:   # implicit if self._num_observations > 1          
                    self.num_observations_on_last_update_normals = self._num_observations 
                    observations = list(self._observations.items())
                    kf_ref = self.kf_ref 
                    idx_ref = self._observations[kf_ref]
                    position = self._pt.copy() 
                else: 
                    skip = True 
        if skip or len(observations)==0:
            return 
                     
        normals = np.array([normalize_vector2(position-kf.Ow) for kf,idx in observations])
        normal = normalize_vector2(np.mean(normals,axis=0))
        #print('normals: ', normals)
        #print('mean normal: ', self.normal)        
  
        level = kf_ref.octaves[idx_ref]
        level_scale_factor = Frame.feature_manager.scale_factors[level]
        dist = np.linalg.norm(position-kf_ref.Ow)
        
        with self._lock_pos:           
            self._max_distance = dist * level_scale_factor
            self._min_distance = self._max_distance / Frame.feature_manager.scale_factors[Frame.feature_manager.num_levels-1]            
            self.normal = normal         
                    
        
    def update_best_descriptor(self,force=False):
        skip = False 
        with self._lock_features:
            if self._is_bad:
                return                          
            if self._num_observations > self.num_observations_on_last_update_des or force:    # implicit if self._num_observations > 1   
                self.num_observations_on_last_update_des = self._num_observations      
                observations = list(self._observations.items())
            else: 
                skip = True 
        if skip:
            return 
        descriptors = [kf.des[idx] for kf,idx in observations if not kf.is_bad]
        N = len(descriptors)
        if N > 2:
            #median_distances = [ np.median([Frame.descriptor_distance(d, descriptors[i]) for d in descriptors]) for i in range(N) ]
            median_distances = [ np.median(Frame.descriptor_distances(descriptors[i], descriptors)) for i in range(N)]
            with self._lock_features:            
                self.des = descriptors[np.argmin(median_distances)].copy()
            #print('descriptors: ', descriptors)
            #print('median_distances: ', median_distances)
            #print('des: ', self.des)        


    def update_info(self):
        with self._lock_features:
            with self._lock_pos:            
                self.update_normal_and_depth()
                self.update_best_descriptor()     
                             
    # predict detection level from map point distance    
    def predict_detection_level(self, dist):
        with self._lock_pos:             
            ratio = self._max_distance/dist
        level = math.ceil(math.log(ratio)/Frame.feature_manager.log_scale_factor)
        if level < 0:
            level = 0
        elif level >= Frame.feature_manager.num_levels:
            level = Frame.feature_manager.num_levels-1
        return level
    
    
# predict detection levels from map point distances    
def predict_detection_levels(points, dists):    
    assert(len(points)==len(dists)) 
    max_distances = np.array([p._max_distance for p in points])  
    ratios = max_distances/dists
    levels = np.ceil(np.log(ratios)/Frame.feature_manager.log_scale_factor).astype(np.intp)
    levels = np.clip(levels,0,Frame.feature_manager.num_levels-1)
    return levels    