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

import cv2
import numpy as np

from scipy.spatial import cKDTree

from threading import RLock

from parameters import Parameters  
from utils import Printer
from collections import defaultdict, OrderedDict, Counter

from frame import Frame 
from camera_pose import CameraPose


class KeyFrameGraph(object):
    def __init__(self):
        self._lock_connections = RLock()           
        # spanning tree
        self.init_parent = False  # is parent initialized? 
        self.parent = None 
        self.children = set()
        # loop edges  
        self.loop_edges = set() 
        self.not_to_erase = False   # if there is a loop edge then you cannot erase this keyframe 
        # covisibility graph 
        self.connected_keyframes_weights = Counter() #defaultdict(int) 
        self.ordered_keyframes_weights = OrderedDict()  # ordered list of connected keyframes (on the basis of the number of map points with this keyframe)
        # 
        self.is_first_connection=True 
        
    # ===============================    
    # spanning tree     
    def add_child(self, keyframe):
        with self._lock_connections:
            self.children.add(keyframe)
    
    def erase_child(self, keyframe):
        with self._lock_connections:
            try: 
                self.children.remove(keyframe)     
            except:
                pass 
            
    def set_parent(self, keyframe):
        with self._lock_connections:
            if self == keyframe: 
                if __debug__:
                    Printer.orange('KeyFrameGraph.set_parent - trying to set self as parent')
                return 
            self.parent = keyframe 
            keyframe.add_child(self)
        
    def get_children(self):
        with self._lock_connections:
            return self.children.copy()
                    
    def get_parent(self):
        with self._lock_connections:
            return self.parent        
        
    def has_child(self, keyframe):            
        with self._lock_connections:
            return keyframe in self.children        
        
               
    # ===============================    
    # loop edges                    
    def add_loop_edge(self, keyframe):
        with self._lock_connections:
            self.not_to_erase = True 
            self.loop_edges.add(keyframe)
        
    def get_loop_edges(self):
        with self._lock_connections:        
            return self.loop_edges.copy()
        
        
    # ===============================    
    # covisibility           
    
    def reset_covisibility(self): 
        self.connected_keyframes_weights = Counter() 
        self.ordered_keyframes_weights = OrderedDict()    
                
    def add_connection(self, keyframe, weigth):
        with self._lock_connections: 
            self.connected_keyframes_weights[keyframe]=weigth
            self.update_best_covisibles()
            
    def erase_connection(self, keyframe):
        with self._lock_connections: 
            try:
                del self.connected_keyframes_weights[keyframe]   
                self.update_best_covisibles()
            except: 
                pass                  
                
    def update_best_covisibles(self):
        with self._lock_connections:         
            self.ordered_keyframes_weights = OrderedDict(sorted(self.connected_keyframes_weights.items(), key=lambda x: x[1], reverse=True)) # order by value (decreasing order)
     
    # get a list of all the keyframe that shares points  
    def get_connected_keyframes(self): 
        with self._lock_connections:          
            return list(self.connected_keyframes_weights.keys()) # returns a copy 
        
    # get an ordered list of covisible keyframes     
    def get_covisible_keyframes(self):
        with self._lock_connections:                  
            return list(self.ordered_keyframes_weights.keys()) # returns a copy   
        
    # get an ordered list of covisible keyframes     
    def get_best_covisible_keyframes(self,N):
        with self._lock_connections:                  
            return list(self.ordered_keyframes_weights.keys())[:N] # returns a copy         
        
    def get_covisible_by_weight(self,weight):  
        with self._lock_connections:          
              return [kf for kf,w in self.ordered_keyframes_weights.items() if w > weight]
          
    def get_weight(self,keyframe): 
        with self._lock_connections:  
            return self.connected_keyframes_weights[keyframe]  
                    

class KeyFrame(Frame,KeyFrameGraph):
    def __init__(self, frame, img=None):
        KeyFrameGraph.__init__(self)
        Frame.__init__(self, img=None, camera=frame.camera, pose=frame.pose, id=frame.id, timestamp=frame.timestamp)   # here we MUST have img=None in order to avoid recomputing keypoint info
                
        if frame.img is not None: 
            self.img = frame.img  # this is already a copy of an image 
        else:
            if img is not None: 
                self.img = img.copy()
                
        self.map = None 
                
        self.is_keyframe = True  
        self.kid = None           # keyframe id 
        
        self._is_bad = False 
        self.to_be_erased = False 
        
        # pose relative to parent (this is computed when bad flag is activated)
        self._pose_Tcp = CameraPose() 

        # share keypoints info with frame (these are computed once for all on frame initialization and they are not changed anymore)
        self.kps     = frame.kps      # keypoint coordinates                  [Nx2]
        self.kpsu    = frame.kpsu     # [u]ndistorted keypoint coordinates    [Nx2]
        self.kpsn    = frame.kpsn     # [n]ormalized keypoint coordinates     [Nx2] (Kinv * [kp,1])    
        self.octaves = frame.octaves  # keypoint octaves                      [Nx1]
        self.sizes   = frame.sizes    # keypoint sizes                        [Nx1] 
        self.angles  = frame.angles   # keypoint angles                       [Nx1] 
        self.des     = frame.des      # keypoint descriptors                  [NxD] where D is the descriptor length 
        
        if hasattr(frame, '_kd'):     
            self._kd = frame._kd 
        else: 
            Printer.orange('KeyFrame %d computing kdtree for input frame %d'%(self.id,frame.id))
            self._kd = cKDTree(self.kpsu)
    
        # map points information arrays (copy points coming from frame)
        self.points   = frame.get_points()     # map points => self.points[idx] is the map point matched with self.kps[idx] (if is not None)
        self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)     # used just in propagate_map_point_matches()   
        
    # associate matched map points to observations    
    def init_observations(self):
        with self._lock_features:           
            for idx,p in enumerate(self.points):
                if p is not None and not p.is_bad:  
                    if p.add_observation(self, idx):
                        p.update_info()                 
        
    def update_connections(self):
        # for all map points of this keyframe check in which other keyframes they are seen
        # build a counter for these other keyframes    
        points = self.get_matched_good_points()
        assert len(points) > 0
        viewing_keyframes = [kf for p in points for kf in p.keyframes() if kf.kid != self.kid] # exclude this keyframe 
        viewing_keyframes = Counter(viewing_keyframes)   
        if not viewing_keyframes: # if empty   (https://www.pythoncentral.io/how-to-check-if-a-list-tuple-or-dictionary-is-empty-in-python/)
            return 
        # order the keyframes 
        covisible_keyframes = viewing_keyframes.most_common() 
        #print('covisible_keyframes: ', covisible_keyframes)
        # get keyframe that shares most points 
        kf_max, w_max = covisible_keyframes[0]
        # if the counter is greater than threshold add connection
        # otherwise add the one with maximum counter             
        with self._lock_connections:          
            if w_max >= Parameters.kMinNumOfCovisiblePointsForCreatingConnection:
                self.connected_keyframes_weights = viewing_keyframes 
                self.ordered_keyframes_weights = OrderedDict()
                for kf,w in covisible_keyframes:
                    if w >= Parameters.kMinNumOfCovisiblePointsForCreatingConnection:
                        kf.add_connection(self,w)
                        self.ordered_keyframes_weights[kf] = w
                    else:
                        break 
            else:
                self.connected_keyframes_weights = Counter({kf_max,w_max}) 
                self.ordered_keyframes_weights = OrderedDict({kf_max,w_max})    
                kf_max.add_connection(self,w_max)        
            # update spanning tree                     
            if self.is_first_connection and self.kid!=0: 
                self.set_parent(kf_max)
                self.is_first_connection = False 
        #print('ordered_keyframes_weights: ', self.ordered_keyframes_weights)                               
            
    @property             
    def is_bad(self): 
        with self._lock_connections:        
            return self._is_bad        

    def set_not_erase(self): 
        with self._lock_connections:          
            not_to_erase = True 
            
    def set_erase(self): 
        with self._lock_connections: 
            if len(self.loop_edges)==0:          
                self.not_to_erase = False        
            if self.to_be_erased: 
                self.set_bad()        

    def set_bad(self): 
        with self._lock_connections: 
            if self.kid == 0: 
                return 
            if self.not_to_erase: 
                self.to_be_erased = True     
                return
                           
            # update covisibility graph 
            for kf_connected in list(self.connected_keyframes_weights.keys()):          
                kf_connected.erase_connection(self)    
                
            for idx,p in enumerate(self.points): 
                if p is not None: 
                    p.remove_observation(self,idx)
                                 
            self.reset_covisibility()
            
            # update spanning tree: each children must be connected to a new parent 
            
            # build a set of parent candidates for the children 
            parent_candidates = set() 
            assert(self.parent is not None)
            parent_candidates.add(self.parent)
            
            # each child must be connected to a new parent (the candidate parent with highest covisibility weight)
            # once a child is connected to a new parent, include the child as new parent candidate for the rest            
            while not len(self.children)==0:            
                w_max = 0
                child_to_connect = None 
                parent_to_connect = None 
                found_connection = False 
                for kf_child in self.children: 
                    if kf_child.is_bad:
                        continue
                    # check if a candidate parent is connected to kf_child and compute the candidate parent with highest covisibility weight                     
                    covisible_keyframes = kf_child.get_covisible_keyframes()
                    for candidate_parent in parent_candidates: 
                        if candidate_parent in covisible_keyframes:
                            w = kf_child.get_weight(candidate_parent)
                            if w > w_max: 
                                w_max = w 
                                child_to_connect = kf_child
                                parent_to_connect = candidate_parent 
                                found_connection = True 
                if found_connection: 
                    child_to_connect.set_parent(parent_to_connect)
                    parent_candidates.add(child_to_connect)
                    self.children.remove(child_to_connect)
                else: 
                    break # stop since there is no connection with covisibility weight>0

            # if a child has no covisibility connections with any parent candidate, connect it with the original parent of this keyframe
            if not len(self.children)==0:
                for kf_child in self.children:  
                    kf_child.set_parent(self.parent)
                    
            self.parent.erase_child(self)
            self._pose_Tcp.update(self.Tcw @ self.parent.Twc)
            self._is_bad = True 
            
        if self.map is not None:
            self.map.remove_keyframe(self)