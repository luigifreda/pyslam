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

from geom_helpers import poseRt, add_ones
from frame import Frame
import time
import numpy as np


class MapPoint(object):
    # A Point is a 3-D point in the world
    # Each Point is observed in multiple Frames

    def __init__(self, mapp, position, color, tid=None):
        self.pt = np.array(position)

        self.frames = []   # list of frames `f` in which this point has been observed
        self.idxs = []     # list of observation indexes `idx` to be used with self.frames, such that 
                           #      for f, idx in zip(self.frames, self.idxs): f.points[idx] = this point

        self.color = np.copy(color)
        self.id = tid if tid is not None else mapp.add_point(self)
        self.is_bad = False   # a point becomes bad when its num_observations < 2 (cannot be considered for bundle ajustment or other operations)
        self.num_observations = 0 

    def homogeneous(self):
        return add_ones(self.pt)

    # return array of corresponding descriptors 
    def des(self):
        return [f.des[idx] for f, idx in zip(self.frames, self.idxs)]

    # minimum distance between input descriptor and map point corresponding descriptors 
    def min_des_distance(self, descriptor):
        return min([Frame.descriptor_distance(d, descriptor) for d in self.des()])

    def delete(self):
        for f, idx in zip(self.frames, self.idxs):  # f.points[idx] = this point
            f.remove_point_observation(idx)  
        self.frames = []
        self.idxs = [] 
        self.is_bad = True            
        del self

    def add_observation(self, frame, idx):
        assert(frame.points[idx] is None)
        assert(frame not in self.frames)
        frame.points[idx] = self
        self.frames.append(frame)
        self.idxs.append(idx)
        self.num_observations += 1
        self.is_bad = (self.num_observations < 2)

    # remove frame observation 
    def remove_observation(self, frame, idx = None):  
        if __debug__: 
            if idx is not None:
                assert(self == frame.points[idx])     
        frame.remove_point(self)
        frame_idx = None 
        try:
            frame_idx = self.frames.index(frame)
        except:
            pass
        if frame_idx is not None: 
            del(self.frames[frame_idx])    
            del(self.idxs[frame_idx])
            self.num_observations = max(0, self.num_observations - 1)
        self.is_bad = (self.num_observations < 2)

