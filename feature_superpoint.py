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
import os
import cv2 
import torch

from demo_superpoint import SuperPointFrontend
from geom_helpers import convertMatPtsToKeyPoints

# get the location of this file!
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

kVerbose = True   

class SuperPointOptions:
    def __init__(self): 
        # default options from demo_superpoints
        self.weights_path=__location__ + '/thirdparty/superpoint/superpoint_v1.pth'
        self.nms_dist=4
        self.conf_thresh=0.015
        self.nn_thresh=0.7
        
        use_cuda = torch.cuda.is_available()
        device = torch.device('cuda' if use_cuda else 'cpu')
        print('SuperPoint using ', device)        
        self.cuda=use_cuda   
    

# interface for pySLAM
class SuperPointFeature2D: 
    def __init__(self): 
        opts = SuperPointOptions()
        print(opts)        
        
        print('SuperPointFeature2D')
        print('==> Loading pre-trained network.')
        # This class runs the SuperPoint network and processes its outputs.
        self.fe = SuperPointFrontend(weights_path=opts.weights_path,
                                nms_dist=opts.nms_dist,
                                conf_thresh=opts.conf_thresh,
                                nn_thresh=opts.nn_thresh,
                                cuda=opts.cuda)
        print('==> Successfully loaded pre-trained network.')
                        
        self.pts = []
        self.kps = []        
        self.des = []
        self.heatmap = [] 
        self.frame = None 
        self.frameFloat = None 
          
    # compute both keypoints and descriptors       
    def detectAndCompute(self, frame, mask=None):  # mask is a fake input 
        self.frame = frame 
        self.frameFloat  = (frame.astype('float32') / 255.)
        self.pts, self.des, self.heatmap = self.fe.run(self.frameFloat)
        self.kps = []
        self.kps = convertMatPtsToKeyPoints(self.pts.T)
        if kVerbose:
            print('detector: SuperPoint, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])      
        return self.kps, self.des.T                 
            
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input  
        if self.frame is not frame:
            self.detectAndCompute(frame)        
        return self.kps
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute_descriptor(self, frame): 
        if self.frame is not frame:
            self.detectAndCompute(frame)
        return self.des.T    
    
    def compute(self, frame, kps=None): # kps is a fake input  
        des = self.compute_descriptor(frame)
        return self.kps, self.des.T
           