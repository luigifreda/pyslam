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

import config
config.cfg.set_lib('superpoint') 

from demo_superpoint import SuperPointFrontend
from threading import RLock

from utils import Printer 


kVerbose = True   


class SuperPointOptions:
    def __init__(self, do_cuda=True): 
        # default options from demo_superpoints
        self.weights_path=config.cfg.root_folder + '/thirdparty/superpoint/superpoint_v1.pth'
        self.nms_dist=4
        self.conf_thresh=0.015
        self.nn_thresh=0.7
        
        use_cuda = torch.cuda.is_available() & do_cuda
        device = torch.device('cuda' if use_cuda else 'cpu')
        print('SuperPoint using ', device)        
        self.cuda=use_cuda   
    
    
# convert matrix of pts into list of keypoints
# N.B.: pts are - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
def convert_superpts_to_keypoints(pts, size=1): 
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints  
        kps = [ cv2.KeyPoint(p[0], p[1], _size=size, _response=p[2]) for p in pts ]                      
    return kps         


def transpose_des(des):
    if des is not None: 
        return des.T 
    else: 
        return None 


# interface for pySLAM 
class SuperPointFeature2D: 
    def __init__(self, do_cuda=True): 
        self.lock = RLock()
        self.opts = SuperPointOptions(do_cuda)
        print(self.opts)        
        
        print('SuperPointFeature2D')
        print('==> Loading pre-trained network.')
        # This class runs the SuperPoint network and processes its outputs.
        self.fe = SuperPointFrontend(weights_path=self.opts.weights_path,
                                nms_dist=self.opts.nms_dist,
                                conf_thresh=self.opts.conf_thresh,
                                nn_thresh=self.opts.nn_thresh,
                                cuda=self.opts.cuda)
        print('==> Successfully loaded pre-trained network.')
                        
        self.pts = []
        self.kps = []        
        self.des = []
        self.heatmap = [] 
        self.frame = None 
        self.frameFloat = None 
        self.keypoint_size = 20  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint 
          
    # compute both keypoints and descriptors       
    def detectAndCompute(self, frame, mask=None):  # mask is a fake input 
        with self.lock: 
            self.frame = frame 
            self.frameFloat  = (frame.astype('float32') / 255.)
            self.pts, self.des, self.heatmap = self.fe.run(self.frameFloat)
            # N.B.: pts are - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
            #print('pts: ', self.pts.T)
            self.kps = convert_superpts_to_keypoints(self.pts.T, size=self.keypoint_size)
            if kVerbose:
                print('detector: SUPERPOINT, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])      
            return self.kps, transpose_des(self.des)                 
            
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input  
        with self.lock:         
            #if self.frame is not frame:
            self.detectAndCompute(frame)        
            return self.kps
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute(self, frame, kps=None, mask=None): # kps is a fake input, mask is a fake input
        with self.lock: 
            if self.frame is not frame:
                Printer.orange('WARNING: SUPERPOINT is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, transpose_des(self.des)
           