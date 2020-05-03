"""
* This file is part of PYSLAM.
* Adapted from https://raw.githubusercontent.com/naver/r2d2/master/extract.py, see the licence therein.  
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

# adapted from from https://raw.githubusercontent.com/naver/r2d2/master/extract.py

import config 
config.cfg.set_lib('r2d2') 

import os, pdb
from PIL import Image
import numpy as np
import torch
import cv2 
from threading import RLock

from r2d2.tools import common
from r2d2.tools.dataloader import norm_RGB
from r2d2.nets.patchnet import *
import argparse

from utils import Printer 


kVerbose = True 


def load_network(model_fn): 
    checkpoint = torch.load(model_fn)
    print("\n>> Creating net = " + checkpoint['net']) 
    net = eval(checkpoint['net'])
    nb_of_weights = common.model_size(net)
    print(f" ( Model size: {nb_of_weights/1000:.0f}K parameters )")

    # initialization
    weights = checkpoint['state_dict']
    net.load_state_dict({k.replace('module.',''):v for k,v in weights.items()})
    return net.eval()


class NonMaxSuppression (torch.nn.Module):
    def __init__(self, rel_thr=0.7, rep_thr=0.7):
        nn.Module.__init__(self)
        self.max_filter = torch.nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.rel_thr = rel_thr
        self.rep_thr = rep_thr
    
    def forward(self, reliability, repeatability, **kw):
        assert len(reliability) == len(repeatability) == 1
        reliability, repeatability = reliability[0], repeatability[0]

        # local maxima
        maxima = (repeatability == self.max_filter(repeatability))

        # remove low peaks
        maxima *= (repeatability >= self.rep_thr)
        maxima *= (reliability   >= self.rel_thr)

        return maxima.nonzero().t()[2:4]


def extract_multiscale( net, img, detector, scale_f=2**0.25, 
                        min_scale=0.0, max_scale=1, 
                        min_size=256, max_size=1024, 
                        verbose=False):
    old_bm = torch.backends.cudnn.benchmark 
    torch.backends.cudnn.benchmark = False # speedup
    
    # extract keypoints at multiple scales
    B, three, H, W = img.shape
    assert B == 1 and three == 3, "should be a batch with a single RGB image"
    
    assert max_scale <= 1
    s = 1.0 # current scale factor
    
    level = 0
    L = []
    X,Y,S,C,Q,D = [],[],[],[],[],[]
    while  s+0.001 >= max(min_scale, min_size / max(H,W)):
        if s-0.001 <= min(max_scale, max_size / max(H,W)):
            nh, nw = img.shape[2:]
            if verbose: print(f"extracting at scale x{s:.02f} = {nw:4d}x{nh:3d} - level {level}")
            # extract descriptors
            with torch.no_grad():
                res = net(imgs=[img])
                
            # get output and reliability map
            descriptors = res['descriptors'][0]
            reliability = res['reliability'][0]
            repeatability = res['repeatability'][0]

            # normalize the reliability for nms
            # extract maxima and descs
            y,x = detector(**res) # nms
            c = reliability[0,0,y,x]
            q = repeatability[0,0,y,x]
            d = descriptors[0,:,y,x].t()
            n = d.shape[0]

            # accumulate multiple scales
            X.append(x.float() * W/nw)
            Y.append(y.float() * H/nh)
            S.append((32/s) * torch.ones(n, dtype=torch.float32, device=d.device))
            C.append(c)
            Q.append(q)
            D.append(d)
            
            L_tmp =level * np.ones(n,dtype=np.int32)
            L = np.concatenate((L, L_tmp), axis=0).astype(np.int32)                
            level += 1
            
        s /= scale_f

        # down-scale the image for next iteration
        nh, nw = round(H*s), round(W*s)
        img = F.interpolate(img, (nh,nw), mode='bilinear', align_corners=False)

    # restore value
    torch.backends.cudnn.benchmark = old_bm

    Y = torch.cat(Y)
    X = torch.cat(X)
    S = torch.cat(S) # scale
    scores = torch.cat(C) * torch.cat(Q) # scores = reliability * repeatability
    XYS = torch.stack([X,Y,S], dim=-1)
    D = torch.cat(D)
    return XYS, D, scores, L


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, sizes, levels): 
    assert(len(pts)==len(scores))
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints  
        kps = [ cv2.KeyPoint(p[0], p[1], _size=sizes[i], _response=scores[i], _octave=levels[i]) for i,p in enumerate(pts) ]                      
    return kps         

 
# TODO: fix the octave field of the output keypoints 
# interface for pySLAM
class R2d2Feature2D: 
    def __init__(self,
                 num_features = 2000, 
                 scale_f   = 2**0.25, 
                 min_size  = 256, 
                 max_size  = 1300, #1024,                   
                 min_scale = 0, 
                 max_scale = 1,                
                 reliability_thr   = 0.7,   
                 repeatability_thr = 0.7,  
                 do_cuda=True):    
        print('Using R2d2Feature2D')    
        self.lock = RLock()             
        self.model_base_path = config.cfg.root_folder + '/thirdparty/r2d2'        
        self.model_weights_path = self.model_base_path + '/models/r2d2_WASF_N16.pt'
        #print('model_weights_path:',self.model_weights_path)
        
        self.pts = []
        self.kps = []        
        self.des = []
        self.frame = None    
                
        self.num_features = num_features
        self.scale_f   = scale_f 
        self.min_size  = min_size 
        self.max_size  = max_size                     
        self.min_scale = min_scale 
        self.max_scale = max_scale            
        self.reliability_thr = reliability_thr
        self.repeatability_thr = repeatability_thr
        self.do_cuda = do_cuda
        if do_cuda:                  
            gpus = [0]
        else:
            gpus = -1 
        self.gpus = gpus 
        self.do_cuda = common.torch_set_gpu(gpus)   
                                            
        print('==> Loading pre-trained network.')   
                  
        self.net = load_network(self.model_weights_path)
        if self.do_cuda: self.net = self.net.cuda()

        # create the non-maxima detector
        self.detector = NonMaxSuppression(rel_thr=reliability_thr, rep_thr=repeatability_thr)  
        
        print('==> Successfully loaded pre-trained network.')            
            
    
    def compute_kps_des(self,img):     
        with self.lock:        
            H, W = img.shape[:2]
            img = norm_RGB(img)[None] 
            if self.do_cuda: img = img.cuda()
            
            # extract keypoints/descriptors for a single image
            xys, desc, scores, levels = extract_multiscale(self.net, img, self.detector,
                scale_f   = self.scale_f, 
                min_scale = self.min_scale, 
                max_scale = self.max_scale,
                min_size  = self.min_size, 
                max_size  = self.max_size, 
                verbose = kVerbose)

            xys = xys.cpu().numpy()
            desc = desc.cpu().numpy()
            scores = scores.cpu().numpy()
            idxs = scores.argsort()[-self.num_features or None:]
            
            selected_xys = xys[idxs]
            self.pts = selected_xys[:,:2]    
            sizes = selected_xys[:,2]   
            des = desc[idxs]          
            scores = scores[idxs]    
            levels = np.array(levels)[idxs] 
            
            kps = convert_pts_to_keypoints(self.pts, scores, sizes, levels)
            return kps, des                  
            
        
    def detectAndCompute(self, frame, mask=None):  #mask is a fake input  
        with self.lock:
            self.frame = frame         
            self.kps, self.des = self.compute_kps_des(frame)            
            if kVerbose:
                print('detector: R2D2 , descriptor: R2D2 , #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])                  
            return self.kps, self.des
    
           
    # return keypoints if available otherwise call detectAndCompute()    
    def detect(self, frame, mask=None):  # mask is a fake input  
        with self.lock:        
            if self.frame is not frame:
                self.detectAndCompute(frame)        
            return self.kps
    
    
    # return descriptors if available otherwise call detectAndCompute()  
    def compute(self, frame, kps=None, mask=None): # kps is a fake input, mask is a fake input
        with self.lock:        
            if self.frame is not frame:
                Printer.orange('WARNING: R2D2 is recomputing both kps and des on last input frame', frame.shape)            
                self.detectAndCompute(frame)
            return self.kps, self.des                 
           

