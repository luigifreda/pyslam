import sys
import os
import cv2 
import torch

import config
config.cfg.set_lib('xfeat') 

from modules.xfeat import XFeat
from threading import RLock

from utils_sys import Printer


kVerbose = True   

def convert_superpts_to_keypoints(pts, scores, size=1): 
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints  
        kps = [ cv2.KeyPoint(int(p[0]), int(p[1]), size=size, response=1) for p in pts ]            
                            
    return kps  

def transpose_des(des):
    if des is not None: 
        return des.T 
    else: 
        return None 
class CVWrapper():
    def __init__(self, mtd):
        self.mtd = mtd
    def detectAndCompute(self, x, mask=None):
        return self.mtd.detectAndCompute(torch.tensor(x).unsqueeze(0).float().unsqueeze(0))[0]

class XFeat2D: 
    def __init__(self): 
        self.lock = RLock()
                
        
        print('====>XFeat')
        # This class runs the SuperPoint network and processes its outputs.
        self.xfeat = XFeat(top_k=2048)
      
                        
        self.pts = []
        self.kps = []        
        self.des = []
        self.heatmap = [] 
        self.frame = None 
        self.frameFloat = None 
        self.keypoint_size = 30  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint 
          
    # compute both keypoints and descriptors       
    def detectAndCompute(self, frame, mask=None):  # mask is a fake input 
        with self.lock: 
            # self.frame = frame
            
            # self.frameFloat  = (frame.astype('float32') / 255.)
            current = CVWrapper(self.xfeat).detectAndCompute(frame)
            kpts, descs =  current['keypoints'].cpu().numpy(), current['descriptors'].cpu().numpy()
        
            self.pts, self.des = kpts, descs
           
            
            # N.B.: pts are - 3xN numpy array with corners [x_i, y_i, confidence_i]^T.
            #print('pts: ', self.pts.T)
            self.kps = convert_superpts_to_keypoints(self.pts, scores=1, size=self.keypoint_size)
           # print(self.kps)
            #print(1/(time.time()-time0))
            #if kVerbose:
            #print('detector: SUPERPOINT, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])  
            #print (self.kps,self.des.shape)    
            return self.kps,self.des 