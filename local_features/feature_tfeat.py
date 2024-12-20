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

import config
config.cfg.set_lib('tfeat') 

import os
import numpy as np
import math 
import cv2
import time 

import torchvision as tv
import torch

import tfeat_model
import tfeat_utils

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

from utils_features import extract_patches_array, extract_patches_array_cpp


kVerbose = True   


# interface for pySLAM
class TfeatFeature2D: 
    def __init__(self, do_cuda=True): 
        print('Using TfeatFeature2D')   
        self.model_base_path = config.cfg.root_folder + '/thirdparty/tfeat/'

        self.do_cuda = do_cuda & torch.cuda.is_available()
        print('cuda:',self.do_cuda)        
        device = torch.device("cuda:0" if self.do_cuda else "cpu")        
        
        torch.set_grad_enabled(False)
        
        # mag_factor is how many times the original keypoint scale
        # is enlarged to generate a patch from a keypoint
        self.mag_factor = 3        
        
        print('==> Loading pre-trained network.')
        #init tfeat and load the trained weights
        self.model = tfeat_model.TNet()
        self.models_path = self.model_base_path + 'pretrained-models'
        self.net_name = 'tfeat-liberty'
        self.model.load_state_dict(torch.load(os.path.join(self.models_path,self.net_name+".params")))
        if self.do_cuda:
            self.model.cuda()
            print('Extracting on GPU')
        else:
            print('Extracting on CPU')
            self.model = model.cpu()        
        self.model.eval()  
        print('==> Successfully loaded pre-trained network.')
    
    
    def compute_des(self, patches):                  
        patches = torch.from_numpy(patches).float()
        patches = torch.unsqueeze(patches,1)
        if self.do_cuda:
            patches = patches.cuda()
        with torch.no_grad():            
            descrs = self.model(patches)
        return descrs.detach().cpu().numpy().reshape(-1, 128)        
    
    
    def compute(self, frame, kps, mask=None):  #mask is a fake input  
        #print('kps: ', kps)
        if len(kps)>0:
            #des = tfeat_utils.describe_opencv(self.model, frame, kps, 32, self.mag_factor)
            # extract the keypoint patches 
            #t = time.time()
            if False: 
                # use python code 
                patches = extract_patches_array(frame, kps, patch_size=32, mag_factor=self.mag_factor)
            else:
                # use faster cpp code 
                patches = extract_patches_array_cpp(frame, kps, patch_size=32, mag_factor=self.mag_factor)
            patches = np.asarray(patches)
            #if kVerbose:
            #    print('patches.shape:',patches.shape)                
            #if kVerbose:                         
            #    print('patch elapsed: ', time.time()-t)
            # compute descriptor by feeeding the full patch tensor to the network              
            des = self.compute_des(patches)            
        else:
            des = []
        if kVerbose:
            print('descriptor: TFEAT, #features: ', len(kps), ', frame res: ', frame.shape[0:2])                  
        return kps, des
           