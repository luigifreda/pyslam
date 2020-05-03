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

import torchvision as tv
import phototour
import torch
from tqdm import tqdm 
import numpy as np
import torch.nn as nn
import math 
import tfeat_model
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import os

import cv2
import tfeat_utils
import numpy as np
import cv2
from matplotlib import pyplot as plt

from geom_helpers import convertMatPtsToKeyPoints

# get the location of this file!
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

kVerbose = True   

# interface for pySLAM
class TfeatFeature2D: 
    def __init__(self): 

        # mag_factor is how many times the original keypoint scale
        # is enlarged to generate a patch from a keypoint
        self.mag_factor = 3

        self.tfeat_base_path = __location__ + '/thirdparty/tfeat/'

        print('TfeatFeature2D')
        print('==> Loading pre-trained network.')
        #init tfeat and load the trained weights
        self.tfeat = tfeat_model.TNet()
        self.models_path = self.tfeat_base_path + 'pretrained-models'
        self.net_name = 'tfeat-liberty'
        self.tfeat.load_state_dict(torch.load(os.path.join(self.models_path,self.net_name+".params")))
        self.tfeat.cuda()
        self.tfeat.eval()
        print('==> Successfully loaded pre-trained network.')
    
    def compute(self, frame, kps):  
        #print('kps: ', kps)
        if len(kps)>0:
            self.des = tfeat_utils.describe_opencv(self.tfeat, frame, kps, 32, self.mag_factor)
        else:
            self.des = []
        if kVerbose:
            print('descriptor: TFEAT, #features: ', len(kps), ', frame res: ', frame.shape[0:2])                  
        return kps, self.des
           