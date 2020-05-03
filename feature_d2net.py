"""
* This file is part of PYSLAM 
* Adapted from https://github.com/mihaidusmanu/d2-net/blob/master/extract_features.py, see the license therein. 
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

# adapted from https://github.com/mihaidusmanu/d2-net/blob/master/extract_features.py

import config
config.cfg.set_lib('d2net') 

import os 
import argparse

import cv2 
import numpy as np

import imageio

from threading import RLock

import torch

from tqdm import tqdm

import scipy
import scipy.io
import scipy.misc

from utils import Printer

from lib.model_test import D2Net
from lib.utils import preprocess_image
from lib.pyramid import process_multiscale



kVerbose = True   


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, size=1): 
    assert(len(pts)==len(scores))
    kps = []
    if pts is not None: 
        # convert matrix [Nx2] of pts into list of keypoints  
        kps = [ cv2.KeyPoint(p[0], p[1], _size=size, _response=scores[i]) for i,p in enumerate(pts) ]                      
    return kps         


# interface for pySLAM 
# from https://github.com/mihaidusmanu/d2-net
# N.B.: The singlescale features require less than 6GB of VRAM for 1200x1600 images. 
#       The multiscale flag can be used to extract multiscale features - for this, we recommend at least 12GB of VRAM.
class D2NetFeature2D: 
    def __init__(self, 
                 use_relu=True,             # remove ReLU after the dense feature extraction module
                 multiscale=False,          # extract multiscale features (read the note above)
                 max_edge=1600,             # maximum image size at network input
                 max_sum_edges=2800,        # maximum sum of image sizes at network input
                 preprocessing='torch',     # image preprocessing (caffe or torch) 
                 do_cuda=True): 
        print('Using D2NetFeature2D')   
        self.lock = RLock()
        self.model_base_path = config.cfg.root_folder + '/thirdparty/d2net/'
        self.models_path = self.model_base_path + 'models/d2_ots.pth'   # best performances obtained with 'd2_ots.pth'
        
        self.use_relu = use_relu
        self.multiscale = multiscale
        self.max_edge = max_edge 
        self.max_sum_edges = max_sum_edges
        self.preprocessing = preprocessing
        
        self.pts = []
        self.kps = []        
        self.des = []
        self.frame = None 
        self.keypoint_size = 20  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint        
        
        self.do_cuda = do_cuda & torch.cuda.is_available()
        print('cuda:',self.do_cuda)        
        self.device = torch.device("cuda:0" if self.do_cuda else "cpu")        
        
        torch.set_grad_enabled(False)
                
        print('==> Loading pre-trained network.')
        # Creating CNN model
        self.model = D2Net(
            model_file=self.models_path,
            use_relu=use_relu,
            use_cuda=do_cuda)
        if self.do_cuda:
            print('Extracting on GPU')
        else:
            print('Extracting on CPU')
        print('==> Successfully loaded pre-trained network.')
    
    
    def compute_kps_des(self, image):   
        with self.lock:         
            print('D2Net image shape:',image.shape)               
            if len(image.shape) == 2:
                    image = image[:, :, np.newaxis]
                    image = np.repeat(image, 3, -1)

            # TODO: switch to PIL.Image due to deprecation of scipy.misc.imresize.
            resized_image = image
            if max(resized_image.shape) > self.max_edge:
                resized_image = scipy.misc.imresize(
                    resized_image,
                    self.max_edge / max(resized_image.shape)
                ).astype('float')
            if sum(resized_image.shape[: 2]) > self.max_sum_edges:
                resized_image = scipy.misc.imresize(
                    resized_image,
                    self.max_sum_edges / sum(resized_image.shape[: 2])
                ).astype('float')

            fact_i = image.shape[0] / resized_image.shape[0]
            fact_j = image.shape[1] / resized_image.shape[1]
            print('scale factors: {}, {}'.format(fact_i,fact_j))

            input_image = preprocess_image(
                resized_image,
                preprocessing=self.preprocessing
            )
            with torch.no_grad():
                if self.multiscale:
                    self.pts, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=self.device
                        ),
                        self.model
                    )
                else:
                    self.pts, scores, descriptors = process_multiscale(
                        torch.tensor(
                            input_image[np.newaxis, :, :, :].astype(np.float32),
                            device=self.device
                        ),
                        self.model,
                        scales=[1]
                    )

            # Input image coordinates
            self.pts[:, 0] *= fact_i
            self.pts[:, 1] *= fact_j
            # i, j -> u, v
            self.pts = self.pts[:, [1, 0, 2]]
            #print('pts.shape: ', self.pts.shape)
            #print('pts:', self.pts)
            
            self.kps = convert_pts_to_keypoints(self.pts, scores, self.keypoint_size)        
            self.des = descriptors 
            return self.kps, self.des 
    
    
    def detectAndCompute(self, frame, mask=None):  #mask is a fake input  
        with self.lock:
            self.frame = frame         
            self.kps, self.des = self.compute_kps_des(frame)            
            if kVerbose:
                print('detector: D2NET, descriptor: D2NET, #features: ', len(self.kps), ', frame res: ', frame.shape[0:2])                  
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
                Printer.orange('WARNING: D2NET is recomputing both kps and des on last input frame', frame.shape)            
                self.detectAndCompute(frame)
            return self.kps, self.des         