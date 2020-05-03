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
import numpy as np 
import cv2

from geom_helpers import imgBlocks

kVerbose = True   

kAdaptorNumRowDivs = 2
kAdaptorNumColDivs = 2

kNumLevelsInitSigma = 12

# BlockAdaptor divides the image in row_divs x col_divs cells and extracts features in each of these cells
class BlockAdaptor(object): 
    def __init__(self, detector, descriptor = None, row_divs = kAdaptorNumRowDivs, col_divs = kAdaptorNumColDivs):    
        self.detector = detector 
        self.descriptor = descriptor 
        self.row_divs = row_divs
        self.col_divs = col_divs 
        self.is_detector_equal_to_descriptor = (self.detector == self.descriptor)

    def detect(self, frame, mask=None):
        if self.row_divs == 1 and self.col_divs == 1: 
            return self.detector.detect(frame, mask)
        else:   
            if kVerbose:             
                print('BlockAdaptor ', self.row_divs, 'x', self.col_divs)
            block_generator = imgBlocks(frame, self.row_divs, self.col_divs)
            kps_global = []
            for b, i, j in block_generator:
                if kVerbose and False:                  
                    print('BlockAdaptor  in block (',i,',',j,')')                 
                kps = self.detector.detect(b)
                #print('adaptor: detected #features: ', len(kps), ' in block (',i,',',j,')')  
                for kp in kps:
                    #print('kp.pt before: ', kp.pt)
                    kp.pt = (kp.pt[0] + j, kp.pt[1] + i)        
                    #print('kp.pt after: ', kp.pt)                                                                     
                    kps_global.append(kp)
            return np.array(kps_global)
        
    def detectAndCompute(self, frame, mask=None):
        if self.row_divs == 1 and self.col_divs == 1: 
            return self.detector.detectAndCompute(frame, mask)
        else:   
            if kVerbose:             
                print('BlockAdaptor ', self.row_divs, 'x', self.col_divs)
            block_generator = imgBlocks(frame, self.row_divs, self.col_divs)
            kps_global = []
            des_global = []
            # TODO: manage mask here 
            for b, i, j in block_generator:
                if kVerbose and False:                  
                    print('BlockAdaptor  in block (',i,',',j,')')    
                if self.is_detector_equal_to_descriptor:             
                    kps, des = self.detector.detectAndCompute(b, mask=None)
                else:
                    kps = self.detector.detect(b)    
                    kps, des = self.descriptor.compute(b, kps)  
                    #print('adaptor: detected #features: ', len(kps), ' in block (',i,',',j,')')  
                for kp in kps:
                    #print('kp.pt before: ', kp.pt)
                    kp.pt = (kp.pt[0] + j, kp.pt[1] + i)        
                    #print('kp.pt after: ', kp.pt)                                                                     
                    kps_global.append(kp)
                if len(des_global)>0:
                    if len(des)>0:
                        des_global = np.vstack([des_global, des])                    
                else:
                    des_global = des                 
            return np.array(kps_global), des_global     


# PyramidAdaptor generate a pyramid of num_levels images and extracts features in each of these images
# TODO: check if a point on one level 'overlaps' with a point on other levels or add such option (DONE by FeatureManager.kdt_nms() )
class PyramidAdaptor(object): 
    def __init__(self, detector, descriptor = None, num_levels = 4, scale_factor = 1.2, use_block_adaptor = False):    
        self.detector = detector 
        self.descriptor = descriptor    
        self.is_detector_equal_to_descriptor = (self.detector == self.descriptor)             
        self.num_levels = num_levels
        self.scale_factor = scale_factor 
        self.cur_pyr = [] 
        self.scale_factors = None
        self.inv_scale_factors = None 
        self.use_block_adaptor = use_block_adaptor
        self.block_adaptor = None 
        if self.use_block_adaptor:
            self.block_adaptor = BlockAdaptor(self.detector, self.descriptor, row_divs = kAdaptorNumRowDivs, col_divs = kAdaptorNumColDivs)            
        self.initSigmaLevels()

    def initSigmaLevels(self): 
        num_levels = max(kNumLevelsInitSigma, self.num_levels)
        self.scale_factors = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.scale_factors[0]=1.0
        for i in range(1,num_levels):
            self.scale_factors[i]=self.scale_factors[i-1]*self.scale_factor
        #print('self.scale_factors: ', self.scale_factors)
        for i in range(num_levels):
            self.inv_scale_factors[i]=1.0/self.scale_factors[i]
        #print('self.inv_scale_factors: ', self.inv_scale_factors)       

    def computerPyramid(self, frame): 
        self.cur_pyr = []
        self.cur_pyr.append(frame) 
        inv_scale = 1./self.scale_factor
        #print('self.num_levels: ', self.num_levels)
        for i in range(1,self.num_levels):
            pyr_cur  = self.cur_pyr[-1]
            pyr_down = cv2.resize(pyr_cur,(0,0),fx=inv_scale,fy=inv_scale)
            self.cur_pyr.append(pyr_down)    

    def detect(self, frame, mask=None):      
        if self.num_levels == 1: 
            return self.detector.detect(frame, mask)
        else:    
            if kVerbose:              
                print('PyramidAdaptor #levels: ', self.num_levels, ', scale: ', self.scale_factor)
            self.computerPyramid(frame)
            kps_global = []
            for i in range(0,self.num_levels):              
                scale = self.scale_factors[i]
                pyr_cur  = self.cur_pyr[i]     
                kps = None 
                if self.block_adaptor is None:        
                    kps = self.detector.detect(pyr_cur)                 
                else:
                    kps = self.block_adaptor.detect(pyr_cur)
                if kVerbose and False:                
                    print("PyramidAdaptor - level", i, ", shape: ", pyr_cur.shape)                     
                for kp in kps:
                    #print('kp.pt before: ', kp.pt)
                    kp.pt = (kp.pt[0]*scale, kp.pt[1]*scale) 
                    kp.size = kp.size*scale   
                    kp.octave = i      
                    #print('kp: ', kp.pt, kp.octave)                                                                     
                    kps_global.append(kp)
            return np.array(kps_global)  
        
    def detectAndCompute(self, frame, mask=None):      
        if self.num_levels == 1: 
            return self.detector.detectAndCompute(frame, mask)
        else:    
            if kVerbose:              
                print('PyramidAdaptor #levels: ', self.num_levels, ', scale: ', self.scale_factor)
            self.computerPyramid(frame)
            kps_global = []
            des_global = []            
            for i in range(0,self.num_levels):              
                scale = self.scale_factors[i]
                pyr_cur  = self.cur_pyr[i]     
                kps = None 
                if self.block_adaptor is None:        
                    #kps, des = self.detector.detectAndCompute(pyr_cur)
                    if self.is_detector_equal_to_descriptor:             
                        kps, des = self.detector.detectAndCompute(pyr_cur)
                    else:
                        kps = self.detector.detect(pyr_cur)    
                        kps, des = self.descriptor.compute(pyr_cur, kps)                         
                else:
                    kps, des = self.block_adaptor.detectAndCompute(pyr_cur)
                if kVerbose and False:                
                    print("PyramidAdaptor - level", i, ", shape: ", pyr_cur.shape)                     
                for kp in kps:
                    #print('kp.pt before: ', kp.pt)
                    kp.pt = (kp.pt[0]*scale, kp.pt[1]*scale) 
                    kp.size = kp.size*scale   
                    kp.octave = i      
                    #print('kp: ', kp.pt, kp.octave)                                                                     
                    kps_global.append(kp)
                if len(des_global)>0:
                    if len(des)>0:
                        des_global = np.vstack([des_global, des])                                       
                else:
                    des_global = des 
            return np.array(kps_global), des_global          
                     

