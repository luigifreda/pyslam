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
import numpy as np 
import cv2
from enum import Enum
from geom_helpers import imgBlocks

kVerbose = True   

kMinNumFeatureDefault = 2000

kAdaptorNumRowDivs = 4
kAdaptorNumColDivs = 4

kNumLevels = 4
kNumLevelsInitSigma = 12
kScaleFactor = 1.2 

kDrawOriginalExtractedFeatures = False  # for debugging 


class FeatureDetectorTypes(Enum):
    SHI_TOMASI = 1
    FAST = 2    
    SIFT = 3
    SURF = 4
    ORB  = 5 
    BRISK = 6
    AKAZE = 7


class FeatureDescriptorTypes(Enum):
    NONE = 0  # used for LK tracker
    SIFT = 1
    SURF = 2
    ORB  = 3  
    BRISK = 4       
    AKAZE = 5


def feature_detector_factory(min_num_features=kMinNumFeatureDefault, 
                             num_levels = kNumLevels, 
                             detector_type = FeatureDetectorTypes.FAST, 
                             descriptor_type = FeatureDescriptorTypes.ORB):
    return FeatureDetector(min_num_features, num_levels, detector_type, descriptor_type)

# BlockAdaptor divides the image in row_divs x col_divs cells and extracts features in each of these cells
class BlockAdaptor(object): 
    def __init__(self, detector, row_divs = kAdaptorNumRowDivs, col_divs = kAdaptorNumColDivs):    
        self.detector = detector 
        self.row_divs = row_divs
        self.col_divs = col_divs 

    def detect(self, frame, mask=None):
        if self.row_divs == 1 and self.col_divs == 1: 
            return self.detector.detect(frame, mask)
        else:   
            if kVerbose:             
                print('BlockAdaptor')
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
            return kps_global

# PyramidAdaptor generate a pyramid of num_levels images and extracts features in each of these images
# TODO: check if a point on one level 'overlaps' with a point on other levels
class PyramidAdaptor(object): 
    def __init__(self, detector, num_levels = 4, scale_factor = 1.2, use_block_adaptor = False):    
        self.detector = detector 
        self.num_levels = num_levels
        self.scale_factor = scale_factor 
        self.cur_pyr = [] 
        self.scale_factors = None
        self.inv_scale_factors = None 
        self.use_block_adaptor = use_block_adaptor
        self.block_adaptor = None 
        if self.use_block_adaptor:
            self.block_adaptor = BlockAdaptor(self.detector, row_divs = kAdaptorNumRowDivs, col_divs = kAdaptorNumColDivs)            
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

    def detect(self, frame, mask=None):      
        if self.num_levels == 1: 
            return self.detector.detect(frame, mask)
        else:    
            if kVerbose:              
                print('PyramidAdaptor')
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
            return kps_global  

    def computerPyramid(self, frame): 
        self.cur_pyr = []
        self.cur_pyr.append(frame) 
        inv_scale = 1./self.scale_factor
        for i in range(1,self.num_levels):
            pyr_cur  = self.cur_pyr[-1]
            pyr_down = cv2.resize(pyr_cur,(0,0),fx=inv_scale,fy=inv_scale)
            self.cur_pyr.append(pyr_down)                         



class ShiTomasiDetector(object): 
    def __init__(self, min_num_features=kMinNumFeatureDefault, quality_level = 0.01, min_coner_distance = 7):
        self.min_num_features = min_num_features
        self.quality_level = quality_level
        self.min_coner_distance = min_coner_distance

    def detect(self, frame, mask=None):                
        pts = cv2.goodFeaturesToTrack(frame, self.min_num_features, self.quality_level, self.min_coner_distance, mask=mask)
        # convert matrix of pts into list of keypoints 
        if pts is not None: 
            kps = [ cv2.KeyPoint(p[0][0], p[0][1], 1) for p in pts ]
        else:
            kps = []
        if kVerbose:
            print('detector: Shi-Tomasi, #features: ', len(kps), ', #ref: ', self.min_num_features)      
        return kps


class FeatureDetector(object):
    def __init__(self, min_num_features=kMinNumFeatureDefault, 
                       num_levels = kNumLevels, 
                       detector_type = FeatureDetectorTypes.SHI_TOMASI,  
                       descriptor_type = FeatureDescriptorTypes.ORB):
        self.detector_type = detector_type 
        self.descriptor_type = descriptor_type

        self.num_levels = num_levels  
        self.scale_factor = kScaleFactor  
        self.initSigmaLevels()

        self.min_num_features = min_num_features
        # at present time pyramid adaptor has the priority and can combine a block adaptor withint itself 
        self.use_bock_adaptor = False 
        self.block_adaptor = None
        self.use_pyramid_adaptor = False 
        self.pyramid_adaptor = None 

        if cv2.__version__.split('.')[0] == '3':
            from cv2.xfeatures2d import SIFT_create, SURF_create 
            from cv2 import ORB_create, BRISK_create, AKAZE_create
        else:
            SIFT_create = cv2.SIFT
            SURF_create = cv2.SURF
            ORB_create = cv2.ORB 
            BRISK_create = cv2.BRISK
            AKAZE_create = cv2.AKAZE            
        self.SIFT_create = SIFT_create
        self.SURF_create = SURF_create
        self.ORB_create = ORB_create 
        self.BRISK_create = BRISK_create            
        self.AKAZE_create = AKAZE_create        

        self.orb_params = dict(nfeatures=min_num_features,
                               scaleFactor=self.scale_factor,
                               nlevels=self.num_levels,
                               patchSize=31,
                               edgeThreshold = 19, 
                               fastThreshold = 20,
                               scoreType=cv2.ORB_HARRIS_SCORE)  #scoreType=cv2.ORB_HARRIS_SCORE, scoreType=cv2.ORB_FAST_SCORE 
        
        self.detector_name = ''
        self.decriptor_name = ''

        # init detector 
        if self.detector_type == FeatureDetectorTypes.SIFT: 
            self._feature_detector = SIFT_create() 
            self.detector_name = 'SIFT'
        elif self.detector_type == FeatureDetectorTypes.SURF:
            self._feature_detector = SURF_create() 
            self.detector_name = 'SURF'            
        elif self.detector_type == FeatureDetectorTypes.ORB:
            self._feature_detector = ORB_create(**self.orb_params) 
            self.detector_name = 'ORB'                
            self.use_bock_adaptor = True    
        elif self.detector_type == FeatureDetectorTypes.BRISK:
            self._feature_detector = BRISK_create(octaves=self.num_levels) 
            self.detector_name = 'BRISK'  
            self.scale_factor = 1.3   # from the BRISK opencv code this seems to be the used scale factor between intra-octave frames 
            self.initSigmaLevels()                 
        elif self.detector_type == FeatureDetectorTypes.AKAZE:
            self._feature_detector = AKAZE_create(nOctaves=self.num_levels) 
            self.detector_name = 'AKAZE'                                                  
        elif self.detector_type == FeatureDetectorTypes.FAST:
            self._feature_detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)  
            self.detector_name = 'FAST'        
            self.use_bock_adaptor = True             
            self.use_pyramid_adaptor = self.num_levels > 1               
        elif self.detector_type == FeatureDetectorTypes.SHI_TOMASI:
            self._feature_detector = ShiTomasiDetector(self.min_num_features)  
            self.detector_name = 'Shi-Tomasi'
            self.use_bock_adaptor = False  
            self.use_pyramid_adaptor = self.num_levels > 1 
        else:
            raise ValueError("Unknown feature extractor %s" % self.detector_type)

        if self.use_bock_adaptor is True:
            self.block_adaptor = BlockAdaptor(self._feature_detector, row_divs = kAdaptorNumRowDivs, col_divs = kAdaptorNumColDivs)

        if self.use_pyramid_adaptor is True:            
            self.pyramid_adaptor = PyramidAdaptor(self._feature_detector, self.num_levels, self.scale_factor, use_block_adaptor=self.use_bock_adaptor)

        # init descriptor  
        if self.descriptor_type == FeatureDescriptorTypes.SIFT: 
            self._feature_descriptor = SIFT_create() 
            self.decriptor_name = 'SIFT'            
        elif self.descriptor_type == FeatureDescriptorTypes.SURF:
            self._feature_descriptor = SURF_create() 
            self.decriptor_name = 'SURF'                  
        elif self.descriptor_type == FeatureDescriptorTypes.ORB:
            self._feature_descriptor = ORB_create(**self.orb_params) 
            self.decriptor_name = 'ORB' 
        elif self.descriptor_type == FeatureDescriptorTypes.BRISK:
            self._feature_descriptor = BRISK_create() 
            self.decriptor_name = 'BRISK'         
        elif self.descriptor_type == FeatureDescriptorTypes.AKAZE:
            self._feature_descriptor = AKAZE_create() 
            self.decriptor_name = 'AKAZE'                                                
        elif self.descriptor_type == FeatureDescriptorTypes.NONE:
            self._feature_descriptor = None              
            self.decriptor_name = 'None'                                     
        else:
            raise ValueError("Unknown feature extractor %s" % self.detector_type)            

    def initSigmaLevels(self): 
        num_levels = max(kNumLevelsInitSigma, self.num_levels)        
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)

        self.scale_factors[0]=1.0
        self.level_sigmas2[0]=1.0
        for i in range(1,num_levels):
            self.scale_factors[i]=self.scale_factors[i-1]*self.scale_factor
            self.level_sigmas2[i]=self.scale_factors[i]*self.scale_factors[i]
        #print('self.scale_factors: ', self.scale_factors)
        for i in range(num_levels):
            self.inv_scale_factors[i]=1.0/self.scale_factors[i]
            self.inv_level_sigmas2[i]=1.0/self.level_sigmas2[i]
        #print('self.inv_scale_factors: ', self.inv_scale_factors)            

    # detect keypoints without computing their descriptors
    # out: kps 
    def detect(self, frame, mask=None): 
        if frame.ndim>2:
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)             
        if self.use_pyramid_adaptor:
            kps = self.pyramid_adaptor.detect(frame, mask)            
        elif self.use_bock_adaptor:
            kps = self.block_adaptor.detect(frame, mask)            
        else:       
            kps = self._feature_detector.detect(frame, mask)                  
        kps = self.satNumberOfFeatures(kps)            
        if kDrawOriginalExtractedFeatures: # draw the original features
            imgDraw = cv2.drawKeypoints(frame, kps, None, color=(0,255,0), flags=0)
            cv2.imshow('detected keypoints',imgDraw)            
        if kVerbose:
            print('detector: ', self.detector_name, ', #features: ', len(kps))    
        return kps        

    # detect keypoints and their descriptors
    # out: kps, des 
    def detectAndCompute(self, frame, mask=None):
        if frame.ndim>2:
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)             
        kps = self.detect(frame, mask)    
        kps, des = self._feature_descriptor.compute(frame, kps)  
        if kVerbose:
            #print('detector: ', self.detector_name, ', #features: ', len(kps))           
            print('descriptor: ', self.decriptor_name, ', #features: ', len(kps))                                
        return kps, des             

    # keep the first 'self.min_num_features' best features
    def satNumberOfFeatures(self, kps):
        if kVerbose:
            print('detector: ', self.detector_name, ', #features: ', len(kps),', ref: ', self.min_num_features)          
        if len(kps) > self.min_num_features:
            # keep the features with the best response 
            kps = sorted(kps, key=lambda x:x.response, reverse=True)[:self.min_num_features]         
            if False: 
                for k in kps:
                    print("response: ", k.response) 
                    print("size: ", k.size)  
                    print("octave: ", k.octave)   
        return kps 