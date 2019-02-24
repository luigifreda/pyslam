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
* along with PYVO. If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np 
import cv2
from enum import Enum
from geom_helpers import imgBlocks

kVerbose = True   
kMinNumFeatureDefault = 2000
kAdaptorNumRowDivs = 4
kAdaptorNumColDivs = 4
kDrawOriginalExtractedFeatures = False  # for debugging 


class FeatureDetectorTypes(Enum):
    SHI_TOMASI = 1
    FAST = 2    
    SIFT = 3
    SURF = 4
    ORB  = 5 


class FeatureDescriptorTypes(Enum):
    NONE = 0  # used for LK tracker
    SIFT = 1
    SURF = 2
    ORB  = 3     


def feature_detector_factory(min_num_features=kMinNumFeatureDefault, detector_type = FeatureDetectorTypes.FAST, descriptor_type = FeatureDescriptorTypes.ORB):
    return FeatureDetector(min_num_features, detector_type, descriptor_type)


class BlockAdaptor(object): 
    def __init__(self, detector, row_divs = kAdaptorNumRowDivs, col_divs = kAdaptorNumColDivs):    
        self.detector = detector 
        self.row_divs = row_divs
        self.col_divs = col_divs 

    def detect(self, frame, mask=None):
        if self.row_divs == 1 and self.col_divs == 1: 
            return self.detector.detect(frame, mask)
        else:    
            block_generator = imgBlocks(frame, self.row_divs, self.col_divs)
            kps_global = []
            for b, i, j in block_generator:
                kps = self.detector.detect(b)
                #print('adaptor: detected #features: ', len(kps), ' in block (',i,',',j,')')  
                for kp in kps:
                    #print('kp.pt before: ', kp.pt)
                    kp.pt = (kp.pt[0] + j, kp.pt[1] + i)        
                    #print('kp.pt after: ', kp.pt)                                                                     
                    kps_global.append(kp)
            return kps_global


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
    def __init__(self, min_num_features=kMinNumFeatureDefault, detector_type = FeatureDetectorTypes.SHI_TOMASI,  descriptor_type = FeatureDescriptorTypes.ORB):
        self.detector_type = detector_type 
        self.descriptor_type = descriptor_type

        self.num_levels = 8  
        self.scaleFactor=1.2  
        self.initSigmaLevels()

        self.min_num_features = min_num_features
        self.use_bock_adaptor = False 
        self.block_adaptor = None

        if cv2.__version__.split('.')[0] == '3':
            from cv2.xfeatures2d import SIFT_create, SURF_create
            from cv2 import ORB_create
        else:
            SIFT_create = cv2.SIFT
            SURF_create = cv2.SURF
            ORB_create = cv2.ORB 
        self.SIFT_create = SIFT_create
        self.SURF_create = SURF_create
        self.ORB_create = ORB_create     

        self.orb_params = dict(nfeatures=min_num_features,
                               scaleFactor=1.2,
                               nlevels=self.num_levels,
                               patchSize=21,
                               edgeThreshold = 21, 
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
        elif self.detector_type == FeatureDetectorTypes.FAST:
            self._feature_detector = cv2.FastFeatureDetector_create(threshold=25, nonmaxSuppression=True)  
            self.detector_name = 'FAST'                    
        elif self.detector_type == FeatureDetectorTypes.SHI_TOMASI:
            self._feature_detector = ShiTomasiDetector(self.min_num_features)  
            self.detector_name = 'Shi-Tomasi'
            #qself.use_bock_adaptor = True                                             
        else:
            raise ValueError("Unknown feature extractor %s" % self.detector_type)

        if self.use_bock_adaptor is True:
            self.block_adaptor = BlockAdaptor(self._feature_detector, row_divs = kAdaptorNumRowDivs, col_divs = kAdaptorNumColDivs)

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
        elif self.descriptor_type == FeatureDescriptorTypes.NONE:
            self._feature_descriptor = None              
            self.decriptor_name = 'None'                                     
        else:
            raise ValueError("Unknown feature extractor %s" % self.detector_type)            

    def initSigmaLevels(self): 
        self.vScaleFactor = np.zeros(self.num_levels)
        self.vLevelSigma2 = np.zeros(self.num_levels)
        self.vInvScaleFactor = np.zeros(self.num_levels)
        self.vInvLevelSigma2 = np.zeros(self.num_levels)

        self.vScaleFactor[0]=1.0
        self.vLevelSigma2[0]=1.0
        for i in range(1,self.num_levels):
            self.vScaleFactor[i]=self.vScaleFactor[i-1]*self.scaleFactor
            self.vLevelSigma2[i]=self.vScaleFactor[i]*self.vScaleFactor[i]
        #print('self.vScaleFactor: ', self.vScaleFactor)
        for i in range(self.num_levels):
            self.vInvScaleFactor[i]=1.0/self.vScaleFactor[i]
            self.vInvLevelSigma2[i]=1.0/self.vLevelSigma2[i]
        #print('self.vInvScaleFactor: ', self.vInvScaleFactor)            

    # detect keypoints without computing their descriptors
    # out: kps 
    def detect(self, frame, mask=None):  
        if self.use_bock_adaptor:
            kps = self.block_adaptor.detect(frame, mask)
        else:       
            kps = self._feature_detector.detect(frame, mask)         
        kps = self.satNumberOfFeatures(kps) 
        if kDrawOriginalExtractedFeatures: # draw the original features
            imgDraw = cv2.drawKeypoints(frame, kps, None, color=(0,255,0), flags=0)
            cv2.imshow('kps',imgDraw)            
        if kVerbose:
            print('detector: ', self.detector_name, ', #features: ', len(kps))    
        return kps        

    # detect keypoints and their descriptors
    # out: kps, des 
    def detectAndCompute(self, frame, mask=None):
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
        return kps 