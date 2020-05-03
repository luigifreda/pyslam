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
import math 
import numpy as np 
import cv2

from scipy.spatial import cKDTree
import numpy as np

from enum import Enum

import parameters  

from geom_helpers import unpackSiftOctaveKps
from feature_manager_adaptors import BlockAdaptor, PyramidAdaptor

try:
    from feature_superpoint import SuperPointFeature2D 
except ImportError:
    print('problems when importing TfeatFeature2D, check the file TROUBLESHOOTING.md')
    
try:
    from feature_tfeat import TfeatFeature2D                     
except ImportError:
    print('problems when importing TfeatFeature2D, check the file TROUBLESHOOTING.md')

kVerbose = True   

kMinNumFeatureDefault = 2000

kNumLevels = 4
kNumLevelsInitSigma = 12
kScaleFactor = 1.2 
kSigmaLevel0 = 1. 

kDrawOriginalExtractedFeatures = False  # for debugging 
kUseKdtNonMaximumSuppressionForKps = parameters.kUseKdtNonMaximumSuppressionForKps 

class FeatureDetectorTypes(Enum):
    SHI_TOMASI = 1
    FAST       = 2    
    SIFT       = 3
    ROOT_SIFT  = 4 
    SURF       = 5
    ORB        = 6 
    BRISK      = 7
    AKAZE      = 8
    FREAK      = 9  # DOES NOT WORK in my setup! "The function/feature is not implemented"
    SUPERPOINT = 10


class FeatureDescriptorTypes(Enum):
    NONE       = 0  # used for LK tracker
    SIFT       = 1
    ROOT_SIFT  = 2    
    SURF       = 3
    ORB        = 4  
    BRISK      = 5       
    AKAZE      = 6
    FREAK      = 7  # DOES NOT WORK in my setup! "The function/feature is not implemented"
    SUPERPOINT = 8    
    TFEAT      = 9


def feature_manager_factory(min_num_features=kMinNumFeatureDefault, 
                            num_levels = kNumLevels,                        # number of pyramid levels or octaves for detector and descriptor
                            scale_factor = kScaleFactor,                    # detection scale factor (if it can be set, otherwise it is automatically computed)
                            detector_type = FeatureDetectorTypes.FAST, 
                            descriptor_type = FeatureDescriptorTypes.ORB):
    return FeatureManager(min_num_features, num_levels, scale_factor, detector_type, descriptor_type)


class ShiTomasiDetector(object): 
    def __init__(self, min_num_features=kMinNumFeatureDefault, quality_level = 0.01, min_coner_distance = 7):
        self.min_num_features = min_num_features
        self.quality_level = quality_level
        self.min_coner_distance = min_coner_distance
        self.blockSize=3

    def detect(self, frame, mask=None):                
        pts = cv2.goodFeaturesToTrack(frame, self.min_num_features, self.quality_level, self.min_coner_distance, blockSize=self.blockSize, mask=mask)
        # convert matrix of pts into list of keypoints 
        if pts is not None: 
            kps = [ cv2.KeyPoint(p[0][0], p[0][1], self.blockSize) for p in pts ]
        else:
            kps = []
        if kVerbose:
            print('detector: Shi-Tomasi, #features: ', len(kps), ', #ref: ', self.min_num_features, ', frame res: ', frame.shape[0:2])      
        return kps


# https://www.robots.ox.ac.uk/~vgg/publications/2012/Arandjelovic12/arandjelovic12.pdf
# adapated from https://www.pyimagesearch.com/2015/04/13/implementing-rootsift-in-python-and-opencv/
class RootSIFTFeature2D:
    def __init__(self, feature):
        # initialize the SIFT feature detector
        self.feature = feature

    def detect(self, frame, mask=None):
        return self.feature.detect(frame, mask)
 
    def transform_descriptors(self, des, eps=1e-7): 
        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        des /= (des.sum(axis=1, keepdims=True) + eps)
        des = np.sqrt(des)        
        return des 
            
    def compute(self, frame, kps, eps=1e-7):
        # compute SIFT descriptors
        (kps, des) = self.feature.compute(frame, kps)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        des = self.transform_descriptors(des)

        # return a tuple of the keypoints and descriptors
        return (kps, des)

    # detect keypoints and their descriptors
    # out: kps, des 
    def detectAndCompute(self, frame, mask=None):
        # compute SIFT keypoints and descriptors
        (kps, des) = self.feature.detectAndCompute(frame, mask)

        # if there are no keypoints or descriptors, return an empty tuple
        if len(kps) == 0:
            return ([], None)

        # apply the Hellinger kernel by first L1-normalizing and 
        # taking the square-root
        des = self.transform_descriptors(des)

        # return a tuple of the keypoints and descriptors
        return (kps, des)


# Manager of both detector and descriptor 
# This exposes methods thath similar to OpenCV::Feature2D, i.e. detect() and detectAndCompute()
class FeatureManager(object):
    def __init__(self, min_num_features=kMinNumFeatureDefault, 
                       num_levels = kNumLevels,                         # number of pyramid levels or octaves for detector and descriptor
                       scale_factor = kScaleFactor,                     # detection scale factor (if it can be set, otherwise it is automatically computed)
                       detector_type = FeatureDetectorTypes.SHI_TOMASI,  
                       descriptor_type = FeatureDescriptorTypes.ORB):
        self.detector_type = detector_type 
        self.descriptor_type = descriptor_type
        
        self._feature_detector = None 
        self.detector_name = ''
        
        self._feature_descriptor = None 
        self.decriptor_name = ''

        self.num_levels = num_levels  
        self.scale_factor = scale_factor  # scale factor bewteen two octaves 
        self.sigma_level0 = kSigmaLevel0  # sigma on first octave 
        self.initSigmaLevels()

        self.min_num_features = min_num_features
        # at present time pyramid adaptor has the priority and can combine a block adaptor withint itself 
        self.use_bock_adaptor = False 
        self.block_adaptor = None
        self.use_pyramid_adaptor = False 
        self.pyramid_adaptor = None 
        
        self.need_nms = False # need or not non-maximum suppression of keypoints 

        print("using opencv ", cv2.__version__)
        # check opencv version in order to use the right modules 
        if cv2.__version__.split('.')[0] == '3':
            from cv2.xfeatures2d import SIFT_create, SURF_create, FREAK_create   
            from cv2 import ORB_create, BRISK_create, AKAZE_create
        else:
            SIFT_create = cv2.SIFT
            SURF_create = cv2.SURF
            ORB_create = cv2.ORB 
            BRISK_create = cv2.BRISK
            AKAZE_create = cv2.AKAZE 
            FREAK_create = cv2.FREAK # TODO: to be checked 

        self.FAST_create = cv2.FastFeatureDetector_create
        self.SIFT_create = SIFT_create
        self.SURF_create = SURF_create
        self.ORB_create = ORB_create 
        self.BRISK_create = BRISK_create            
        self.AKAZE_create = AKAZE_create   
        self.FREAK_create = FREAK_create   

        self.orb_params = dict(nfeatures=min_num_features,
                               scaleFactor=self.scale_factor,
                               nlevels=self.num_levels,
                               patchSize=31,
                               edgeThreshold = 31, #19, 
                               fastThreshold = 20,
                               firstLevel = 0,
                               WTA_K = 2,
                               scoreType=cv2.ORB_HARRIS_SCORE)  #scoreType=cv2.ORB_HARRIS_SCORE, scoreType=cv2.ORB_FAST_SCORE 
        
        # --------------- #
        # init detector 
        # --------------- #
        if self.detector_type == FeatureDetectorTypes.SIFT:
            self.detector_name = 'SIFT'
            self._feature_detector = self.SIFT_create(nOctaveLayers=3)  
            # N.B.: The number of octaves is computed automatically from the image resolution, here we set the number of layers in each octave.
            #  from https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
            self.intra_layer_factor = 1.2599  # num layers = nOctaves*nOctaveLayers  scale=2^(1/nOctaveLayers) = 1.2599  
            self.scale_factor = 2             # scale factor between octaves  
            self.sigma_level0 = 1.6  # from https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html                                                
        elif self.detector_type == FeatureDetectorTypes.ROOT_SIFT: 
            self.detector_name = 'ROOT_SIFT'          
            sift_detector = self.SIFT_create(nOctaveLayers=3)  
            # N.B.: The number of octaves is computed automatically from the image resolution, here we set the number of layers in each octave.
            #  from https://docs.opencv.org/3.4/d5/d3c/classcv_1_1xfeatures2d_1_1SIFT.html
            self.intra_layer_factor = 1.2599   # num layers = nOctaves*nOctaveLayers  scale=2^(1/nOctaveLayers) = 1.2599  
            self.scale_factor = 2              # scale factor between octaves  
            self.sigma_level0 = 1.6  # from https://docs.opencv.org/3.1.0/da/df5/tutorial_py_sift_intro.html                
            self._feature_detector = RootSIFTFeature2D(sift_detector)  
        elif self.detector_type == FeatureDetectorTypes.SURF:
            self.detector_name = 'SURF'              
            self._feature_detector = self.SURF_create(nOctaves = self.num_levels, nOctaveLayers=3)  
            self.intra_layer_factor = 1.2599   # num layers = nOctaves*nOctaveLayers  scale=2^(1/nOctaveLayers) = 1.2599  
            self.scale_factor = 2              # scale factor between octaves               
        elif self.detector_type == FeatureDetectorTypes.ORB:
            self.detector_name = 'ORB'              
            self._feature_detector = self.ORB_create(**self.orb_params)               
            self.use_bock_adaptor = False   # add a block adaptor? your choice ...
        elif self.detector_type == FeatureDetectorTypes.BRISK:
            self.detector_name = 'BRISK'              
            self._feature_detector = self.BRISK_create(octaves=self.num_levels) 
            #self.intra_layer_factor = 1.3   # from the BRISK opencv code this seems to be the used scale factor between intra-octave frames  
            self.intra_layer_factor = math.sqrt(2) # approx, num layers = nOctaves*nOctaveLayers, from the BRISK paper there are octave ci and intra-octave di layers, t(ci)=2^i, t(di)=2^i * 1.5    
            self.scale_factor = 2                  # scale factor between octaves                  
        elif self.detector_type == FeatureDetectorTypes.AKAZE:
            self.detector_name = 'AKAZE'               
            self._feature_detector = self.AKAZE_create(nOctaves=self.num_levels) 
            self.scale_factor = 2                  # scale factor between octaves    
        elif self.detector_type == FeatureDetectorTypes.FREAK: 
            self.detector_name = 'FREAK'                
            self._feature_detector = self.FREAK_create(nOctaves=self.num_levels)
            self.scale_factor = 2                  # scale factor between octaves 
        elif self.detector_type == FeatureDetectorTypes.SUPERPOINT: 
            self.detector_name = 'SUPERPOINT'                 
            self._feature_detector = SuperPointFeature2D()   
            self.use_pyramid_adaptor = self.num_levels > 1    
            self.need_nms = self.num_levels > 1                                                                     
        elif self.detector_type == FeatureDetectorTypes.FAST:
            self.detector_name = 'FAST'                  
            self._feature_detector = self.FAST_create(threshold=25, nonmaxSuppression=True)    
            self.use_bock_adaptor = False  # override a block adaptor?           
            self.use_pyramid_adaptor = self.num_levels > 1         
            self.need_nms = self.num_levels > 1       
        elif self.detector_type == FeatureDetectorTypes.SHI_TOMASI:
            self.detector_name = 'SHI-TOMASI'            
            self._feature_detector = ShiTomasiDetector(self.min_num_features)  
            self.use_bock_adaptor = False  # override a block adaptor?
            self.use_pyramid_adaptor = self.num_levels > 1 
            self.need_nms = self.num_levels > 1 
        else:
            raise ValueError("Unknown feature detector %s" % self.detector_type)

        self.initSigmaLevels()      
                    
        # --------------- #
        # init descriptor 
        # --------------- #        
        if self.descriptor_type == FeatureDescriptorTypes.SIFT: 
            self.decriptor_name = 'SIFT'               
            self._feature_descriptor = self.SIFT_create()  
        elif self.descriptor_type == FeatureDescriptorTypes.ROOT_SIFT: 
            self.decriptor_name = 'ROOT_SIFT'                 
            self._feature_descriptor = RootSIFTFeature2D(self.SIFT_create)                     
        elif self.descriptor_type == FeatureDescriptorTypes.SURF:
            self.decriptor_name = 'SURF'                              
            self._feature_descriptor = self.SURF_create(nOctaveLayers=self.num_levels)         
        elif self.descriptor_type == FeatureDescriptorTypes.ORB:
            self.decriptor_name = 'ORB'                                        
            self._feature_descriptor = self.ORB_create(**self.orb_params) 
        elif self.descriptor_type == FeatureDescriptorTypes.BRISK:
            self.decriptor_name = 'BRISK'              
            self._feature_descriptor = self.BRISK_create(octaves=self.num_levels)        
        elif self.descriptor_type == FeatureDescriptorTypes.AKAZE:
            self.decriptor_name = 'AKAZE'              
            self._feature_descriptor = self.AKAZE_create(nOctaves=self.num_levels) 
        elif self.descriptor_type == FeatureDescriptorTypes.FREAK: 
            self.decriptor_name = 'FREAK'             
            self._feature_descriptor = self.FREAK_create(nOctaves=self.num_levels)   
        elif self.descriptor_type == FeatureDescriptorTypes.SUPERPOINT: 
            self.decriptor_name = 'SUPERPOINT'               
            if self.detector_type != FeatureDetectorTypes.SUPERPOINT: 
                raise ValueError("At the present time, you cannot use SuperPoint descriptor without SuperPoint detector!\nPlease, select SuperPoint as both descriptor and detector!")
            self._feature_descriptor = self._feature_detector  # reuse the same SuperPointDector object                                     
        elif self.descriptor_type == FeatureDescriptorTypes.TFEAT: 
            self.decriptor_name = 'TFEAT'                
            self._feature_descriptor = TfeatFeature2D()                                                                      
        elif self.descriptor_type == FeatureDescriptorTypes.NONE:
            self.decriptor_name = 'NONE'                 
            self._feature_descriptor = None                                              
        else:
            raise ValueError("Unknown feature descriptor %s" % self.detector_type)    
        
        # check if detector and descriptor manager are identical (have same name)
        self.is_detector_equal_to_descriptor = False
        if self.detector_name == self.decriptor_name:  
            self.is_detector_equal_to_descriptor = True  
            self._feature_descriptor = self._feature_detector
            
        if self.use_bock_adaptor:
            self.block_adaptor = BlockAdaptor(self._feature_detector, self._feature_descriptor)

        if self.use_pyramid_adaptor:            
            self.pyramid_adaptor = PyramidAdaptor(self._feature_detector, self._feature_descriptor, self.num_levels, self.scale_factor, use_block_adaptor=self.use_bock_adaptor)
            

    # initialize scale factors, sigmas for each octave level (used for managing image pyramids)
    def initSigmaLevels(self): 
        num_levels = max(kNumLevelsInitSigma, self.num_levels)        
        self.scale_factors = np.zeros(num_levels)
        self.level_sigmas2 = np.zeros(num_levels)
        self.inv_scale_factors = np.zeros(num_levels)
        self.inv_level_sigmas2 = np.zeros(num_levels)

        # TODO: in the SIFT case, this sigma management could be refined. 
        #        SIFT method has layers with intra-layer scale factor = math.sqrt(2)
        self.scale_factors[0]=1.0
        self.level_sigmas2[0]=self.sigma_level0*self.sigma_level0
        for i in range(1,num_levels):
            self.scale_factors[i]=self.scale_factors[i-1]*self.scale_factor
            self.level_sigmas2[i]=self.scale_factors[i]*self.scale_factors[i]*self.level_sigmas2[i-1]
        #print('self.scale_factors: ', self.scale_factors)
        for i in range(num_levels):
            self.inv_scale_factors[i]=1.0/self.scale_factors[i]
            self.inv_level_sigmas2[i]=1.0/self.level_sigmas2[i]
        #print('self.inv_scale_factors: ', self.inv_scale_factors)            

    # detect keypoints without computing their descriptors
    # out: kps 
    def detect(self, frame, mask=None): 
        if frame.ndim>2: # check if we have to convert to gray image 
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)             
        if self.use_pyramid_adaptor:  # detection with pyramid adaptor (priority w.r.t. simple block adaptor)
            kps = self.pyramid_adaptor.detect(frame, mask)            
        elif self.use_bock_adaptor:   # detection with block adaptor 
            kps = self.block_adaptor.detect(frame, mask)            
        else:                         # standard detection      
            kps = self._feature_detector.detect(frame, mask) 
        # keep the first 'self.min_num_features' best features or apply NMS 
        if self.need_nms and kUseKdtNonMaximumSuppressionForKps:      
            kps, _ = self.kdt_nms(kps)                   
        else:
            kps, _ = self.satNumberOfFeatures(kps)                 
        if kDrawOriginalExtractedFeatures: # draw the original features
            imgDraw = cv2.drawKeypoints(frame, kps, None, color=(0,255,0), flags=0)
            cv2.imshow('detected keypoints',imgDraw)            
        if kVerbose:
            print('detector: ', self.detector_name, ', #features: ', len(kps))    
        return kps        

    # detect keypoints and their descriptors
    # out: kps, des 
    def detectAndCompute(self, frame, mask=None):
        if frame.ndim>2: # check if we have to convert to gray image 
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2GRAY)  
        if self.use_pyramid_adaptor:  # detectAndCompute with pyramid adaptor (priority w.r.t. simple block adaptor)
            kps, des = self.pyramid_adaptor.detectAndCompute(frame, mask)            
        elif self.use_bock_adaptor:   # detectAndCompute with block adaptor 
            kps, des = self.block_adaptor.detectAndCompute(frame, mask)            
        else:                         # standard detectAndCompute  
            if self.is_detector_equal_to_descriptor:                             
                kps, des = self._feature_detector.detectAndCompute(frame, mask)   
                if kVerbose:
                    #print('using optimized detectAndCompute()')
                    print('detector: ', self.detector_name, ', #features: ', len(kps))           
                    print('descriptor: ', self.decriptor_name, ', #features: ', len(kps))                      
            else:
                # detector and descriptor are different => call them separately 
                # first, detect keypoint locations  
                kps = self.detect(frame, mask)    
                # then, compute descriptors  
                kps, des = self._feature_descriptor.compute(frame, kps)  
                if kVerbose:
                    #print('detector: ', self.detector_name, ', #features: ', len(kps))           
                    print('descriptor: ', self.decriptor_name, ', #features: ', len(kps))                                           
        # keep the first 'self.min_num_features' best features or apply NMS 
        if self.need_nms and kUseKdtNonMaximumSuppressionForKps: 
            kps, des = self.kdt_nms(kps,des)   
        else:
            kps, des = self.satNumberOfFeatures(kps,des)          
        if self.detector_type == FeatureDetectorTypes.SIFT or self.detector_type == FeatureDetectorTypes.ROOT_SIFT:
            unpackSiftOctaveKps(kps)                                        
        return kps, des           

    # keep the first 'self.min_num_features' best features
    # TODO: improve it by using similar computations to kdt_nms
    def satNumberOfFeatures(self, kps, des=None):    
        if len(kps) > self.min_num_features:
            # keep the features with the best response 
            if des is None: 
                kps = sorted(kps, key=lambda x:x.response, reverse=True)[:self.min_num_features]     
            else:            
                # sort by score to keep highest score features 
                print('sat with des')
                neg_responses = [-kp.response for kp in kps]
                order = np.argsort(neg_responses)       
                kps = np.array(kps)[order].tolist()[:self.min_num_features]       
                des = (np.array(des)[order])[:self.min_num_features]            
            if kVerbose:
                print('detector [sat]: ', self.detector_name, ', #features: ', len(kps),', #max: ', self.min_num_features)                        
            if False: 
                for k in kps:
                    print("response: ", k.response) 
                    print("size: ", k.size)  
                    print("octave: ", k.octave)           
        return kps, des 
     
    # kdtree-based non-maximum suppression of keypoints 
    # adapted and optimized from https://stackoverflow.com/questions/9210431/well-distributed-features-using-opencv/50267891
    def kdt_nms(self, kps, des=None, r=parameters.kKdtNmsRadius, k_max=parameters.kKdtNmsKmax):
        """ Use kd-tree to perform local non-maximum suppression of key-points
        kps - key points obtained by one of openCVs 2d features detectors (SIFT, SURF, AKAZE etc..)
        r - the radius of points to query for removal
        k_max - maximum points retreived in single query
        """
        
        if kps is None:
            return kps, des
        
        # sort by score to keep highest score features 
        neg_responses = [-kp.response for kp in kps]
        order = np.argsort(neg_responses)       
        kps = np.array(kps)[order].tolist()

        # create kd-tree for quick NN queries
        data = np.array([list(kp.pt) for kp in kps])
        kd_tree = cKDTree(data)

        # perform NMS using kd-tree, by querying points by score order, 
        # and removing neighbors from future queries
        N = len(kps)
        idx_removed = set()
        for i in range(N):
            if i in idx_removed:
                continue

            dist, inds = kd_tree.query(data[i,:],k=k_max,distance_upper_bound=r)
            for j in inds: 
                if j>i:
                    idx_removed.add(j)

        idx_remaining = [i for i in range(0,len(kps)) if i not in idx_removed]
        kp_out = np.array(kps)[idx_remaining].tolist()
        des_out = None
        if des is not None:
            des = des[order]
            des_out = des[idx_remaining]
        if len(kp_out) > self.min_num_features:
            kp_out = kp_out[:self.min_num_features]
            if des_out is not None:
                des_out = des_out[:self.min_num_features]
        if kVerbose:
            #print('Remaining #features: ',len(kp_filtered),' of ',N, ' kps')
            print('detector [NMS]: ', self.detector_name, ', #features: ', len(kp_out),', #max: ', self.min_num_features)        
        if False: 
            for k in kp_out:
                print("response: ", k.response) 
                print("size: ", k.size)  
                print("octave: ", k.octave)                       
        return kp_out, des_out 
