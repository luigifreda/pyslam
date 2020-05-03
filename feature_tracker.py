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

import numpy as np 
import cv2
from enum import Enum

from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from utils import Printer, import_from
from utils_geom import hamming_distance, hamming_distances, l2_distance, l2_distances
from parameters import Parameters 


kMinNumFeatureDefault = 2000
kLkPyrOpticFlowNumLevelsMin = 3   # maximal pyramid level number for LK optic flow 
kRatioTest = Parameters.kFeatureMatchRatioTest


class FeatureTrackerTypes(Enum):
    LK        = 0   # Lucas Kanade pyramid optic flow (use pixel patch as "descriptor" and matching by optimization)
    DES_BF    = 1   # descriptor-based, brute force matching with knn 
    DES_FLANN = 2   # descriptor-based, FLANN-based matching 


def feature_tracker_factory(num_features=kMinNumFeatureDefault, 
                            num_levels = 1,                                 # number of pyramid levels or octaves for detector and descriptor   
                            scale_factor = 1.2,                             # detection scale factor (if it can be set, otherwise it is automatically computed)
                            detector_type = FeatureDetectorTypes.FAST, 
                            descriptor_type = FeatureDescriptorTypes.ORB, 
                            match_ratio_test = kRatioTest,
                            tracker_type = FeatureTrackerTypes.LK):
    if tracker_type == FeatureTrackerTypes.LK:
        return LkFeatureTracker(num_features=num_features, 
                                num_levels = num_levels, 
                                scale_factor = scale_factor, 
                                detector_type = detector_type, 
                                descriptor_type = descriptor_type, 
                                match_ratio_test = match_ratio_test,                                
                                tracker_type = tracker_type)
    else: 
        return DescriptorFeatureTracker(num_features=num_features, 
                                        num_levels = num_levels, 
                                        scale_factor = scale_factor, 
                                        detector_type = detector_type, 
                                        descriptor_type = descriptor_type,
                                        match_ratio_test = match_ratio_test,    
                                        tracker_type = tracker_type)
    return None 


class FeatureTrackingResult(object): 
    def __init__(self):
        self.kps_ref = None          # all reference keypoints (numpy array Nx2)
        self.kps_cur = None          # all current keypoints   (numpy array Nx2)
        self.des_cur = None          # all current descriptors (numpy array NxD)
        self.idxs_ref = None         # indexes of matches in kps_ref so that kps_ref_matched = kps_ref[idxs_ref]  (numpy array of indexes)
        self.idxs_cur = None         # indexes of matches in kps_cur so that kps_cur_matched = kps_cur[idxs_cur]  (numpy array of indexes)
        self.kps_ref_matched = None  # reference matched keypoints, kps_ref_matched = kps_ref[idxs_ref]
        self.kps_cur_matched = None  # current matched keypoints, kps_cur_matched = kps_cur[idxs_cur]


# Base class for a feature tracker.
# It mainly contains a feature manager and a feature matcher. 
class FeatureTracker(object): 
    def __init__(self, num_features=kMinNumFeatureDefault, 
                       num_levels = 1,                                   # number of pyramid levels for detector and descriptor  
                       scale_factor = 1.2,                               # detection scale factor (if it can be set, otherwise it is automatically computed) 
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.ORB,
                       match_ratio_test = kRatioTest, 
                       tracker_type = FeatureTrackerTypes.LK):
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.tracker_type = tracker_type

        self.feature_manager = None      # it contains both detector and descriptor  
        self.matcher = None              # it contain descriptors matching methods based on BF, FLANN, etc.
                
    @property
    def num_features(self):
        return self.feature_manager.num_features
    
    @property
    def num_levels(self):
        return self.feature_manager.num_levels    
    
    @property
    def scale_factor(self):
        return self.feature_manager.scale_factor    
    
    @property
    def norm_type(self):
        return self.feature_manager.norm_type       
    
    @property
    def descriptor_distance(self):
        return self.feature_manager.descriptor_distance               
    
    @property
    def descriptor_distances(self):
        return self.feature_manager.descriptor_distances               
    
    # out: keypoints and descriptors 
    def detectAndCompute(self, frame, mask): 
        return None, None 

    # out: FeatureTrackingResult()
    def track(self, image_ref, image_cur, kps_ref, des_ref):
        return FeatureTrackingResult()             


# Lucas-Kanade Tracker: it uses raw pixel patches as "descriptors" and track/"match" by using Lucas Kanade pyr optic flow 
class LkFeatureTracker(FeatureTracker): 
    def __init__(self, num_features=kMinNumFeatureDefault, 
                       num_levels = 3,                             # number of pyramid levels for detector  
                       scale_factor = 1.2,                         # detection scale factor (if it can be set, otherwise it is automatically computed) 
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.NONE, 
                       match_ratio_test = kRatioTest,
                       tracker_type = FeatureTrackerTypes.LK):                         
        super().__init__(num_features=num_features, 
                         num_levels=num_levels, 
                         scale_factor=scale_factor, 
                         detector_type=detector_type, 
                         descriptor_type=descriptor_type, 
                         tracker_type=tracker_type)
        self.feature_manager = feature_manager_factory(num_features=num_features, 
                                                       num_levels=num_levels, 
                                                       scale_factor=scale_factor, 
                                                       detector_type=detector_type, 
                                                       descriptor_type=descriptor_type)   
        #if num_levels < 3:
        #    Printer.green('LkFeatureTracker: forcing at least 3 levels on LK pyr optic flow') 
        #    num_levels = 3          
        optic_flow_num_levels = max(kLkPyrOpticFlowNumLevelsMin,num_levels)
        Printer.green('LkFeatureTracker: num levels on LK pyr optic flow: ', optic_flow_num_levels)
        # we use LK pyr optic flow for matching     
        self.lk_params = dict(winSize  = (21, 21), 
                              maxLevel = optic_flow_num_levels,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))        

    # out: keypoints and empty descriptors
    def detectAndCompute(self, frame, mask=None):
        return self.feature_manager.detect(frame, mask), None  

    # out: FeatureTrackingResult()
    def track(self, image_ref, image_cur, kps_ref, des_ref = None):
        kps_cur, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, kps_ref, None, **self.lk_params)  #shape: [k,2] [k,1] [k,1]
        st = st.reshape(st.shape[0])
        res = FeatureTrackingResult()    
        #res.idxs_ref = (st == 1)
        res.idxs_ref = [i for i,v in enumerate(st) if v== 1]
        res.idxs_cur = res.idxs_ref.copy()       
        res.kps_ref_matched = kps_ref[res.idxs_ref] 
        res.kps_cur_matched = kps_cur[res.idxs_cur]  
        res.kps_ref = res.kps_ref_matched  # with LK we follow feature trails hence we can forget unmatched features 
        res.kps_cur = res.kps_cur_matched
        res.des_cur = None                      
        return res         
        

# Extract features by using desired detector and descriptor, match keypoints by using desired matcher on computed descriptors
class DescriptorFeatureTracker(FeatureTracker): 
    def __init__(self, num_features=kMinNumFeatureDefault, 
                       num_levels = 1,                                    # number of pyramid levels for detector  
                       scale_factor = 1.2,                                # detection scale factor (if it can be set, otherwise it is automatically computed)                
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.ORB,
                       match_ratio_test = kRatioTest, 
                       tracker_type = FeatureTrackerTypes.DES_FLANN):
        super().__init__(num_features=num_features, 
                         num_levels=num_levels, 
                         scale_factor=scale_factor, 
                         detector_type=detector_type, 
                         descriptor_type=descriptor_type, 
                         match_ratio_test = match_ratio_test,
                         tracker_type=tracker_type)
        self.feature_manager = feature_manager_factory(num_features=num_features, 
                                                       num_levels=num_levels, 
                                                       scale_factor=scale_factor, 
                                                       detector_type=detector_type, 
                                                       descriptor_type=descriptor_type)                     

        if tracker_type == FeatureTrackerTypes.DES_FLANN:
            self.matching_algo = FeatureMatcherTypes.FLANN
        elif tracker_type == FeatureTrackerTypes.DES_BF:
            self.matching_algo = FeatureMatcherTypes.BF
        else:
            raise ValueError("Unmanaged matching algo for feature tracker %s" % self.tracker_type)                   
                    
        # init matcher 
        self.matcher = feature_matcher_factory(norm_type=self.norm_type, ratio_test=match_ratio_test, type=self.matching_algo)        


    # out: keypoints and descriptors 
    def detectAndCompute(self, frame, mask=None):
        return self.feature_manager.detectAndCompute(frame, mask) 


    # out: FeatureTrackingResult()
    def track(self, image_ref, image_cur, kps_ref, des_ref):
        kps_cur, des_cur = self.detectAndCompute(image_cur)
        # convert from list of keypoints to an array of points 
        kps_cur = np.array([x.pt for x in kps_cur], dtype=np.float32) 
    
        idxs_ref, idxs_cur = self.matcher.match(des_ref, des_cur)  #knnMatch(queryDescriptors,trainDescriptors)
        #print('num matches: ', len(matches))

        res = FeatureTrackingResult()
        res.kps_ref = kps_ref  # all the reference keypoints  
        res.kps_cur = kps_cur  # all the current keypoints       
        res.des_cur = des_cur  # all the current descriptors         
        
        res.kps_ref_matched = np.asarray(kps_ref[idxs_ref]) # the matched ref kps  
        res.idxs_ref = np.asarray(idxs_ref)                  
        
        res.kps_cur_matched = np.asarray(kps_cur[idxs_cur]) # the matched cur kps  
        res.idxs_cur = np.asarray(idxs_cur)
        
        return res                 
