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
from feature_detector import feature_detector_factory, FeatureDetectorTypes, FeatureDescriptorTypes
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes
from helpers import Printer
from geom_helpers import hamming_distance, l2_distance


class TrackerTypes(Enum):
    LK        = 0   # use patch as "descriptor" and match by using Lucas Kanade pyramid optic flow 
    DES_BF    = 1   # descriptor-based, brute force matching 
    DES_FLANN = 2   # descriptor-based, FLANN-based matching (should be faster)


kMinNumFeatureDefault = 2000


def feature_tracker_factory(min_num_features=kMinNumFeatureDefault, 
                            num_levels = 1, 
                            detector_type = FeatureDetectorTypes.FAST, 
                            descriptor_type = FeatureDescriptorTypes.ORB, 
                            tracker_type = TrackerTypes.LK):
    if tracker_type == TrackerTypes.LK:
        return LkFeatureTracker(min_num_features=min_num_features, num_levels = num_levels, detector_type=detector_type)
    else: 
        return DescriptorFeatureTracker(min_num_features=min_num_features, num_levels = num_levels, detector_type = detector_type, descriptor_type = descriptor_type, tracker_type = tracker_type)
    return None 


class TrackResult(object): 
    def __init__(self):
        self.kp_ref = None          # all reference keypoints 
        self.kp_cur = None          # all current keypoints
        self.des_cur = None         # all current descriptors 
        self.idx_ref = None         # indexes of matches in kp_ref so that kp_ref_matched = kp_ref[idx_ref]
        self.idx_cur = None         # indexes of matches in kp_cur so that kp_cur_matched = kp_cur[idx_cur]
        self.kp_ref_matched = None  # reference matched keypoints 
        self.kp_cur_matched = None  # current matched keypoints 


# base class 
class FeatureTracker(object): 
    def __init__(self, min_num_features=kMinNumFeatureDefault, 
                       num_levels = 1,
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.ORB, 
                       tracker_type = TrackerTypes.LK):
        self.min_num_features = min_num_features
        self.num_levels = num_levels 
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.tracker_type = tracker_type

        self.detector = None 
        self.descriptor_distance = None  # pointer function 

    # out: keypoints and descriptors 
    def detect(self, frame, mask): 
        return None, None 

    # out: kp1, kp2, des2
    def track(self, image_ref, image_cur, kp_ref, des_ref):
        return None, None, None             


# use patch as "descriptor" and track/"match" by using Lucas Kanade pyr optic flow 
class LkFeatureTracker(FeatureTracker): 
    def __init__(self, min_num_features=kMinNumFeatureDefault, 
                       num_levels = 3,
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.NONE, 
                       tracker_type = TrackerTypes.LK):   
        if num_levels < 3:
            Printer.green('LkFeatureTracker: forcing at least 3 levels') 
            num_levels = 3                        
        super().__init__(min_num_features, num_levels, detector_type, descriptor_type, tracker_type)
        self.detector = feature_detector_factory(min_num_features, num_levels, detector_type, descriptor_type)   
        # we use LK pyr optic flow for matching     
        self.lk_params = dict(winSize  = (21, 21), 
                              maxLevel = self.num_levels,
                              criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))        

    # out: keypoints and empty descriptors
    def detect(self, frame, mask=None):
        return self.detector.detect(frame, mask), None  

    # out: kp_ref_matched, kp_cur_matched, empty des_cur_matched
    def track(self, image_ref, image_cur, kp_ref, des_ref = None):
        kp_cur, st, err = cv2.calcOpticalFlowPyrLK(image_ref, image_cur, kp_ref, None, **self.lk_params)  #shape: [k,2] [k,1] [k,1]
        st = st.reshape(st.shape[0])
        res = TrackResult()    
        #res.idx_ref = (st == 1)
        res.idx_ref = [i for i,v in enumerate(st) if v== 1]
        res.idx_cur = res.idx_ref.copy()       
        res.kp_ref_matched = kp_ref[res.idx_ref] 
        res.kp_cur_matched = kp_cur[res.idx_cur]  
        res.kp_ref = res.kp_ref_matched  # with LK we follow feature trails hence we can forget unmatched features 
        res.kp_cur = res.kp_cur_matched
        res.des_cur = None                      
        return res         
        

# extract keypoints by using desired detector and descriptor, match keypoints by using desired matcher 
class DescriptorFeatureTracker(FeatureTracker): 
    def __init__(self, min_num_features=kMinNumFeatureDefault, 
                       num_levels = 1,
                       detector_type = FeatureDetectorTypes.FAST, 
                       descriptor_type = FeatureDescriptorTypes.ORB, 
                       tracker_type = TrackerTypes.DES_FLANN):
        super().__init__(min_num_features, num_levels, detector_type, descriptor_type, tracker_type)
        self.detector = feature_detector_factory(min_num_features=min_num_features, num_levels=num_levels, detector_type=detector_type, descriptor_type=descriptor_type)

        if descriptor_type == FeatureDescriptorTypes.ORB:
            self.norm_type = cv2.NORM_HAMMING
        elif descriptor_type == FeatureDescriptorTypes.BRISK:
            self.norm_type = cv2.NORM_HAMMING   
        elif descriptor_type == FeatureDescriptorTypes.AKAZE:
            self.norm_type = cv2.NORM_HAMMING       
        elif descriptor_type == FeatureDescriptorTypes.FREAK:
            self.norm_type = cv2.NORM_HAMMING                                           
        elif descriptor_type == FeatureDescriptorTypes.SURF:
            self.norm_type = cv2.NORM_L2            
        elif descriptor_type == FeatureDescriptorTypes.SIFT: 
            self.norm_type = cv2.NORM_L2
        else:
            raise ValueError("Unmanaged norm type for feature tracker %s" % self.tracker_type)      

        if self.norm_type == cv2.NORM_HAMMING:
            self.descriptor_distance = hamming_distance 
        if self.norm_type == cv2.NORM_L2:
            self.descriptor_distance = l2_distance            

        if tracker_type == TrackerTypes.DES_FLANN:
            self.matching_algo = FeatureMatcherTypes.FLANN
        elif tracker_type == TrackerTypes.DES_BF:
            self.matching_algo = FeatureMatcherTypes.BF
        else:
            raise ValueError("Unmanaged matching algo for feature tracker %s" % self.tracker_type)               
        
        self.matcher = feature_matcher_factory(norm_type=self.norm_type, type=self.matching_algo)        

    # out: keypoints and descriptors 
    def detect(self, frame, mask=None):
        return self.detector.detectAndCompute(frame, mask) 

    # out: kp_ref_matched, kp_cur_matched, des_cur_matched
    def track(self, image_ref, image_cur, kp_ref, des_ref):
        kp_cur, des_cur = self.detect(image_cur)
        # convert from list of keypoints to an array of points 
        kp_cur = np.array([x.pt for x in kp_cur], dtype=np.float32) 

        idx_ref, idx_cur = self.matcher.match(des_ref, des_cur)  #knnMatch(queryDescriptors,trainDescriptors)
        #print('num matches: ', len(matches))

        res = TrackResult()
        res.kp_ref = kp_ref  # let's keep all the original ref kps  
        res.kp_cur = kp_cur  # let's keep all the original cur kps       
        res.kp_ref_matched = np.asarray(kp_ref[idx_ref]) # here we put the matched ones 
        res.kp_cur_matched = np.asarray(kp_cur[idx_cur]) # here we put the matched ones 
        res.idx_cur = np.asarray(idx_cur)
        res.idx_ref = np.asarray(idx_ref)
        res.des_cur = des_cur         
        return res                 
