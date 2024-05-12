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
from parameters import Parameters  
from enum import Enum
from collections import defaultdict
import config
config.cfg.set_lib('xfeat') 

from modules.xfeat import XFeat

kRatioTest = Parameters.kFeatureMatchRatioTest
kVerbose = False

class FeatureMatcherTypes(Enum):
    NONE = 0
    BF = 1     
    FLANN = 2
    XFEAT = 3


def feature_matcher_factory(norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=kRatioTest, type=FeatureMatcherTypes.FLANN):
    if type == FeatureMatcherTypes.BF:
        return BfFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type == FeatureMatcherTypes.FLANN:
        return FlannFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type ==FeatureMatcherTypes.XFEAT:
        return  XFeatMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    return None 


"""
N.B.: 
The result of matches = matcher.knnMatch() is a list of cv2.DMatch objects. 
A DMatch object has the following attributes:
    DMatch.distance - Distance between descriptors. The lower, the better it is.
    DMatch.trainIdx - Index of the descriptor in train descriptors
    DMatch.queryIdx - Index of the descriptor in query descriptors
    DMatch.imgIdx - Index of the train image.
"""        


# base class 
import torch
class FeatureMatcher(object): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.BF):
        self.type = type 
        self.norm_type = norm_type 
        self.cross_check = cross_check   # apply cross check 
        self.matches = []
        self.ratio_test = ratio_test 
        self.matcher = None 
        self.matcher_name = ''
        
        
    # input: des1 = queryDescriptors, des2= trainDescriptors
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    def match(self, des1, des2, ratio_test=None):
        if kVerbose:
            print(self.matcher_name,', norm ', self.norm_type) 
        #print('des1.shape:',des1.shape,' des2.shape:',des2.shape)    
        #print('des1.dtype:',des1.dtype,' des2.dtype:',des2.dtype)
        #print(self.type)
        if self.type == FeatureMatcherTypes.XFEAT:
            d1_tensor = torch.tensor(des1, dtype=torch.float32)  # Specify dtype if needed
            d2_tensor = torch.tensor(des2, dtype=torch.float32)  # Specify dtype if needed

            # If the original tensors were on a GPU, you should move the new tensors to GPU as well
            # d1_tensor = d1_tensor.to('cuda')  # Use 'cuda' or 'cuda:0' if your device is a GPU
            # d2_tensor = d2_tensor.to('cuda') 
            idx0, idx1 = self.matcher.match(d1_tensor, d2_tensor, 0.93) 
            return idx0.cpu(), idx1.cpu()                  
        # matches = self.matcher.knnMatch(des1, des2, k=2)  #knnMatch(queryDescriptors,trainDescriptors)
        # self.matches = matches
        # return self.goodMatches(matches, des1, des2, ratio_test)          
    
    
   
    def goodMatchesOneToOne(self, matches, des1, des2, ratio_test=None):
        len_des2 = len(des2)
        idx1, idx2 = [], []  
        # good_matches = []           
        if ratio_test is None: 
            ratio_test = self.ratio_test
        if matches is not None:         
            float_inf = float('inf')
            dist_match = defaultdict(lambda: float_inf)   
            index_match = dict()  
            for m, n in matches:
                if m.distance > ratio_test * n.distance:
                    continue     
                dist = dist_match[m.trainIdx]
                if dist == float_inf: 
                    # trainIdx has not been matched yet
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idx2)-1
                else:
                    if m.distance < dist: 
                        # we have already a match for trainIdx: if stored match is worse => replace it
                        #print("double match on trainIdx: ", m.trainIdx)
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx) 
                        idx1[index]=m.queryIdx
                        idx2[index]=m.trainIdx                        
        return idx1, idx2



    def goodMatches(self, matches, des1, des2, ratio_test=None): 
        #return self.goodMatchesSimple(matches, des1, des2, ratio_test)   # <= N.B.: this generates problem in SLAM since it can produce matches where a trainIdx index is associated to two (or more) queryIdx indexes
        return self.goodMatchesOneToOne(matches, des1, des2, ratio_test)


# Brute-Force Matcher 
class BfFeatureMatcher(FeatureMatcher): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.BF):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        self.matcher = cv2.BFMatcher(norm_type, cross_check)     
        self.matcher_name = 'BfFeatureMatcher'   


class XFeatMatcher(FeatureMatcher):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.XFEAT):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        self.matcher = XFeat()    
        self.matcher_name = 'XFeatFeatureMatcher'   

# Flann Matcher 
class FlannFeatureMatcher(FeatureMatcher): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.FLANN):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        if norm_type == cv2.NORM_HAMMING:
            # FLANN parameters for binary descriptors 
            FLANN_INDEX_LSH = 6
            self.index_params= dict(algorithm = FLANN_INDEX_LSH,   # Multi-Probe LSH: Efficient Indexing for High-Dimensional Similarity Search
                        table_number = 6,      # 12
                        key_size = 12,         # 20
                        multi_probe_level = 1) # 2            
        if norm_type == cv2.NORM_L2: 
            # FLANN parameters for float descriptors 
            FLANN_INDEX_KDTREE = 1
            self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 4)  
        self.search_params = dict(checks=32)   # or pass empty dictionary                 
        self.matcher = cv2.FlannBasedMatcher(self.index_params, self.search_params)  
        self.matcher_name = 'FlannFeatureMatcher'                                                

