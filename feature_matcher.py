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

kRatioTest = 0.7 
kVerbose = True 

class FeatureMatcherTypes(Enum):
    NONE = 0
    BF = 1     
    FLANN = 2


def feature_matcher_factory(norm_type=cv2.NORM_HAMMING, cross_check = False, type=FeatureMatcherTypes.FLANN):
    if type == FeatureMatcherTypes.BF:
        return BfFeatureMatcher(norm_type, cross_check)
    if type == FeatureMatcherTypes.FLANN:
        return FlannFeatureMatcher(norm_type)
    return None 


# base class 
class FeatureMatcher(object): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, type = FeatureMatcherTypes.BF):
        self.type = type 
        self.norm_type = norm_type 
        self.cross_check = cross_check 

    def match(self, des1, des2):
        return None 

    # des1 = queryDescriptors, des2= trainDescriptors
    def goodMatches(self, matches, des1, des2):
        len_des2 = len(des2)
        idx1, idx2 = [], []        
        if matches is not None:         
            #good_matches = []
            flag_match = np.full(len_des2, False, dtype=bool)
            dist_match = np.zeros(len_des2)
            index_match = np.full(len_des2, 0, dtype=int)
            m = None # match to insert 
            for lm in matches:
                if len(lm)==0:
                    continue    
                elif len(lm)==1:
                    m = lm[0]
                else:
                    # we have two mathes for queryIdx 
                    if lm[0].distance < kRatioTest * lm[1].distance:
                        m = lm[0]
                        #print("match1: %d, %d" % (lm[0].queryIdx,lm[0].trainIdx))
                        #print("match2: %d, %d" % (lm[1].queryIdx,lm[1].trainIdx))                        
                    else:
                        continue     
                if not flag_match[m.trainIdx]: 
                    # trainIdx has not been matched yet
                    flag_match[m.trainIdx] = True
                    dist_match[m.trainIdx] = m.distance
                    #good_matches.append(m)
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idx2)-1
                else:
                    # we have already a match for trainIdx 
                    # if stored match is worse => replace it
                    if dist_match[m.trainIdx] > m.distance:
                        #print("double match on trainIdx: ", m.trainIdx)
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx) 
                        #good_matches[index] = m 
                        idx1[index]=m.queryIdx
                        idx2[index]=m.trainIdx                        
            #return good_matches, idx1, idx2
            return idx1, idx2
        else:
            #return matches, idx1, idx2                             
            return idx1, idx2

    # des1 = queryDescriptors, des2= trainDescriptors
    def goodMatchesSimple(self, matches, des1, des2):
        idx1, idx2 = [], []           
        if matches is not None: 
            # Apply ratio test
            #good_matches = []
            for m in matches:
                if len(m)==0:
                    continue    
                elif len(m)==1:
                    #good_matches.append(m[0]) 
                    idx1.append(m[0].queryIdx)
                    idx2.append(m[0].trainIdx)                    
                else:
                    m1 = m[0]
                    m2 = m[1]
                    if m1.distance < kRatioTest*m2.distance:
                        #good_matches.append(m1) 
                        idx1.append(m1.queryIdx)
                        idx2.append(m1.trainIdx)                            
            #return good_matches, idx1, idx2
            return idx1, idx2
        else:
            #return matches, idx1, idx2                             
            return idx1, idx2 


class BfFeatureMatcher(FeatureMatcher): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, type = FeatureMatcherTypes.BF):
        super().__init__(norm_type, cross_check, type)
        self.bf = cv2.BFMatcher(norm_type, cross_check)        

    def match(self, des1, des2):
        if kVerbose:
            print('BfFeatureMatcher, norm ', self.norm_type)        
        matches = self.bf.knnMatch(des1, des2, k=2)   #knnMatch(queryDescriptors,trainDescriptors)
        return self.goodMatches(matches, des1, des2)


class FlannFeatureMatcher(FeatureMatcher): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, type = FeatureMatcherTypes.FLANN):
        super().__init__(norm_type, cross_check, type)
        if norm_type == cv2.NORM_HAMMING:
            # FLANN parameters for binary descriptors 
            FLANN_INDEX_LSH = 6
            self.index_params= dict(algorithm = FLANN_INDEX_LSH,
                        table_number = 6,      # 12
                        key_size = 12,         # 20
                        multi_probe_level = 1) # 2            
        if norm_type == cv2.NORM_L2: 
            # FLANN parameters for float descriptors 
            FLANN_INDEX_KDTREE = 1
            self.index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)  
        self.search_params = dict(checks=50)   # or pass empty dictionary                 
        self.flann = cv2.FlannBasedMatcher(self.index_params, self.search_params)                                         

    def match(self, des1, des2):
        if kVerbose:
            print('FlannFeatureMatcher')
        matches = self.flann.knnMatch(des1, des2, k=2)  #knnMatch(queryDescriptors,trainDescriptors)
        return self.goodMatches(matches, des1, des2)             