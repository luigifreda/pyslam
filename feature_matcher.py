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
import torch
from utils_sys import Printer, import_from
from parameters import Parameters  
from enum import Enum
from collections import defaultdict

from frame import Frame
import config
config.cfg.set_lib('xfeat') 
config.cfg.set_lib('lightglue')

XFeat = import_from('modules.xfeat', 'XFeat')
LightGlue = import_from('lightglue', 'LightGlue')


kRatioTest = Parameters.kFeatureMatchRatioTest
kVerbose = False

class FeatureMatcherTypes(Enum):
    NONE      = 0
    BF        = 1      # Brute force
    FLANN     = 2      # FLANN-based 
    XFEAT     = 3      # "XFeat: Accelerated Features for Lightweight Image Matching"
    LIGHTGLUE = 4      # "LightGlue: Local Feature Matching at Light Speed"


def feature_matcher_factory(norm_type=cv2.NORM_HAMMING, cross_check=False, ratio_test=kRatioTest, type=FeatureMatcherTypes.FLANN):
    if type == FeatureMatcherTypes.BF:
        return BfFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type == FeatureMatcherTypes.FLANN:
        return FlannFeatureMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type ==FeatureMatcherTypes.XFEAT:
        return  XFeatMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    if type == FeatureMatcherTypes.LIGHTGLUE:
        return LightGlueMatcher(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
    return None 



class MatcherUtils: 
    # input: des1 = query-descriptors, des2 = train-descriptors
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.: this returns matches where each trainIdx index is associated to only one queryIdx index
    @staticmethod    
    def goodMatchesOneToOne(matches, des1, des2, ratio_test=0.7):
        len_des2 = len(des2)
        idx1, idx2 = [], []           
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

    # input: des1 = query-descriptors, des2 = train-descriptors
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.: this may return matches where a trainIdx index is associated to two (or more) queryIdx indexes
    @staticmethod    
    def goodMatchesSimple(matches, des1, des2, ratio_test=0.7):
        idx1, idx2 = [], []                  
        if matches is not None: 
            for m,n in matches:
                if m.distance < ratio_test * n.distance:
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)                                                         
        return idx1, idx2 


    # input: des1 = query-descriptors, des2 = train-descriptors, kps1 = query-keypoints, kps2 = train-keypoints 
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.0: cross checking can be also enabled with the BruteForce Matcher below 
    # N.B.1: after matching there is a model fitting with fundamental matrix estimation 
    # N.B.2: fitting a fundamental matrix has problems in the following cases: [see Hartley/Zisserman Book]
    # - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie a ruled quadric 
    # - degenerate motions such a pure rotation (a sufficient parallax is required) or an infinitesimal viewpoint change (where the translation is almost zero)
    # N.B.3: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return a correct rotation    
    # Adapted from https://github.com/lzx551402/geodesc/blob/master/utils/opencvhelper.py 
    @staticmethod    
    def matchWithCrossCheckAndModelFit(matcher, des1, des2, kps1, kps2, ratio_test=0.7, cross_check=True, err_thld=1, info=''):
        """Compute putative and inlier matches.
        Args:
            feat: (n_kpts, 128) Local features.
            cv_kpts: A list of keypoints represented as cv2.KeyPoint.
            ratio_test: The threshold to apply ratio test.
            cross_check: (True by default) Whether to apply cross check.
            err_thld: Epipolar error threshold.
            info: Info to print out.
        Returns:
            good_matches: Putative matches.
            mask: The mask to distinguish inliers/outliers on putative matches.
        """
        idx1, idx2 = [], []          
            
        init_matches1 = matcher.knnMatch(des1, des2, k=2)
        init_matches2 = matcher.knnMatch(des2, des1, k=2)

        good_matches = []

        for i,(m1,n1) in enumerate(init_matches1):
            cond = True
            if cross_check:
                cond1 = cross_check and init_matches2[m1.trainIdx][0].trainIdx == i
                cond *= cond1
            if ratio_test is not None:
                cond2 = m1.distance <= ratio_test * n1.distance
                cond *= cond2
            if cond:
                good_matches.append(m1)
                idx1.append(m1.queryIdx)
                idx2.append(m1.trainIdx)

        if type(kps1) is list and type(kps2) is list:
            good_kps1 = np.array([kps1[m.queryIdx].pt for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx].pt for m in good_matches])
        elif type(kps1) is np.ndarray and type(kps2) is np.ndarray:
            good_kps1 = np.array([kps1[m.queryIdx] for m in good_matches])
            good_kps2 = np.array([kps2[m.trainIdx] for m in good_matches])
        else:
            raise Exception("Keypoint type error!")
            exit(-1)

        ransac_method = None 
        try: 
            ransac_method = cv2.USAC_MSAC 
        except: 
            ransac_method = cv2.RANSAC
        _, mask = cv2.findFundamentalMat(good_kps1, good_kps2, ransac_method, err_thld, confidence=0.999)
        n_inlier = np.count_nonzero(mask)
        print(info, 'n_putative', len(good_matches), 'n_inlier', n_inlier)
        return idx1, idx2, good_matches, mask
    
    
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
class FeatureMatcher(object): 
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.BF):
        self.type = type 
        self.norm_type = norm_type 
        self.cross_check = cross_check   # apply cross check 
        self.matches = []
        self.ratio_test = ratio_test 
        self.matcher = None 
        self.matcher_name = ''
        self.matcherLG = None
        self.deviceLG = None
        if self.type == FeatureMatcherTypes.LIGHTGLUE:
            if self.matcherLG is None:
                LightGlue.pruning_keypoint_thresholds['cuda']
                self.matcherLG = LightGlue(features="superpoint",n_layers=2).eval().to('cuda')
            if self.deviceLG is None: 
                self.deviceLG = torch.device("cuda" if torch.cuda.is_available() else "cpu")        
        
        
    # input: des1 = queryDescriptors, des2= trainDescriptors
    # output: idx1, idx2  (vectors of corresponding indexes in des1 and des2, respectively)
    def match(self, frame : Frame, des1, des2, kps1 = None, kps2 = None, ratio_test=None):
        #print('des1.shape:',des1.shape,' des2.shape:',des2.shape)    
        #print('des1.dtype:',des1.dtype,' des2.dtype:',des2.dtype)
        #print(self.type)               
        print('matcher: ', self.type.name)         
        if ratio_test is None:
            ratio_test = self.ratio_test        
        if kVerbose:
            print(self.matcher_name,', norm ', self.norm_type) 
        if self.type == FeatureMatcherTypes.LIGHTGLUE:
            d1={
            'keypoints': torch.tensor(kps1,device='cuda').unsqueeze(0),
            'descriptors': torch.tensor(des2,device='cuda').unsqueeze(0),
            'image_size': torch.tensor(frame.shape, device='cuda').unsqueeze(0)
            }
            d2={
            'keypoints': torch.tensor(kps2,device='cuda').unsqueeze(0),
            'descriptors': torch.tensor(des1,device='cuda').unsqueeze(0),
            'image_size': torch.tensor(frame.shape, device='cuda').unsqueeze(0)
            }
             
            matches01 = self.matcherLG({"image0": d1, "image1": d2})
            #print(matches01['matches'])
            idx0 = matches01['matches'][0][:, 0].cpu().tolist()
            idx1 = matches01['matches'][0][:, 1].cpu().tolist()
            #print(des1.shape,len(idx0),len(idx1))
            return idx1, idx0
            # print(d1['keypoints'].shape, d1['descriptors'].shape, d1['image_size'].shape)
            # print(d2['keypoints'].shape, d2['descriptors'].shape, d2['image_size'].shape)
        elif self.type == FeatureMatcherTypes.XFEAT:
            d1_tensor = torch.tensor(des1, dtype=torch.float32)  # Specify dtype if needed
            d2_tensor = torch.tensor(des2, dtype=torch.float32)  # Specify dtype if needed
            # If the original tensors were on a GPU, you should move the new tensors to GPU as well
            # d1_tensor = d1_tensor.to('cuda')  # Use 'cuda' or 'cuda:0' if your device is a GPU
            # d2_tensor = d2_tensor.to('cuda') 
            idx0, idx1 = self.matcher.match(d1_tensor, d2_tensor, 0.93) 
            return idx0.cpu(), idx1.cpu()                  
        else: 
            matches = self.matcher.knnMatch(des1, des2, k=2)  #knnMatch(queryDescriptors,trainDescriptors)
            self.matches = matches
            return self.goodMatches(matches, des1, des2, ratio_test)          
            
    def goodMatches(self, matches, des1, des2, ratio_test=None): 
        if ratio_test is None:
            ratio_test = self.ratio_test
        #return goodMatchesSimple(matches, des1, des2, ratio_test)   # <= N.B.: this generates problem in SLAM since it can produce matches where a trainIdx index is associated to two (or more) queryIdx indexes
        return MatcherUtils.goodMatchesOneToOne(matches, des1, des2, ratio_test)

    
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

class LightGlueMatcher(FeatureMatcher):
    def __init__(self, norm_type=cv2.NORM_HAMMING, cross_check = False, ratio_test=kRatioTest, type = FeatureMatcherTypes.LIGHTGLUE):
        super().__init__(norm_type=norm_type, cross_check=cross_check, ratio_test=ratio_test, type=type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.matcher = LightGlue(features="superpoint").eval().to(device) 
        self.matcher_name = 'LightGlueFeatureMatcher'   

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

