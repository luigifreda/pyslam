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
import os                
import numpy as np 
import cv2
import platform
import torch

from utils_sys import Printer, import_from
from utils_data import AtomicCounter
from utils_serialization import SerializableEnum, register_class
from config_parameters import Parameters  

from collections import defaultdict

from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo

import kornia as K
import kornia.feature as KF
import numpy as np

import config
config.cfg.set_lib('xfeat') 
config.cfg.set_lib('lightglue')

XFeat = import_from('modules.xfeat', 'XFeat')
LightGlue = import_from('lightglue', 'LightGlue')


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'
kLogsFolder = kRootFolder + '/logs'



kRatioTest = Parameters.kFeatureMatchRatioTest
kVerbose = False


@register_class
class FeatureMatcherTypes(SerializableEnum):
    NONE      = 0
    LK        = 1      # Lucas-Kanade tracking
    BF        = 2      # Brute force
    FLANN     = 3      # FLANN-based 
    XFEAT     = 4      # "XFeat: Accelerated Features for Lightweight Image Matching"
    LIGHTGLUE = 5      # "LightGlue: Local Feature Matching at Light Speed"
    LOFTR     = 6      # "LoFTR: Efficient Local Feature Matching with Transformers" (based on kornia)


def feature_matcher_factory(norm_type=cv2.NORM_HAMMING, 
                            cross_check=False, 
                            ratio_test=kRatioTest, 
                            matcher_type=FeatureMatcherTypes.FLANN,
                            detector_type=FeatureDetectorTypes.NONE,
                            descriptor_type=FeatureDescriptorTypes.NONE):
    if matcher_type == FeatureMatcherTypes.BF:
        return BfFeatureMatcher(norm_type=norm_type, 
                                cross_check=cross_check, 
                                ratio_test=ratio_test, 
                                matcher_type=matcher_type,
                                detector_type=detector_type,
                                descriptor_type=descriptor_type)
    elif matcher_type == FeatureMatcherTypes.FLANN:
        return FlannFeatureMatcher(norm_type=norm_type, 
                                   cross_check=cross_check, 
                                   ratio_test=ratio_test, 
                                   matcher_type=matcher_type,
                                   detector_type=detector_type,
                                   descriptor_type=descriptor_type)
    elif matcher_type ==FeatureMatcherTypes.XFEAT:
        return  XFeatMatcher(norm_type=norm_type, 
                             cross_check=cross_check, 
                             ratio_test=ratio_test, 
                             matcher_type=matcher_type,
                             detector_type=detector_type,
                             descriptor_type=descriptor_type)
    elif matcher_type == FeatureMatcherTypes.LIGHTGLUE:
        return LightGlueMatcher(norm_type=norm_type, 
                                cross_check=cross_check, 
                                ratio_test=ratio_test, 
                                matcher_type=matcher_type,
                                detector_type=detector_type,
                                descriptor_type=descriptor_type)
    elif matcher_type == FeatureMatcherTypes.LOFTR:
        return LoFTRMatcher(norm_type=norm_type, 
                                cross_check=cross_check, 
                                ratio_test=ratio_test, 
                                matcher_type=matcher_type,
                                detector_type=detector_type,
                                descriptor_type=descriptor_type)    
    return None 


# ==============================================================================

class MatcherUtils: 
    # input: 
    #   matches: list of cv2.DMatch (expected k=2 for knn search)
    #   des1 = query-descriptors, 
    #   des2 = train-descriptors
    # output: 
    #   idxs1, idxs2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.: this returns matches where each trainIdx index is associated to only one queryIdx index
    @staticmethod    
    def goodMatchesOneToOne(matches, des1, des2, ratio_test=0.7):
        #len_des2 = len(des2)
        idxs1, idxs2 = [], []           
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
                    idxs1.append(m.queryIdx)
                    idxs2.append(m.trainIdx)
                    index_match[m.trainIdx] = len(idxs2)-1
                else:
                    if m.distance < dist: 
                        # we have already a match for trainIdx: if stored match is worse => replace it
                        #print("double match on trainIdx: ", m.trainIdx)
                        index = index_match[m.trainIdx]
                        assert(idxs2[index] == m.trainIdx) 
                        idxs1[index]=m.queryIdx
                        idxs2[index]=m.trainIdx
        return np.array(idxs1), np.array(idxs2)

    # input: des1 = query-descriptors, des2 = train-descriptors
    # output: idxs1, idxs2  (vectors of corresponding indexes in des1 and des2, respectively)
    # N.B.: this may return matches where a trainIdx index is associated to two (or more) queryIdx indexes
    @staticmethod    
    def goodMatchesSimple(matches, des1, des2, ratio_test=0.7):
        idxs1, idxs2 = [], []                  
        if matches is not None: 
            for m,n in matches:
                if m.distance < ratio_test * n.distance:
                    idxs1.append(m.queryIdx)
                    idxs2.append(m.trainIdx)                                                         
        return np.array(idxs1), np.array(idxs2) 


    @staticmethod
    def rowMatches(matcher, kps1, des1, kps2, des2, max_matching_distance, 
            max_row_distance=Parameters.kStereoMatchingMaxRowDistance, max_disparity=100):
        idxs1, idxs2 = [], []  
        matches = matcher.match(np.array(des1), np.array(des2))
        for m in matches:          
            pt1 = kps1[m.queryIdx]
            pt2 = kps2[m.trainIdx]
            if (m.distance < max_matching_distance and 
                abs(pt1[1] - pt2[1]) < max_row_distance and 
                abs(pt1[0] - pt2[0]) < max_disparity):        # epipolar constraint + max disparity check
                idxs1.append(m.queryIdx)
                idxs2.append(m.trainIdx)   
        return np.array(idxs1), np.array(idxs2) 
    
    @staticmethod
    def rowMatchesWithRatioTest(matcher, kps1, des1, kps2, des2, max_matching_distance, 
            max_row_distance=Parameters.kStereoMatchingMaxRowDistance, max_disparity=100, ratio_test=0.7):
        idxs1, idxs2 = [], []  
        matches =  matcher.knnMatch(np.array(des1), np.array(des2), k=2) 
        for m,n in matches:          
            pt1 = kps1[m.queryIdx]
            pt2 = kps2[m.trainIdx]
            if (m.distance < max_matching_distance and 
                abs(pt1[1] - pt2[1]) < max_row_distance and 
                abs(pt1[0] - pt2[0]) < max_disparity):         # epipolar constraint + max disparity check
                if m.distance < ratio_test * n.distance:                
                    idxs1.append(m.queryIdx)
                    idxs2.append(m.trainIdx)   
        return np.array(idxs1), np.array(idxs2)     
    
    @staticmethod
    def filterNonRowMatches(kps1, idxs1, kps2, idxs2, max_row_distance=Parameters.kStereoMatchingMaxRowDistance, max_disparity=100):  
        assert(len(idxs1) == len(idxs2))
        out_idxs1, out_idxs2 = [], []   
        for idx1, idx2 in zip(idxs1, idxs2):
            pt1 = kps1[idx1]
            pt2 = kps2[idx2]
            if abs(pt1[1] - pt2[1]) < max_row_distance and abs(pt1[0] - pt2[0]) < max_disparity:   # epipolar constraint + max disparity check
                out_idxs1.append(idx1)
                out_idxs2.append(idx2)   
        return np.array(out_idxs1), np.array(out_idxs2)
    
    # input: des1 = query-descriptors, des2 = train-descriptors, kps1 = query-keypoints, kps2 = train-keypoints 
    # output: idxs1, idxs2  (vectors of corresponding indexes in des1 and des2, respectively)
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
        idxs1, idxs2 = [], []          
            
        init_matches1 = matcher.knnMatch(des1, des2, k=2)
        init_matches2 = matcher.knnMatch(des2, des1, k=2)

        good_matches = []

        for i, (m1, n1) in enumerate(init_matches1):
            if cross_check and init_matches2[m1.trainIdx][0].trainIdx != i:
                continue
            if ratio_test is not None and m1.distance > ratio_test * n1.distance:
                continue
            good_matches.append(m1)
            idxs1.append(m1.queryIdx)
            idxs2.append(m1.trainIdx)        

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
        return idxs1, idxs2, good_matches, mask
    
# ==============================================================================

class FeatureMatchingResult(object): 
    def __init__(self):
        self.kps1 = None          # all reference keypoints (numpy array Nx2)
        self.kps2 = None          # all current keypoints   (numpy array Nx2)
        self.lafs1 = None         # all reference LAFS (Local Affine Features), if available (numpy array Nx2x2)
        self.lafs2 = None         # all current LAFS (Local Affine Features), if available (numpy array Nx2x2)
        self.resps1 = None        # all reference responses, if available (numpy array Nx1)
        self.resps2 = None        # all current responses, if available (numpy array Nx1)
        self.des1 = None          # all reference descriptors (numpy array NxD)
        self.des2 = None          # all current descriptors (numpy array NxD)
        self.idxs1 = None         # indices of matches in kps_ref so that kps_ref_matched = kps_ref[idxs_ref]  (numpy array of indexes)
        self.idxs2 = None         # indices of matches in kps_cur so that kps_cur_matched = kps_cur[idxs_cur]  (numpy array of indexes)

            
# base class 
class FeatureMatcher(object): 
    global_counter=AtomicCounter()
    def __init__(self,
                 norm_type=cv2.NORM_HAMMING,
                 cross_check = False,
                 ratio_test=kRatioTest,
                 matcher_type = FeatureMatcherTypes.BF,
                 detector_type=FeatureDetectorTypes.NONE,
                 descriptor_type=FeatureDescriptorTypes.NONE):
        self.matcher_type = matcher_type 
        self.detector_type = detector_type
        self.descriptor_type = descriptor_type
        self.norm_type = norm_type 
        self.cross_check = cross_check   # apply cross check 
        self.ratio_test = ratio_test 
        self.matcher = None 
        self.parallel = True
        self.matcher_name = ''
                    
    # input: des1 = queryDescriptors, des2= trainDescriptors
    # output: idxs1, idxs2  (vectors of corresponding indexes in des1 and des2, respectively)
    def match(self, img1, img2, des1, des2, kps1=None, kps2=None, ratio_test=None, 
              row_matching=False, max_disparity=None, data=None):
                        
        result = FeatureMatchingResult()
        result.des1 = des1
        result.des2 = des2
        result.kps1 = kps1
        result.kps2 = kps2
        if kVerbose:
            print(self.matcher_name,', norm ', self.norm_type)             
            print('matcher: ', self.matcher_type.name)  
            if img1 is not None: 
                print(f'img1.shape: {img1.shape}')
            print('des1.shape:',des1.shape,' des1.dtype:',des1.dtype) 
            print('des2.shape:',des2.shape,' des2.dtype:',des2.dtype)    
            if kps1 is not None and isinstance(kps1, np.ndarray):         
                print('kps1.shape:',kps1.shape,' kps1.dtype:',kps1.dtype)    
            if kps2 is not None and isinstance(kps2, np.ndarray):
                print('kps2.shape:',kps2.shape,' kps2.dtype:',kps2.dtype)                           
        if ratio_test is None:
            ratio_test = self.ratio_test 
        # TODO: Use inheritance here instead of using if-else.   
        # NOTE: Not using inheritance for now since the interface is not optimal yet and it may change.
        # ===========================================================   
        if self.matcher_type == FeatureMatcherTypes.LIGHTGLUE:            
            # TODO: add row epipolar check for row matching
            scales1 = None
            scales2 = None
            oris1 = None 
            oris2 = None
            if kps1 is None and kps2 is None:
                print('FeatureMatcher.match: kps1 and kps2 are None')
                return result
            else: 
                # convert from list of keypoints to an array of points if needed
                if not isinstance(kps1, np.ndarray) or kps1.dtype != np.float32:
                    if self.detector_type == FeatureDetectorTypes.LIGHTGLUESIFT:
                        scales1 = np.array([x.size for x in kps1], dtype=np.float32)
                        oris1 = np.array([x.angle for x in kps1], dtype=np.float32)
                    print(f'kps1: {kps1}')
                    kps1 = np.array([x.pt for x in kps1], dtype=np.float32)
                    if kVerbose:
                        print('kps1.shape:',kps1.shape,' kps1.dtype:',kps1.dtype)
                if not isinstance(kps2, np.ndarray) or kps2.dtype != np.float32:
                    if self.detector_type == FeatureDetectorTypes.LIGHTGLUESIFT:                    
                        scales2 = np.array([x.size for x in kps2], dtype=np.float32)
                        oris2 = np.array([x.angle for x in kps2], dtype=np.float32)                    
                    kps2 = np.array([x.pt for x in kps2], dtype=np.float32)
                    if kVerbose: 
                        print('kps2.shape:',kps2.shape,' kps2.dtype:',kps2.dtype)
            if kVerbose:
                print(f'image1.shape: {img1.shape}, image2.shape: {img2.shape}')
            img1_shape = img1.shape[0:2]
            d0={
            'keypoints': torch.tensor(kps1, device=self.torch_device).unsqueeze(0),
            'descriptors': torch.tensor(des1, device=self.torch_device).unsqueeze(0),
            'image_size': torch.tensor(img1_shape, device=self.torch_device).unsqueeze(0)
            }            
            if scales1 is not None and oris1 is not None:
                d0['scales'] = torch.tensor(scales1, device=self.torch_device).unsqueeze(0)
                d0['oris'] = torch.tensor(oris1, device=self.torch_device).unsqueeze(0)
            d1={
            'keypoints': torch.tensor(kps2, device=self.torch_device).unsqueeze(0),
            'descriptors': torch.tensor(des2, device=self.torch_device).unsqueeze(0),
            'image_size': torch.tensor(img1_shape, device=self.torch_device).unsqueeze(0)
            }
            if scales2 is not None and oris2 is not None:
                d1['scales'] = torch.tensor(scales2, device=self.torch_device).unsqueeze(0)
                d1['oris'] = torch.tensor(oris2, device=self.torch_device).unsqueeze(0)             
            matches01 = self.matcher({"image0": d0, "image1": d1})
            #print(matches01['matches'])
            idxs0 = matches01['matches'][0][:, 0].cpu().tolist()
            idxs1 = matches01['matches'][0][:, 1].cpu().tolist()
            result.idxs1 = np.array(idxs0)
            result.idxs2 = np.array(idxs1)
            if row_matching: 
                result.idxs1, result.idxs2 = MatcherUtils.filterNonRowMatches(kps1, result.idxs1, kps2, result.idxs2, max_disparity=max_disparity)            
            if kVerbose:
                print(f'#result.idxs1: {result.idxs1.shape}, #result.idxs2: {result.idxs2.shape}')            
            return result
        # ===========================================================
        elif self.matcher_type == FeatureMatcherTypes.XFEAT:
            d1_tensor = torch.tensor(des1, dtype=torch.float32)  # Specify dtype if needed
            d2_tensor = torch.tensor(des2, dtype=torch.float32)  # Specify dtype if needed
            # If the original tensors were on a GPU, you should move the new tensors to GPU as well
            # d1_tensor = d1_tensor.to('cuda')  # Use 'cuda' or 'cuda:0' if your device is a GPU
            # d2_tensor = d2_tensor.to('cuda') 
            min_cossim = 0.82 # default in xfeat code 
            idx0, idxs1 = self.matcher.match(d1_tensor, d2_tensor, min_cossim=min_cossim)    
            result.idxs1 = idx0.cpu()
            result.idxs2 = idxs1.cpu()
            if row_matching: 
                result.idxs1, result.idxs2 = MatcherUtils.filterNonRowMatches(kps1, result.idxs1, kps2, result.idxs2, max_disparity=max_disparity)
            return result
        # ===========================================================
        elif self.matcher_type == FeatureMatcherTypes.LOFTR:
            if img1.ndim>2:
                img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
            if img2.ndim>2:
                img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2GRAY)                
            img1 = K.image_to_tensor(img1, False).to(self.torch_device).float() / 255.          
            img2 = K.image_to_tensor(img2, False).to(self.torch_device).float() / 255.                
            matching_input = {"image0": img1, "image1": img2}
            out_matching = self.matcher(matching_input)
            kps1 = out_matching['keypoints0'].cpu().numpy()
            kps1 = np.array([ cv2.KeyPoint(int(p[0]), int(p[1]), size=1, response=1) for p in kps1 ])
            kps2 = out_matching['keypoints1'].cpu().numpy()
            kps2 = np.array([ cv2.KeyPoint(int(p[0]), int(p[1]), size=1, response=1) for p in kps2 ])
            #idxs = out_matching['batch_indexes'].cpu().numpy()
            #print(f'idxs.shape: {idxs.shape}, idxs.dtype: {idxs.dtype}')
            result.kps1 = kps1
            result.kps2 = kps2
            result.idxs1 = np.arange(len(kps1), dtype=np.int32)
            result.idxs2 = np.arange(len(kps2), dtype=np.int32)
            if row_matching: 
                result.idxs1, result.idxs2 = MatcherUtils.filterNonRowMatches(kps1, result.idxs1, kps2, result.idxs2, max_disparity=max_disparity)            
            return result         
        # ===========================================================      
        else:
            matcher = cv2.BFMatcher(self.norm_type, self.cross_check) #if self.parallel else self.matcher            
            if not row_matching:
                """
                The result of matches = matcher.knnMatch() is a list of cv2.DMatch objects. 
                A DMatch object has the following attributes:
                    DMatch.distance - Distance between descriptors. The lower, the better it is.
                    DMatch.trainIdx - Index of the descriptor in train descriptors
                    DMatch.queryIdx - Index of the descriptor in query descriptors
                    DMatch.imgIdx - Index of the train image.
                """            
                matches = matcher.knnMatch(des1, des2, k=2)  #knnMatch(queryDescriptors,trainDescriptors)
                #return MatcherUtils.goodMatchesSimple(matches, des1, des2, ratio_test)   # <= N.B.: this generates problem in SLAM since it can produce matches where a trainIdx index is associated to two (or more) queryIdx indexes
                idxs1, idxs2 = MatcherUtils.goodMatchesOneToOne(matches, des1, des2, ratio_test)                    
            else: 
                assert(max_disparity is not None)
                # we perform row matching for stereo images (matching rectified left and right images)
                max_descriptor_distance = 0.75 * FeatureInfo.max_descriptor_distance[self.descriptor_type]  # for rectified stereo matching we assume the matching descriptors have in general a small relative distance 
                if ratio_test < 1.0:
                    idxs1, idxs2 = MatcherUtils.rowMatchesWithRatioTest(matcher, kps1, des1, kps2, des2, max_descriptor_distance, max_disparity=max_disparity, ratio_test=ratio_test)
                else:                             
                    idxs1, idxs2 = MatcherUtils.rowMatches(matcher, kps1, des1, kps2, des2, max_descriptor_distance, max_disparity=max_disparity)
            
            # Change suggested here https://github.com/luigifreda/pyslam/issues/125#issuecomment-2555299806
            #result.idxs1 = idxs1
            #result.idxs2 = idxs2
            result.idxs1 = idxs1.astype(np.int64)
            result.idxs2 = idxs2.astype(np.int64)                              
            return result      

    
# ==============================================================================S
# Brute-Force Matcher 
class BfFeatureMatcher(FeatureMatcher): 
    def __init__(self, 
                 norm_type=cv2.NORM_HAMMING, 
                 cross_check = False, 
                 ratio_test=kRatioTest, 
                 matcher_type = FeatureMatcherTypes.BF,
                 detector_type=FeatureDetectorTypes.NONE,
                 descriptor_type=FeatureDescriptorTypes.NONE):
        super().__init__(norm_type=norm_type, 
                         cross_check=cross_check, 
                         ratio_test=ratio_test, 
                         matcher_type=matcher_type,
                         detector_type=detector_type,
                         descriptor_type=descriptor_type)
        self.matcher = cv2.BFMatcher(norm_type, cross_check)     
        self.matcher_name = 'BfFeatureMatcher'   
        Printer.green(f'matcher: {self.matcher_name} - norm_type: {norm_type}, cross_check: {cross_check}, ratio_test: {ratio_test}')


# ==============================================================================
# Flann Matcher 
class FlannFeatureMatcher(FeatureMatcher): 
    def __init__(self, 
                 norm_type=cv2.NORM_HAMMING, 
                 cross_check = False, 
                 ratio_test=kRatioTest, 
                 matcher_type = FeatureMatcherTypes.FLANN,
                 detector_type=FeatureDetectorTypes.NONE,
                 descriptor_type=FeatureDescriptorTypes.NONE):
        super().__init__(norm_type=norm_type, 
                         cross_check=cross_check, 
                         ratio_test=ratio_test, 
                         matcher_type=matcher_type,
                         detector_type=detector_type,
                         descriptor_type=descriptor_type)
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
        Printer.green(f'matcher: {self.matcher_name} - norm_type: {norm_type}, cross_check: {cross_check}, ratio_test: {ratio_test}')


# ==============================================================================
class XFeatMatcher(FeatureMatcher):
    def __init__(self, 
                 norm_type=cv2.NORM_L2, 
                 cross_check = False, 
                 ratio_test=kRatioTest, 
                 matcher_type = FeatureMatcherTypes.XFEAT,
                 detector_type=FeatureDetectorTypes.NONE,
                 descriptor_type=FeatureDescriptorTypes.NONE):
        super().__init__(norm_type=norm_type, 
                         cross_check=cross_check, 
                         ratio_test=ratio_test, 
                         matcher_type=matcher_type,
                         detector_type=detector_type,
                         descriptor_type=descriptor_type)
        self.matcher = XFeat()    
        self.matcher_name = 'XFeatFeatureMatcher'   
        Printer.green(f'matcher: {self.matcher_name}')
        

# ==============================================================================
class LightGlueMatcher(FeatureMatcher):
    def __init__(self, 
                 norm_type=cv2.NORM_L2, 
                 cross_check = False, 
                 ratio_test=kRatioTest, 
                 matcher_type = FeatureMatcherTypes.LIGHTGLUE,
                 detector_type=FeatureDetectorTypes.SUPERPOINT,
                 descriptor_type=FeatureDescriptorTypes.NONE):
        super().__init__(norm_type=norm_type, 
                         cross_check=cross_check, 
                         ratio_test=ratio_test, 
                         matcher_type=matcher_type,
                         detector_type=detector_type,
                         descriptor_type=descriptor_type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.torch_device = device
        if self.torch_device == 'cuda':
            LightGlue.pruning_keypoint_thresholds['cuda']      
        features_string = None 
        if detector_type == FeatureDetectorTypes.SUPERPOINT:
            features_string = 'superpoint'
        elif detector_type == FeatureDetectorTypes.DISK:
            features_string = 'disk'  
        elif detector_type == FeatureDetectorTypes.ALIKED:
            features_string = 'aliked'            
        elif detector_type == FeatureDetectorTypes.LIGHTGLUESIFT:
            features_string = 'sift'                  
        else:
            raise ValueError(f'LightGlue: Unmanaged detector type: {detector_type.name}')
        self.matcher = LightGlue(features=features_string,n_layers=2).eval().to(device) 
        self.matcher_name = 'LightGlueFeatureMatcher'   
        print('device: ', self.torch_device)  
        Printer.green(f'matcher: {self.matcher_name}')
        

# ==============================================================================
class LoFTRMatcher(FeatureMatcher):
    def __init__(self, 
                 norm_type=cv2.NORM_L2, 
                 cross_check = False, 
                 ratio_test=kRatioTest, 
                 matcher_type = FeatureMatcherTypes.LOFTR,
                 detector_type=FeatureDetectorTypes.NONE,
                 descriptor_type=FeatureDescriptorTypes.NONE):
        super().__init__(norm_type=norm_type, 
                         cross_check=cross_check, 
                         ratio_test=ratio_test, 
                         matcher_type=matcher_type,
                         detector_type=detector_type,
                         descriptor_type=descriptor_type)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        #device = 'cpu' # force cpu mode
        self.torch_device = device
        if self.torch_device == 'cuda':
            torch.cuda.empty_cache()
        # https://kornia.readthedocs.io/en/latest/feature.html#kornia.feature.LoFTR
        self.matcher = KF.LoFTR('outdoor').eval().to(device)  
        self.matcher_name = 'LoFTRMatcher' 
        print('device: ', self.torch_device)    
        Printer.green(f'matcher: {self.matcher_name}')
