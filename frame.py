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

import cv2
import numpy as np
from scipy.spatial import cKDTree

from skimage.measure import ransac
from geom_helpers import add_ones, poseRt, normalize
from helpers import myjet

from feature_detector import feature_detector_factory, FeatureDetectorTypes, FeatureDescriptorTypes

import constants 


class Frame(object):
    kMinNumFeatureDefault = 2000
    detector = feature_detector_factory(min_num_features=kMinNumFeatureDefault,
                                        detector_type=FeatureDetectorTypes.FAST,
                                        descriptor_type=FeatureDescriptorTypes.ORB)
    next_id = 0
    def __init__(self, mapp, img, K, Kinv, DistCoef, pose=np.eye(4), tid=None, des=None):
        self.H, self.W = img.shape[0:2]
        self.K = np.array(K)  
        # K = [[fx, 0,cx],
        #      [ 0,fy,cy],
        #      [ 0, 0, 1]]
        self.fx = K[0][0]
        self.fy = K[1][1]
        self.cx = K[0][2]
        self.cy = K[1][2]
        self.Kinv = np.array(Kinv)
        self.D = np.array(DistCoef)

        self.pose = np.array(pose)  # Tcw

        # self.kps       keypoints
        # self.kpsu      [u]ndistorted keypoints
        # self.kpsn      [n]ormalized keypoints
        # self.octaves   keypoint octaves 
        # self.des       keypoint descriptors
        # self.points    map points
        # self.outliers  outliers flags for map points 

        if img is not None:
            self.h, self.w = img.shape[0:2]
            if des is None:
                # convert to gray image 
                if img.ndim>2:
                    img = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)       
                self.kps, self.des = Frame.detector.detectAndCompute(img)
                # convert from a list of keypoints to an array of points and octaves  
                kps_data = np.array([ [x.pt[0], x.pt[1], x.octave] for x in self.kps], dtype=np.float32)
                self.octaves = np.uint32(kps_data[:,2].copy())
                #print('octaves: ', self.octaves)               
                self.kps = kps_data[:,:2].copy()                   
                self.kpsu = cv2.undistortPoints(np.expand_dims(self.kps, axis=1), self.K, self.D, None, self.K)
                self.kpsu = self.kpsu.ravel().reshape(self.kps.shape[0], 2)
                #print('kpsu diff: ', self.kpsu-self.kps)
            else:
                assert len(des) < 256
                self.kpsu, self.des = des, np.array(list(range(len(des)))*32, np.uint8).reshape(32, len(des)).T
            self.kpsn = normalize(self.Kinv, self.kpsu)
            self.points = [None]*len(self.kpsu)  # init points
            self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)
        else:
            # fill in later
            self.h, self.w = 0, 0
            self.kpsu, self.des, self.points = None, None, None
        self.id = Frame.next_id #tid if tid is not None else mapp.add_frame(self)
        Frame.next_id+=1

    # project a list of N map points on this frame
    # out: 2xN image points 
    def project_map_points(self, points):
        points4d = np.array([p.homogeneous() for p in points])
        projs = np.dot(np.dot(self.K, self.pose[:3,:]), points4d.T).T
        projs = projs[:, 0:2] / projs[:, 2:]    
        return projs

    # project a single map point on this frame
    # out: 2x1 image points 
    def project_map_point(self, p):   
        proj = np.dot(np.dot(self.K, self.pose[:3,:]), p.homogeneous().T).T
        proj = proj[0:2]/ proj[2]    
        return proj    

    def reset_outlier_map_points(self):
        for i,p in enumerate(self.points):
            if p is not None and self.outliers[i] is True:          
                self.points[i] = None 
                self.outliers[i] = False 

    def remove_point_observation(self, idx):
        self.points[idx] = None 
        self.outliers[idx] = False         

    def remove_point(self, p):
        p_idx = None 
        try: 
            p_idx = self.points.index(p)
        except:
            pass 
        if p_idx is not None: 
            self.points[p_idx] = None 
            self.outliers[p_idx] = False  

    def reset_points(self):
        self.points = [None]*len(self.kpsu) 
        self.outliers = np.full(self.kpsu.shape[0], False, dtype=bool)                 

    # KD tree of unnormalized keypoints
    @property
    def kd(self):
        if not hasattr(self, '_kd'):
            self._kd = cKDTree(self.kpsu)
        return self._kd

    def delete(self):
        del self

    def draw_feature_trails(self, img):
        # draw annotations on the image
        for i1 in range(len(self.kps)):
            u1, v1 = int(round(self.kps[i1][0])), int(round(self.kps[i1][1]))
            if self.points[i1] is not None:
                # there's a corresponding 3D point
                if len(self.points[i1].frames) >= 5:
                    cv2.circle(img, (u1, v1), color=(0, 255, 0), radius=3)
                else:
                    cv2.circle(img, (u1, v1), color=(0, 128, 0), radius=3)
                # draw the trail (for each keypoint, its 9 corresponding points in previous frames)
                pts = []
                lfid = None  # last frame id
                for f, idx in zip(self.points[i1].frames[-9:][::-1], self.points[i1].idxs[-9:][::-1]):
                    if lfid is not None and lfid-1 != f.id:
                        break
                    pts.append(tuple(map(lambda x: int(round(x)), f.kps[idx])))
                    lfid = f.id
                if len(pts) >= 2:
                    cv2.polylines(img, np.array([pts], dtype=np.int32), False, myjet[len(pts)]*255, thickness=1, lineType=16)
            else:
                # no corresponding 3D point
                cv2.circle(img, (u1, v1), color=(0, 0, 0), radius=3)
        return img        

# match frames f1 and f2
# out: a vector of match pairs [ ... [match1i, match2i] ... ]
def match_frames(f1, f2):
    bf = cv2.BFMatcher(cv2.NORM_HAMMING)

    matches = bf.knnMatch(f1.des, f2.des, k=2) # knnMatch(queryDescriptors, trainDescriptors)

    #good_point_matches = []
    idx1, idx2 = [], []
    flag_match = np.full(len(f2.des), False, dtype=bool)
    dist_match = np.zeros(len(f2.des))
    index_match = np.full(len(f2.des), 0, dtype=int)

    for m, n in matches:
        # apply Lowe's ratio test
        if m.distance < constants.kMatchRatioTest * n.distance:
            #p1 = f1.kpsn[m.queryIdx]
            #p2 = f2.kpsn[m.trainIdx]
            if m.distance < constants.kMaxOrbDistanceMatch:  # remove absolute distance threshold!
                if not flag_match[m.trainIdx]:
                    flag_match[m.trainIdx] = True
                    dist_match[m.trainIdx] = m.distance
                    idx1.append(m.queryIdx)
                    idx2.append(m.trainIdx)
                    #good_point_matches.append((p1, p2))
                    index_match[m.trainIdx] = len(idx1)-1
                else:
                    # stored match is worse => replace it
                    if dist_match[m.trainIdx] > m.distance:
                        index = index_match[m.trainIdx]
                        assert(idx2[index] == m.trainIdx)                        
                        idx1[index] = m.queryIdx
                        idx2[index] = m.trainIdx
                        #good_point_matches[index] = (p1, p2)

    assert len(idx1) >= 8
    #good_point_matches = np.array(good_point_matches)
    idx1 = np.array(idx1)
    idx2 = np.array(idx2)
    return idx1, idx2
