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
import time
import cv2
from enum import Enum

from frame import Frame, match_frames

import optimizer_g2o

from map_point import MapPoint
from map import Map
from geom_helpers import triangulate, add_ones, poseRt
from pinhole_camera import Camera, PinholeCamera
from initializer import Initializer

import constants 

kVerbose=True     
kMinNumFeature = 2000
kRansacThresholdNormalized = 0.0003  # metric threshold used for normalized image coordinates 
kUseEssentialMatrixEstimation = True # more robust RANSAC given the five point algorithm 
kRansacProb = 0.999
kUseGroundTruthScale = False 
kLocalWindow = 10
kUseMotionModel = False 

class SlamStage(Enum):
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    LOST=3

class SLAM(object):
    def __init__(self, camera, detector, grountruth = None):
        self.cam = camera 

        self.map = Map()

        Frame.detector = detector # set the static field of the class 

        self.W, self.H = camera.width, camera.height 
        self.K = camera.K
        self.Kinv = camera.Kinv
        self.D = camera.D        # distortion coefficients [k1, k2, p1, p2, k3]

        self.stage = SlamStage.NO_IMAGES_YET

        self.intializer = Initializer()

        self.cur_R = None # current rotation w.r.t. world frame 
        self.cur_t = None # current translation w.r.t. world frame 

        self.trueX, self.trueY, self.trueZ = None, None, None
        self.grountruth = grountruth

        self.mask_match = None 

        self.velocity = None 
 
        self.init_history = True 
        self.poses = []       # history of poses
        self.t0_est = None    # history of estimated translations      
        self.t0_gt = None     # history of ground truth translations (if available)
        self.traj3d_est = []  # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = []   # history of estimated ground truth translations centered w.r.t. first one                  

    def estimatePose(self, kpn_ref, kpn_cur):	     
        E, self.mask_match = cv2.findEssentialMat(kpn_cur, kpn_ref, focal=1, pp=(0., 0.), method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)                         
        _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0., 0.))   
        return poseRt(R,t.T)  # Rrc,trc (cur with respect to 'ref' frame)         

    def track(self, img, frame_id, pose=None, verts=None):

        assert img.shape[0:2] == (self.H, self.W)        
        
        start_time = time.time()

        f_cur = Frame(self.map, img, self.K, self.Kinv, self.D, des=verts) 

        if self.stage == SlamStage.NO_IMAGES_YET: 
            self.map.add_frame(f_cur) # add first frame in map 
            self.stage = SlamStage.NOT_INITIALIZED
            return # EXIT (jump to second frame)

        #if self.stage == SlamStage.NOT_INITIALIZED:
        #    f_ref = self.map.frames[0]  # first frame in map (to be used for initialization)
        #else:            
        f_ref = self.map.frames[-1] # last frame in map
        
        # find image point matches
        idx_cur, idx_ref = match_frames(f_cur, f_ref)
        
        if self.stage == SlamStage.NOT_INITIALIZED:
            initializer_output, is_ok = self.intializer.init(f_cur, f_ref, idx_cur, idx_ref, img)
            if is_ok:
                self.map.add_frame(f_cur) # first add the frame and update its id then add points 
                new_pts_count,_ = self.map.add_points(initializer_output.points4d, None, f_cur, f_ref, initializer_output.idx_cur, initializer_output.idx_ref, img)
                print("map: initialized %d new points" % (new_pts_count))                   
                self.stage = SlamStage.OK               
            return # EXIT (jump to next frame)
        else:
            self.map.add_frame(f_cur)  # simply add f_cur to map 
        
        # estimate inter frame camera motion 
        Mrc = self.estimatePose(f_ref.kpsn[idx_ref], f_cur.kpsn[idx_cur])
        Mcr = np.linalg.inv(poseRt(Mrc[:3, :3], Mrc[:3, 3]))
        f_cur.pose = np.dot(Mcr, f_ref.pose)

        # remove outliers from matches by using the mask computed with inter frame pose estimation 
        mask_index = [ i for i,v in enumerate(self.mask_match) if v > 0] 
        num_inliers = len(mask_index)
        print('num inliers: ', num_inliers)
        idx_ref = idx_ref[mask_index]
        idx_cur = idx_cur[mask_index]

        # kinematic model 
        self.velocity = np.dot(f_ref.pose, np.linalg.inv(self.map.frames[-2].pose))
        predicted_pose = np.dot(self.velocity, f_ref.pose)
                
        # set intial guess for pose optimization 
        #f_cur.pose = f_ref.pose.copy()  # get the last pose as an initial guess for optimization
        if kUseMotionModel is True:
            f_cur.pose[:,3] = predicted_pose[:,3].copy() # keep the estimated rotation and override translation 
        else:
            f_cur.pose[:,3] = f_ref.pose[:,3].copy() # keep the estimated rotation and override translation             

        # let's populate the map points of the current frame: use the point image matches with previous frame and add map point observations in current frame
        num_found_map_pts_inter_frame = 0
        for i, idx in enumerate(idx_ref):
            if f_ref.points[idx] is not None: 
                f_ref.points[idx].add_observation(f_cur, idx_cur[i])
                num_found_map_pts_inter_frame += 1
        
        print("matched %d map points" % num_found_map_pts_inter_frame)   

        # pose optimization 
        pose_opt_error, pose_is_ok = optimizer_g2o.poseOptimization(f_cur, verbose=False)
        print("pose opt err: %f,  ok: %d" % (pose_opt_error, int(pose_is_ok)) )        

        # discard outliers detected in pose optimization 
        f_cur.reset_outlier_map_points()

        # TODO: implement a proper local mapping  
        #num_found_map_pts = self.searchMapByProjection(f_cur)
        num_found_map_pts = self.searchLocalFramesByProjection(f_cur, local_window = kLocalWindow)        
        
        # TODO: this triangulation should be done from keyframes!
        # triangulate the points we don't have matches for
        good_pts4d = np.array([f_cur.points[i] is None for i in idx_cur])

        # do triangulation in global frame
        pts4d = triangulate(f_cur.pose, f_ref.pose, f_cur.kpsn[idx_cur], f_ref.kpsn[idx_ref])
        good_pts4d &= np.abs(pts4d[:, 3]) != 0
        good_pts4d &= f_ref.outliers[idx_ref] == False 
        pts4d /= pts4d[:, 3:]       # homogeneous 3-D coords

        new_pts_count,_ = self.map.add_points(pts4d, good_pts4d, f_cur, f_ref, idx_cur, idx_ref, img, check_parallax=True)
        print("map: added %d new points (%d searched by projection)" % (new_pts_count, num_found_map_pts))
            
        err = self.map.localOptimize(local_window=kLocalWindow)
        print("local optimization:   %f units of error" % err)

        # optimize the map
        useBA = False
        if f_cur.id >= 4 and f_cur.id % 5 == 0 and useBA:
            err = self.map.optimize()  # verbose=True)
            print("optimize:   %f units of error" % err)

        print("map: %d points, %d frames" % (len(self.map.points), len(self.map.frames)))
        print("time: %.2f ms" % ((time.time()-start_time)*1000.0))
        #self.updateHistory()

    # search by projection the map points of local frames on frame f_cur
    def searchLocalFramesByProjection(self, f_cur, local_window = kLocalWindow):
        # TODO: implement a proper local mapping 
        # take the points in the last N frame 
        points = []
        point_id_set = set()
        frames = self.map.frames[-local_window:]
        for f in frames:
            f_points = [p for p,outlier in zip(f.points,f.outliers) if (p is not None) and (outlier is not True)] 
            for p in f_points: 
                if p.id not in point_id_set:
                    points.append(p)
                    point_id_set.add(p.id)
        print('searching %d map points' % len(points))
        return self.searchByProjection(points, f_cur)  

    # search by projection all the map points on frame f_cur
    def searchMapByProjection(self, f_cur):
        return self.searchByProjection(self.map.points, f_cur)                

    # search by projection the input map points on frame f_cur
    def searchByProjection(self, points, f_cur):
        found_pts_count = 0
        if len(points) > 0:

            # project the points on frame f_cur
            projs = f_cur.project_map_points(points)

            # check if points are visible 
            visible_pts = (projs[:, 0] > 0) & (projs[:, 0] < f_cur.W) & \
                          (projs[:, 1] > 0) & (projs[:, 1] < f_cur.H)

            for i, p in enumerate(points):
                if not visible_pts[i] or p.is_bad is True:
                    # point not visible in frame
                    continue
                if f_cur in p.frames:
                    # we already matched this map point to this frame
                    continue
                for m_idx in f_cur.kd.query_ball_point(projs[i], constants.kMaxReprojectionDistance):
                    #print('checking : ', m_idx)
                    # if no point associated and not outlier 
                    if f_cur.points[m_idx] is None: # or f_cur.outliers[m_idx] is False:
                        orb_dist = p.orb_distance(f_cur.des[m_idx])
                        #print('b_dist : ', orb_dist)
                        if orb_dist < constants.kMaxOrbDistanceSearchByReproj:
                            p.add_observation(f_cur, m_idx)
                            found_pts_count += 1
                            break      
        return found_pts_count;            

    # get current translation scale from ground-truth if this is set 
    def getAbsoluteScale(self, frame_id):  
        if self.grountruth is not None and kUseGroundTruthScale:
            self.trueX, self.trueY, self.trueZ, scale = self.grountruth.getPoseAndAbsoluteScale(frame_id)
            return scale
        else:
            self.trueX = 0 
            self.trueY = 0 
            self.trueZ = 0
            return 1

    def updateHistory(self):
        f_cur = self.map.frames[-1]
        self.cur_R = f_cur.pose[:3,:3].T
        self.cur_t = np.dot(-self.cur_R,f_cur.pose[:3,3])
        if (self.init_history is True) and (self.trueX is not None):
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
            self.t0_gt  = np.array([self.trueX, self.trueY, self.trueZ])           # starting translation 
        if (self.t0_est is not None) and (self.t0_gt is not None):             
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(p)
            self.traj3d_gt.append([self.trueX-self.t0_gt[0], self.trueY-self.t0_gt[1], self.trueZ-self.t0_gt[2]])            
            self.poses.append(poseRt(self.cur_R, p))    
