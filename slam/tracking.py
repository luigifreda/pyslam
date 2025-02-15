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
import os
import numpy as np
import time

#import json
import ujson as json

from collections import defaultdict, Counter
from itertools import chain

import cv2
import g2o

from config_parameters import Parameters  

from frame import Frame, FeatureTrackerShared, match_frames
from keyframe import KeyFrame
from map_point import MapPoint
from map import Map

from search_points import propagate_map_point_matches
from search_points import search_map_by_projection, search_frame_by_projection

from local_mapping import LocalMapping
from initializer import Initializer
import optimizer_g2o

from loop_closing import LoopClosing

from timer import TimerFps

from dataset import SensorType

from slam_dynamic_config import SLAMDynamicConfig
from motion_model import MotionModel, MotionModelDamping

from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_tracker import feature_tracker_factory, FeatureTracker, FeatureTrackerTypes 

from utils_serialization import SerializableEnum, SerializationJSON, register_class
from utils_sys import Printer, Logging
from utils_draw import draw_feature_matches
from utils_geom import poseRt, inv_T
from utils_geom_2views import estimate_pose_ess_mat
from utils_features import ImageGrid

from slam_commons import SlamState

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from slam import Slam, SlamState  # Only imported when type checking, not at runtime


kVerbose = True     
kTimerVerbose = False 
kShowFeatureMatches = False 

kLocalMappingOnSeparateThread = Parameters.kLocalMappingOnSeparateThread 
kTrackingWaitForLocalMappingToGetIdle = Parameters.kTrackingWaitForLocalMappingToGetIdle

kLogKFinfoToFile = True 

kUseDynamicDesDistanceTh = True  

kRansacThresholdNormalized = 0.0004  # 0.0003 # metric threshold used for normalized image coordinates 
kRansacProb = 0.999
kNumMinInliersEssentialMat = 8

kUseGroundTruthScale = False 

kNumMinInliersPoseOptimizationTrackFrame = 10
kNumMinInliersPoseOptimizationTrackLocalMap = 20
kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle = 50 # defines bad/weak tracking condition


kUseMotionModel = Parameters.kUseMotionModel or Parameters.kUseSearchFrameByProjection
kUseSearchFrameByProjection = Parameters.kUseSearchFrameByProjection and not Parameters.kUseEssentialMatrixFitting         
kUseEssentialMatrixFitting = Parameters.kUseEssentialMatrixFitting      
       
kNumMinObsForKeyFrameDefault = 3


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'
kLogsFolder = kRootFolder + '/logs'


if not kVerbose:
    def print(*args, **kwargs):
        pass 


class TrackingHistory(object):
    def __init__(self):
        self.relative_frame_poses = []  # list of relative frame poses w.r.t reference keyframes as g2o.Isometry3d() (see camera_pose.py)
        self.kf_references = []         # list of reference keyframes  
        self.timestamps = []            # list of frame timestamps
        self.ids = []
        self.slam_states = []           # list of slam states 

    def reset(self):
        self.relative_frame_poses.clear()
        self.kf_references.clear()
        self.timestamps.clear()
        self.ids.clear()
        self.slam_states.clear()


class Tracking:
    def __init__(self, slam: 'Slam'):
        
        if kShowFeatureMatches: 
            Frame.is_store_imgs = True 
                    
        self.slam = slam
        
        self.trackingWaitForLocalMappingSleepTime = Parameters.kTrackingWaitForLocalMappingSleepTime
        
        self.intializer = Initializer(self.sensor_type)
        
        self.motion_model = MotionModel()  # motion model for current frame pose prediction without damping  
        #self.motion_model = MotionModelDamping()  # motion model for current frame pose prediction with damping       
        
        self.dyn_config = SLAMDynamicConfig(self.feature_tracker.feature_manager.max_descriptor_distance)
        self.descriptor_distance_sigma = self.feature_tracker.feature_manager.max_descriptor_distance
        self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMap 
        if self.sensor_type == SensorType.RGBD:
            self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMapRgbd   
        
        self.max_frames_between_kfs = int(slam.camera.fps) if slam.camera.fps is not None else 1
        self.min_frames_between_kfs = 0         
        
        # params read and set by Slam
        self.far_points_threshold = None    
        self.use_fov_centers_based_kf_generation = False
        self.max_fov_centers_distance = -1        

        self.state = SlamState.NO_IMAGES_YET
        
        self.num_matched_kps = None                             # current number of matched keypoints 
        self.num_inliers = None                                 # current number of matched points 
        self.num_matched_map_points = None                      # current number of matched map points
        self.num_matched_map_points_in_last_pose_opt = None     # current number of matched map points (matched and found valid in last pose optimization)     
        self.num_kf_ref_tracked_points = None                   # number of tracked points in k_ref (considering a minimum number of observations)      
        
        self.last_num_static_stereo_map_points = None
        
        self.last_reloc_frame_id = -float('inf')
        
        self.mask_match = None 

        self.pose_is_ok = False 
        self.mean_pose_opt_chi2_error = None
        self.predicted_pose = None 
        self.velocity = None 
        
        self.f_cur = None 
        self.idxs_cur = None 
        self.f_ref = None 
        self.idxs_ref = None 
                
        self.kf_ref = None    # reference keyframe (in general, different from last keyframe depending on the used approach)
        self.kf_last = None   # last keyframe  
        self.kid_last_BA = -1 # last keyframe id when performed BA 
        
        self.local_keyframes = [] # local keyframes 
        self.local_points    = [] # local points 
        self.vo_points       = [] # visual odometry points  
         
        self.tracking_history = TrackingHistory()
 
        self.timer_verbose = kTimerVerbose  # set this to True if you want to print timings 
        self.timer_main_track = TimerFps('Track', is_verbose = self.timer_verbose)
        self.timer_pose_opt = TimerFps('Pose optimization', is_verbose = self.timer_verbose)
        self.timer_seach_frame_proj = TimerFps('Search frame by proj', is_verbose = self.timer_verbose) 
        self.timer_match = TimerFps('Match', is_verbose = self.timer_verbose)                   
        self.timer_pose_est = TimerFps('Ess mat pose estimation', is_verbose = self.timer_verbose)
        self.timer_frame = TimerFps('Frame', is_verbose = self.timer_verbose)
        self.timer_seach_map = TimerFps('Search map', is_verbose = self.timer_verbose)   
        
        self.time_track = None  
        
        self.init_history = True     # need to init history?
        self.poses = []              # history of poses
        self.pose_timestamps = []    # history of pose timestamps        
        self.t0_est = None           # history of estimated translations      
        self.t0_gt = None            # history of ground truth translations (if available)
        self.traj3d_est = []         # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = []          # history of estimated ground truth translations centered w.r.t. first one                 

        self.cur_R = None # current rotation Rwc w.r.t. world frame   
        self.cur_t = None # current translation twc w.r.t. world frame 
        self.gt_x, self.gt_y, self.gt_z = None, None, None
        
        if kLogKFinfoToFile:
            self.kf_info_logger = Logging.setup_file_logger('kf_info_logger', kLogsFolder + '/kf_info.log', formatter=Logging.simple_log_formatter)
                 
                 
    @property
    def feature_tracker(self):
        return self.slam.feature_tracker
    
    @property
    def map(self):
        return self.slam.map
    
    @property
    def camera(self):
        return self.slam.camera
    
    @property
    def sensor_type(self):
        return self.slam.sensor_type
    
    @property
    def local_mapping(self):
        return self.slam.local_mapping
    
    def reset(self):
        print('Tracking: reset...')           

        self.intializer.reset()

        self.motion_model.reset()
                        
        self.state = SlamState.NO_IMAGES_YET
        
        self.num_matched_kps = None                             # current number of matched keypoints 
        self.num_inliers = None                                 # current number of matched points 
        self.num_matched_map_points = None                      # current number of matched map points        
        self.num_matched_map_points_in_last_pose_opt = None     # current number of matched map points (matched and found valid in current pose optimization)     
        self.num_kf_ref_tracked_points = None                   # number of tracked points in k_ref (considering a minimum number of observations)      
        
        self.last_num_static_stereo_map_points = None
                
        self.mask_match = None 

        self.pose_is_ok = False 
        self.mean_pose_opt_chi2_error = None
        self.predicted_pose = None 
        self.velocity = None 
                        
        self.f_cur = None 
        self.idxs_cur = None 
        self.f_ref = None 
        self.idxs_ref = None 
                
        self.kf_ref = None    # reference keyframe (in general, different from last keyframe depending on the used approach)
        self.kf_last = None   # last keyframe  
        self.kid_last_BA = -1 # last keyframe id when performed BA 
                
        self.local_keyframes.clear()
        self.local_points.clear()
        self.vo_points.clear()
        
        self.tracking_history.reset()
                
        self.init_history = True        
        self.poses.clear()
        self.pose_timestamps.clear()
        self.t0_est = None           # history of estimated translations      
        self.t0_gt = None            # history of ground truth translations (if available)        
        self.traj3d_est.clear()
        self.traj3d_gt.clear()
        
        self.cur_R = None # current rotation w.r.t. world frame  
        self.cur_t = None # current translation w.r.t. world frame 
        self.gt_x, self.gt_y, self.gt_z = None, None, None
                

    # estimate a pose from a fitted essential mat; 
    # since we do not have an interframe translation scale, this fitting can be used to detect outliers, estimate interframe orientation and translation direction 
    # N.B. read the NBs of the method estimate_pose_ess_mat(), where the limitations of this method are explained  
    def estimate_pose_by_fitting_ess_mat(self, f_ref, f_cur, idxs_ref, idxs_cur): 
        if len(idxs_ref) == 0 or len(idxs_cur) == 0:
            Printer.red('idxs_ref or idxs_cur is empty')
            return idxs_ref, idxs_cur
        
        # N.B.: in order to understand the limitations of fitting an essential mat, read the comments of the method self.estimate_pose_ess_mat() 
        self.timer_pose_est.start()
        ransac_method = None 
        try: 
            ransac_method = cv2.USAC_MSAC 
        except: 
            ransac_method = cv2.RANSAC 
        try:
            # estimate inter frame camera motion by using found keypoint matches 
            # output of the following function is:  Trc = [Rrc, trc] with ||trc||=1  where c=cur, r=ref  and  pr = Trc * pc 
            Mrc, self.mask_match = estimate_pose_ess_mat(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur], 
                                                        method=ransac_method, prob=kRansacProb, threshold=kRansacThresholdNormalized)   
        except Exception as e:
            Printer.red(f'Error in estimate_pose_ess_mat: {e}')
            return idxs_ref, idxs_cur
            
        #Mcr = np.linalg.inv(poseRt(Mrc[:3, :3], Mrc[:3, 3]))   
        Mcr = inv_T(Mrc)
        estimated_Tcw = np.dot(Mcr, f_ref.pose)
        self.timer_pose_est.refresh()      

        # remove outliers from keypoint matches by using the mask computed with inter frame pose estimation        
        mask_idxs = (self.mask_match.ravel() == 1)
        self.num_inliers = sum(mask_idxs)
        print('# inliers: ', self.num_inliers )
        idxs_ref = idxs_ref[mask_idxs]
        idxs_cur = idxs_cur[mask_idxs]

        # if there are not enough inliers do not use the estimated pose 
        if self.num_inliers < kNumMinInliersEssentialMat:
            #f_cur.update_pose(f_ref.pose) # reset estimated pose to previous frame 
            Printer.red('Essential mat: not enough inliers!')  
        else:
            # use the estimated pose as an initial guess for the subsequent pose optimization 
            # set only the estimated rotation (essential mat computation does not provide a scale for the translation, see above) 
            #f_cur.pose[:3,:3] = estimated_Tcw[:3,:3] # copy only the rotation 
            #f_cur.pose[:,3] = f_ref.pose[:,3].copy() # override translation with ref frame translation 
            Rcw = estimated_Tcw[:3,:3] # copy only the rotation 
            tcw = f_ref.pose[:3,3]     # override translation with ref frame translation          
            f_cur.update_rotation_and_translation(Rcw, tcw)     
        return  idxs_ref, idxs_cur


    def pose_optimization(self, f_cur, name=''):
        print('pose opt %s ' % (name) ) 
        pose_before=f_cur.pose.copy() 
        # f_cur pose optimization 1  (here we use f_cur pose as first guess and exploit the matched map points of f_ref )
        self.timer_pose_opt.start()          
        self.mean_pose_opt_chi2_error, self.pose_is_ok, self.num_matched_map_points_in_last_pose_opt = optimizer_g2o.pose_optimization(f_cur, verbose=False)
        self.timer_pose_opt.pause()
        print('     error^2: %f,  ok: %d' % (self.mean_pose_opt_chi2_error, int(self.pose_is_ok)) ) 
        
        if not self.pose_is_ok: 
            # if current pose optimization failed, reset f_cur pose             
            f_cur.update_pose(pose_before)                         
         
        return self.pose_is_ok, self.mean_pose_opt_chi2_error
    
    
    # track camera motion of f_cur w.r.t. f_ref 
    def track_previous_frame(self, f_ref: Frame, f_cur: Frame):            
        print('>>>> tracking previous frame ...')        
        is_search_frame_by_projection_failure = False 
        use_search_frame_by_projection = self.motion_model.is_ok and kUseSearchFrameByProjection and kUseMotionModel
        
        if use_search_frame_by_projection: 
            # search frame by projection: match map points observed in f_ref with keypoints of f_cur
            print('search frame by projection') 
            search_radius = Parameters.kMaxReprojectionDistanceFrame  
            
            # NOTE: the following two lines are commented for the moment since they seem to provide less stable tracking
            # if self.sensor_type != SensorType.STEREO:
            #     search_radius = 2*Parameters.kMaxReprojectionDistanceFrame        
            
            f_cur.reset_points()               
            self.timer_seach_frame_proj.start()
            idxs_ref, idxs_cur, num_found_map_pts = search_frame_by_projection(f_ref, f_cur,
                                                                             max_reproj_distance=search_radius,
                                                                             max_descriptor_distance=self.descriptor_distance_sigma,
                                                                             is_monocular=(self.sensor_type == SensorType.MONOCULAR))
            self.timer_seach_frame_proj.refresh()  
            self.num_matched_kps = len(idxs_cur)    
            print("# matched map points in prev frame: %d " % self.num_matched_kps)
                                    
            # if not enough map point matches consider a larger search radius 
            if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                f_cur.remove_frame_views(idxs_cur)
                f_cur.reset_points()   
                idxs_ref, idxs_cur, num_found_map_pts = search_frame_by_projection(f_ref, f_cur,
                                                                                 max_reproj_distance=2*search_radius,
                                                                                 max_descriptor_distance=0.5*self.descriptor_distance_sigma,
                                                                                 is_monocular=(self.sensor_type == SensorType.MONOCULAR))
                self.num_matched_kps = len(idxs_cur)    
                Printer.orange("# matched map points in prev frame (wider search 1): %d " % self.num_matched_kps)                   
                                                
            if kShowFeatureMatches and True: 
                img_matches = draw_feature_matches(f_ref.img, f_cur.img, 
                                                   f_ref.kps[idxs_ref], f_cur.kps[idxs_cur], 
                                                   f_ref.sizes[idxs_ref], f_cur.sizes[idxs_cur],
                                                    horizontal=False)
                cv2.imshow('tracking frame by projection - matches', img_matches)
                cv2.waitKey(1)                
                        
            if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                f_cur.remove_frame_views(idxs_cur)
                f_cur.reset_points()                   
                is_search_frame_by_projection_failure = True                   
                Printer.red('Not enough matches in search frame by projection: ', self.num_matched_kps)
            else:   
                # search frame by projection was successful 
                if kUseDynamicDesDistanceTh: 
                    self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stats(f_ref, f_cur, idxs_ref, idxs_cur)         
                              
                # store tracking info (for possible reuse)                                                    
                self.idxs_ref = idxs_ref 
                self.idxs_cur = idxs_cur 
                                         
                # f_cur pose optimization 1:  
                # here, we use f_cur pose as first guess and exploit the matched map point of f_ref 
                self.pose_optimization(f_cur,'proj-frame-frame')
                 # update matched map points; discard outliers detected in last pose optimization 
                self.num_matched_map_points = f_cur.clean_outlier_map_points()   
                #print('     # num_matched_map_points_in_last_pose_opt: %d' % (self.num_matched_map_points_in_last_pose_opt) )
                #print('     # matched points: %d' % (self.num_matched_map_points) )
                                      
                if not self.pose_is_ok or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame:
                    Printer.red(f'failure in tracking previous frame, # matched map points: {self.num_matched_map_points}')                    
                    self.pose_is_ok = False                                                                                                   
        
        if not use_search_frame_by_projection or is_search_frame_by_projection_failure:
            self.track_reference_frame(f_ref, f_cur,'match-frame-frame')                        
                          
                                                           
    # track camera motion of f_cur w.r.t. f_ref
    # estimate motion by matching keypoint descriptors                    
    def track_reference_frame(self, f_ref, f_cur, name=''):
        print('>>>> tracking reference %d ...' %(f_ref.id))        
        if f_ref is None:
            return 
        # find keypoint matches between f_cur and kf_ref   
        print('matching keypoints with ', FeatureTrackerShared.feature_matcher.matcher_type.name)              
        self.timer_match.start()
        matching_result = match_frames(f_cur, f_ref) 
        idxs_cur, idxs_ref = np.asarray(matching_result.idxs1), np.asarray(matching_result.idxs2)           
        self.timer_match.refresh()          
        self.num_matched_kps = idxs_cur.shape[0]    
        print("# keypoints matched: %d " % self.num_matched_kps)  
        if kUseEssentialMatrixFitting: 
            # estimate camera orientation and inlier matches by fitting and essential matrix (see the limitations above)             
            idxs_ref, idxs_cur = self.estimate_pose_by_fitting_ess_mat(f_ref, f_cur, idxs_ref, idxs_cur)      
        
        if kUseDynamicDesDistanceTh: 
            self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stats(f_ref, f_cur, idxs_ref, idxs_cur)        
                               
        # propagate map point matches from kf_ref to f_cur  (do not override idxs_ref, idxs_cur)
        num_found_map_pts_inter_frame, idx_ref_prop, idx_cur_prop = propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur, 
                                                                                                max_descriptor_distance=self.descriptor_distance_sigma) 
        print("# matched map points in prev frame: %d " % num_found_map_pts_inter_frame)      
                
        if kShowFeatureMatches and True: 
            img_matches = draw_feature_matches(f_ref.img, f_cur.img, 
                                               f_ref.kps[idx_ref_prop], f_cur.kps[idx_cur_prop], 
                                               f_ref.sizes[idx_ref_prop], f_cur.sizes[idx_cur_prop],
                                               horizontal=False)
            cv2.imshow('tracking frame (no projection) - matches', img_matches)
            cv2.waitKey(1)      
                                
        # store tracking info (for possible reuse)              
        self.idxs_ref = idxs_ref 
        self.idxs_cur = idxs_cur   
                                    
        # f_cur pose optimization using last matches with kf_ref:  
        # here, we use first guess of f_cur pose and propated map point matches from f_ref (matched keypoints) 
        self.pose_optimization(f_cur, name)  
        # update matched map points; discard outliers detected in last pose optimization 
        self.num_matched_map_points = f_cur.clean_outlier_map_points()   
        #print('     # num_matched_map_points_in_last_pose_opt: %d' % (self.num_matched_map_points_in_last_pose_opt) ) 
        #print('     # matched points: %d' % (self.num_matched_map_points) )               
        if not self.pose_is_ok or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame:
            f_cur.remove_frame_views(idxs_cur)
            f_cur.reset_points()               
            Printer.red(f'failure in tracking reference {f_ref.id}, # matched map points: {self.num_matched_map_points}')  
            self.pose_is_ok = False            
        
        
    # track camera motion of f_cur w.r.t. given keyframe
    # estimate motion by matching keypoint descriptors                    
    def track_keyframe(self, keyframe, f_cur, name='match-frame-keyframe'): 
        f_cur.update_pose(self.f_ref.pose.copy()) # start pose optimization from last frame pose                    
        self.track_reference_frame(keyframe, f_cur, name)
        
    def update_local_map(self):            
        self.f_cur.clean_bad_map_points()
        #self.local_points = self.map.local_map.get_points()
        self.kf_ref, self.local_keyframes, self.local_points = self.map.local_map.get_frame_covisibles(self.f_cur)
        if self.kf_ref is not None:       
            self.f_cur.kf_ref = self.kf_ref  
        
    # track camera motion of f_cur w.r.t. the built local map  
    # find matches between {local map points} (points in the built local map) and {unmatched keypoints of f_cur}   
    def track_local_map(self, f_cur: Frame): 
        if self.map.local_map.is_empty():
            return 
        print('>>>> tracking local map...')
        self.timer_seach_map.start()
                
        self.update_local_map()
        
        if self.local_points is None or len(self.local_points) == 0:
            self.pose_is_ok = False
            return
        
        # use the updated local map to search for matches between {local map points} and {unmatched keypoints of f_cur}
        num_found_map_pts, reproj_err_frame_map_sigma, matched_points_frame_idxs = search_map_by_projection(self.local_points, f_cur,
                                    max_reproj_distance=self.reproj_err_frame_map_sigma, #Parameters.kMaxReprojectionDistanceMap, 
                                    max_descriptor_distance=self.descriptor_distance_sigma,
                                    ratio_test=Parameters.kMatchRatioTestMap,
                                    far_points_threshold=self.far_points_threshold)          
        self.timer_seach_map.refresh()
        #print('reproj_err_sigma: ', reproj_err_frame_map_sigma, ' used: ', self.reproj_err_frame_map_sigma)        
        print(f"# matched map points in local map: {num_found_map_pts}, perc%: {100*num_found_map_pts/len(self.local_points):.2f}")                   
        print("# local map points ", self.map.local_map.num_points())         
        
        if kShowFeatureMatches and True: 
            img_matched_trails = f_cur.draw_feature_trails(f_cur.img.copy(), matched_points_frame_idxs, trail_max_length=3) 
            cv2.imshow('tracking local map - matched trails', img_matched_trails)
            cv2.waitKey(1)          
             
        # f_cur pose optimization 2 with all the matched local map points 
        self.pose_optimization(f_cur,'proj-map-frame')    
        self.num_matched_map_points = f_cur.update_map_points_statistics(self.sensor_type)  # here we reset outliers only in the case of STEREO; in other cases, 
                                                                                        # we let them reach the keyframe generation 
                                                                                        # and then bundle adjustment will possible decide if remove them or not;
                                                                                        # only after keyframe generation the outliers are cleaned!
        #print('     # num_matched_points: %d' % (self.num_matched_map_points) )
        if not self.pose_is_ok or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackLocalMap:
            Printer.red(f'failure in tracking local map, # matched map points: {self.num_matched_map_points}') 
            self.pose_is_ok = False                                        
        
        #if kUseDynamicDesDistanceTh: 
        #    self.reproj_err_frame_map_sigma = self.dyn_config.update_reproj_err_map_stats(reproj_err_frame_map_sigma)                         


    # store frame history in order to retrieve the complete camera trajectory 
    def update_tracking_history(self):
        if self.state==SlamState.OK:        
            isometry3d_Tcr = self.f_cur.isometry3d * self.f_cur.kf_ref.isometry3d.inverse() # pose of current frame w.r.t. current reference keyframe kf_ref 
            self.tracking_history.relative_frame_poses.append(isometry3d_Tcr)
            self.tracking_history.kf_references.append(self.kf_ref)     
            self.tracking_history.timestamps.append(self.f_cur.timestamp)
            self.tracking_history.ids.append(self.f_cur.id)           
        else:
            if len(self.tracking_history.relative_frame_poses) > 0:
                self.tracking_history.relative_frame_poses.append(self.tracking_history.relative_frame_poses[-1])
                self.tracking_history.kf_references.append(self.tracking_history.kf_references[-1])   
                self.tracking_history.timestamps.append(self.tracking_history.timestamps[-1])
                self.tracking_history.ids.append(self.tracking_history.ids[-1])                                              
        self.tracking_history.slam_states.append(self.state)                
          

    def clean_vo_points(self):
        for p in self.vo_points:
            p.set_bad() 
            p.delete()
        self.vo_points.clear()
      
    def need_new_keyframe(self, f_cur: Frame):
        
        # If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if self.local_mapping.is_stopped() or self.local_mapping.is_stop_requested():
            return False
                
        num_keyframes = self.map.num_keyframes()
        
        # Do not insert keyframes if not enough frames have passed from last relocalisation
        if f_cur.id < self.last_reloc_frame_id + self.max_frames_between_kfs and num_keyframes > self.max_frames_between_kfs:
            return False
        
        nMinObs = kNumMinObsForKeyFrameDefault
        if num_keyframes <= 2:
            nMinObs = 2  # if just two keyframes then we can have just two observations 
        num_kf_ref_tracked_points = self.kf_ref.num_tracked_points(nMinObs)  # number of tracked points in k_ref
        #num_f_cur_tracked_points = f_cur.num_matched_inlier_map_points()     # number of inliers in f_cur
        num_f_cur_tracked_points = self.num_matched_map_points if self.num_matched_map_points is not None else 0 # updated in the last self.track_local_map()
        tracking_info_message = f'F({f_cur.id}) #matched points: {num_f_cur_tracked_points}, KF({self.kf_ref.id}) #matched points: {num_kf_ref_tracked_points}'
        Printer.green(tracking_info_message)
        
        if kLogKFinfoToFile:
            self.kf_info_logger.info(tracking_info_message)
        
        self.num_kf_ref_tracked_points = num_kf_ref_tracked_points

        is_local_mapping_idle = self.local_mapping.is_idle()  
        local_mapping_queue_size = self.local_mapping.queue_size()        
        print('is_local_mapping_idle: ', is_local_mapping_idle,', local_mapping_queue_size: ', local_mapping_queue_size)                                    
                
        # Check how many "close" points are being tracked and how many could be potentially created.
        num_non_tracked_close = 0 
        num_tracked_close = 0 
        # Create a mask for tracked points (not None and not an outlier)
        tracked_mask = (f_cur.points != None) & (~f_cur.outliers)        
        if self.sensor_type!=SensorType.MONOCULAR:
            # Create a mask to identify valid depth values within the threshold
            depth_mask = (f_cur.depths > 0) & (f_cur.depths < f_cur.camera.depth_threshold)
            # Create a mask for tracked points (not None and not an outlier)
            #tracked_mask = (f_cur.points != None) & (~f_cur.outliers)
            # Count points that are close and tracked
            num_tracked_close = np.sum(depth_mask & tracked_mask)
            # Count points that are close but not tracked
            num_non_tracked_close = np.sum(depth_mask & ~tracked_mask)
            
        is_need_to_insert_close = (num_tracked_close < Parameters.kNumMinTrackedClosePointsForNewKfNonMonocular) and \
                                  (num_non_tracked_close > Parameters.kNumMaxNonTrackedClosePointsForNewKfNonMonocular)
                         
        #  Thresholds
        thRefRatio = Parameters.kThNewKfRefRatioStereo
        if num_keyframes < 2:
            thRefRatio = 0.4

        if self.sensor_type == SensorType.MONOCULAR:
            thRefRatio = Parameters.kThNewKfRefRatio
                                                                    
        # condition 1a: more than "max_frames_between_kfs" have passed from last keyframe insertion                                        
        cond1a = f_cur.id >= (self.kf_last.id + self.max_frames_between_kfs) 
        
        # condition 1b: more than "min_frames_between_kfs" have passed and local mapping is idle
        cond1b = (f_cur.id >= (self.kf_last.id + self.min_frames_between_kfs)) & is_local_mapping_idle          
        #cond1b = (f_cur.id >= (self.kf_last.id + self.min_frames_between_kfs)) 
                  
        # condition 1c: tracking is weak 1
        cond1c = (self.sensor_type!=SensorType.MONOCULAR) and (num_f_cur_tracked_points<num_kf_ref_tracked_points*Parameters.kThNewKfRefRatioNonMonocualar or is_need_to_insert_close) 
        
        # condition 1d: tracking image coverage is weak 
        # we divide the image in 3x2 cells and check that each cell is filled by at least one point (the partition is assumed to be gross in order not to generate too many KFs)
        cond1d = False 
        if Parameters.kUseFeatureCoverageControlForNewKf:
            image_grid = ImageGrid(self.camera.width, self.camera.height, num_div_x=3, num_div_y=2)
            image_grid.add_points(f_cur.kps[tracked_mask])
            num_uncovered_cells = image_grid.num_cells_uncovered(num_min_points=1)  
            cond1d = (num_uncovered_cells > 1)    
            if True:
                cv2.namedWindow('grid_img', cv2.WINDOW_NORMAL)
                cv2.imshow('grid_img', image_grid.get_grid_img())
                cv2.waitKey(1)

        # condition 2: few tracked features compared to reference keyframe 
        cond2 = (num_f_cur_tracked_points < num_kf_ref_tracked_points * thRefRatio or is_need_to_insert_close) \
                 and (num_f_cur_tracked_points > Parameters.kNumMinPointsForNewKf)
                 
        # condition 3: distance to closest fov center is too big
        cond3 = False
        if self.use_fov_centers_based_kf_generation:
            if num_f_cur_tracked_points > Parameters.kNumMinPointsForNewKf:
                # compute distance to closest fov center
                close_kfs = self.local_keyframes
                if not self.kf_last in close_kfs:
                    close_kfs.append(self.kf_last)
                if len(close_kfs)>0:
                    close_fov_centers_w = np.array([kf.fov_center_w.flatten() for kf in close_kfs if kf.fov_center_w is not None])            
                    if close_fov_centers_w.shape[0] > 0: 
                        dists = np.linalg.norm(close_fov_centers_w - f_cur.fov_center_w.flatten(), axis=1)
                        min_dist = np.min(dists)
                        cond3 = min_dist > self.max_fov_centers_distance
        
        #print(f'KF conditions: cond1a: {cond1a}, cond1b: {cond1b}, cond1c: {cond1c}, cond1d: {cond1d}, cond2: {cond2}')
        condition_checks = ( (cond1a or cond1b or cond1c or cond1d) and cond2 ) or cond3    
                                                        
        if condition_checks:
            if is_local_mapping_idle:
                return True 
            else: 
                self.local_mapping.interrupt_optimization()
                if self.sensor_type == SensorType.MONOCULAR: 
                    if local_mapping_queue_size <= 3:
                        return True
                    else:                  
                        return False    
                else: 
                    return False 
        else: 
            return False 

    def create_new_keyframe(self, f_cur: Frame, img,  img_right=None, depth=None):
        if not self.local_mapping.set_not_stop(True):
            return
                                      
        kf_new = KeyFrame(f_cur, img, img_right, depth)                                     
        self.kf_last = kf_new  
        self.kf_ref = kf_new 
        f_cur.kf_ref = kf_new                  
        
        Printer.green(f'Adding new KF with id {kf_new.id}, img shape: {img.shape if img is not None else None}, img_right shape: {img_right.shape if img_right is not None else None}, depth shape: {depth.shape if depth is not None else None}')
        if kLogKFinfoToFile:
            self.kf_info_logger.info('adding new KF with frame id % d: ' %(f_cur.id))  
                    
        self.map.add_keyframe(kf_new)   # add kf_cur to map 
        
        if self.sensor_type != SensorType.MONOCULAR:
            self.create_and_add_stereo_map_points_on_new_kf(f_cur, kf_new, img)
        
        self.local_mapping.push_keyframe(kf_new, img, img_right, depth)         
        
        self.local_mapping.set_not_stop(False)

    def relocalize(self, f_cur: Frame, img):
        Printer.green(f'Relocalizing frame id: {f_cur.id}...')
        return self.slam.loop_closing.relocalize(f_cur, img)
 
                      
    def create_vo_points_on_last_frame(self):
        if self.sensor_type == SensorType.MONOCULAR or self.kf_last.id == self.f_ref.id or self.f_ref.depths is None: 
            return
        # Create "visual odometry" MapPoints
        # Sort points according to their measured depth by the stereo/RGB-D sensor
        valid_depths_and_idxs = [(z, i) for i, z in enumerate(self.f_ref.depths) if z > 0]
        if len(valid_depths_and_idxs)==0:
            return
        # code similar to the one in create_and_add_stereo_map_points_on_new_kf()
        valid_depths_and_idxs.sort() # increasing-depth order 
        
        sorted_z_values, sorted_idx_values = zip(*valid_depths_and_idxs) # unpack the sorted z values and i values into separate lists
        sorted_z_values = np.array(sorted_z_values, dtype=np.float32)
        sorted_idx_values = np.array(sorted_idx_values, dtype=np.int32)
                                
        N = 100
        # create new map points where the depth is smaller than the prefixed depth threshold 
        #        or at least N new points with the closest depths
        mask_depths_smaller_than_th = sorted_z_values < self.f_ref.camera.depth_threshold
        mask_first_N_points = np.zeros(len(sorted_z_values), dtype=bool)[:min(N, len(sorted_z_values))] = True
        mask_first_selection = np.logical_or(mask_depths_smaller_than_th, mask_first_N_points)
        
        sorted_z_values = sorted_z_values[mask_first_selection]
        sorted_idx_values = sorted_idx_values[mask_first_selection]
        sorted_points = self.f_ref.points[sorted_idx_values]
                    
        # get the points that are None or where the num of observations i< smaller than 1
        vector_num_mp_observations = np.array([ p.num_observations if p is not None else 0 for p in sorted_points], dtype=np.int32)
        mask_where_to_create_new_map_points = vector_num_mp_observations<1
            
        sorted_z_values = sorted_z_values[mask_where_to_create_new_map_points]
        sorted_idx_values = sorted_idx_values[mask_where_to_create_new_map_points]
        sorted_points = sorted_points[mask_where_to_create_new_map_points]     
        
        # finally, let's create the new points on the last frame 
        pts3d, pts3d_mask = self.f_ref.unproject_points_3d(sorted_idx_values, transform_in_world=True)
        num_added_points = 0
        for i, p in enumerate(pts3d):
            if not pts3d_mask[i]:
                continue
            color = (0, 0, 255)
            # add the point to this map                 
            mp = MapPoint(p[0:3], color, self.f_ref, sorted_idx_values[i]) 
            self.f_ref.points[sorted_idx_values[i]] = mp # add point to the frame
            num_added_points += 1
        print(f'Added #new VO points: {num_added_points}')
           
        
    # kf is a newly created keyframe starting from frame f
    def create_and_add_stereo_map_points_on_new_kf(self, f: Frame, kf: KeyFrame, img):
        if self.sensor_type != SensorType.MONOCULAR and kf.depths is not None: 
            valid_depths_and_idxs = [(z, i) for i, z in enumerate(kf.depths) if z > 0]
            valid_depths_and_idxs.sort() # increasing-depth order 
            
            if len(valid_depths_and_idxs)==0:
                Printer.yellow('[create_and_add_stereo_map_points_on_new_kf] no valid depths and idxs found, returning')
                return
            
            sorted_z_values, sorted_idx_values = zip(*valid_depths_and_idxs) # unpack the sorted z values and i values into separate lists
            sorted_z_values = np.array(sorted_z_values, dtype=np.float32)
            sorted_idx_values = np.array(sorted_idx_values, dtype=np.int32)
                                    
            N = 100
            # create new map points where the depth is smaller than the prefixed depth threshold 
            #        or at least N new points with the closest depths
            mask_depths_smaller_than_th = sorted_z_values < kf.camera.depth_threshold
            mask_first_N_points = np.zeros(len(sorted_z_values), dtype=bool)
            mask_first_N_points[:min(N, len(sorted_z_values))] = True  # set True for the first N points otherwise set all True if len(sorted_z_values) < N
            mask_first_selection = np.logical_or(mask_depths_smaller_than_th, mask_first_N_points)
            
            sorted_z_values = sorted_z_values[mask_first_selection]
            sorted_idx_values = sorted_idx_values[mask_first_selection]
            sorted_points = kf.points[sorted_idx_values]
                        
            # get the points that are None or where the num of observations is smaller than 1
            vector_num_mp_observations = np.array([ p.num_observations if p is not None else 0 for p in sorted_points], dtype=np.int32)
            mask_where_to_create_new_map_points = vector_num_mp_observations<1
             
            sorted_z_values = sorted_z_values[mask_where_to_create_new_map_points]
            sorted_idx_values = sorted_idx_values[mask_where_to_create_new_map_points]
            #sorted_points = sorted_points[mask_where_to_create_new_map_points]
                                                
            # we need to reset the points in the originary frame 
            null_array = np.full(len(sorted_idx_values), None, dtype=object)
            f.points[sorted_idx_values] = null_array
            
            pts3d, pts3d_mask = f.unproject_points_3d(sorted_idx_values, transform_in_world=True)
            num_added_points = self.map.add_stereo_points(pts3d, pts3d_mask, f, kf, sorted_idx_values, img)
            self.last_num_static_stereo_map_points = num_added_points
        

    # Since we do not have real-time performances, we can slow-down things and make tracking wait till local mapping gets idle  
    # N.B.: this function must be called outside 'with self.map.update_lock' blocks, 
    #       since both self.track() and the local-mapping optimization use the RLock 'map.update_lock'    
    #       => they cannot wait for each other once map.update_lock is locked (deadlock)                        
    def wait_for_local_mapping(self, timeout=1.0):                   
        if kTrackingWaitForLocalMappingToGetIdle:                        
            #while not self.local_mapping.is_idle() or self.local_mapping.queue_size()>0:       
            if not self.local_mapping.is_idle():       
                print('>>>> waiting for local mapping...')                         
                self.local_mapping.wait_idle(print=print, timeout=timeout)
        else:
            # if we are close to bad tracking give local mapping more time
            if self.num_matched_map_points is not None and self.num_matched_map_points < kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle:
                if self.local_mapping.queue_size()>0:
                    Printer.orange(">>>> close to bad tracking: forcing waiting for local mapping...")
                    self.local_mapping.wait_idle(print=print, timeout=timeout)
                            
            if self.local_mapping.queue_size()>0: # and self.trackingWaitForLocalMappingSleepTime>0:
                print(f'>>>> waiting for local mapping idle (queue_size={self.local_mapping.queue_size()})...')  
                self.local_mapping.wait_idle(print=print, timeout=timeout)
                
        # check again for debug                     
        #is_local_mapping_idle = self.local_mapping.is_idle()  
        #local_mapping_queue_size = self.local_mapping.queue_size()        
        #print('is_local_mapping_idle: ', is_local_mapping_idle,', local_mapping_queue_size: ', local_mapping_queue_size)             


    # Here, pose estimates are saved online: At each frame, the current pose estimate is saved. 
    # Note that in other frameworks, pose estimates may be saved at the end of the dataset playback 
    # so that each pose estimate is refined multiple times by LBA and BA over the multiple window optimizations that cover it.  
    def update_history(self):
        f_cur = self.map.get_frame(-1)
        self.cur_R = f_cur.pose[:3,:3].T
        self.cur_t = np.dot(-self.cur_R,f_cur.pose[:3,3])
        if self.init_history is True:
            self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
            if self.gt_x is not None:
                self.t0_gt  = np.array([self.gt_x, self.gt_y, self.gt_z])           # starting translation 
        if self.t0_est is not None:            
            p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
            self.traj3d_est.append(p)
            if self.t0_gt is not None: 
                self.traj3d_gt.append([self.gt_x-self.t0_gt[0], self.gt_y-self.t0_gt[1], self.gt_z-self.t0_gt[2]])            
            self.poses.append(poseRt(self.cur_R, p))
            self.pose_timestamps.append(f_cur.timestamp)


    # @ main track method @
    def track(self, img, img_right, depth, img_id, timestamp=None):
        Printer.cyan(f'@tracking {self.sensor_type.name}, img id: {img_id}, frame id: {Frame.next_id()}, state: {self.state.name}')
        time_start = time.time()
                
        # check image size is coherent with camera params 
        print(f'img.shape: {img.shape}, camera: {self.camera.height}x{self.camera.width}')
        if depth is not None: 
            print("depth.shape: ", depth.shape)
        if img_right is not None:
            print("img_right.shape: ", img_right.shape)
        assert img.shape[0:2] == (self.camera.height, self.camera.width)   
        if timestamp is not None: 
            print('timestamp: ', timestamp)  
        
        self.timer_main_track.start()
        
        # at initialization time is better to use more extracted features     
        if self.state != SlamState.OK:
            FeatureTrackerShared.feature_tracker.set_double_num_features() 
        else:
            FeatureTrackerShared.feature_tracker.set_normal_num_features()

        # build current frame 
        self.timer_frame.start()        
        f_cur = Frame(self.camera, img, img_right=img_right, depth=depth, timestamp=timestamp, img_id=img_id) 
        self.f_cur = f_cur 
        #print("frame: ", f_cur.id)        
        self.timer_frame.refresh()   
        
        # reset indexes of matches 
        self.idxs_ref = [] 
        self.idxs_cur = []           
        
        if self.state == SlamState.NO_IMAGES_YET: 
            # push first frame in the inizializer 
            self.intializer.init(f_cur, img, img_right, depth) 
            self.state = SlamState.NOT_INITIALIZED
            return # EXIT (jump to second frame)
        
        if self.state == SlamState.NOT_INITIALIZED:
            # try to inizialize 
            initializer_output, intializer_is_ok = self.intializer.initialize(f_cur, img, img_right, depth)
            if intializer_is_ok:
                kf_ref = initializer_output.kf_ref
                kf_cur = initializer_output.kf_cur
                        
                # add the two initialized frames in the map 
                self.map.keyframe_origins.add(kf_ref)
                self.map.add_frame(kf_ref) # add first frame in map and update its frame id
                self.map.add_frame(kf_cur) # add second frame in map and update its frame id  
                # add the two initialized frames as keyframes in the map               
                self.map.add_keyframe(kf_ref) # add first keyframe in map and update its kid
                self.map.add_keyframe(kf_cur) # add second keyframe in map and update its kid                     
                kf_ref.init_observations()
                kf_cur.init_observations()
                
                # add points in map
                new_pts_count,_,_ = self.map.add_points(initializer_output.pts, None, kf_cur, kf_ref, initializer_output.idxs_cur, initializer_output.idxs_ref, img, do_check=False)
                #new_pts_count = self.map.add_stereo_points(initializer_output.pts, None, f_cur, kf_cur, np.arange(len(initializer_output.pts), dtype=np.int), img)
                
                Printer.green(f"map: initialized with kfs {kf_ref.id}, {kf_cur.id} and {new_pts_count} new map points")                 
                
                # update covisibility graph connections 
                kf_ref.update_connections()
                kf_cur.update_connections()     
                
                # update tracking info            
                self.f_cur = kf_cur 
                self.f_cur.kf_ref = kf_ref                         
                self.kf_ref = kf_cur  # set reference keyframe 
                self.kf_last = kf_cur # set last added keyframe                                     
                self.map.local_map.update(self.kf_ref)
                self.state = SlamState.OK      
                
                self.update_tracking_history()
                self.motion_model.update_pose(kf_cur.timestamp,kf_cur.position,kf_cur.quaternion)
                self.motion_model.is_ok = False   # after initialization you cannot use motion model for next frame pose prediction (time ids of initialized poses may not be consecutive)
                
                self.intializer.reset()
                
                if kUseDynamicDesDistanceTh: 
                    self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stats(kf_ref, kf_cur, initializer_output.idxs_ref, initializer_output.idxs_cur)                     
            return # EXIT (jump to next frame)
                
        # get previous frame in map as reference        
        f_ref = self.map.get_frame(-1) 
        #f_ref_2 = self.map.get_frame(-2)
        self.f_ref = f_ref 
        
        # add current frame f_cur to map                  
        self.map.add_frame(f_cur)          
        self.f_cur.kf_ref = self.kf_ref  
        
        # reset pose state flag 
        self.pose_is_ok = False 
                            
        # HACK: Since loop closing may be not fast enough (when adjusting the loop) in python (and tracking is not in real-time) => give loop closing more time to process stuff 
        if self.slam.loop_closing is not None:
            if self.slam.loop_closing.is_closing():
                self.local_mapping.wait_idle(print=print)
                self.slam.loop_closing.wait_if_closing()
                print('...loop closing done')                            
                            
        # HACK: Since local mapping may be not fast enough in python (and tracking is not in real-time) => give local mapping more time to process stuff  
        self.wait_for_local_mapping()  # N.B.: this must be outside the `with self.map.update_lock:` block
                                    
        with self.map.update_lock:
            
            # DEBUG:
            # if img_id == 50:
            # # if self.map.num_keyframes() > 50: # for ibow 
            #     self.state = SlamState.LOST # force to lost state for testing relocalization
            
            if self.state == SlamState.OK:
                
                # check for map point replacements in previous frame f_ref (some points might have been replaced by local mapping during point fusion)
                self.f_ref.check_replaced_map_points()
                                                                            
                # udpdate (velocity) old motion model                                             # c1=ref_ref, c2=ref, c3=cur;  c=cur, r=ref
                #self.velocity = np.dot(f_ref.pose, inv_T(f_ref_2.pose))                          # Tc2c1 = Tc2w * Twc1   (predicted Tcr)
                #self.predicted_pose = g2o.Isometry3d(np.dot(self.velocity, f_ref.pose))          # Tc3w = Tc2c1 * Tc2w   (predicted Tcw)        
                                                        
                # set intial guess for current pose optimization                         
                if kUseMotionModel and self.motion_model.is_ok:
                    print('using motion model for next pose prediction')                   
                    # update f_ref pose according to its reference keyframe (the pose of the reference keyframe could be updated by local mapping)
                    self.f_ref.update_pose(self.tracking_history.relative_frame_poses[-1] * self.f_ref.kf_ref.isometry3d)         
                    if Parameters.kUseVisualOdometryPoints:                         
                        self.create_vo_points_on_last_frame()
                    # predict pose by using motion model 
                    self.predicted_pose,_ = self.motion_model.predict_pose(timestamp, self.f_ref.position, self.f_ref.orientation)            
                    f_cur.update_pose(self.predicted_pose)
                else:
                    print('setting f_cur.pose <-- f_ref.pose')
                    # use reference frame pose as initial guess 
                    f_cur.update_pose(f_ref.pose)
                                                        
                # track camera motion from f_ref to f_cur
                self.track_previous_frame(f_ref, f_cur)
                
                if not self.pose_is_ok:
                    # if previous track didn't go well then track the camera motion from kf_ref to f_cur 
                    self.track_keyframe(self.kf_ref, f_cur)
                    
            else:
                if self.state != SlamState.INIT_RELOCALIZE:
                    self.state = SlamState.RELOCALIZE
                if self.relocalize(f_cur, img):
                    if self.state != SlamState.INIT_RELOCALIZE:
                        self.last_reloc_frame_id = f_cur.id                    
                    self.state = SlamState.OK
                    self.pose_is_ok = True
                    self.kf_ref = f_cur.kf_ref # right side updated by self.relocalize()
                    self.kf_last = self.kf_ref
                    Printer.green('Relocalization successful')
                else: 
                    Printer.red('Relocalization failed')
                                            
            # now, having a better estimate of f_cur pose, we can find more map point matches: 
            # find matches between {local map points} (points in the local map) and {unmatched keypoints of f_cur}
            if self.pose_is_ok: 
                self.track_local_map(f_cur)
            
            # update slam state 
            if self.pose_is_ok:
                self.state=SlamState.OK          
            else:
                if self.state==SlamState.OK:               
                    self.state=SlamState.LOST
                    Printer.red('tracking failure')
                                
            # update motion model state     
            self.motion_model.is_ok = self.pose_is_ok                    
                                
            if self.pose_is_ok:   # if tracking was successful
                
                # update motion model                     
                self.motion_model.update_pose(timestamp, f_cur.position, f_cur.quaternion)  
                                                                    
                f_cur.clean_vo_map_points() # clean VO matches
                self.clean_vo_points()      # clean VO points
                        
                # do we need a new KeyFrame? 
                need_new_kf = self.need_new_keyframe(f_cur)
                                    
                if need_new_kf:
                    self.create_new_keyframe(f_cur, img, img_right, depth)
                                        
                    if not kLocalMappingOnSeparateThread:
                        self.local_mapping.step()
                                                                               
                else: 
                    Printer.yellow('NOT KF')      
                    
                # From ORBSLAM2: 
                # Clean outliers once keyframe generation has been managed:
                # we allow points with high innovation (considered outliers by the Huber Function)
                # pass to the new keyframe, so that bundle adjustment will finally decide
                # if they are outliers or not. We don't want next frame to estimate its position
                # with those points so we discard them in the frame.                
                f_cur.clean_outlier_map_points()     
                
                                  
        # end block {with self.map.update_lock:}  
                                             
        # NOTE: this reset must be outside the block {with self.map.update_lock:}  
        need_reset = self.slam.reset_requested or (self.state == SlamState.LOST and self.map.num_keyframes_session() <= 5)                  
        if need_reset: 
            Printer.yellow('\nTracking: SLAM resetting...\n')
            state_before_reset = self.state              
            self.slam.reset_session()
            if state_before_reset == SlamState.LOST and self.map.is_reloaded():
                self.state = SlamState.INIT_RELOCALIZE
            Printer.yellow('\nTracking: SLAM reset done\n')
            return
                                                                  
        if self.f_cur.kf_ref is None:
            self.f_cur.kf_ref = self.kf_ref
                                
        self.update_tracking_history()    # must stay after having updated slam state (self.state)                                                                  
        self.update_history()
        Printer.green("map: %d points, %d keyframes" % (self.map.num_points(), self.map.num_keyframes()))
        #self.update_history()   
                        
        self.timer_main_track.refresh()
        elapsed_time = time.time() - time_start
        self.time_track = elapsed_time
        print('Tracking: elapsed_time: ', elapsed_time)     