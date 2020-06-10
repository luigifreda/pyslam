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
import time
from enum import Enum

from collections import defaultdict, Counter
from itertools import chain

import cv2
import g2o

from parameters import Parameters  

from frame import Frame, match_frames
from keyframe import KeyFrame
from map_point import MapPoint
from map import Map

from search_points import propagate_map_point_matches
from search_points import search_map_by_projection, search_frame_by_projection

from local_mapping import LocalMapping
from initializer import Initializer
import optimizer_g2o

from timer import TimerFps

from slam_dynamic_config import SLAMDynamicConfig
from motion_model import MotionModel, MotionModelDamping

from feature_tracker import FeatureTrackerTypes 

from utils import Printer, getchar, Logging
from utils_draw import draw_feature_matches
from utils_geom import triangulate_points, poseRt, normalize_vector, inv_T, triangulate_normalized_points, estimate_pose_ess_mat


kVerbose = True     
kTimerVerbose = False 
kDebugDrawMatches = False 

kLocalMappingOnSeparateThread = Parameters.kLocalMappingOnSeparateThread 
kTrackingWaitForLocalMappingToGetIdle = Parameters.kTrackingWaitForLocalMappingToGetIdle
kTrackingWaitForLocalMappingSleepTime = Parameters.kTrackingWaitForLocalMappingSleepTime

kLogKFinfoToFile = True 

kUseDynamicDesDistanceTh = True  

kRansacThresholdNormalized = 0.0003  # 0.0003 # metric threshold used for normalized image coordinates 
kRansacProb = 0.999
kNumMinInliersEssentialMat = 8

kUseGroundTruthScale = False 

kNumMinInliersPoseOptimizationTrackFrame = 10
kNumMinInliersPoseOptimizationTrackLocalMap = 20

kUseMotionModel = Parameters.kUseMotionModel or Parameters.kUseSearchFrameByProjection
kUseSearchFrameByProjection = Parameters.kUseSearchFrameByProjection and not Parameters.kUseEssentialMatrixFitting         
kUseEssentialMatrixFitting = Parameters.kUseEssentialMatrixFitting      
       
kNumMinObsForKeyFrameDefault = 3


if not kVerbose:
    def print(*args, **kwargs):
        pass 


class SlamState(Enum):
    NO_IMAGES_YET=0,
    NOT_INITIALIZED=1,
    OK=2,
    LOST=3


class TrackingHistory(object):
    def __init__(self):
        self.relative_frame_poses = []  # list of relative frame poses as g2o.Isometry3d() (see camera_pose.py)
        self.kf_references = []         # list of reference keyframes  
        self.timestamps = []            # list of frame timestamps 
        self.slam_states = []           # list of slam states 


# main slam class containing all the required modules 
class Slam(object):
    def __init__(self, camera, feature_tracker, groundtruth = None):    
        self.init_feature_tracker(feature_tracker)
        self.camera = camera 
        self.map = Map()
        self.local_mapping = LocalMapping(self.map)
        if kLocalMappingOnSeparateThread:
            self.local_mapping.start()
        self.groundtruth = groundtruth  # not actually used here; could be used for evaluating performances 
        self.tracking = Tracking(self)

        
    def quit(self):
        if kLocalMappingOnSeparateThread:
            self.local_mapping.quit()                       


    def init_feature_tracker(self, tracker):
        Frame.set_tracker(tracker) # set the static field of the class 
        if kUseEssentialMatrixFitting:
            Printer.orange('forcing feature matcher ratio_test to 0.8')
            tracker.matcher.ratio_test = 0.8
        if tracker.tracker_type == FeatureTrackerTypes.LK:
            raise ValueError("You cannot use Lukas-Kanade tracker in this SLAM approach!")  
        
        
    # @ main track method @
    def track(self, img, frame_id, timestamp=None):
        return self.tracking.track(img,frame_id,timestamp)


class Tracking(object):
    def __init__(self, system):
        
        if kDebugDrawMatches: 
            Frame.is_store_imgs = True 
                    
        self.system = system                     
        self.camera = system.camera 
        self.map = system.map
        
        self.local_mapping = system.local_mapping
                                
        self.intializer = Initializer()
        
        self.motion_model = MotionModel()  # motion model for current frame pose prediction without damping  
        #self.motion_model = MotionModelDamping()  # motion model for current frame pose prediction with damping       
        
        self.dyn_config = SLAMDynamicConfig()
        self.descriptor_distance_sigma = Parameters.kMaxDescriptorDistance 
        self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMap        
        
        self.max_frames_between_kfs = int(system.camera.fps) 
        self.min_frames_between_kfs = 0         

        self.state = SlamState.NO_IMAGES_YET
        
        self.num_matched_kps = None           # current number of matched keypoints 
        self.num_inliers = None               # current number of matched points 
        self.num_matched_map_points = None         # current number of matched map points (matched and found valid in current pose optimization)     
        self.num_kf_ref_tracked_points = None # number of tracked points in k_ref (considering a minimum number of observations)      
        
        self.mask_match = None 

        self.pose_is_ok = False 
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
         
        self.tracking_history = TrackingHistory()
 
        self.timer_verbose = kTimerVerbose  # set this to True if you want to print timings 
        self.timer_main_track = TimerFps('Track', is_verbose = self.timer_verbose)
        self.timer_pose_opt = TimerFps('Pose optimization', is_verbose = self.timer_verbose)
        self.timer_seach_frame_proj = TimerFps('Search frame by proj', is_verbose = self.timer_verbose) 
        self.timer_match = TimerFps('Match', is_verbose = self.timer_verbose)                   
        self.timer_pose_est = TimerFps('Ess mat pose estimation', is_verbose = self.timer_verbose)
        self.timer_frame = TimerFps('Frame', is_verbose = self.timer_verbose)
        self.timer_seach_map = TimerFps('Search map', is_verbose = self.timer_verbose)     
        
        self.init_history = True 
        self.poses = []       # history of poses
        self.t0_est = None    # history of estimated translations      
        self.t0_gt = None     # history of ground truth translations (if available)
        self.traj3d_est = []  # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = []   # history of estimated ground truth translations centered w.r.t. first one                 

        self.cur_R = None # current rotation w.r.t. world frame  
        self.cur_t = None # current translation w.r.t. world frame 
        self.trueX, self.trueY, self.trueZ = None, None, None
        self.groundtruth = system.groundtruth  # not actually used here; could be used for evaluating performances 
        
        if kLogKFinfoToFile:
            self.kf_info_logger = Logging.setup_file_logger('kf_info_logger', 'kf_info.log',formatter=Logging.simple_log_formatter)
                 

    # estimate a pose from a fitted essential mat; 
    # since we do not have an interframe translation scale, this fitting can be used to detect outliers, estimate interframe orientation and translation direction 
    # N.B. read the NBs of the method estimate_pose_ess_mat(), where the limitations of this method are explained  
    def estimate_pose_by_fitting_ess_mat(self, f_ref, f_cur, idxs_ref, idxs_cur): 
        # N.B.: in order to understand the limitations of fitting an essential mat, read the comments of the method self.estimate_pose_ess_mat() 
        self.timer_pose_est.start()
        # estimate inter frame camera motion by using found keypoint matches 
        # output of the following function is:  Trc = [Rrc, trc] with ||trc||=1  where c=cur, r=ref  and  pr = Trc * pc 
        Mrc, self.mask_match = estimate_pose_ess_mat(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur], 
                                                     method=cv2.RANSAC, prob=kRansacProb, threshold=kRansacThresholdNormalized)   
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
        pose_opt_error, self.pose_is_ok, self.num_matched_map_points = optimizer_g2o.pose_optimization(f_cur, verbose=False)
        self.timer_pose_opt.pause()
        print('     error^2: %f,  ok: %d' % (pose_opt_error, int(self.pose_is_ok)) ) 
        
        if not self.pose_is_ok: 
            # if current pose optimization failed, reset f_cur pose             
            f_cur.update_pose(pose_before)                         
         
        return self.pose_is_ok   
    
    
    # track camera motion of f_cur w.r.t. f_ref 
    def track_previous_frame(self, f_ref, f_cur):            
        print('>>>> tracking previous frame ...')        
        is_search_frame_by_projection_failure = False 
        use_search_frame_by_projection = self.motion_model.is_ok and kUseSearchFrameByProjection and kUseMotionModel
        
        if use_search_frame_by_projection: 
            # search frame by projection: match map points observed in f_ref with keypoints of f_cur
            print('search frame by projection') 
            search_radius = Parameters.kMaxReprojectionDistanceFrame          
            f_cur.reset_points()               
            self.timer_seach_frame_proj.start()
            idxs_ref, idxs_cur, num_found_map_pts = search_frame_by_projection(f_ref, f_cur,
                                                                             max_reproj_distance=search_radius,
                                                                             max_descriptor_distance=self.descriptor_distance_sigma)
            self.timer_seach_frame_proj.refresh()  
            self.num_matched_kps = len(idxs_cur)    
            print("# matched map points in prev frame: %d " % self.num_matched_kps)
                                    
            # if not enough map point matches consider a larger search radius 
            if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                f_cur.remove_frame_views(idxs_cur)
                f_cur.reset_points()   
                idxs_ref, idxs_cur, num_found_map_pts = search_frame_by_projection(f_ref, f_cur,
                                                                                 max_reproj_distance=2*search_radius,
                                                                                 max_descriptor_distance=0.5*self.descriptor_distance_sigma)
                self.num_matched_kps = len(idxs_cur)    
                Printer.orange("# matched map points in prev frame (wider search): %d " % self.num_matched_kps)    
                                                
            if kDebugDrawMatches and True: 
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
                    self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stat(f_ref, f_cur, idxs_ref, idxs_cur)         
                              
                # store tracking info (for possible reuse)                                                    
                self.idxs_ref = idxs_ref 
                self.idxs_cur = idxs_cur 
                                         
                # f_cur pose optimization 1:  
                # here, we use f_cur pose as first guess and exploit the matched map point of f_ref 
                self.pose_optimization(f_cur,'proj-frame-frame')
                 # update matched map points; discard outliers detected in last pose optimization 
                num_matched_points = f_cur.clean_outlier_map_points()   
                print('     # num_matched_map_points: %d' % (self.num_matched_map_points) )
                #print('     # matched points: %d' % (num_matched_points) )
                                      
                if not self.pose_is_ok or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame:
                    Printer.red('failure in tracking previous frame, # matched map points: ', self.num_matched_map_points)                    
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
        print('matching keypoints with ', Frame.feature_matcher.type.name)              
        self.timer_match.start()
        idxs_cur, idxs_ref = match_frames(f_cur, f_ref) 
        self.timer_match.refresh()          
        self.num_matched_kps = idxs_cur.shape[0]    
        print("# keypoints matched: %d " % self.num_matched_kps)  
        if kUseEssentialMatrixFitting: 
            # estimate camera orientation and inlier matches by fitting and essential matrix (see the limitations above)             
            idxs_ref, idxs_cur = self.estimate_pose_by_fitting_ess_mat(f_ref, f_cur, idxs_ref, idxs_cur)      
        
        if kUseDynamicDesDistanceTh: 
            self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stat(f_ref, f_cur, idxs_ref, idxs_cur)        
                               
        # propagate map point matches from kf_ref to f_cur  (do not override idxs_ref, idxs_cur)
        num_found_map_pts_inter_frame, idx_ref_prop, idx_cur_prop = propagate_map_point_matches(f_ref, f_cur, idxs_ref, idxs_cur, 
                                                                                                max_descriptor_distance=self.descriptor_distance_sigma) 
        print("# matched map points in prev frame: %d " % num_found_map_pts_inter_frame)      
                
        if kDebugDrawMatches and True: 
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
        num_matched_points = f_cur.clean_outlier_map_points()   
        print('      # num_matched_map_points: %d' % (self.num_matched_map_points) ) 
        #print('     # matched points: %d' % (num_matched_points) )               
        if not self.pose_is_ok or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame:
            f_cur.remove_frame_views(idxs_cur)
            f_cur.reset_points()               
            Printer.red('failure in tracking reference %d, # matched map points: %d' %(f_ref.id,self.num_matched_map_points))  
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
        self.f_cur.kf_ref = self.kf_ref  
        
    # track camera motion of f_cur w.r.t. the built local map  
    # find matches between {local map points} (points in the built local map) and {unmatched keypoints of f_cur}   
    def track_local_map(self, f_cur): 
        if self.map.local_map.is_empty():
            return 
        print('>>>> tracking local map...')
        self.timer_seach_map.start()
                
        self.update_local_map()        
        
        num_found_map_pts, reproj_err_frame_map_sigma, matched_points_frame_idxs = search_map_by_projection(self.local_points, f_cur,
                                    max_reproj_distance=self.reproj_err_frame_map_sigma, #Parameters.kMaxReprojectionDistanceMap, 
                                    max_descriptor_distance=self.descriptor_distance_sigma,
                                    ratio_test=Parameters.kMatchRatioTestMap) # use the updated local map          
        self.timer_seach_map.refresh()
        #print('reproj_err_sigma: ', reproj_err_frame_map_sigma, ' used: ', self.reproj_err_frame_map_sigma)        
        print("# new matched map points in local map: %d " % num_found_map_pts)                   
        print("# local map points ", self.map.local_map.num_points())         
        
        if kDebugDrawMatches and True: 
            img_matched_trails = f_cur.draw_feature_trails(f_cur.img.copy(), matched_points_frame_idxs, trail_max_length=3) 
            cv2.imshow('tracking local map - matched trails', img_matched_trails)
            cv2.waitKey(1)          
             
        # f_cur pose optimization 2 with all the matched local map points 
        self.pose_optimization(f_cur,'proj-map-frame')    
        f_cur.update_map_points_statistics()  # here we do not reset outliers; we let them reach the keyframe generation 
                                              # and then bundle adjustment will possible decide if remove them or not;
                                              # only after keyframe generation the outliers are cleaned!
        print('     # num_matched_map_points: %d' % (self.num_matched_map_points) )
        if not self.pose_is_ok or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackLocalMap:
            Printer.red('failure in tracking local map, # matched map points: ', self.num_matched_map_points) 
            self.pose_is_ok = False                                        
        
        #if kUseDynamicDesDistanceTh: 
        #    self.reproj_err_frame_map_sigma = self.dyn_config.update_reproj_err_map_stat(reproj_err_frame_map_sigma)                         


    # store frame history in order to retrieve the complete camera trajectory 
    def update_tracking_history(self):
        if self.state==SlamState.OK:        
            isometry3d_Tcr = self.f_cur.isometry3d * self.f_cur.kf_ref.isometry3d.inverse() # pose of current frame w.r.t. current reference keyframe kf_ref 
            self.tracking_history.relative_frame_poses.append(isometry3d_Tcr)
            self.tracking_history.kf_references.append(self.kf_ref)     
            self.tracking_history.timestamps.append(self.f_cur.timestamp)               
        else: 
            self.tracking_history.relative_frame_poses.append(self.tracking_history.relative_frame_poses[-1])
            self.tracking_history.kf_references.append(self.tracking_history.kf_references[-1])   
            self.tracking_history.timestamps.append(self.tracking_history.timestamps[-1])                                              
        self.tracking_history.slam_states.append(self.state)                
          

      
    def need_new_keyframe(self, f_cur):
        num_keyframes = self.map.num_keyframes()
        nMinObs = kNumMinObsForKeyFrameDefault
        if num_keyframes <= 2:
            nMinObs = 2  # if just two keyframes then we can have just two observations 
        num_kf_ref_tracked_points = self.kf_ref.num_tracked_points(nMinObs)  # number of tracked points in k_ref
        num_f_cur_tracked_points = f_cur.num_matched_inlier_map_points()     # number of inliers in f_cur 
        Printer.purple('F(%d) #points: %d, KF(%d) #points: %d ' %(f_cur.id, num_f_cur_tracked_points, self.kf_ref.id, num_kf_ref_tracked_points))
        
        if kLogKFinfoToFile:
            self.kf_info_logger.info('F(%d) #points: %d, KF(%d) #points: %d ' %(f_cur.id, num_f_cur_tracked_points, self.kf_ref.id, num_kf_ref_tracked_points))
        
        self.num_kf_ref_tracked_points = num_kf_ref_tracked_points

        is_local_mapping_idle = self.local_mapping.is_idle()  
        local_mapping_queue_size = self.local_mapping.queue_size()        
        print('is_local_mapping_idle: ', is_local_mapping_idle,', local_mapping_queue_size: ', local_mapping_queue_size)                                    
                                                        
        # condition 1: more than "max_frames_between_kfs" have passed from last keyframe insertion                                        
        cond1 = f_cur.id >= (self.kf_last.id + self.max_frames_between_kfs) 
        
        # condition 2: more than "min_frames_between_kfs" have passed and local mapping is idle
        cond2 = (f_cur.id >= (self.kf_last.id + self.min_frames_between_kfs)) & is_local_mapping_idle          
        #cond2 = (f_cur.id >= (self.kf_last.id + self.min_frames_between_kfs)) 
                  
        # condition 3: few tracked features compared to reference keyframe 
        cond3 = (num_f_cur_tracked_points < num_kf_ref_tracked_points * Parameters.kThNewKfRefRatio) and (num_f_cur_tracked_points > Parameters.kNumMinPointsForNewKf)
        
        #print('KF conditions: %d %d %d' % (cond1, cond2, cond3) )
        ret = (cond1 or cond2 ) and cond3    
                                                
        if ret:
            if is_local_mapping_idle:
                return True 
            else: 
                self.local_mapping.interrupt_optimization()
                if True: 
                    if local_mapping_queue_size <= 3:
                        return True
                    else:                  
                        return False    
                else: 
                    return False 
        else: 
            return False 


    # @ main track method @
    def track(self, img, frame_id, timestamp=None):
        Printer.cyan('@tracking')
        time_start = time.time()
                
        # check image size is coherent with camera params 
        print("img.shape: ", img.shape)
        print("camera ", self.camera.height," x ", self.camera.width)
        assert img.shape[0:2] == (self.camera.height, self.camera.width)   
        if timestamp is not None: 
            print('timestamp: ', timestamp)  
        
        self.timer_main_track.start()

        # build current frame 
        self.timer_frame.start()        
        f_cur = Frame(img, self.camera, timestamp=timestamp) 
        self.f_cur = f_cur 
        print("frame: ", f_cur.id)        
        self.timer_frame.refresh()   
        
        # reset indexes of matches 
        self.idxs_ref = [] 
        self.idxs_cur = []           
        
        if self.state == SlamState.NO_IMAGES_YET: 
            # push first frame in the inizializer 
            self.intializer.init(f_cur) 
            self.state = SlamState.NOT_INITIALIZED
            return # EXIT (jump to second frame)
        
        if self.state == SlamState.NOT_INITIALIZED:
            # try to inizialize 
            initializer_output, intializer_is_ok = self.intializer.initialize(f_cur, img)
            if intializer_is_ok:
                kf_ref = initializer_output.kf_ref
                kf_cur = initializer_output.kf_cur        
                # add the two initialized frames in the map 
                self.map.add_frame(kf_ref) # add first frame in map and update its frame id
                self.map.add_frame(kf_cur) # add second frame in map and update its frame id  
                # add the two initialized frames as keyframes in the map               
                self.map.add_keyframe(kf_ref) # add first keyframe in map and update its kid
                self.map.add_keyframe(kf_cur) # add second keyframe in map and update its kid                        
                kf_ref.init_observations()
                kf_cur.init_observations()
                # add points in map 
                new_pts_count,_,_ = self.map.add_points(initializer_output.pts, None, kf_cur, kf_ref, initializer_output.idxs_cur, initializer_output.idxs_ref, img, do_check=False)
                Printer.green("map: initialized %d new points" % (new_pts_count))                 
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
                    self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stat(kf_ref, kf_cur, initializer_output.idxs_ref, initializer_output.idxs_cur)                     
            return # EXIT (jump to next frame)
        
        # get previous frame in map as reference        
        f_ref   = self.map.get_frame(-1) 
        #f_ref_2 = self.map.get_frame(-2)
        self.f_ref = f_ref 
        
        # add current frame f_cur to map                  
        self.map.add_frame(f_cur)          
        self.f_cur.kf_ref = self.kf_ref  
        
        # reset pose state flag 
        self.pose_is_ok = False 
                            
        with self.map.update_lock:
            # check for map point replacements in previous frame f_ref (some points might have been replaced by local mapping during point fusion)
            self.f_ref.check_replaced_map_points()
                                                  
            if kUseDynamicDesDistanceTh:         
                print('descriptor_distance_sigma: ', self.descriptor_distance_sigma)
                self.local_mapping.descriptor_distance_sigma = self.descriptor_distance_sigma
                    
            # udpdate (velocity) old motion model                                             # c1=ref_ref, c2=ref, c3=cur;  c=cur, r=ref
            #self.velocity = np.dot(f_ref.pose, inv_T(f_ref_2.pose))                          # Tc2c1 = Tc2w * Twc1   (predicted Tcr)
            #self.predicted_pose = g2o.Isometry3d(np.dot(self.velocity, f_ref.pose))          # Tc3w = Tc2c1 * Tc2w   (predicted Tcw)        
                                                    
            # set intial guess for current pose optimization                         
            if kUseMotionModel and self.motion_model.is_ok:
                print('using motion model for next pose prediction')                   
                # update f_ref pose according to its reference keyframe (the pose of the reference keyframe could be updated by local mapping)
                self.f_ref.update_pose(self.tracking_history.relative_frame_poses[-1] * self.f_ref.kf_ref.isometry3d)                                  
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
                                            
            # now, having a better estimate of f_cur pose, we can find more map point matches: 
            # find matches between {local map points} (points in the local map) and {unmatched keypoints of f_cur}
            if self.pose_is_ok: 
                self.track_local_map(f_cur)
                
        # end block {with self.map.update_lock:}
        
        # TODO: add relocalization 

        # HACK: since local mapping is not fast enough in python (and tracking is not in real-time) => give local mapping more time to process stuff  
        self.wait_for_local_mapping()  # N.B.: this must be outside the `with self.map.update_lock:` block

        with self.map.update_lock:
            
            # update slam state 
            if self.pose_is_ok:
                self.state=SlamState.OK          
            else:                
                self.state=SlamState.LOST
                Printer.red('tracking failure')
                
            # update motion model state     
            self.motion_model.is_ok = self.pose_is_ok                    
                                
            if self.pose_is_ok:   # if tracking was successful
                
                # update motion model                     
                self.motion_model.update_pose(timestamp, f_cur.position, f_cur.quaternion)  
                                                                    
                f_cur.clean_vo_map_points()
                        
                # do we need a new KeyFrame? 
                need_new_kf = self.need_new_keyframe(f_cur)
                                    
                if need_new_kf: 
                    Printer.green('adding new KF with frame id % d: ' %(f_cur.id))
                    if kLogKFinfoToFile:
                        self.kf_info_logger.info('adding new KF with frame id % d: ' %(f_cur.id))                
                    kf_new = KeyFrame(f_cur, img)                                     
                    self.kf_last = kf_new  
                    self.kf_ref = kf_new 
                    f_cur.kf_ref = kf_new                  
                    
                    self.local_mapping.push_keyframe(kf_new) 
                    if not kLocalMappingOnSeparateThread:
                        self.local_mapping.do_local_mapping()                                      
                else: 
                    Printer.yellow('NOT KF')      
                    
                # From ORBSLAM2: 
                # Clean outliers once keyframe generation has been managed:
                # we allow points with high innovation (considered outliers by the Huber Function)
                # pass to the new keyframe, so that bundle adjustment will finally decide
                # if they are outliers or not. We don't want next frame to estimate its position
                # with those points so we discard them in the frame.                
                f_cur.clean_outlier_map_points()                    
                                
            if self.f_cur.kf_ref is None:
                self.f_cur.kf_ref = self.kf_ref  
                                    
            self.update_tracking_history()    # must stay after having updated slam state (self.state)                                                                  
                    
            Printer.green("map: %d points, %d keyframes" % (self.map.num_points(), self.map.num_keyframes()))
            #self.update_history()
            
            self.timer_main_track.refresh()
            
            duration = time.time() - time_start
            print('tracking duration: ', duration)            
        
        
    # Since we do not have real-time performances, we can slow-down things and make tracking wait till local mapping gets idle  
    # N.B.: this function must be called outside 'with self.map.update_lock' blocks, 
    #       since both self.track() and the local-mapping optimization use the RLock 'map.update_lock'    
    #       => they cannot wait for each other once map.update_lock is locked (deadlock)                        
    def wait_for_local_mapping(self):                   
        if kTrackingWaitForLocalMappingToGetIdle:                        
            #while not self.local_mapping.is_idle() or self.local_mapping.queue_size()>0:       
            if not self.local_mapping.is_idle():       
                print('>>>> waiting for local mapping...')                         
                self.local_mapping.wait_idle()
        else:
            if not self.local_mapping.is_idle() and kTrackingWaitForLocalMappingSleepTime>0:
                print('>>>> sleeping for local mapping...')                    
                time.sleep(kTrackingWaitForLocalMappingSleepTime)  
        # check again for debug                     
        #is_local_mapping_idle = self.local_mapping.is_idle()  
        #local_mapping_queue_size = self.local_mapping.queue_size()        
        #print('is_local_mapping_idle: ', is_local_mapping_idle,', local_mapping_queue_size: ', local_mapping_queue_size)             


    # def update_history(self):
    #     f_cur = self.map.get_frame(-1)
    #     self.cur_R = f_cur.pose[:3,:3].T
    #     self.cur_t = np.dot(-self.cur_R,f_cur.pose[:3,3])
    #     if (self.init_history is True) and (self.trueX is not None):
    #         self.t0_est = np.array([self.cur_t[0], self.cur_t[1], self.cur_t[2]])  # starting translation 
    #         self.t0_gt  = np.array([self.trueX, self.trueY, self.trueZ])           # starting translation 
    #     if (self.t0_est is not None) and (self.t0_gt is not None):             
    #         p = [self.cur_t[0]-self.t0_est[0], self.cur_t[1]-self.t0_est[1], self.cur_t[2]-self.t0_est[2]]   # the estimated traj starts at 0
    #         self.traj3d_est.append(p)
    #         self.traj3d_gt.append([self.trueX-self.t0_gt[0], self.trueY-self.t0_gt[1], self.trueZ-self.t0_gt[2]])            
    #         self.poses.append(poseRt(self.cur_R, p))    


    # get current translation scale from ground-truth if this is set 
    # def get_absolute_scale(self, frame_id):  
    #     if self.groundtruth is not None and kUseGroundTruthScale:
    #         self.trueX, self.trueY, self.trueZ, scale = self.groundtruth.getPoseAndAbsoluteScale(frame_id)
    #         return scale
    #     else:
    #         self.trueX = 0 
    #         self.trueY = 0 
    #         self.trueZ = 0
    #         return 1

