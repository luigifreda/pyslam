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
import cv2
from enum import Enum

from frame import Frame, match_frames
from keyframe import KeyFrame

from collections import deque

from map_point import MapPoint
from map import Map
from utils_geom import triangulate_points, triangulate_normalized_points, add_ones, poseRt, inv_T
from camera  import Camera, PinholeCamera
from utils_sys import Printer
from utils_features import ImageGrid
from parameters import Parameters  
from dataset import SensorType

from utils_draw import draw_feature_matches

kVerbose=True     
kRansacThresholdNormalized = 0.0004  # metric threshold used for normalized image coordinates 
kRansacProb = 0.999

kMinIdDistBetweenIntializingFrames = 2
kMaxIdDistBetweenIntializingFrames = 6   # N.B.: worse performances with values smaller than 5!

kShowFeatureMatches = False # show the feature matches during initialization

kFeatureMatchRatioTestInitializer = Parameters.kFeatureMatchRatioTestInitializer

kNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved = 10

kMaxLenFrameDeque = 20


class InitializerOutput(object):
    def __init__(self):    
        self.pts = None    # 3d points [Nx3]
        self.kf_cur = None 
        self.kf_ref = None 
        self.idxs_cur = None 
        self.idxs_ref = None 


class Initializer(object):
    def __init__(self, sensor_type=SensorType.MONOCULAR):
        self.sensor_type = sensor_type
        self.mask_match = None
        self.mask_recover = None 
        self.frames = deque(maxlen=kMaxLenFrameDeque)  # deque with max length, it is thread-safe      
        #self.idx_f_ref = 0   # index of the reference frame in self.frames buffer  
        self.f_ref = None        
                
        self.num_min_features = Parameters.kInitializerNumMinFeatures if self.sensor_type == SensorType.MONOCULAR else Parameters.kInitializerNumMinFeaturesStereo
        self.num_min_triangulated_points = Parameters.kInitializerNumMinTriangulatedPoints       
        self.num_failures = 0 
        
        self.check_frame_distance = True
        
        if kShowFeatureMatches: 
            Frame.is_store_imgs = True         
        
    def reset(self):
        self.frames.clear()
        self.f_ref = None  

    # fit essential matrix E with RANSAC such that:  p2.T * E * p1 = 0  where  E = [t21]x * R21
    # out: Trc  homogeneous transformation matrix with respect to 'ref' frame,  pr_= Trc * pc_
    # N.B.1: trc is estimated up to scale (i.e. the algorithm always returns ||trc||=1, we need a scale in order to recover a translation which is coherent with previous estimated poses)
    # N.B.2: this function has problems in the following cases: [see Hartley/Zisserman Book]
    # - 'geometrical degenerate correspondences', e.g. all the observed features lie on a plane (the correct model for the correspondences is an homography) or lie a ruled quadric 
    # - degenerate motions such a pure rotation (a sufficient parallax is required) or anum_edges viewpoint change (where the translation is almost zero)
    # N.B.3: the five-point algorithm (used for estimating the Essential Matrix) seems to work well in the degenerate planar cases [Five-Point Motion Estimation Made Easy, Hartley]
    # N.B.4: as reported above, in case of pure rotation, this algorithm will compute a useless fundamental matrix which cannot be decomposed to return the rotation     
    # N.B.5: the OpenCV findEssentialMat function uses the five-point algorithm solver by D. Nister => hence it should work well in the degenerate planar cases
    def estimatePose(self, kpn_ref, kpn_cur):	     
        ransac_method = None
        try: 
            ransac_method = cv2.USAC_MSAC 
        except: 
            ransac_method = cv2.RANSAC  
        # here, the essential matrix algorithm uses the five-point algorithm solver by D. Nister (see the notes and paper above )               
        E, self.mask_match = cv2.findEssentialMat(kpn_cur, kpn_ref, focal=1, pp=(0., 0.), method=ransac_method, prob=kRansacProb, threshold=kRansacThresholdNormalized)                         
        _, R, t, mask = cv2.recoverPose(E, kpn_cur, kpn_ref, focal=1, pp=(0., 0.))                                                     
        return poseRt(R,t.T)  # Trc  homogeneous transformation matrix with respect to 'ref' frame,  pr_= Trc * pc_        

    # push the first image
    def init(self, f_cur : Frame, img_cur):
        self.frames.append(f_cur)    
        self.f_ref = f_cur
                  
    # actually initialize having two available images 
    def initialize(self, f_cur: Frame, img_cur):

        if self.num_failures > kNumOfFailuresAfterWichNumMinTriangulatedPointsIsHalved: 
            self.num_min_triangulated_points = 2.0/3 * Parameters.kInitializerNumMinTriangulatedPoints
            #self.num_failures = 0
            #self.check_frame_distance = False
            Printer.orange('Initializer: reduced min num triangulated features to ', self.num_min_triangulated_points)            

        # prepare the output 
        out = InitializerOutput()
        is_ok = False 

        #print('num frames: ', len(self.frames))
        
        # if too many frames have passed, move the current idx_f_ref forward 
        # this is just one very simple policy that can be used 
        if self.f_ref is not None: 
            if f_cur.id - self.f_ref.id > kMaxIdDistBetweenIntializingFrames: 
                self.f_ref = self.frames[-1]  # take last frame in the buffer
                #self.idx_f_ref = len(self.frames)-1  # take last frame in the buffer
                #self.idx_f_ref = self.frames.index(self.f_ref)  # since we are using a deque, the code of the previous commented line is not valid anymore 
                #print('*** idx_f_ref:',self.idx_f_ref)            
        #self.f_ref = self.frames[self.idx_f_ref] 
        f_ref = self.f_ref 
        #print('ref fid: ',self.f_ref.id,', curr fid: ', f_cur.id, ', idxs_ref: ', self.idxs_ref)
                
        # append current frame 
        self.frames.append(f_cur)

        if self.check_frame_distance and self.sensor_type == SensorType.MONOCULAR:
            if f_cur.id - f_ref.id < kMinIdDistBetweenIntializingFrames:
                Printer.red('Inializer: ko - init frames are too close!') 
                self.num_failures += 1
                return out, is_ok

        # if the current frames do no have enough features exit 
        if len(f_ref.kps) < self.num_min_features or len(f_cur.kps) < self.num_min_features:
            Printer.red('Inializer: ko - not enough features!') 
            self.num_failures += 1
            return out, is_ok

        # find keypoint matches
        matching_result = match_frames(f_cur, f_ref, kFeatureMatchRatioTestInitializer)      
        idxs_cur, idxs_ref = np.asarray(matching_result.idxs1), np.asarray(matching_result.idxs2)      
    
        if kShowFeatureMatches: # debug frame matching
            frame_img_matches = draw_feature_matches(img_cur, self.f_ref.img, f_cur.kps[idxs_cur], self.f_ref.kps[idxs_ref], horizontal=False)
            #cv2.namedWindow('stereo_img_matches', cv2.WINDOW_NORMAL)
            cv2.imshow('initializer frame matches', frame_img_matches)
            cv2.waitKey(1)   

        print('|------------')        
        #print('deque ids: ', [f.id for f in self.frames])
        print('initializing frames ', f_cur.id, ', ', f_ref.id)
        print("# keypoint matches: ", len(idxs_cur))  
                
        Trc = self.estimatePose(f_ref.kpsn[idxs_ref], f_cur.kpsn[idxs_cur])
        Tcr = inv_T(Trc)  # Tcr w.r.t. ref frame 
        f_ref.update_pose(np.eye(4))        
        f_cur.update_pose(Tcr)

        # remove outliers from keypoint matches by using the mask computed with inter frame pose estimation        
        mask_idxs = (self.mask_match.ravel() == 1)
        self.num_inliers = sum(mask_idxs)
        print('# keypoint inliers: ', self.num_inliers )
        idx_ref_inliers = idxs_ref[mask_idxs]        
        idx_cur_inliers = idxs_cur[mask_idxs]

        # create a temp map for initializing 
        map = Map()
        f_ref.reset_points()
        f_cur.reset_points()
        
        #map.add_frame(f_ref)        
        #map.add_frame(f_cur)  
        
        kf_ref = KeyFrame(f_ref)
        kf_cur = KeyFrame(f_cur, img_cur)        
        map.add_keyframe(kf_ref)        
        map.add_keyframe(kf_cur)      
        
        pts3d = None 
        mask_pts3d = None
        do_check = True
        
        if self.sensor_type == SensorType.RGBD or self.sensor_type == SensorType.STEREO:                    
            # unproject 3D points by using available depths 
            pts3d, mask_pts3d = kf_ref.unproject_points_3d(idx_ref_inliers)
            
            num_valid_pts3d = np.sum(mask_pts3d) 
            if num_valid_pts3d < Parameters.kInitializerNumMinNumPointsForPnPWithDepth:
                return out, is_ok
            
            # solve pnp to get a pose of the current frame w.r.t. the reference frame (the scale is defined by the depths of the reference frame)
            mask_pts3d = mask_pts3d.flatten()
            print(f'init PnP - #pts3d: {len(pts3d)}, #mask_pts3d: {sum(mask_pts3d)}')
            success, rvec_rc, tvec_rc = cv2.solvePnP(pts3d[mask_pts3d], kf_cur.kps[idx_cur_inliers[mask_pts3d]], kf_cur.camera.K, kf_cur.camera.D, flags=cv2.SOLVEPNP_EPNP)
            if success:
                rot_matrix_rc, _ = cv2.Rodrigues(rvec_rc)
                Trc = poseRt(rot_matrix_rc, tvec_rc.flatten())
            else: 
                Trc = np.eye(4)
            print(f'init PnP - success: {success}, Trc: {Trc}')                
            kf_cur.update_pose(Trc)
            f_cur.update_pose(Trc)
                
            do_check = False # in this case, we do not check reprojection errors and other related stuff
        else: 
            pts3d, mask_pts3d = triangulate_normalized_points(kf_cur.Tcw, kf_ref.Tcw, kf_cur.kpsn[idx_cur_inliers], kf_ref.kpsn[idx_ref_inliers])

        new_pts_count, mask_points, added_map_points = map.add_points(pts3d, mask_pts3d, kf_cur, kf_ref, idx_cur_inliers, idx_ref_inliers, img_cur, do_check=do_check, cos_max_parallax=Parameters.kCosMaxParallaxInitializer)
        print("# triangulated points: ", new_pts_count)   
                        
        if new_pts_count > self.num_min_triangulated_points:   
            
            reproj_error_before = map.compute_mean_reproj_error()
            print(f'reprojection error before: {reproj_error_before}')
                               
            reproj_error_after = map.optimize(verbose=False, rounds=20, use_robust_kernel=True)
            print(f'init optimization error^2: {reproj_error_after}')        
            
            num_map_points = len(map.points)
            print("# map points:   %d" % num_map_points)   
                            
            is_ok = reproj_error_after < reproj_error_before and  num_map_points > self.num_min_triangulated_points        

            if is_ok:
                # after optimization
                out_pts3d = [p.pt for p in added_map_points] # get the optimized points
                out_pts3d = np.array(out_pts3d).reshape(-1,3)
                
                #reproj_error_after = map.compute_mean_reproj_error(added_map_points)
                #print(f'reprojection error after2: {reproj_error_after}')  
                            
                out.pts = out_pts3d 
                out.kf_cur = kf_cur
                out.idxs_cur = idx_cur_inliers[mask_points]        
                out.kf_ref = kf_ref 
                out.idxs_ref = idx_ref_inliers[mask_points]

            if is_ok and self.sensor_type == SensorType.MONOCULAR: 
                median_depth = kf_cur.compute_points_median_depth(out_pts3d)
                baseline = np.linalg.norm(kf_cur.Ow - kf_ref.Ow)               
                is_ok = median_depth > 0 and baseline > 0
                
            if is_ok and self.sensor_type == SensorType.MONOCULAR:    
                ratioDepthBaseline = median_depth/baseline                       
                print(f'ratioDepthBaseline: {ratioDepthBaseline}, median_depth: {median_depth}, baseline: {baseline}') 
                if ratioDepthBaseline > Parameters.kInitializerMinRatioDepthBaseline:
                    is_ok = False     
                
            if is_ok and self.sensor_type == SensorType.MONOCULAR:             
                # set scene median depth to equal desired_median_depth'                
                desired_median_depth = Parameters.kInitializerDesiredMedianDepth
                depth_scale = desired_median_depth/median_depth 
                print(f'forcing current median depth {median_depth} to {desired_median_depth}, depth_scale: {depth_scale}') 
                out.pts[:,:3] = out.pts[:,:3] * depth_scale  # scale points 
                tcw = kf_cur.tcw * depth_scale  # scale initial baseline    
                kf_cur.update_translation(tcw)                
            
            # if is_ok:
            #     image_grid = ImageGrid(kf_cur.camera.width, kf_cur.camera.height, num_div_x=3, num_div_y=2)
            #     image_grid.add_points(kf_cur.kps[out.idxs_cur])
            #     # num_uncovered_cells = image_grid.num_cells_uncovered(num_min_points=1)      
            #     # print(f'# uncovered cells: {num_uncovered_cells}')          
            #     # #is_ok = image_grid.is_each_cell_covered(num_min_points=1)     
            #     is_ok = image_grid.is_each_cell_covered(num_min_points=1)   
            #     if True:
            #         cv2.namedWindow('grid_img', cv2.WINDOW_NORMAL)
            #         cv2.imshow('grid_img', image_grid.get_grid_img())
            #         cv2.waitKey(1)
            
        map.delete()
  
        if is_ok:
            Printer.green('Inializer: ok!')    
        else:
            self.num_failures += 1            
            Printer.red('Inializer: ko!')                         
        print('|------------')               
        return out, is_ok
