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

from collections import defaultdict 

import os
import numpy as np

from frame import Frame, FeatureTrackerShared, compute_frame_matches, prepare_input_data_for_pnpsolver
from rotation_histogram import filter_matches_with_histogram_orientation
import optimizer_gtsam
import optimizer_g2o
from utils_sys import Printer, Logging
from loop_detector_base import LoopDetectorOutput
from search_points import search_frame_by_projection

from timer import TimerFps
from config_parameters import Parameters

import logging

import traceback
import pnpsolver


kVerbose = True
kTimerVerbose = False # set this to True if you want to print timings 
kPrintTrackebackDetails = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + '/..'

    
pose_optimization = optimizer_gtsam.pose_optimization if Parameters.kOptimizationFrontEndUseGtsam else optimizer_g2o.pose_optimization


# Relocalizer working on loop detection output 
class Relocalizer: 
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op    
    
    def __init__(self):
        self.timer = TimerFps('Relocalizer', is_verbose = kTimerVerbose)
        self.init_print()
        
    def init_print(self):
        if kVerbose:
            if Parameters.kRelocalizationDebugAndPrintToFile:
                # redirect the prints of local mapping to the file logs/relocalization.log 
                # you can watch the output in separate shell by running:
                # $ tail -f logs/relocalization.log 
                
                logging_file = Parameters.kLogsFolder + '/relocalization.log'
                Relocalizer.local_logger = Logging.setup_file_logger('relocalization_logger', logging_file, formatter=Logging.simple_log_formatter)
                def print_file(*args, **kwargs):
                    message = ' '.join(str(arg) for arg in args)  # Convert all arguments to strings and join with spaces                
                    return Relocalizer.local_logger.info(message, **kwargs)
            else:
                def print_file(*args, **kwargs):
                    message = ' '.join(str(arg) for arg in args)  # Convert all arguments to strings and join with spaces                
                    return print(message, **kwargs)
            Relocalizer.print = staticmethod(print_file)  


    def relocalize(self, frame: Frame, detection_output: LoopDetectorOutput, keyframes_map: dict): 
        try:        
            if detection_output is None or len(detection_output.candidate_idxs) == 0:
                msg = 'None output' if detection_output is None else 'No candidates'
                Relocalizer.print(f'Relocalizer: {msg} with frame {frame.id}')
                return False 
            
            Relocalizer.print(f'Relocalizer: Detected candidates: {frame.id} with {detection_output.candidate_idxs}')
            reloc_candidate_kfs = [keyframes_map[idx] for idx in detection_output.candidate_idxs if idx in keyframes_map] # get back the keyframes from their ids
            
            kp_match_idxs = defaultdict(lambda: (None,None))   # dictionary of keypointmatches  (kf_i, kf_j) -> (idxs_i,idxs_j)              

            self.timer.start()
            kp_match_idxs = compute_frame_matches(frame, reloc_candidate_kfs, kp_match_idxs, 
                                                  do_parallel=Parameters.kRelocalizationParallelKpsMatching, \
                                                  max_workers=Parameters.kRelocalizationParallelKpsMatchingNumWorkers, \
                                                  ratio_test=Parameters.kRelocalizationFeatureMatchRatioTest, \
                                                  print_fun=print)
                                
            solvers = []
            solvers_input = []
            considered_candidates = []
            mp_match_idxs = defaultdict(lambda: (None,None))   # dictionary of map point matches  (kf_i, kf_j) -> (idxs_i,idxs_j)            
            for i,kf in enumerate(reloc_candidate_kfs):
                if kf.id == frame.id or kf.is_bad:
                    continue 
                
                # extract matches from precomputed map  
                idxs_frame, idxs_kf = kp_match_idxs[(frame,kf)]        
                assert(len(idxs_frame)==len(idxs_kf))
                
                # if features have descriptors with orientation then let's check the matches with a rotation histogram
                if FeatureTrackerShared.oriented_features:
                    #num_matches_before = len(idxs_frame)
                    valid_match_idxs = filter_matches_with_histogram_orientation(idxs_frame, idxs_kf, frame, kf)
                    if len(valid_match_idxs)>0:
                        idxs_frame = idxs_frame[valid_match_idxs]
                        idxs_kf = idxs_kf[valid_match_idxs]       
                    #print(f'Relocalizer: rotation histogram filter: #matches ({frame.id},{kf.id}): before {num_matches_before}, after {len(idxs_frame)}')             
                
                num_matches = len(idxs_frame)
                Relocalizer.print(f'Relocalizer: num_matches ({frame.id},{kf.id}): {num_matches}')
                
                if(num_matches<Parameters.kRelocalizationMinKpsMatches):
                    Relocalizer.print(f'Relocalizer: skipping keyframe {kf.id} with too few matches ({num_matches}) (min: {Parameters.kRelocalizationMinKpsMatches})')
                    continue 
                
                points_3d_w, points_2d, sigmas2, idxs1, idxs2 = \
                    prepare_input_data_for_pnpsolver(frame, kf, idxs_frame, idxs_kf, print=print)
                
                
                # fill the dictionary of map point matches (its content needs to be cleaned up later with found inliers)
                mp_match_idxs[(frame,kf)] = (idxs1, idxs2)
                                
                solver_input_data = pnpsolver.PnPsolverInput()
                solver_input_data.points_2d = points_2d
                solver_input_data.points_3d = points_3d_w
                solver_input_data.sigmas2 = sigmas2
                solver_input_data.fx = frame.camera.fx
                solver_input_data.fy = frame.camera.fy
                solver_input_data.cx = frame.camera.cx
                solver_input_data.cy = frame.camera.cy
                
                num_correspondences = len(points_2d)
                if num_correspondences < 4:
                    Relocalizer.print(f'Relocalizer: skipping keyframe {kf.id} with too few correspondences ({num_correspondences}) (min: 4)')
                    continue                
                
                #print(f'Relocalizer: initializing MLPnPsolver for keyframe {kf.id}, num correspondences: {num_correspondences}')                
                solver = pnpsolver.MLPnPsolver(solver_input_data)
                solver.set_ransac_parameters(0.99,10,300,6,0.5,5.991)
                
                solvers.append(solver)
                solvers_input.append(solver_input_data)
                considered_candidates.append(kf)                
                
            # check if candidates get a valid solution
            success_relocalization_kf = None
            for i, kf in enumerate(considered_candidates):
                # perform 5 ransac iterations on each solver
                Relocalizer.print(f'Relocalizer: performing MLPnPsolver iterations for keyframe {kf.id}')                      
                ok, Tcw, is_no_more, inlier_flags, num_inliers = solvers[i].iterate(5)
                if not ok or is_no_more:
                    continue                 
                inlier_flags = np.array(inlier_flags,dtype=bool)  # from from int8 to bool

                # we got a valid pose solution => let's optimize it
                frame.update_pose(Tcw)
                 
                idxs_frame, idxs_kf = mp_match_idxs[(frame,kf)]
                for j, idx in enumerate(idxs_frame):
                    if inlier_flags[j]:
                        frame.points[idx] = kf.points[idxs_kf[j]]
                    else: 
                        frame.points[idx] = None
                idxs_kf_inliers = idxs_kf[inlier_flags]
                        
                pose_before=frame.pose.copy()        
                mean_pose_opt_chi2_error, pose_is_ok, num_matched_map_points = pose_optimization(frame, verbose=False)
                Relocalizer.print(f'Relocalizer: pos opt1: error^2: {mean_pose_opt_chi2_error},  ok: {pose_is_ok}, #inliers: {num_matched_map_points}') 
                
                if not pose_is_ok: 
                    # if current pose optimization failed, reset f_cur pose             
                    frame.update_pose(pose_before)
                    continue
                
                if num_matched_map_points < Parameters.kRelocalizationPoseOpt1MinMatches:
                    continue
                
                for i in range(len(frame.points)):
                    if frame.outliers[i]:
                        frame.points[i] = None
            
                # if few inliers, search by projection in a coarse window and optimize again
                if num_matched_map_points < Parameters.kRelocalizationDoPoseOpt2NumInliers:
                    idxs_kf, idxs_frame, num_new_found_map_points = search_frame_by_projection(kf, frame,
                                                                    max_reproj_distance=Parameters.kRelocalizationMaxReprojectionDistanceMapSearchCoarse,
                                                                    max_descriptor_distance=Parameters.kMaxDescriptorDistance,
                                                                    ratio_test = Parameters.kRelocalizationFeatureMatchRatioTestLarge,
                                                                    is_monocular=True, # must be True
                                                                    already_matched_ref_idxs=idxs_kf_inliers)
                    
                    if num_matched_map_points + num_new_found_map_points >= Parameters.kRelocalizationDoPoseOpt2NumInliers:
                        pose_before=frame.pose.copy()        
                        mean_pose_opt_chi2_error, pose_is_ok, num_matched_map_points = pose_optimization(frame, verbose=False)
                        Relocalizer.print(f'Relocalizer: pos opt2: error^2: {mean_pose_opt_chi2_error},  ok: {pose_is_ok}, #inliers: {num_matched_map_points}') 
                        
                        if not pose_is_ok: 
                            # if current pose optimization failed, reset f_cur pose             
                            frame.update_pose(pose_before)
                            continue    
                        
                        # if many inliers but still not enough, search by projection again in a narrower window
                        # the camera has been already optimized with many points
                        if num_matched_map_points>30 and num_matched_map_points<Parameters.kRelocalizationDoPoseOpt2NumInliers: 
                            matched_ref_idxs = np.flatnonzero(frame.points!=None)
                            
                            idxs_kf, idxs_frame, num_new_found_map_points = search_frame_by_projection(kf, frame,
                                                                            max_reproj_distance=Parameters.kRelocalizationMaxReprojectionDistanceMapSearchFine,
                                                                            max_descriptor_distance=0.7*Parameters.kMaxDescriptorDistance,
                                                                            ratio_test = Parameters.kRelocalizationFeatureMatchRatioTestLarge,
                                                                            is_monocular=True, # must be True
                                                                            already_matched_ref_idxs=matched_ref_idxs)
                                                
                            # final optimization 
                            if num_matched_map_points + num_new_found_map_points >= Parameters.kRelocalizationDoPoseOpt2NumInliers:
                                pose_before=frame.pose.copy()        
                                mean_pose_opt_chi2_error, pose_is_ok, num_matched_map_points = pose_optimization(frame, verbose=False)
                                Relocalizer.print(f'Relocalizer: pos opt3: error^2: {mean_pose_opt_chi2_error},  ok: {pose_is_ok}, #inliers: {num_matched_map_points}') 
                                
                                if not pose_is_ok: 
                                    # if current pose optimization failed, reset f_cur pose             
                                    frame.update_pose(pose_before)
                                    continue 
                                
                if num_matched_map_points >= Parameters.kRelocalizationDoPoseOpt2NumInliers:
                    success_relocalization_kf = kf
                    break
            
            res = False    
            if success_relocalization_kf is None:
                Relocalizer.print('Relocalizer: failed')
                res = False
            else:
                frame.kf_ref = success_relocalization_kf
                Relocalizer.print(f'Relocalizer: success: connected frame id: {frame.id} to keyframe id: {frame.kf_ref.id}')
                res = True                                                       
                
            self.timer.refresh()
            Relocalizer.print(f'Relocalizer: elapsed time: {self.timer.last_elapsed}')
            return res
                
        except Exception as e:           
            Relocalizer.print(f'Relocalizer: EXCEPTION: {e} !!!')   
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                Relocalizer.print(f'\t traceback details: {traceback_details}')
                
        return False              