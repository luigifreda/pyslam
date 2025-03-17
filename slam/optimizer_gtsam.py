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

import math 
import numpy as np
import sys
import time

import platform
import threading 
import multiprocessing as mp
import traceback

import gtsam
import gtsam_factors
from gtsam.symbol_shorthand import X, L
import numpy as np
import math

import g2o  

from utils_sys import Printer
from utils_geom import inv_T
from utils_mp import MultiprocessingManager

from frame import FeatureTrackerShared
from map_point import MapPoint
from keyframe import KeyFrame


# gtsam helper functions 
def vector6(x, y, z, a, b, c):
    """Create 6d double numpy array."""
    return np.array([x, y, z, a, b, c], dtype=float)
    
# constants (user variables) 
# prior_model = gtsam.noiseModel.Diagonal.Variances(vector6(1e-6, 1e-6, 1e-6, 1e-4, 1e-4, 1e-4))
# odom_model = gtsam.noiseModel.Diagonal.Variances(vector6(1e-2, 1e-2, 1e-2, 1e-2, 1e-2, 1e-2))
# loop_model = gtsam.noiseModel.Diagonal.Variances(vector6(0.5, 0.5, 0.5, 0.5, 0.5, 0.5))
# robust_loop_model = gtsam.noiseModel.Robust.Create(gtsam.noiseModel.mEstimator.Cauchy.Create(1), loop_model)
# NOTE: alternative robust kernels 
# gtsam.noiseModel.mEstimator.Cauchy(2.0)  # Good for extremely noisy data
# gtsam.noiseModel.mEstimator.Tukey(4.0)  # Even more robust but suppresses large errors heavily
# gtsam.noiseModel.mEstimator.GemanMcClure(1.0)  # Works well for heavy-tailed noise

# ------------------------------------------------------------------------------------------

# Continuously syncs the g2o.Flag with the shared boolean variable.
def sync_flag_fun(abort_flag, mp_abort_flag, print=print):
    print('sync_flag_fun: starting...')
    try:    
        while not abort_flag.value:
            # If the flag's value doesn't match the shared_bool, update it
            if mp_abort_flag.value != abort_flag.value:
                abort_flag.value = mp_abort_flag.value
                print(f'sync_flag_fun: Flag updated to: {abort_flag.value}')
            time.sleep(0.003)
        print('sync_flag_fun: done...')
    except Exception as e:
        print(f'sync_flag_fun: EXCEPTION: {e}')
        traceback_details = traceback.format_exc()
        print(f'\t traceback details: {traceback_details}')      


# ------------------------------------------------------------------------------------------

# optimize pixel reprojection error, bundle adjustment
def bundle_adjustment(keyframes, points, local_window, fixed_points=False, \
                      rounds=10, loop_kf_id=0, \
                      use_robust_kernel=False, \
                      abort_flag=None, mp_abort_flag=None, \
                      result_dict=None, \
                      verbose=False,
                      print=print):
    if local_window is None:
        local_frames = keyframes
    else:
        local_frames = keyframes[-local_window:]
        
    robust_rounds = rounds // 2 if use_robust_kernel else 0        
    final_rounds = rounds - robust_rounds
    print(f'bundle_adjustment: rounds: {rounds}, robust_rounds: {robust_rounds}, final_rounds: {final_rounds}')
    
    if abort_flag is None:
        abort_flag = g2o.Flag()
        
    sync_flag_thread = None
    if mp_abort_flag is not None: 
        # Create a thread for keeping abort_flag in sync with mp_abort_flag.
        # Why? The g2o-abort-flag passed (via pickling) to a launched parallel process (via multiprocessing module) is just a different instance that is not kept in sync 
        # with its source instance in the parent process. This means we don't succeed to abort the BA when set the source instance. 
        sync_flag_thread = threading.Thread(target=sync_flag_fun, args=(abort_flag, mp_abort_flag, print))
        sync_flag_thread.daemon = True  # Daemonize thread so it exits when the main thread does
        sync_flag_thread.start()    
    
    # Create GTSAM factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    
    sigma_for_fixed = 1e-9 # sigma used for fixed entities
        
    # Huber loss parameters for robust optimization
    th_huber_mono = np.sqrt(5.991)  # chi-square 2 DOFs
    th_huber_stereo = np.sqrt(7.815)  # chi-square 3 DOFs
        
    keyframe_keys = {}
    point_keys = {}
    graph_factors = {}        
    
    # Add keyframes as pose variables
    for kf in (local_frames if fixed_points else keyframes): # if points are fixed then consider just the local frames, otherwise we need all frames or at least two frames for each point
        if kf.is_bad:
            continue
        pose_key = gtsam.symbol('x', kf.kid)        
        keyframe_keys[kf] = pose_key
                
        pose = gtsam.Pose3(gtsam.Rot3(kf.Rwc), gtsam.Point3(*kf.Ow))
        initial_estimates.insert(pose_key, pose)
        
        if kf.kid == 0 or kf not in local_frames:
            graph.add(gtsam.PriorFactorPose3(pose_key, pose, gtsam.noiseModel.Isotropic.Sigma(6, sigma_for_fixed)))
    
    num_edges = 0
    num_bad_edges = 0   

    # Add points as graph vertices
    for p in points:
        if p is None or p.is_bad: # do not consider bad points   
            continue
        point_key = gtsam.symbol('p', p.id)
        point_keys[p] = point_key
        point_position = gtsam.Point3(p.pt[0:3])
        initial_estimates.insert(point_key, point_position)
        
        if fixed_points:
            graph.add(gtsam.PriorFactorPoint3(point_key, point_position, gtsam.noiseModel.Isotropic.Sigma(3, sigma_for_fixed)))
        
        # Add measurement factors
        for kf, idx in p.observations():
            # if kf.is_bad:  # redundant since we check kf is in graph_keyframes (selected as non-bad)
            #     continue             
            if kf not in keyframe_keys:
                continue
            
            pose_key = keyframe_keys[kf]
            
            is_stereo_obs = kf.kps_ur is not None and kf.kps_ur[idx] > 0
            #invSigma2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[kf.octaves[p_idx]]
            sigma = FeatureTrackerShared.feature_manager.level_sigmas[kf.octaves[idx]]
            
            robust_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(th_huber_mono if not is_stereo_obs else th_huber_stereo),  # You can also try gtsam.noiseModel.mEstimator.Cauchy(2.0)
                gtsam.noiseModel.Isotropic.Sigma(2 if not is_stereo_obs else 3, sigma)
            ) 
            iso_noise = gtsam.noiseModel.Isotropic.Sigma(2 if not is_stereo_obs else 3, sigma)     
        
            robust_factor, iso_factor = None, None
            if is_stereo_obs:
                calib = gtsam.Cal3_S2Stereo(kf.camera.fx, kf.camera.fy, 0, kf.camera.cx, kf.camera.cy, kf.camera.b)
                measurement = gtsam.StereoPoint2(kf.kpsu[idx][0], kf.kps_ur[idx], kf.kpsu[idx][1]) # uL, uR, v
                robust_factor = gtsam.GenericStereoFactor(measurement, robust_noise, pose_key, point_key, calib)
                iso_factor = gtsam.GenericStereoFactor(measurement, iso_noise, pose_key, point_key, calib)
            else:
                calib = gtsam.Cal3_S2(kf.camera.fx, kf.camera.fy, 0, kf.camera.cx, kf.camera.cy)
                measurement = gtsam.Point2(kf.kpsu[idx][0], kf.kpsu[idx][1])
                robust_factor = gtsam.GenericProjectionFactorCal3_S2(measurement, robust_noise, pose_key, point_key, calib)
                iso_factor = gtsam.GenericProjectionFactorCal3_S2(measurement, iso_noise, pose_key, point_key, calib)
                
            graph.add(robust_factor)
            graph_factors[(robust_factor,iso_factor)] = (p,kf,idx,is_stereo_obs) # one has kf.points[idx] == p
            num_edges += 1
    
    if abort_flag.value:
        return -1, result_dict

    chi2Mono = 5.991 # chi-square 2 DOFs
    chi2Stereo = 7.815 # chi-square 3 DOFs
    
    initial_mean_squared_error = graph.error(initial_estimates)/max(num_edges,1)

    if robust_rounds > 0:
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(robust_rounds)
        if verbose:
            params.setVerbosityLM("SUMMARY")

        # Optimize using Levenberg-Marquardt
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)        
        result = optimizer.optimize()
        
        inlier_factors = []
        # check inliers observation 
        for factor_pair, factor_data in graph_factors.items(): 
            robust_factor, iso_factor = factor_pair
            p, kf, idx, is_stereo = factor_data
            
            chi2 = 2.0 * robust_factor.error(result) # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian. 
            
            chi2_check_failure = chi2 > (chi2Stereo if is_stereo else chi2Mono)
            point_key = point_keys[p]
            pose_key = keyframe_keys[kf]            
            point_position = np.array(result.atPoint3(point_key)).reshape(3,1)
            pose_wc = np.array(result.atPose3(pose_key).matrix().reshape(4,4)) # Twc
            pose_cw = inv_T(pose_wc)
            Pc = (pose_cw[:3,:3] @ point_position + pose_cw[:3,3].reshape(3,1)).T
            uv, depth = kf.camera.project(Pc)
             
            if chi2_check_failure or not depth>0:
                num_bad_edges += 1
            else:
                inlier_factors.append(iso_factor) # now we just use iso_factors
    else:
        result = initial_estimates
        
    if abort_flag.value:
        return -1, result_dict        
        
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(final_rounds)
    if verbose:
        params.setVerbosityLM("SUMMARY")

    if robust_rounds > 0:
        # rebuild the graph with only inlier factors and optimize
        final_graph = gtsam.NonlinearFactorGraph()
        for f in inlier_factors:
            final_graph.add(f)
            
        # Ensure all keyframe poses are added back
        for kf, pose_key in keyframe_keys.items():
            if not final_graph.exists(pose_key):
                final_graph.add(gtsam.PriorFactorPose3(pose_key, result.atPose3(pose_key), gtsam.noiseModel.Isotropic.Sigma(6, sigma_for_fixed)))

        # Ensure all point variables are added back
        for p, point_key in point_keys.items():
            if not final_graph.exists(point_key):
                final_graph.add(gtsam.PriorFactorPoint3(point_key, result.atPoint3(point_key), gtsam.noiseModel.Isotropic.Sigma(3, sigma_for_fixed)))            
    else: 
        # use the original graph
        final_graph = graph
                     
    optimizer = gtsam.LevenbergMarquardtOptimizer(final_graph, result, params)
    new_result = optimizer.optimize()
    result = new_result
        
    # shut down the sync thread if used and still running
    if sync_flag_thread is not None and sync_flag_thread.is_alive:
        abort_flag.value = True # force the sync thread to exit            
        sync_flag_thread.join() #timeout=0.005)    
        

    # if result_dict is not None then fill in the result dictionary 
    # instead of changing the keyframes and points
    keyframe_updates = None
    point_updates = None
    if result_dict is not None:
        keyframe_updates, point_updates = {}, {}
    
    # put frames back
    if keyframe_updates is not None:
        # store the updates in a dictionary
        for kf in keyframe_keys:
            pose_key = keyframe_keys[kf]
            pose_estimated = result.atPose3(pose_key)
            Twc = pose_estimated.matrix().reshape(4,4)
            Tcw = inv_T(Twc)     
            keyframe_updates[kf.id] = Tcw
    else:
        for kf in keyframe_keys:
            pose_key = keyframe_keys[kf]
            pose_estimated = result.atPose3(pose_key)
            Twc = pose_estimated.matrix().reshape(4,4)
            Tcw = inv_T(Twc)                
            if loop_kf_id == 0:
                # direct update on map
                kf.update_pose(Tcw)
            else:
                # update for loop closure
                kf.Tcw_GBA = Tcw
                kf.GBA_kf_id = loop_kf_id

    # put points back
    if not fixed_points:
        if point_updates is not None:
            # store the updates in a dictionary
            for p in point_keys:
                point_key = point_keys[p]
                point_updates[p.id] = np.array(result.atPoint3(point_key))
        else:
            if loop_kf_id == 0:
                for p in point_keys:
                    # direct update on map
                    point_key = point_keys[p]
                    p.update_position(np.array(result.atPoint3(point_key)))
                    p.update_normal_and_depth(force=True)
            else: 
                for p in point_keys:
                    # update for loop closure
                    point_key = point_keys[p]
                    p.pt_GBA = np.array(np.array(result.atPoint3(point_key)))
                    p.GBA_kf_id = loop_kf_id
            
    num_active_edges = num_edges-num_bad_edges            
    mean_squared_error = final_graph.error(result)/num_active_edges
    
    print(f'bundle_adjustment: mean_squared_error: {mean_squared_error}, initial_mean_squared_error: {initial_mean_squared_error}, num_edges: {num_edges},  num_bad_edges: {num_bad_edges} (perc: {num_bad_edges/num_edges*100:.2f}%)')
    
    if result_dict is not None:
        result_dict['keyframe_updates'] = keyframe_updates
        result_dict['point_updates'] = point_updates

    return mean_squared_error, result_dict
    

# ------------------------------------------------------------------------------------------


def global_bundle_adjustment(keyframes, points, rounds=10, loop_kf_id=0, use_robust_kernel=False, \
                             abort_flag=None, mp_abort_flag=None, result_dict=None, verbose=False, print=print):
    fixed_points=False
    mean_squared_error, result_dict = bundle_adjustment(keyframes, points, local_window=None, \
                                           fixed_points=fixed_points, rounds=rounds, loop_kf_id=loop_kf_id, use_robust_kernel=use_robust_kernel, \
                                           abort_flag=abort_flag, mp_abort_flag=mp_abort_flag, \
                                           result_dict=result_dict, verbose=verbose, print=print)
    return mean_squared_error, result_dict



def global_bundle_adjustment_map(map, rounds=10, loop_kf_id=0, use_robust_kernel=False, \
                                 abort_flag=None, mp_abort_flag=None, result_dict=None, verbose=False, print=print):
    fixed_points=False
    keyframes = map.get_keyframes()
    points = map.get_points()
    return global_bundle_adjustment(keyframes=keyframes, points=points, rounds=rounds, loop_kf_id=loop_kf_id, use_robust_kernel=use_robust_kernel, \
                             abort_flag=abort_flag, mp_abort_flag=mp_abort_flag, result_dict=result_dict, verbose=verbose, print=print)


# ------------------------------------------------------------------------------------------

def resectioning_mono_factor_py(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:

    def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: list[np.ndarray]) -> np.ndarray:
        pose = v.atPose3(pose_key)
        camera = gtsam.PinholeCameraCal3_S2(pose, calib)
        try:
            #print(f'p: {p}, proj: {camera.project(P)}')
            if H is None or len(H) == 0:
                return camera.project(P) - p
            Dpose = np.zeros((2, 6), order="F")
            Dpoint = np.zeros((2, 3), order="F")
            Dcal = np.zeros((2, 5), order="F")
            result = camera.project(P, Dpose, Dpoint, Dcal) - p
            H[0] = Dpose
            if len(H) > 1:
                H[1] = Dpoint            
        except Exception as e:
            Printer.red(f"[resectioning_mono_factor]: Exception: {e}")
            result = np.zeros((2, 1), order="F")
            if H is not None and len(H) > 0:
                H[0] = np.zeros((2, 6), order="F")
                if len(H) > 1:
                    H[1] = np.zeros((2, 3), order="F")                
        return result

    return gtsam.CustomFactor(noise_model, gtsam.KeyVector([pose_key]), error_func)

def resectioning_mono_factor(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    return gtsam_factors.ResectioningFactor(noise_model, pose_key, calib, p, P)


def resectioning_stereo_factor_py(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2Stereo,
    p: gtsam.StereoPoint2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:

    def error_func(this: gtsam.CustomFactor, v: gtsam.Values, H: list[np.ndarray]) -> np.ndarray:
        pose = v.atPose3(pose_key)
        camera = gtsam.StereoCamera(pose, calib)
        p_vec = p.vector()
        try:
            #print(f'p_vec: {p_vec}, proj: {camera.project(P).vector()}')
            if H is None or len(H) == 0:
                return camera.project(P).vector() - p_vec
            Dpose = np.zeros((3, 6), order="F")
            Dpoint = np.zeros((3, 3), order="F")
            result = camera.project2(P, Dpose, Dpoint).vector() - p_vec
            H[0] = Dpose
            if len(H) > 1:
                H[1] = Dpoint            
        except Exception as e:
            Printer.red(f"[resectioning_stereo_factor]: Exception: {e}")
            result = np.zeros((3, 1), order="F")
            if H is not None and len(H) > 0:
                H[0] = np.zeros((3, 6), order="F")
                if len(H) > 1:
                    H[1] = np.zeros((3, 3), order="F")                
        return result

    return gtsam.CustomFactor(noise_model, gtsam.KeyVector([pose_key]), error_func)

def resectioning_stereo_factor(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2Stereo,
    p: gtsam.StereoPoint2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    return gtsam_factors.ResectioningFactorStereo(noise_model, pose_key, calib, p, P)


class PoseOptimizerGTSAM:
    def __init__(self, frame, use_robust_factors=True):
        self.frame = frame
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.factor_tuples = {}
        self.num_factors = 0
        self.use_robust_factors = use_robust_factors
                
        self.K_mono = gtsam.Cal3_S2(frame.camera.fx, frame.camera.fy, 0, frame.camera.cx, frame.camera.cy)
        self.K_stereo = gtsam.Cal3_S2Stereo(frame.camera.fx, frame.camera.fy, 0, frame.camera.cx, frame.camera.cy, frame.camera.b)

        self.thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS 
        self.thHuberStereo = math.sqrt(7.815) # chi-square 3 DOFS 
        
        self.add_mono_factor = resectioning_mono_factor
        self.add_stereo_factor = resectioning_stereo_factor
        if platform.system() == "Darwin":
            self.add_mono_factor  = resectioning_mono_factor_py 
            self.add_stereo_factor  = resectioning_stereo_factor_py       

    def add_pose_node(self):
        pose_initial = gtsam.Pose3(gtsam.Rot3(self.frame.Rwc), gtsam.Point3(*self.frame.Ow))
        self.initial.insert(X(0), pose_initial)
        # NOTE: there is no need to set a prior here
        #noise_prior = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)  
        #self.graph.add(gtsam.PriorFactorPose3(X(0), pose_initial, noise_prior))

    def add_observations(self):        
        with MapPoint.global_lock:
            for idx, p in enumerate(self.frame.points):
                if p is None:
                    continue

                self.frame.outliers[idx] = False
                is_stereo_obs = self.frame.kps_ur is not None and self.frame.kps_ur[idx] > 0

                robust_factor, iso_factor = None, None
                #invSigma2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[self.frame.octaves[idx]]
                sigma = FeatureTrackerShared.feature_manager.level_sigmas[self.frame.octaves[idx]]
                
                robust_noise = gtsam.noiseModel.Robust.Create(
                    gtsam.noiseModel.mEstimator.Huber.Create(self.thHuberMono if not is_stereo_obs else self.thHuberStereo),  # You can also try gtsam.noiseModel.mEstimator.Cauchy(2.0)
                    gtsam.noiseModel.Isotropic.Sigma(2 if not is_stereo_obs else 3, sigma)
                )
                
                iso_noise = gtsam.noiseModel.Isotropic.Sigma(2 if not is_stereo_obs else 3, sigma)
                
                # Add the observation factor
                if is_stereo_obs:
                    robust_factor = self.add_stereo_factor(robust_noise, X(0), self.K_stereo, gtsam.StereoPoint2(self.frame.kpsu[idx][0], self.frame.kps_ur[idx], self.frame.kpsu[idx][1]), gtsam.Point3(*p.pt))
                    iso_factor = self.add_stereo_factor(iso_noise, X(0), self.K_stereo, gtsam.StereoPoint2(self.frame.kpsu[idx][0], self.frame.kps_ur[idx], self.frame.kpsu[idx][1]), gtsam.Point3(*p.pt))
                else:
                    robust_factor = self.add_mono_factor(robust_noise, X(0), self.K_mono, gtsam.Point2(self.frame.kpsu[idx][0], self.frame.kpsu[idx][1]), gtsam.Point3(*p.pt))
                    iso_factor = self.add_mono_factor(iso_noise, X(0), self.K_mono, gtsam.Point2(self.frame.kpsu[idx][0], self.frame.kpsu[idx][1]), gtsam.Point3(*p.pt))

                used_factor = robust_factor if self.use_robust_factors else iso_factor
                self.graph.add(used_factor) # initially we use robust factors 
                self.factor_tuples[p] = (robust_factor, iso_factor, idx)
                self.num_factors += 1

    def optimize(self, rounds=10, verbose=False):
        if self.num_factors < 3:
            Printer.red('pose_optimization: not enough correspondences!')
            return 0, False, 0

        chi2Mono = 5.991  # Chi-squared 2 DOFs
        chi2Stereo = 7.815 # chi-squared 3 DOFs   
        
        is_ok = True 
    
        params = gtsam.LevenbergMarquardtParams()
        #params.setlambdaInitial(1e-9)
        params.setMaxIterations(rounds)
        if verbose:
            params.setVerbosityLM("SUMMARY")
            
        #initial_error = self.graph.error(self.initial)
         
        result, result_prev = None, None
        cost, cost_prev = None, float("inf")
        num_inliers = 0     
        
        for it in range(4):
            optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
            result_prev = result
            result = optimizer.optimize()
            
            #marginals = gtsam.Marginals(self.graph, result)

            cost = self.graph.error(result)  # Compute new cost
            cost_change = cost_prev - cost
            
            if cost_change <= 0 or np.isinf(cost):
                Printer.orange(f"pose_optimization: Warning: Cost did not decrease or is not finite at iteration {it}! Previous: {cost_prev}, Current: {cost}")
                result = result_prev 
                is_ok = False           
                break                     
            
            cost_prev = cost
            
            num_bad_point_edges = 0  
            total_inlier_error = 0.0 
            num_inliers = 0  
            inlier_factors = []

            for p, factor_tuple in self.factor_tuples.items():
                robust_factor, iso_factor, idx = factor_tuple
                used_factor = robust_factor if self.use_robust_factors else iso_factor
                chi2 = 2.0 * used_factor.error(result) # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian. 
                
                is_stereo_obs = self.frame.kps_ur is not None and self.frame.kps_ur[idx]>0

                chi2_check_failure = chi2 > (chi2Stereo if is_stereo_obs else chi2Mono)
                if chi2_check_failure:
                    self.frame.outliers[idx] = True
                    num_bad_point_edges += 1
                else:
                    self.frame.outliers[idx] = False
                    total_inlier_error += chi2  # Sum error only for inliers
                    num_inliers += 1    
                    inlier_factors.append((robust_factor, iso_factor))
            
            if num_inliers < 10:
                Printer.red('pose_optimization: stopped - not enough edges!')  
                result = result_prev 
                is_ok = False           
                break       
                        
            if it ==2: 
                self.use_robust_factors = False # last iterations use isotropic factors
                
            if it < 3:    
                # rebuild the graph with only inlier factors (in gtsam is not possible to deactivate the outlier factors)      
                new_graph = gtsam.NonlinearFactorGraph()
                #print(f'pose_optimization: it {it}, #inliers {num_inliers}/{num_bad_point_edges} #pose: {current_pose_estimated.matrix().reshape(4,4)}')
                for robust_f, iso_f in inlier_factors:
                    f = robust_f if self.use_robust_factors else iso_f
                    new_graph.add(f)
                self.initial = result                      
                self.graph = new_graph           
        
        print(f'pose optimization: available {self.num_factors} points, found {num_bad_point_edges} bad points')   
        num_valid_points = self.num_factors - num_bad_point_edges
        if num_valid_points == 0:
            Printer.red('pose_optimization: all the available correspondences are bad!')
            result = result_prev
            is_ok = False

        # update pose estimation
        if result is not None: 
            is_ok = True
            pose_estimated = result.atPose3(X(0))
            Twc = pose_estimated.matrix().reshape(4,4)
            self.frame.update_pose(inv_T(Twc))
        else: 
            is_ok = False

        mean_squared_error = total_inlier_error / max(num_inliers, 1) if num_inliers > 0 else -1
        return mean_squared_error, is_ok, num_valid_points
    
    
    def run(self, verbose=False, rounds=10):
        self.add_pose_node()
        self.add_observations()
        return self.optimize(rounds, verbose)

def pose_optimization(frame, verbose=False, rounds=10):
    optimizer = PoseOptimizerGTSAM(frame)
    return optimizer.run(verbose, rounds)


# ------------------------------------------------------------------------------------------


# local bundle adjustment (optimize points reprojection error)
# - frames and points are optimized
# - frames_ref are fixed 
def local_bundle_adjustment(keyframes, points, keyframes_ref=[], fixed_points=False, verbose=False, rounds=10, abort_flag=g2o.Flag(), mp_abort_flag=None, map_lock=None):    
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()
    
    sigma_for_fixed = 1e-9 # sigma used for fixed points
        
    # Huber loss parameters for robust optimization
    th_huber_mono = np.sqrt(5.991)  # chi-square 2 DOFs
    th_huber_stereo = np.sqrt(7.815)  # chi-square 3 DOFs

    good_keyframes = [kf for kf in keyframes if not kf.is_bad] + [kf for kf in keyframes_ref if not kf.is_bad]
    good_points = [p for p in points if p is not None and not p.is_bad] # and any(f in keyframes for f in p.keyframes())]
       
    keyframe_keys = {}
    point_keys = {}

    # Add keyframe (camera pose) vertices
    for kf in good_keyframes:
        pose_key = gtsam.symbol('x', kf.kid)
        keyframe_keys[kf] = pose_key
        
        pose = gtsam.Pose3(gtsam.Rot3(kf.Rwc), gtsam.Point3(*kf.Ow))
        initial_estimates.insert(pose_key, pose)
        
        if kf.kid == 0 or kf in keyframes_ref:
            graph.add(gtsam.PriorFactorPose3(pose_key, pose, gtsam.noiseModel.Isotropic.Sigma(6, sigma_for_fixed)))

    num_edges = 0
    num_bad_edges = 0
    graph_factors = {}
    
    # Add 3D point vertices
    for p in good_points:
        point_key = gtsam.symbol('p', p.id)
        point_keys[p] = point_key
        initial_estimates.insert(point_key, gtsam.Point3(p.pt[:3]))

        if fixed_points:
            graph.add(gtsam.PriorFactorPoint3(point_key, gtsam.Point3(p.pt[:3]), gtsam.noiseModel.Isotropic.Sigma(3, sigma_for_fixed)))

        # add edges
        good_observations = [(kf, p_idx) for kf, p_idx in p.observations() if not kf.is_bad and kf in keyframe_keys]
        
        # Add reprojection factors
        for kf, p_idx in good_observations:
            pose_key = keyframe_keys[kf]

            assert(kf.get_point_match(p_idx) is p)
            
            is_stereo_obs = kf.kps_ur is not None and kf.kps_ur[p_idx] > 0
            #invSigma2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[kf.octaves[p_idx]]
            sigma = FeatureTrackerShared.feature_manager.level_sigmas[kf.octaves[p_idx]]
            
            robust_noise = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(th_huber_mono if not is_stereo_obs else th_huber_stereo),  # You can also try gtsam.noiseModel.mEstimator.Cauchy(2.0)
                gtsam.noiseModel.Isotropic.Sigma(2 if not is_stereo_obs else 3, sigma)
            )   
            
            iso_noise = gtsam.noiseModel.Isotropic.Sigma(2 if not is_stereo_obs else 3, sigma)     
        
            robust_factor, iso_factor = None, None
            if is_stereo_obs:
                calib = gtsam.Cal3_S2Stereo(kf.camera.fx, kf.camera.fy, 0, kf.camera.cx, kf.camera.cy, kf.camera.b)
                measurement = gtsam.StereoPoint2(kf.kpsu[p_idx][0], kf.kps_ur[p_idx], kf.kpsu[p_idx][1]) # uL, uR, v
                robust_factor = gtsam.GenericStereoFactor(measurement, robust_noise, pose_key, point_key, calib)
                iso_factor = gtsam.GenericStereoFactor(measurement, iso_noise, pose_key, point_key, calib)
            else:
                calib = gtsam.Cal3_S2(kf.camera.fx, kf.camera.fy, 0, kf.camera.cx, kf.camera.cy)
                measurement = gtsam.Point2(kf.kpsu[p_idx][0], kf.kpsu[p_idx][1])
                robust_factor = gtsam.GenericProjectionFactorCal3_S2(measurement, robust_noise, pose_key, point_key, calib)
                iso_factor = gtsam.GenericProjectionFactorCal3_S2(measurement, iso_noise, pose_key, point_key, calib)
                
            graph.add(robust_factor)
            graph_factors[(robust_factor,iso_factor)] = (p,kf,p_idx,is_stereo_obs) # one has kf.points[p_idx] == p
            num_edges += 1

    if abort_flag.value:
        return -1,0

    chi2Mono = 5.991 # chi-square 2 DOFs
    chi2Stereo = 7.815 # chi-square 3 DOFs
        
    params = gtsam.LevenbergMarquardtParams()
    params.setMaxIterations(5)
    if verbose:
        params.setVerbosityLM("SUMMARY")
            
    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
    result = optimizer.optimize()
            
    if not abort_flag.value:

        inlier_factors = []    
        # check inliers observation 
        for factor_pair, factor_data in graph_factors.items(): 
            robust_factor, iso_factor = factor_pair
            p, kf, p_idx, is_stereo = factor_data
            
            # if p.is_bad: # redundant check since the considered points come from good_points
            #     continue 
            
            chi2 = 2.0 * robust_factor.error(result) # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian. 
            
            chi2_check_failure = chi2 > (chi2Stereo if is_stereo else chi2Mono)
            point_key = point_keys[p]
            pose_key = keyframe_keys[kf]
            point_position = np.array(result.atPoint3(point_key)).reshape(3,1)
            pose_wc = np.array(result.atPose3(pose_key).matrix().reshape(4,4)) # Twc
            pose_cw = inv_T(pose_wc)
            Pc = (pose_cw[:3,:3] @ point_position + pose_cw[:3,3].reshape(3,1)).T
            uv, depth = kf.camera.project(Pc) 
            
            if chi2_check_failure or not depth>0:
                num_bad_edges += 1
            else:
                inlier_factors.append(iso_factor) # now we just use iso_factors              

        # optimize again without outliers 
        
        params = gtsam.LevenbergMarquardtParams()
        params.setMaxIterations(rounds)
        if verbose:
            params.setVerbosityLM("SUMMARY")

        # rebuild the graph with only inlier factors and optimize
        new_graph = gtsam.NonlinearFactorGraph()
        for f in inlier_factors:
            new_graph.add(f)   

        # Ensure all keyframe poses are added back
        for kf, pose_key in keyframe_keys.items():
            if not new_graph.exists(pose_key):
                new_graph.add(gtsam.PriorFactorPose3(pose_key, result.atPose3(pose_key), gtsam.noiseModel.Isotropic.Sigma(6, sigma_for_fixed)))

        # Ensure all point variables are added back
        for p, point_key in point_keys.items():
            if not new_graph.exists(point_key):
                new_graph.add(gtsam.PriorFactorPoint3(point_key, result.atPoint3(point_key), gtsam.noiseModel.Isotropic.Sigma(3, sigma_for_fixed)))

        new_optimizer = gtsam.LevenbergMarquardtOptimizer(new_graph, result, params)
        new_result = new_optimizer.optimize()
        result = new_result

    # search for final outlier observations and clean map  
    num_bad_observations = 0  # final bad observations
    outliers_factors_data = []
    
    total_error = 0
    num_inlier_observations = 0
    
    for factor_pair, factor_data in graph_factors.items(): 
        robust_factor, iso_factor = factor_pair
        p, kf, p_idx, is_stereo = factor_data
        
        # if p.is_bad: # redundant check since the considered points come from good_points
        #     continue         
        
        assert(kf.get_point_match(p_idx) is p) 
        
        chi2 = 2.0 * iso_factor.error(result) # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian. 
        
        chi2_check_failure = chi2 > (chi2Stereo if is_stereo else chi2Mono)
        point_key = point_keys[p]
        pose_key = keyframe_keys[kf]
        point_position = np.array(result.atPoint3(point_key)).reshape(3,1)
        pose_wc = np.array(result.atPose3(pose_key).matrix().reshape(4,4)) # Twc
        pose_cw = inv_T(pose_wc)
        Pc = (pose_cw[:3,:3] @ point_position + pose_cw[:3,3].reshape(3,1)).T
        uv, depth = kf.camera.project(Pc) 
        
        if chi2_check_failure or not depth>0:         
            num_bad_observations += 1
            outliers_factors_data.append(factor_data) 
        else:
            num_inlier_observations += 1      
            total_error += chi2

    if map_lock is None: 
        map_lock = threading.RLock() # put a fake lock     
    
    with map_lock:      
        # remove outlier observations 
        for p, kf, p_idx, is_stereo in outliers_factors_data:
            p_f = kf.get_point_match(p_idx)
            if p_f is not None:
                assert(p_f is p)
                p.remove_observation(kf,p_idx)
                # the following instruction is now included in p.remove_observation()
                #f.remove_point(p)   # it removes multiple point instances (if these are present)   
                #f.remove_point_match(p_idx) # this does not remove multiple point instances, but now there cannot be multiple instances any more

        # put frames back
        for kf in keyframe_keys:
            pose_key = keyframe_keys[kf]
            pose_estimated = result.atPose3(pose_key)
            Twc = pose_estimated.matrix().reshape(4,4)
            kf.update_pose(inv_T(Twc))            
            kf.lba_count += 1

        # put points back
        if not fixed_points:
            for p in point_keys:
                point_key = point_keys[p]
                p.update_position(np.array(result.atPoint3(point_key)))
                p.update_normal_and_depth(force=True)

    num_active_edges = num_inlier_observations # num_edges-num_bad_edges
    mean_squared_error = total_error/max(num_active_edges,1) # opt.active_chi2()/max(num_active_edges,1)

    return mean_squared_error, num_bad_observations/max(num_edges,1)    