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

from threading import RLock
import multiprocessing as mp

import g2o

from utils_geom import poseRt
from frame import Frame
from utils_sys import Printer
from map_point import MapPoint


# ------------------------------------------------------------------------------------------

# optimize pixel reprojection error, bundle adjustment
def bundle_adjustment(keyframes, points, local_window, fixed_points=False, verbose=False, rounds=10, use_robust_kernel=False, abort_flag=g2o.Flag()):
    if local_window is None:
        local_frames = keyframes
    else:
        local_frames = keyframes[-local_window:]

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    #block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())        
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)
    opt.set_force_stop_flag(abort_flag)

    thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS 
    thHuberStereo = math.sqrt(7.815)  # chi-square 3 DOFS 

    graph_keyframes, graph_points = {}, {}

    # add frame vertices to graph
    for kf in (local_frames if fixed_points else keyframes):    # if points are fixed then consider just the local frames, otherwise we need all frames or at least two frames for each point
        if kf.is_bad:
            continue 
        #print('adding vertex frame ', f.id, ' to graph')
        se3 = g2o.SE3Quat(kf.Rcw, kf.tcw)
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
        v_se3.set_fixed(kf.kid==0 or kf not in local_frames) #(use f.kid here!)
        opt.add_vertex(v_se3)

        # confirm pose correctness
        #est = v_se3.estimate()
        #assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        #assert np.allclose(pose[0:3, 3], est.translation())

        graph_keyframes[kf] = v_se3

    num_edges = 0
    
    # add point vertices to graph 
    for p in points:
        assert(p is not None)        
        if p.is_bad:  # do not consider bad points   
            continue
        if __debug__:        
            if not any([f in keyframes for f in p.keyframes()]):  
                Printer.red('point without a viewing frame!!')
                continue        
        #print('adding vertex point ', p.id,' to graph')
        v_p = g2o.VertexSBAPointXYZ()    
        v_p.set_id(p.id * 2 + 1)  # odd ids
        v_p.set_estimate(p.pt[0:3])
        v_p.set_marginalized(True)
        v_p.set_fixed(fixed_points)
        opt.add_vertex(v_p)
        graph_points[p] = v_p

        # add edges
        for kf, idx in p.observations():
            if kf.is_bad:
                continue 
            if kf not in graph_keyframes:
                continue
            
            #print('adding edge between point ', p.id,' and frame ', f.id)
            is_stereo_obs = kf.kps_ur is not None and kf.kps_ur[idx]>0
            
            edge = None 
            invSigma2 = Frame.feature_manager.inv_level_sigmas2[kf.octaves[idx]]
            
            if is_stereo_obs: 
                edge = g2o.EdgeStereoSE3ProjectXYZ()
                edge.set_vertex(0, v_p)
                edge.set_vertex(1, graph_keyframes[kf])
                obs = [kf.kpsu[idx][0], kf.kpsu[idx][1], kf.kps_ur[idx]]
                edge.set_measurement(obs)
                
                edge.set_information(np.eye(3)*invSigma2)
                if use_robust_kernel:
                    edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberStereo))

                edge.fx = kf.camera.fx 
                edge.fy = kf.camera.fy
                edge.cx = kf.camera.cx
                edge.cy = kf.camera.cy       
                edge.bf = kf.camera.bf         
            else: 
                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_vertex(0, v_p)
                edge.set_vertex(1, graph_keyframes[kf])
                edge.set_measurement(kf.kpsu[idx])
                
                edge.set_information(np.eye(2)*invSigma2)
                if use_robust_kernel:
                    edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

                edge.fx = kf.camera.fx 
                edge.fy = kf.camera.fy
                edge.cx = kf.camera.cx
                edge.cy = kf.camera.cy

            opt.add_edge(edge)
            num_edges += 1

    if verbose:
        opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(rounds)

    # put frames back
    for kf in graph_keyframes:
        est = graph_keyframes[kf].estimate()
        #R = est.rotation().matrix()
        #t = est.translation()
        #f.update_pose(poseRt(R, t))
        kf.update_pose(g2o.Isometry3d(est.orientation(), est.position()))

    # put points back
    if not fixed_points:
        for p in graph_points:
            p.update_position(np.array(graph_points[p].estimate()))
            p.update_normal_and_depth(force=True)
            
    mean_squared_error = opt.active_chi2()/max(num_edges,1)

    return mean_squared_error


# ------------------------------------------------------------------------------------------

# optimize points reprojection error:
# - frame pose is optimized 
# - 3D points observed in frame are considered fixed
# output: 
# - mean_squared_error 
# - is_ok: is the pose optimization successful? 
# - num_valid_points: number of inliers detected by the optimization 
# N.B.: access frames from tracking thread, no need to lock frame fields 
def pose_optimization(frame, verbose=False, rounds=10):

    is_ok = True 

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    #block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    #block_solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())    
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())       
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)

    thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS 
    thHuberStereo = math.sqrt(7.815) # chi-square 3 DOFS 

    point_edge_pairs = {}
    num_point_edges = 0

    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_estimate(g2o.SE3Quat(frame.Rcw, frame.tcw))
    v_se3.set_id(0)  
    v_se3.set_fixed(False)
    opt.add_vertex(v_se3)

    with MapPoint.global_lock:
        # add point vertices to graph 
        for idx, p in enumerate(frame.points):
            if p is None:  
                continue

            # reset outlier flag 
            frame.outliers[idx] = False
            is_stereo_obs = frame.kps_ur is not None and frame.kps_ur[idx]>0
            
            # add edge
            edge = None 
            invSigma2 = Frame.feature_manager.inv_level_sigmas2[frame.octaves[idx]]
                        
            if is_stereo_obs: 
                #print('adding stereo edge between point ', p.id,' and frame ', frame.id)
                edge = g2o.EdgeStereoSE3ProjectXYZOnlyPose()
                obs = [frame.kpsu[idx][0], frame.kpsu[idx][1], frame.kps_ur[idx]] # u,v,ur
                edge.set_vertex(0, opt.vertex(0))
                edge.set_measurement(obs)
                
                edge.set_information(np.eye(3)*invSigma2)
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberStereo))

                edge.fx = frame.camera.fx 
                edge.fy = frame.camera.fy
                edge.cx = frame.camera.cx
                edge.cy = frame.camera.cy
                edge.bf = frame.camera.bf
                edge.Xw = p.pt[0:3]                
            else: 
                #print('adding mono edge between point ', p.id,' and frame ', frame.id)
                edge = g2o.EdgeSE3ProjectXYZOnlyPose()

                edge.set_vertex(0, opt.vertex(0))
                edge.set_measurement(frame.kpsu[idx])

                edge.set_information(np.eye(2)*invSigma2)
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

                edge.fx = frame.camera.fx 
                edge.fy = frame.camera.fy
                edge.cx = frame.camera.cx
                edge.cy = frame.camera.cy
                edge.Xw = p.pt[0:3]
            
            opt.add_edge(edge)

            point_edge_pairs[p] = (edge, idx) # one edge per point 
            num_point_edges += 1

    if num_point_edges < 3:
        Printer.red('pose_optimization: not enough correspondences!') 
        is_ok = False 
        return 0, is_ok, 0

    if verbose:
        opt.set_verbose(True)

    # perform 4 optimizations: 
    # after each optimization we classify observation as inlier/outlier;
    # at the next optimization, outliers are not included, but at the end they can be classified as inliers again
    chi2Mono = 5.991 # chi-square 2 DOFs
    chi2Stereo = 7.815 # chi-square 3 DOFs
    num_bad_point_edges = 0

    for it in range(4):
        v_se3.set_estimate(g2o.SE3Quat(frame.Rcw, frame.tcw))
        opt.initialize_optimization()        
        opt.optimize(rounds)

        num_bad_point_edges = 0

        for p, edge_pair in point_edge_pairs.items(): 
            edge, idx = edge_pair
            if frame.outliers[idx]:
                edge.compute_error()

            chi2 = edge.chi2()
            
            is_stereo_obs = frame.kps_ur is not None and frame.kps_ur[idx]>0
            
            chi2_check_failure = chi2 > chi2Mono if not is_stereo_obs else chi2 > chi2Stereo
            if chi2_check_failure:
                frame.outliers[idx] = True 
                edge.set_level(1)
                num_bad_point_edges +=1
            else:
                frame.outliers[idx] = False
                edge.set_level(0)                                

            if it == 2:
                edge.set_robust_kernel(None)

        if len(opt.edges()) < 10:
            Printer.red('pose_optimization: stopped - not enough edges!')   
            is_ok = False           
            break                 
    
    print('pose optimization: available ', num_point_edges, ' points, found ', num_bad_point_edges, ' bad points')     
    if num_point_edges == num_bad_point_edges:
        Printer.red('pose_optimization: all the available correspondences are bad!')           
        is_ok = False      

    # update pose estimation
    if is_ok: 
        est = v_se3.estimate()
        # R = est.rotation().matrix()
        # t = est.translation()
        # frame.update_pose(poseRt(R, t))
        frame.update_pose(g2o.Isometry3d(est.orientation(), est.position()))

    # since we have only one frame here, each edge corresponds to a single distinct point
    num_valid_points = num_point_edges - num_bad_point_edges   
    
    mean_squared_error = opt.active_chi2()/max(num_valid_points,1)

    return mean_squared_error, is_ok, num_valid_points


# ------------------------------------------------------------------------------------------

# local bundle adjustment (optimize points reprojection error)
# - frames and points are optimized
# - frames_ref are fixed 
def local_bundle_adjustment(keyframes, points, keyframes_ref=[], fixed_points=False, verbose=False, rounds=10, abort_flag=g2o.Flag(), map_lock=None):

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    #block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())  
    #block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())          
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)
    opt.set_force_stop_flag(abort_flag)

    thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS
    thHuberStereo = math.sqrt(7.815)  # chi-square 3 DOFS 

    graph_keyframes, graph_points, graph_edges = {}, {}, {}

    good_keyframes = [kf for kf in keyframes if not kf.is_bad] + \
                     [kf for kf in keyframes_ref if not kf.is_bad]
    
    # add frame vertices to graph
    for kf in good_keyframes:    
        #print('adding vertex frame ', f.id, ' to graph')
        se3 = g2o.SE3Quat(kf.Rcw, kf.tcw)
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
        v_se3.set_fixed(kf.kid==0 or kf in keyframes_ref)  # (use f.kid here!)
        opt.add_vertex(v_se3)
        graph_keyframes[kf] = v_se3     
           
        # confirm pose correctness
        #est = v_se3.estimate()
        #assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        #assert np.allclose(pose[0:3, 3], est.translation())         

    num_edges = 0
    num_bad_edges = 0

    good_points = [p for p in points if p is not None and not p.is_bad and any(f in keyframes for f in p.keyframes())]

    # add point vertices to graph 
    #for p in points:
    for p in good_points:
        
        # assert(p is not None)
        # if p.is_bad:  # do not consider bad points             
        #     continue  
        # if not any([f in keyframes for f in p.keyframes()]):  
        #     Printer.orange('point %d without a viewing keyframe in input keyframes!!' %(p.id))
        #     #Printer.orange('         keyframes: ',p.observations_string())
        #     continue
        
        #print('adding vertex point ', p.id,' to graph')
        v_p = g2o.VertexSBAPointXYZ()    
        v_p.set_id(p.id * 2 + 1)  # odd ids
        v_p.set_estimate(p.pt[0:3])
        v_p.set_marginalized(True)
        v_p.set_fixed(fixed_points)
        opt.add_vertex(v_p)
        graph_points[p] = v_p

        # add edges
        good_observations = [(kf, p_idx) for kf, p_idx in p.observations() if not kf.is_bad and kf in graph_keyframes]
        
        #for kf, p_idx in p.observations():
        for kf, p_idx in good_observations:            
            # if kf.is_bad:
            #     continue 
            # if kf not in graph_keyframes:
            #     continue
            if __debug__:      
                p_f = kf.get_point_match(p_idx)
                if p_f != p:
                    print('frame: ', kf.id, ' missing point ', p.id, ' at index p_idx: ', p_idx)                    
                    if p_f is not None:
                        print('p_f:', p_f)
                    print('p:',p)
            assert(kf.get_point_match(p_idx) is p)            
            
            #print('adding edge between point ', p.id,' and frame ', f.id)
      
            is_stereo_obs = kf.kps_ur is not None and kf.kps_ur[p_idx]>0
            invSigma2 = Frame.feature_manager.inv_level_sigmas2[kf.octaves[p_idx]]
            
            if is_stereo_obs:
                edge = g2o.EdgeStereoSE3ProjectXYZ()
                obs = [kf.kpsu[p_idx][0], kf.kpsu[p_idx][1], kf.kps_ur[p_idx]]
                edge.set_measurement(obs)
                
                edge.set_information(np.eye(3)*invSigma2)
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberStereo))
            
                edge.bf = kf.camera.bf                
            else: 
                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_measurement(kf.kpsu[p_idx])

                edge.set_information(np.eye(2)*invSigma2)
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

            edge.fx, edge.fy, edge.cx, edge.cy = kf.camera.fx, kf.camera.fy, kf.camera.cx, kf.camera.cy
                
            edge.set_vertex(0, v_p)
            edge.set_vertex(1, graph_keyframes[kf])
            opt.add_edge(edge)

            graph_edges[edge] = (p,kf,p_idx,is_stereo_obs) # one has kf.points[p_idx] == p
            num_edges += 1            

    if verbose:
        opt.set_verbose(True)

    if abort_flag.value:
        return -1,0

    # initial optimization 
    opt.initialize_optimization()
    opt.optimize(5)

    chi2Mono = 5.991 # chi-square 2 DOFs
    chi2Stereo = 7.815 # chi-square 3 DOFs
            
    if not abort_flag.value:

        # check inliers observation 
        for edge, edge_data in graph_edges.items(): 
            p, kf, p_idx, is_stereo = edge_data
            
            # if p.is_bad: # redundant check since the considered points come from good_points
            #     continue 
            
            edge_chi2 = edge.chi2()
            chi2_check_failure = edge_chi2 > chi2Mono if not is_stereo else edge_chi2 > chi2Stereo
            if chi2_check_failure or not edge.is_depth_positive():
                edge.set_level(1)
                num_bad_edges += 1
            edge.set_robust_kernel(None)

        # optimize again without outliers 
        opt.initialize_optimization()
        opt.optimize(rounds)

    # search for final outlier observations and clean map  
    num_bad_observations = 0  # final bad observations
    outliers_edge_data = []
    
    chi2_limits = {True: chi2Stereo, False: chi2Mono}
        
    for edge, edge_data in graph_edges.items(): 
        p, kf, p_idx, is_stereo = edge_data
        
        # if p.is_bad: # redundant check since the considered points come from good_points
        #     continue         
        
        assert(kf.get_point_match(p_idx) is p) 
        
        if edge.chi2() > chi2_limits[is_stereo] or not edge.is_depth_positive():         
            num_bad_observations += 1
            outliers_edge_data.append(edge_data)       

    if map_lock is None: 
        map_lock = RLock() # put a fake lock 
        
    with map_lock:      
        # remove outlier observations 
        for p, kf, p_idx, is_stereo in outliers_edge_data:
            p_f = kf.get_point_match(p_idx)
            if p_f is not None:
                assert(p_f is p)
                p.remove_observation(kf,p_idx)
                # the following instruction is now included in p.remove_observation()
                #f.remove_point(p)   # it removes multiple point instances (if these are present)   
                #f.remove_point_match(p_idx) # this does not remove multiple point instances, but now there cannot be multiple instances any more

        # put frames back
        for kf in graph_keyframes:
            est = graph_keyframes[kf].estimate()
            #R = est.rotation().matrix()
            #t = est.translation()
            #f.update_pose(poseRt(R, t))
            kf.update_pose(g2o.Isometry3d(est.orientation(), est.position()))

        # put points back
        if not fixed_points:
            for p in graph_points:
                p.update_position(np.array(graph_points[p].estimate()))
                p.update_normal_and_depth(force=True)

    active_edges = num_edges-num_bad_edges
    mean_squared_error = opt.active_chi2()/max(active_edges,1)

    return mean_squared_error, num_bad_observations/max(num_edges,1)


# ------------------------------------------------------------------------------------------

# parallel local bundle adjustment (optimize points reprojection error)
# - frames and points are optimized
# - frames_ref are fixed 

# This function will handle the multiprocessing part of the optimization
def lba_optimization_process(result_dict, good_keyframes, keyframes_ref, good_points, fixed_points, verbose, rounds, abort_flag, map_lock):
    try:
        # create g2o optimizer
        opt = g2o.SparseOptimizer()
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        #block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())  
        #block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())          
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        opt.set_algorithm(solver)
        opt.set_force_stop_flag(abort_flag)
        
        graph_keyframes, graph_points, graph_edges = {}, {}, {}
            
        # add frame vertices to graph
        for kf in good_keyframes.values():    
            #print('adding vertex frame ', f.id, ' to graph')
            se3 = g2o.SE3Quat(kf.Rcw, kf.tcw)
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_estimate(se3)
            v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
            v_se3.set_fixed(kf.kid==0 or kf in keyframes_ref)  # (use f.kid here!)
            opt.add_vertex(v_se3)
            graph_keyframes[kf] = v_se3     
            
            # confirm pose correctness
            #est = v_se3.estimate()
            #assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
            #assert np.allclose(pose[0:3, 3], est.translation())         

        thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS
        thHuberStereo = math.sqrt(7.815)  # chi-square 3 DOFS 
        
        num_edges = 0
        num_bad_edges = 0

        # Add point vertices to the graph
        for p in good_points.values():
            v_p = g2o.VertexSBAPointXYZ()
            v_p.set_id(p.id * 2 + 1)
            v_p.set_estimate(p.pt[0:3])
            v_p.set_marginalized(True)
            v_p.set_fixed(fixed_points)
            opt.add_vertex(v_p)
            graph_points[p] = v_p

            # add edges
            good_observations = [(kf, p_idx) for kf, p_idx in p.observations() if not kf.is_bad and kf in graph_keyframes]
            for kf, p_idx in good_observations:
                
                if __debug__:      
                    p_f = kf.get_point_match(p_idx)
                    if p_f != p:
                        print('frame: ', kf.id, ' missing point ', p.id, ' at index p_idx: ', p_idx)                    
                        if p_f is not None:
                            print('p_f:', p_f)
                        print('p:',p)
                assert(kf.get_point_match(p_idx) is p)            
                
                #print('adding edge between point ', p.id,' and frame ', f.id)
                            
                is_stereo_obs = kf.kps_ur is not None and kf.kps_ur[p_idx] > 0
                invSigma2 = Frame.feature_manager.inv_level_sigmas2[kf.octaves[p_idx]]

                if is_stereo_obs:
                    edge = g2o.EdgeStereoSE3ProjectXYZ()
                    obs = [kf.kpsu[p_idx][0], kf.kpsu[p_idx][1], kf.kps_ur[p_idx]]
                    edge.set_measurement(obs)
                    
                    edge.set_information(np.eye(3) * invSigma2)
                    edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberStereo))
                    
                    edge.bf = kf.camera.bf
                else:
                    edge = g2o.EdgeSE3ProjectXYZ()
                    edge.set_measurement(kf.kpsu[p_idx])
                    
                    edge.set_information(np.eye(2) * invSigma2)
                    edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

                edge.fx, edge.fy, edge.cx, edge.cy = kf.camera.fx, kf.camera.fy, kf.camera.cx, kf.camera.cy
                
                edge.set_vertex(0, v_p)
                edge.set_vertex(1, graph_keyframes[kf])
                opt.add_edge(edge)
                
                graph_edges[edge] = (p, kf, p_idx, is_stereo_obs)
                num_edges += 1

        if verbose:
            opt.set_verbose(True)

        if abort_flag.value:
            print('lba_optimization_process - aborting optimization')
            result_dict['mean_squared_error'] = -1
            result_dict['perc_bad_observations'] = 0        
            return

        # Initial optimization
        opt.initialize_optimization()
        opt.optimize(5)

        chi2Mono = 5.991 # chi-square 2 DOFs
        chi2Stereo = 7.815 # chi-square 3 DOFs

        if not abort_flag.value:

            # check inliers observation 
            for edge, edge_data in graph_edges.items(): 
                p, kf, p_idx, is_stereo = edge_data
                
                # if p.is_bad: # redundant check since the considered points come from good_points
                #     continue 
                
                edge_chi2 = edge.chi2()
                chi2_check_failure = edge_chi2 > chi2Mono if not is_stereo else edge_chi2 > chi2Stereo
                if chi2_check_failure or not edge.is_depth_positive():
                    edge.set_level(1)
                    num_bad_edges += 1
                edge.set_robust_kernel(None)

            # optimize again without outliers 
            opt.initialize_optimization()
            opt.optimize(rounds)

        # search for final outlier observations and clean map  
        num_bad_observations = 0  # final bad observations
        outliers_edge_data = []
        
        chi2_limits = {True: chi2Stereo, False: chi2Mono}
            
        for edge, edge_data in graph_edges.items(): 
            p, kf, p_idx, is_stereo = edge_data
            
            # if p.is_bad: # redundant check since the considered points come from good_points
            #     continue         
            
            assert(kf.get_point_match(p_idx) is p) 
            
            if edge.chi2() > chi2_limits[is_stereo] or not edge.is_depth_positive():         
                num_bad_observations += 1
                outliers_edge_data.append(edge_data)       

        # if map_lock is None: 
        #     map_lock = RLock() # put a fake lock 
            
        # with map_lock:      
        #     # remove outlier observations 
        #     for p, kf, p_idx, is_stereo in outliers_edge_data:
        #         p_f = kf.get_point_match(p_idx)
        #         if p_f is not None:
        #             assert(p_f is p)
        #             p.remove_observation(kf,p_idx)
        #             # the following instruction is now included in p.remove_observation()
        #             #f.remove_point(p)   # it removes multiple point instances (if these are present)   
        #             #f.remove_point_match(p_idx) # this does not remove multiple point instances, but now there cannot be multiple instances any more

        #     # put frames back
        #     for kf in graph_keyframes:
        #         est = graph_keyframes[kf].estimate()
        #         #R = est.rotation().matrix()
        #         #t = est.translation()
        #         #f.update_pose(poseRt(R, t))
        #         kf.update_pose(g2o.Isometry3d(est.orientation(), est.position()))

        #     # put points back
        #     if not fixed_points:
        #         for p in graph_points:
        #             p.update_position(np.array(graph_points[p].estimate()))
        #             p.update_normal_and_depth(force=True)

        active_edges = num_edges-num_bad_edges
        mean_squared_error = opt.active_chi2()/max(active_edges,1)
        
        # Final results: keyframe poses and point positions
        keyframe_poses = {kf.kid: g2o.Isometry3d(graph_keyframes[kf].estimate().orientation(), graph_keyframes[kf].estimate().position()).matrix() for kf in graph_keyframes}
        point_positions = {p.id: graph_points[p].estimate() for p in graph_points}

        outliers_edge_data_out = [(p_idx, kf.kid) for p, kf, p_idx, is_stereo in outliers_edge_data]
        
        result_dict['keyframe_poses'] = keyframe_poses
        result_dict['point_positions'] = point_positions
        result_dict['outliers_edge_data_out'] = outliers_edge_data_out
        result_dict['mean_squared_error'] = mean_squared_error
        result_dict['perc_bad_observations'] = num_bad_observations/max(num_edges,1)
        print('lba_optimization_process - completed')
    except Exception as e:
        
        Printer.red(f'lba_optimization_process - error: {e}')
        result_dict['mean_squared_error'] = -1
        result_dict['perc_bad_observations'] = 0   
        
def local_bundle_adjustment_parallel(keyframes, points, keyframes_ref=[], fixed_points=False, verbose=False, rounds=10, abort_flag=g2o.Flag(), map_lock=None):
    # good_keyframes = [kf for kf in keyframes if not kf.is_bad] + \
    #                  [kf for kf in keyframes_ref if not kf.is_bad]
    
    good_keyframes = {kf.kid: kf for kf in keyframes if not kf.is_bad}
    good_keyframes.update({kf.kid: kf for kf in keyframes_ref if not kf.is_bad})    
        
    #good_points = [p for p in points if p is not None and not p.is_bad and any(f in keyframes for f in p.keyframes())]
    good_points = {p.id: p for p in points if p is not None and not p.is_bad and any(f in keyframes for f in p.keyframes())}
    
    # Use Manager to store the results from the subprocess
    manager = mp.Manager()
    result_dict = manager.dict()

    # Start the optimization process
    p = mp.Process(target=lba_optimization_process, args=(result_dict, good_keyframes, keyframes_ref, good_points, fixed_points, verbose, rounds, abort_flag, map_lock))
    p.start()
    p.join(timeout=0.5)
    print('local_bundle_adjustment_parallel - joined')
    
    if result_dict['mean_squared_error'] != -1:
        # Extract the keyframe poses and point positions
        keyframe_poses = result_dict['keyframe_poses']
        point_positions = result_dict['point_positions']
        outliers_edge_data_out = result_dict['outliers_edge_data_out']

        # Update the main process map with the new poses and point positions
        if map_lock is None:
            map_lock = RLock()

        with map_lock:
            
            # remove outlier observations 
            for p_idx, kf_kid in outliers_edge_data_out:
                kf = good_keyframes[kf_kid]
                p_f = kf.get_point_match(p_idx)
                if p_f is not None:
                    assert(p_f.id == p_idx)
                    p_f.remove_observation(kf,p_idx)
                    # the following instruction is now included in p.remove_observation()
                    #f.remove_point(p)   # it removes multiple point instances (if these are present)   
                    #f.remove_point_match(p_idx) # this does not remove multiple point instances, but now there cannot be multiple instances any more
            
            # put frames back        
            for kf in good_keyframes.values():
                if kf.kid in keyframe_poses:
                    kf.update_pose(keyframe_poses[kf.kid])

            # put points back
            if not fixed_points:
                for p in good_points.values():
                    if p is not None and p.id in point_positions:
                        p.update_position(np.array(point_positions[p.id]))
                        p.update_normal_and_depth(force=True)

        # Return success indicator
        return result_dict['mean_squared_error'], result_dict['perc_bad_observations']
    else: 
        Printer.red(f'local_bundle_adjustment_parallel - error: {result_dict}')
        return -1, 0

