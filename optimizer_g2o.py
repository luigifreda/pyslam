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

import g2o

from utils_geom import poseRt
from frame import Frame
from utils import Printer
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

    thHuberMono = math.sqrt(5.991);  # chi-square 2 DOFS 

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
            edge = g2o.EdgeSE3ProjectXYZ()
            edge.set_vertex(0, v_p)
            edge.set_vertex(1, graph_keyframes[kf])
            edge.set_measurement(kf.kpsu[idx])
            invSigma2 = Frame.feature_manager.inv_level_sigmas2[kf.octaves[idx]]
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

    #robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))  # chi-square 2 DOFs
    thHuberMono = math.sqrt(5.991);  # chi-square 2 DOFS 

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

            # add edge
            #print('adding edge between point ', p.id,' and frame ', frame.id)
            edge = g2o.EdgeSE3ProjectXYZOnlyPose()

            edge.set_vertex(0, opt.vertex(0))
            edge.set_measurement(frame.kpsu[idx])
            invSigma2 = Frame.feature_manager.inv_level_sigmas2[frame.octaves[idx]]
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
            
            if chi2 > chi2Mono:
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

    #robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))  # chi-square 2 DOFs
    thHuberMono = math.sqrt(5.991);  # chi-square 2 DOFS 

    graph_keyframes, graph_points = {}, {}

    # add frame vertices to graph
    for kf in keyframes:    
        if kf.is_bad:
            continue 
        #print('adding vertex frame ', f.id, ' to graph')
        se3 = g2o.SE3Quat(kf.Rcw, kf.tcw)
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
        v_se3.set_fixed(kf.kid==0)  # (use f.kid here!)
        opt.add_vertex(v_se3)
        graph_keyframes[kf] = v_se3     
           
        # confirm pose correctness
        #est = v_se3.estimate()
        #assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        #assert np.allclose(pose[0:3, 3], est.translation())
        
    # add reference frame vertices to graph
    for kf in keyframes_ref:    
        if kf.is_bad:
            continue 
        #print('adding vertex frame ', f.id, ' to graph')
        se3 = g2o.SE3Quat(kf.Rcw, kf.tcw)
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
        v_se3.set_fixed(True)
        opt.add_vertex(v_se3)
        graph_keyframes[kf] = v_se3             

    graph_edges = {}
    num_edges = 0
    num_bad_edges = 0

    # add point vertices to graph 
    for p in points:
        assert(p is not None)
        if p.is_bad:  # do not consider bad points             
            continue  
        if not any([f in keyframes for f in p.keyframes()]):  
            Printer.orange('point %d without a viewing keyframe in input keyframes!!' %(p.id))
            #Printer.orange('         keyframes: ',p.observations_string())
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
        for kf, p_idx in p.observations():
            if kf.is_bad:
                continue 
            if kf not in graph_keyframes:
                continue
            if __debug__:      
                p_f = kf.get_point_match(p_idx)
                if p_f != p:
                    print('frame: ', kf.id, ' missing point ', p.id, ' at index p_idx: ', p_idx)                    
                    if p_f is not None:
                        print('p_f:', p_f)
                    print('p:',p)
            assert(kf.get_point_match(p_idx) is p)            
            #print('adding edge between point ', p.id,' and frame ', f.id)
            edge = g2o.EdgeSE3ProjectXYZ()
            edge.set_vertex(0, v_p)
            edge.set_vertex(1, graph_keyframes[kf])
            edge.set_measurement(kf.kpsu[p_idx])
            invSigma2 = Frame.feature_manager.inv_level_sigmas2[kf.octaves[p_idx]]
            edge.set_information(np.eye(2)*invSigma2)
            edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

            edge.fx = kf.camera.fx 
            edge.fy = kf.camera.fy
            edge.cx = kf.camera.cx
            edge.cy = kf.camera.cy

            opt.add_edge(edge)

            graph_edges[edge] = (p,kf,p_idx) # one has kf.points[p_idx] == p
            num_edges += 1            

    if verbose:
        opt.set_verbose(True)

    if abort_flag.value:
        return -1,0

    # initial optimization 
    opt.initialize_optimization()
    opt.optimize(5)
    
    if not abort_flag.value:
        chi2Mono = 5.991 # chi-square 2 DOFs

        # check inliers observation 
        for edge, edge_data in graph_edges.items(): 
            p = edge_data[0]
            if p.is_bad:
                continue 
            if edge.chi2() > chi2Mono or not edge.is_depth_positive():
                edge.set_level(1)
                num_bad_edges += 1
            edge.set_robust_kernel(None)

        # optimize again without outliers 
        opt.initialize_optimization()
        opt.optimize(rounds)

    # search for final outlier observations and clean map  
    num_bad_observations = 0  # final bad observations
    outliers_data = []
    for edge, edge_data in graph_edges.items(): 
        p, kf, p_idx = edge_data
        if p.is_bad:
            continue         
        assert(kf.get_point_match(p_idx) is p) 
        if edge.chi2() > chi2Mono or not edge.is_depth_positive():         
            num_bad_observations += 1
            outliers_data.append(edge_data)       

    if map_lock is None: 
        map_lock = RLock() # put a fake lock 
        
    with map_lock:      
        # remove outlier observations 
        for d in outliers_data:
            p, kf, p_idx = d
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
    mean_squared_error = opt.active_chi2()/active_edges

    return mean_squared_error, num_bad_observations/max(num_edges,1)