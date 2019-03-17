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

import g2o

from geom_helpers import poseRt
from frame import Frame
from helpers import Printer


# ------------------------------------------------------------------------------------------

# optimize pixel reprojection error, bundle adjustment
def optimization(frames, points, local_window, fixed_points=False, verbose=False, rounds=40, use_robust_kernel=False):
    if local_window is None:
        local_frames = frames
    else:
        local_frames = frames[-local_window:]

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    thHuberMono = math.sqrt(5.991);  # chi-square 2 DOFS 

    graph_frames, graph_points = {}, {}

    # add frame vertices to graph
    for f in (local_frames if fixed_points else frames):    # if points are fixed then consider just the local frames, otherwise we need all frames or at least two frames for each point
        #print('adding vertex frame ', f.id, ' to graph')
        pose = f.pose
        se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(f.id * 2)  # even ids
        v_se3.set_fixed(f.id < 1 or f not in local_frames)
        opt.add_vertex(v_se3)

        # confirm pose correctness
        #est = v_se3.estimate()
        #assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        #assert np.allclose(pose[0:3, 3], est.translation())

        graph_frames[f] = v_se3

    # add point vertices to graph 
    for p in points:
        if p.is_bad and not fixed_points:
            continue
        if not any([f in local_frames for f in p.frames]):  
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
        for f, idx in zip(p.frames, p.idxs):
            if f not in graph_frames:
                continue
            #print('adding edge between point ', p.id,' and frame ', f.id)
            edge = g2o.EdgeSE3ProjectXYZ()
            edge.set_vertex(0, v_p)
            edge.set_vertex(1, graph_frames[f])
            edge.set_measurement(f.kpsu[idx])
            invSigma2 = Frame.detector.inv_level_sigmas2[f.octaves[idx]]
            edge.set_information(np.eye(2)*invSigma2)
            if use_robust_kernel:
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

            edge.fx = f.fx 
            edge.fy = f.fy
            edge.cx = f.cx
            edge.cy = f.cy

            opt.add_edge(edge)

    if verbose:
        opt.set_verbose(True)
    opt.initialize_optimization()
    opt.optimize(rounds)

    # put frames back
    for f in graph_frames:
        est = graph_frames[f].estimate()
        R = est.rotation().matrix()
        t = est.translation()
        f.pose = poseRt(R, t)

    # put points back
    if not fixed_points:
        for p in graph_points:
            p.pt = np.array(graph_points[p].estimate())

    return opt.active_chi2()


# ------------------------------------------------------------------------------------------

# optimize pixel reprojection error:
# frame pose is optimized 
# 3D points observed in frame are fixed
def poseOptimization(frame, verbose=False, rounds=10):

    is_ok = True 

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    #robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))  # chi-square 2 DOFs
    thHuberMono = math.sqrt(5.991);  # chi-square 2 DOFS 

    point_edge_pairs = {}
    num_point_edges = 0

    se3 = g2o.SE3Quat(frame.pose[0:3, 0:3], frame.pose[0:3, 3])
    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_estimate(se3)
    v_se3.set_id(0)  
    v_se3.set_fixed(False)
    opt.add_vertex(v_se3)

    # add point vertices to graph 
    for idx, p in enumerate(frame.points):
        if p is None:  # do not use p.is_bad here since a single point observation is ok for pose optimization 
            continue

        frame.outliers[idx] = False 

        # add edge
        #print('adding edge between point ', p.id,' and frame ', frame.id)
        edge = g2o.EdgeSE3ProjectXYZOnlyPose()

        edge.set_vertex(0, opt.vertex(0))
        edge.set_measurement(frame.kpsu[idx])
        invSigma2 = Frame.detector.inv_level_sigmas2[frame.octaves[idx]]
        edge.set_information(np.eye(2)*invSigma2)
        edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

        edge.fx = frame.fx 
        edge.fy = frame.fy
        edge.cx = frame.cx
        edge.cy = frame.cy
        edge.Xw = p.pt[0:3]
        
        opt.add_edge(edge)

        point_edge_pairs[p] = (edge, idx) # one edge per point 
        num_point_edges += 1

    if num_point_edges < 3:
        Printer.red('poseOptimization: not enough correspondences!') 
        is_ok = False 
        return 0, is_ok, 0

    if verbose:
        opt.set_verbose(True)

    # We perform 4 optimizations, after each optimization we classify observation as inlier/outlier
    # At the next optimization, outliers are not included, but at the end they can be classified as inliers again.
    chi2Mono = 5.991 # chi-square 2 DOFs
    num_bad_points = 0

    for it in range(4):
        opt.initialize_optimization()        
        opt.optimize(rounds)

        num_bad_points = 0

        for p, edge_pair in point_edge_pairs.items(): 
            if frame.outliers[edge_pair[1]] is True:
                edge_pair[0].compute_error()

            chi2 = edge_pair[0].chi2()
            if chi2 > chi2Mono:
                frame.outliers[edge_pair[1]] = True 
                edge_pair[0].set_level(1)
                num_bad_points +=1
            else:
                frame.outliers[edge_pair[1]] = False
                edge_pair[0].set_level(0)                                

            if it == 2:
                edge_pair[0].set_robust_kernel(None)

        if len(opt.edges()) < 10:
            Printer.red('poseOptimization: stopped - not enough edges!')   
            is_ok = False           
            break                 
    
    print('pose optimization: initial ', num_point_edges, ' points, found ', num_bad_points, ' bad points')     
    if num_point_edges == num_bad_points:
        Printer.red('poseOptimization: all the initial correspondences are bad!')           
        is_ok = False      

    # update pose estimation
    if is_ok is True: 
        est = v_se3.estimate()
        R = est.rotation().matrix()
        t = est.translation()
        frame.pose = poseRt(R, t)

    num_valid_points = num_point_edges - num_bad_points

    return opt.active_chi2(), is_ok, num_valid_points


# ------------------------------------------------------------------------------------------

# locally optimize pixel reprojection error, bundle adjustment
# frames, points are optimized
# frames_ref are fixed 
def localOptimization(frames, points, frames_ref=[], fixed_points=False, verbose=False, rounds=10):

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(solver)
    opt.set_algorithm(solver)

    #robust_kernel = g2o.RobustKernelHuber(np.sqrt(5.991))  # chi-square 2 DOFs
    thHuberMono = math.sqrt(5.991);  # chi-square 2 DOFS 

    graph_frames, graph_points = {}, {}

    all_frames = frames + frames_ref

    # add frame vertices to graph
    for f in all_frames:    
        #print('adding vertex frame ', f.id, ' to graph')
        pose = f.pose
        se3 = g2o.SE3Quat(pose[0:3, 0:3], pose[0:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(f.id * 2)  # even ids
        v_se3.set_fixed(f.id < 1 or f in frames_ref)
        opt.add_vertex(v_se3)
        graph_frames[f] = v_se3        
        # confirm pose correctness
        #est = v_se3.estimate()
        #assert np.allclose(pose[0:3, 0:3], est.rotation().matrix())
        #assert np.allclose(pose[0:3, 3], est.translation())

    graph_edges = {}
    num_point_edges = 0

    # add point vertices to graph 
    for p in points:
        assert(p is not None)
        if p.is_bad and not fixed_points:  # do not consider bad points unless they are fixed 
            continue
        if not any([f in frames for f in p.frames]):  # this is redundant now 
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
        for f, p_idx in zip(p.frames, p.idxs):
            assert(f.points[p_idx] == p)
            if f not in graph_frames:
                continue
            #print('adding edge between point ', p.id,' and frame ', f.id)
            edge = g2o.EdgeSE3ProjectXYZ()
            edge.set_vertex(0, v_p)
            edge.set_vertex(1, graph_frames[f])
            edge.set_measurement(f.kpsu[p_idx])
            invSigma2 = Frame.detector.inv_level_sigmas2[f.octaves[p_idx]]
            edge.set_information(np.eye(2)*invSigma2)
            edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

            edge.fx = f.fx 
            edge.fy = f.fy
            edge.cx = f.cx
            edge.cy = f.cy

            opt.add_edge(edge)

            graph_edges[edge] = (p,f,p_idx) # f.points[p_idx] == p
            num_point_edges += 1            

    if verbose:
        opt.set_verbose(True)

    # initial optimization 
    opt.initialize_optimization()
    opt.optimize(5)

    chi2Mono = 5.991 # chi-square 2 DOFs

    # check inliers observation 
    for edge, edge_data in graph_edges.items(): 
        p = edge_data[0]
        if p.is_bad is True:
            continue 
        if edge.chi2() > chi2Mono or not edge.is_depth_positive():
            edge.set_level(1)
        edge.set_robust_kernel(None)

    # optimize again without outliers 
    opt.initialize_optimization()
    opt.optimize(rounds)

    # clean map observations 
    num_bad_observations = 0 
    outliers_data = []
    for edge, edge_data in graph_edges.items(): 
        p, f, p_idx = edge_data
        if p.is_bad is True:
            continue         
        assert(f.points[p_idx] == p) 
        if edge.chi2() > chi2Mono or not edge.is_depth_positive():         
            num_bad_observations += 1
            outliers_data.append( (p,f,p_idx) )       

    for d in outliers_data:
        (p,f,p_idx) = d
        assert(f.points[p_idx] == p)
        p.remove_observation(f,p_idx)
        f.remove_point(p)      

    # put frames back
    for f in graph_frames:
        est = graph_frames[f].estimate()
        R = est.rotation().matrix()
        t = est.translation()
        f.pose = poseRt(R, t)

    # put points back
    if not fixed_points:
        for p in graph_points:
            p.pt = np.array(graph_points[p].estimate())

    return opt.active_chi2(), num_bad_observations/num_point_edges