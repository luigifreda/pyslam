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
import cv2

import threading
import multiprocessing as mp
import traceback

from pyslam.config_parameters import (
    Parameters,
)  # TODO(dvdmc): do we want parameters to be used in this file?
import g2o

from pyslam.semantics.semantic_mapping_shared import (
    SemanticMappingShared,
)  # TODO(dvdmc): do we want semnatics to be used in this file?
from pyslam.utilities.geometry import poseRt

from pyslam.utilities.logging import Printer
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.drawing import draw_histogram

from .sim3_pose import Sim3Pose
from .feature_tracker_shared import FeatureTrackerShared
from .map_point import MapPoint
from .keyframe import KeyFrame


# ------------------------------------------------------------------------------------------


# Continuously syncs the g2o.Flag with the shared boolean variable.
def sync_flag_fun(abort_flag, mp_abort_flag, print=print):
    print("sync_flag_fun: starting...")
    try:
        while not abort_flag.value:
            # If the flag's value doesn't match the shared_bool, update it
            if mp_abort_flag.value != abort_flag.value:
                abort_flag.value = mp_abort_flag.value
                print(f"sync_flag_fun: Flag updated to: {abort_flag.value}")
            time.sleep(0.003)
        print("sync_flag_fun: done...")
    except Exception as e:
        print(f"sync_flag_fun: EXCEPTION: {e}")
        traceback_details = traceback.format_exc()
        print(f"\t traceback details: {traceback_details}")


# ------------------------------------------------------------------------------------------


# optimize pixel reprojection error, bundle adjustment
def bundle_adjustment(
    keyframes,
    points,
    local_window_size,
    fixed_points=False,
    rounds=10,
    loop_kf_id=0,
    use_robust_kernel=False,
    abort_flag=None,
    mp_abort_flag=None,
    result_dict=None,
    verbose=False,
    print=print,
):
    """
    Optimize pixel reprojection error, bundle adjustment.
    Returns:
    - mean_squared_error
    - result_dict: filled dictionary with the updates of the keyframes and points if provided in the input
    """
    if local_window_size is None:
        local_frames = keyframes
    else:
        local_frames = keyframes[-local_window_size:]

    robust_rounds = rounds // 2 if use_robust_kernel else 0
    final_rounds = rounds - robust_rounds
    print(
        f"bundle_adjustment: rounds: {rounds}, robust_rounds: {robust_rounds}, final_rounds: {final_rounds}"
    )

    if abort_flag is None:
        abort_flag = g2o.Flag()

    sync_flag_thread = None
    if mp_abort_flag is not None:
        # Create a thread for keeping abort_flag in sync with mp_abort_flag.
        # Why? The g2o-abort-flag passed (via pickling) to a launched parallel process (via multiprocessing module) is just a
        # different instance that is not kept in sync with its source instance in the parent process. This means we don't
        # succeed to abort the BA when set the source instance.
        sync_flag_thread = threading.Thread(
            target=sync_flag_fun, args=(abort_flag, mp_abort_flag, print)
        )
        sync_flag_thread.daemon = True  # Daemonize thread so it exits when the main thread does
        sync_flag_thread.start()

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)
    opt.set_force_stop_flag(abort_flag)

    thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS
    thHuberStereo = math.sqrt(7.815)  # chi-square 3 DOFS

    graph_keyframes, graph_points, graph_edges = {}, {}, {}

    # add frame vertices to graph
    for kf in (
        local_frames if fixed_points else keyframes
    ):  # if points are fixed then consider just the local frames, otherwise we need all frames or at least two frames for each point
        if kf.is_bad():
            continue
        # print('adding vertex frame ', f.id, ' to graph')
        kf_Tcw = kf.Tcw()
        se3 = g2o.SE3Quat(kf_Tcw[:3, :3], kf_Tcw[:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
        v_se3.set_fixed(kf.kid == 0 or kf not in local_frames)  # (use f.kid here!)
        opt.add_vertex(v_se3)

        graph_keyframes[kf] = v_se3

    num_edges = 0
    num_bad_edges = 0

    inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2
    eye2 = np.eye(2)
    eye3 = np.eye(3)

    # add point vertices to graph
    for p in points:
        if p is None or p.is_bad():  # do not consider bad points
            continue
        # if __debug__:
        #     if not any([f in keyframes for f in p.keyframes()]):
        #         Printer.red('point without a viewing frame!!')
        #         continue
        # print('adding vertex point ', p.id,' to graph')
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(p.id * 2 + 1)  # odd ids
        v_p.set_estimate(p.pt()[0:3].copy())
        v_p.set_marginalized(True)
        v_p.set_fixed(fixed_points)
        opt.add_vertex(v_p)
        graph_points[p] = v_p

        # add edges
        for kf, idx in p.observations():
            # if kf.is_bad():  # redundant since we check kf is in graph_keyframes (selected as non-bad)
            #     continue
            try:
                v_se3_kf = graph_keyframes[kf]
            except KeyError:
                continue

            kf_kpsu_idx = kf.kpsu[idx]
            kf_kps_ur_idx = kf.kps_ur[idx] if kf.kps_ur is not None else -1

            # print('adding edge between point ', p.id,' and frame ', f.id)
            is_stereo_obs = kf_kps_ur_idx >= 0
            invSigma2 = inv_level_sigmas2[kf.octaves[idx]]

            if Parameters.kUseSemanticsInOptimization and kf.kps_sem is not None:
                invSigma2 *= SemanticMappingShared.get_semantic_weight(kf.kps_sem[idx])

            camera = kf.camera

            if is_stereo_obs:
                edge = g2o.EdgeStereoSE3ProjectXYZ()
                edge.set_vertex(0, v_p)
                edge.set_vertex(1, v_se3_kf)
                obs = [kf_kpsu_idx[0], kf_kpsu_idx[1], kf_kps_ur_idx]
                edge.set_measurement(obs)

                edge.set_information(eye3 * invSigma2)
                if use_robust_kernel:
                    edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberStereo))

                edge.fx = camera.fx
                edge.fy = camera.fy
                edge.cx = camera.cx
                edge.cy = camera.cy
                edge.bf = camera.bf
            else:
                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_vertex(0, v_p)
                edge.set_vertex(1, v_se3_kf)
                edge.set_measurement(kf_kpsu_idx)

                edge.set_information(eye2 * invSigma2)
                if use_robust_kernel:
                    edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

                edge.fx = camera.fx
                edge.fy = camera.fy
                edge.cx = camera.cx
                edge.cy = camera.cy

            opt.add_edge(edge)
            graph_edges[edge] = is_stereo_obs
            num_edges += 1

    if abort_flag.value:
        return -1, result_dict

    if verbose:
        opt.set_verbose(True)

    chi2Mono = 5.991  # chi-square 2 DOFs
    chi2Stereo = 7.815  # chi-square 3 DOFs

    opt.initialize_optimization()
    opt.compute_active_errors()
    initial_mean_squared_error = opt.active_chi2() / max(num_edges, 1)

    if robust_rounds > 0:
        opt.optimize(robust_rounds)
        # check inliers observation
        for edge, is_stereo in graph_edges.items():

            edge_chi2 = edge.chi2()
            chi2_check_failure = (edge_chi2 > chi2Stereo) if is_stereo else (edge_chi2 > chi2Mono)
            if chi2_check_failure or not edge.is_depth_positive():
                edge.set_level(1)
                num_bad_edges += 1
            edge.set_robust_kernel(None)

    if abort_flag.value:
        return -1, result_dict

    opt.initialize_optimization()
    opt.optimize(final_rounds)

    # shut down the sync thread if used and still running
    if sync_flag_thread is not None and sync_flag_thread.is_alive():
        abort_flag.value = True  # force the sync thread to exit
        sync_flag_thread.join()  # timeout=0.005)

    # if result_dict is not None then fill in the result dictionary
    # instead of changing the keyframes and points
    keyframe_updates = None
    point_updates = None
    if result_dict is not None:
        keyframe_updates, point_updates = {}, {}

    # put frames back
    if keyframe_updates is not None:
        # store the updates in a dictionary
        for kf, v_se3 in graph_keyframes.items():
            est = v_se3.estimate()
            R, t = est.rotation().matrix(), est.translation()
            T = poseRt(R, t)
            keyframe_updates[kf.id] = T
    else:
        for kf, v_se3 in graph_keyframes.items():
            est = v_se3.estimate()
            R, t = est.rotation().matrix(), est.translation()
            T = poseRt(R, t)
            if loop_kf_id == 0:
                # direct update on map
                kf.update_pose(T)
            else:
                # update for loop closure
                kf.Tcw_GBA = T
                kf.is_Tcw_GBA_valid = True
                kf.GBA_kf_id = loop_kf_id

    # put points back
    if not fixed_points:
        if point_updates is not None:
            # store the updates in a dictionary
            for p, v_p in graph_points.items():
                point_updates[p.id] = np.array(v_p.estimate())
        else:
            if loop_kf_id == 0:
                for p, v_p in graph_points.items():
                    # direct update on map
                    p.update_position(np.array(v_p.estimate()))
                    p.update_normal_and_depth(force=True)
            else:
                for p, v_p in graph_points.items():
                    # update for loop closure
                    p.pt_GBA = np.array(v_p.estimate())
                    p.is_pt_GBA_valid = True
                    p.GBA_kf_id = loop_kf_id

    num_active_edges = num_edges - num_bad_edges
    mean_squared_error = opt.active_chi2() / max(num_active_edges, 1)

    print(
        f"bundle_adjustment: mean_squared_error: {mean_squared_error}, initial_mean_squared_error: {initial_mean_squared_error}, num_edges: {num_edges},  num_bad_edges: {num_bad_edges} (perc: {num_bad_edges/num_edges*100:.2f}%)"
    )

    if result_dict is not None:
        result_dict["keyframe_updates"] = keyframe_updates
        result_dict["point_updates"] = point_updates

    return mean_squared_error, result_dict


# ------------------------------------------------------------------------------------------


def global_bundle_adjustment(
    keyframes,
    points,
    rounds=10,
    loop_kf_id=0,
    use_robust_kernel=False,
    abort_flag=None,
    mp_abort_flag=None,
    result_dict=None,
    verbose=False,
    print=print,
):
    fixed_points = False
    mean_squared_error, result_dict = bundle_adjustment(
        keyframes,
        points,
        local_window_size=None,
        fixed_points=fixed_points,
        rounds=rounds,
        loop_kf_id=loop_kf_id,
        use_robust_kernel=use_robust_kernel,
        abort_flag=abort_flag,
        mp_abort_flag=mp_abort_flag,
        result_dict=result_dict,
        verbose=verbose,
        print=print,
    )
    return mean_squared_error, result_dict


def global_bundle_adjustment_map(
    map,
    rounds=10,
    loop_kf_id=0,
    use_robust_kernel=False,
    abort_flag=None,
    mp_abort_flag=None,
    result_dict=None,
    verbose=False,
    print=print,
):
    # fixed_points=False
    keyframes = map.get_keyframes()
    points = map.get_points()
    return global_bundle_adjustment(
        keyframes=keyframes,
        points=points,
        rounds=rounds,
        loop_kf_id=loop_kf_id,
        use_robust_kernel=use_robust_kernel,
        abort_flag=abort_flag,
        mp_abort_flag=mp_abort_flag,
        result_dict=result_dict,
        verbose=verbose,
        print=print,
    )


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
    # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverDenseSE3())
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)

    thHuberMono = math.sqrt(5.991)  # chi-squared 2 DOFS
    thHuberStereo = math.sqrt(7.815)  # chi-squared 3 DOFS

    edge_data = []
    num_point_edges = 0

    frame_Tcw = frame.Tcw()
    Rcw = frame_Tcw[:3, :3]
    tcw = frame_Tcw[:3, 3]

    v_se3 = g2o.VertexSE3Expmap()
    v_se3.set_estimate(g2o.SE3Quat(Rcw.copy(), tcw.copy()))
    v_se3.set_id(0)
    v_se3.set_fixed(False)
    opt.add_vertex(v_se3)

    inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2
    eye2 = np.eye(2)
    eye3 = np.eye(3)

    fx = frame.camera.fx
    fy = frame.camera.fy
    cx = frame.camera.cx
    cy = frame.camera.cy
    bf = frame.camera.bf

    with MapPoint.global_lock:
        # add point vertices to graph
        for idx, p in enumerate(frame.points):
            if p is None:
                continue

            frame_kpsu_idx = frame.kpsu[idx]
            frame_kps_ur_idx = frame.kps_ur[idx] if frame.kps_ur is not None else -1

            # reset outlier flag
            frame.outliers[idx] = False
            is_stereo_obs = frame_kps_ur_idx >= 0

            # add edge
            edge = None
            invSigma2 = inv_level_sigmas2[frame.octaves[idx]]

            if Parameters.kUseSemanticsInOptimization and frame.kps_sem is not None:
                invSigma2 *= SemanticMappingShared.get_semantic_weight(frame.kps_sem[idx])

            if is_stereo_obs:
                # print('adding stereo edge between point ', p.id,' and frame ', frame.id)
                edge = g2o.EdgeStereoSE3ProjectXYZOnlyPose()
                obs = [frame_kpsu_idx[0], frame_kpsu_idx[1], frame_kps_ur_idx]  # u,v,ur
                edge.set_vertex(0, v_se3)  # opt.vertex(0))
                edge.set_measurement(obs)

                edge.set_information(eye3 * invSigma2)
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberStereo))

                edge.fx = fx
                edge.fy = fy
                edge.cx = cx
                edge.cy = cy
                edge.bf = bf
                edge.Xw = p.pt()[0:3]
            else:
                # print('adding mono edge between point ', p.id,' and frame ', frame.id)
                edge = g2o.EdgeSE3ProjectXYZOnlyPose()

                edge.set_vertex(0, v_se3)  # opt.vertex(0))
                edge.set_measurement(frame_kpsu_idx)

                edge.set_information(eye2 * invSigma2)
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

                edge.fx = fx
                edge.fy = fy
                edge.cx = cx
                edge.cy = cy
                edge.Xw = p.pt()[0:3]

            opt.add_edge(edge)

            edge_data.append((edge, idx, is_stereo_obs))  # one edge per point
            num_point_edges += 1

    if num_point_edges < 3:
        Printer.red("pose_optimization: not enough correspondences!")
        is_ok = False
        return 0, is_ok, 0

    if verbose:
        opt.set_verbose(True)

    # perform 4 optimizations:
    # after each optimization we classify observation as inlier/outlier;
    # at the next optimization, outliers are not included, but at the end they can be classified as inliers again
    chi2Mono = 5.991  # chi-squared 2 DOFs
    chi2Stereo = 7.815  # chi-squared 3 DOFs
    num_bad_point_edges = 0

    for it in range(4):
        v_se3.set_estimate(g2o.SE3Quat(Rcw.copy(), tcw.copy()))
        opt.initialize_optimization()
        opt.optimize(rounds)

        num_bad_point_edges = 0

        for edge, idx, is_stereo_obs in edge_data:
            if frame.outliers[idx]:
                edge.compute_error()

            chi2 = edge.chi2()

            # is_stereo_obs = frame.kps_ur is not None and frame.kps_ur[idx]>=0

            chi2_check_failure = (chi2 > chi2Stereo) if is_stereo_obs else (chi2 > chi2Mono)
            if chi2_check_failure:
                frame.outliers[idx] = True
                edge.set_level(1)
                num_bad_point_edges += 1
            else:
                frame.outliers[idx] = False
                edge.set_level(0)

            if it == 2:
                edge.set_robust_kernel(None)

        if len(opt.edges()) < 10:
            Printer.red("pose_optimization: stopped - not enough edges!")
            # is_ok = False
            break

    print(
        f"pose optimization: available {num_point_edges} points, found {num_bad_point_edges} bad points"
    )
    num_valid_points = (
        num_point_edges - num_bad_point_edges
    )  # len([e for e in opt.edges() if e.level() == 0])
    if num_valid_points < 10:
        Printer.red("pose_optimization: not enough edges!")
        is_ok = False

    ratio_bad_points = num_bad_point_edges / max(num_point_edges, 1)
    if num_valid_points > 15 and ratio_bad_points > Parameters.kMaxOutliersRatioInPoseOptimization:
        Printer.red(
            f"pose_optimization: percentage of bad points is too high: {ratio_bad_points*100:.2f}%"
        )
        is_ok = False

    # update pose estimation
    if is_ok:
        est = v_se3.estimate()
        R = est.rotation().matrix()
        t = est.translation()
        frame.update_pose(poseRt(R, t))

    draw_chi2_histograms = False  # debug and visualization of chi2 values
    if draw_chi2_histograms:
        chi2_mono_vals = []
        chi2_stereo_vals = []
        for edge, idx, is_stereo_obs in edge_data:
            chi2 = edge.chi2()
            if is_stereo_obs:
                chi2_stereo_vals.append(chi2)
            else:
                chi2_mono_vals.append(chi2)

        # Draw and show histograms
        if chi2_mono_vals:
            hist_img_mono = draw_histogram(
                chi2_mono_vals,
                bins=10,
                delta=chi2Mono,
                min_value=0,
                max_value=chi2Mono * 10,
                color=(255, 0, 0),
            )
            cv2.imshow("Monocular chi2 errors", hist_img_mono)
        if chi2_stereo_vals:
            hist_img_stereo = draw_histogram(
                chi2_stereo_vals,
                bins=10,
                delta=chi2Stereo,
                min_value=0,
                max_value=chi2Stereo * 10,
                color=(0, 255, 0),
            )
            cv2.imshow("Stereo chi2 errors", hist_img_stereo)
        cv2.waitKey(1)

    # since we have only one frame here, each edge corresponds to a single distinct point
    # num_valid_points = num_point_edges - num_bad_point_edges
    mean_squared_error = opt.active_chi2() / max(num_valid_points, 1)

    return mean_squared_error, is_ok, num_valid_points


# ------------------------------------------------------------------------------------------


# local bundle adjustment (optimize points reprojection error)
# - frames and points are optimized
# - frames_ref are fixed
def local_bundle_adjustment(
    keyframes: list[KeyFrame],
    points: list[MapPoint],
    keyframes_ref: list[KeyFrame] | None = None,
    fixed_points: bool = False,
    verbose: bool = False,
    rounds: int = 10,
    abort_flag: g2o.Flag | None = None,
    mp_abort_flag=None,
    map_lock=None,
):
    """
    Local bundle adjustment (optimize points reprojection error)
    Returns:
    - mean_squared_error
    - ratio_bad_observations: percentage of bad observations
    """
    from .local_mapping import LocalMapping

    print = LocalMapping.print

    if abort_flag is None:
        abort_flag = g2o.Flag()

    if keyframes_ref is None:
        keyframes_ref = []

    # create g2o optimizer
    opt = g2o.SparseOptimizer()
    # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
    block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
    # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
    solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
    opt.set_algorithm(solver)
    opt.set_force_stop_flag(abort_flag)

    good_keyframes = [kf for kf in keyframes if not kf.is_bad()] + [
        kf for kf in keyframes_ref if not kf.is_bad()
    ]
    good_points = [
        p for p in points if p is not None and not p.is_bad()
    ]  # and any(f in keyframes for f in p.keyframes())]

    thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS
    thHuberStereo = math.sqrt(7.815)  # chi-square 3 DOFS

    graph_keyframes, graph_points, graph_edges = {}, {}, {}

    # add frame vertices to graph
    for kf in good_keyframes:
        # print('adding vertex frame ', f.id, ' to graph')
        kf_Tcw = kf.Tcw()
        se3 = g2o.SE3Quat(kf_Tcw[:3, :3], kf_Tcw[:3, 3])
        v_se3 = g2o.VertexSE3Expmap()
        v_se3.set_estimate(se3)
        v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
        v_se3.set_fixed(kf.kid == 0 or kf in keyframes_ref)  # (use f.kid here!)
        opt.add_vertex(v_se3)
        graph_keyframes[kf] = v_se3

    num_edges = 0
    num_bad_edges = 0

    eye2 = np.eye(2)
    eye3 = np.eye(3)

    inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2

    # add point vertices to graph
    # for p in points:
    for p in good_points:

        # assert(p is not None)
        # if p.is_bad():  # do not consider bad points
        #     continue
        # if not any([f in keyframes for f in p.keyframes()]):
        #     Printer.orange('point %d without a viewing keyframe in input keyframes!!' %(p.id))
        #     #Printer.orange('         keyframes: ',p.observations_string())
        #     continue

        # print('adding vertex point ', p.id,' to graph')
        v_p = g2o.VertexSBAPointXYZ()
        v_p.set_id(p.id * 2 + 1)  # odd ids
        v_p.set_estimate(p.pt()[0:3].copy())
        v_p.set_marginalized(True)
        v_p.set_fixed(fixed_points)
        opt.add_vertex(v_p)
        graph_points[p] = v_p

        for kf, p_idx in p.observations():
            if kf.is_bad():
                continue

            try:
                v_se3_kf = graph_keyframes[kf]
            except KeyError:
                continue

            # if __debug__:
            #     p_f = kf.get_point_match(p_idx)
            #     if p_f != p:
            #         print("frame: ", kf.id, " missing point ", p.id, " at index p_idx: ", p_idx)
            #         if p_f is not None:
            #             print("p_f:", p_f)
            #         print("p:", p)
            assert kf.get_point_match(p_idx) is p

            # print('adding edge between point ', p.id,' and frame ', f.id)
            kf_kpsu_p_idx = kf.kpsu[p_idx]
            kf_kps_ur_p_idx = kf.kps_ur[p_idx] if kf.kps_ur is not None else -1

            is_stereo_obs = kf_kps_ur_p_idx >= 0
            invSigma2 = inv_level_sigmas2[kf.octaves[p_idx]]

            if Parameters.kUseSemanticsInOptimization and kf.kps_sem is not None:
                invSigma2 *= SemanticMappingShared.get_semantic_weight(kf.kps_sem[p_idx])

            camera = kf.camera

            if is_stereo_obs:
                edge = g2o.EdgeStereoSE3ProjectXYZ()
                obs = [kf_kpsu_p_idx[0], kf_kpsu_p_idx[1], kf_kps_ur_p_idx]
                edge.set_measurement(obs)

                edge.set_information(eye3 * invSigma2)
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberStereo))

                edge.bf = camera.bf
            else:
                edge = g2o.EdgeSE3ProjectXYZ()
                edge.set_measurement(kf_kpsu_p_idx)

                edge.set_information(eye2 * invSigma2)
                edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

            edge.fx, edge.fy, edge.cx, edge.cy = camera.fx, camera.fy, camera.cx, camera.cy

            edge.set_vertex(0, v_p)
            edge.set_vertex(1, v_se3_kf)
            opt.add_edge(edge)

            graph_edges[edge] = (p, kf, p_idx, is_stereo_obs)  # one has kf.points[p_idx] == p
            num_edges += 1

    if verbose:
        opt.set_verbose(True)

    if abort_flag.value:
        return -1, 0

    print(
        f"local_bundle_adjustment: starting optimization with {len(graph_keyframes)} keyframes, {len(graph_points)} points, {num_edges} edges"
    )

    # initial optimization
    opt.initialize_optimization()
    opt.optimize(5)

    chi2Mono = 5.991  # chi-square 2 DOFs
    chi2Stereo = 7.815  # chi-square 3 DOFs

    if not abort_flag.value:

        # check inliers observation
        for edge, (p, kf, p_idx, is_stereo) in graph_edges.items():

            # if p.is_bad(): # redundant check since the considered points come from good_points
            #     continue

            edge_chi2 = edge.chi2()
            chi2_check_failure = (edge_chi2 > chi2Stereo) if is_stereo else (edge_chi2 > chi2Mono)
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

    for edge, (p, kf, p_idx, is_stereo) in graph_edges.items():

        # if p.is_bad(): # redundant check since the considered points come from good_points
        #     continue

        assert kf.get_point_match(p_idx) is p

        if edge.chi2() > chi2_limits[is_stereo] or not edge.is_depth_positive():
            num_bad_observations += 1
            outliers_edge_data.append((p, kf, p_idx, is_stereo))

    if map_lock is None:
        map_lock = threading.RLock()  # put a fake lock

    with map_lock:
        # remove outlier observations
        for p, kf, p_idx, is_stereo in outliers_edge_data:
            p_f = kf.get_point_match(p_idx)
            if p_f is not None:
                assert p_f is p
                p.remove_observation(kf, p_idx, map_no_lock=True)
                # the following instruction is now included in p.remove_observation()
                # f.remove_point(p)   # it removes multiple point instances (if these are present)
                # f.remove_point_match(p_idx) # this does not remove multiple point instances, but now there cannot be multiple instances any more

        # put frames back
        for kf, v_se3 in graph_keyframes.items():
            est = v_se3.estimate()
            R = est.rotation().matrix()
            t = est.translation()
            kf.update_pose(poseRt(R, t))
            kf.lba_count += 1

        # put points back
        if not fixed_points:
            for p, v_p in graph_points.items():
                p.update_position(np.array(v_p.estimate()))
                p.update_normal_and_depth(force=True)

    num_active_edges = num_edges - num_bad_edges
    mean_squared_error = opt.active_chi2() / max(num_active_edges, 1)

    return mean_squared_error, num_bad_observations / max(num_edges, 1)


# ------------------------------------------------------------------------------------------


# Parallel-process local bundle adjustment (optimize points reprojection error)
# - frames and points are optimized
# - frames_ref are fixed
# This function will handle the multiprocessing part of the optimization.
# it is launched by local_bundle_adjustment_parallel()
def lba_optimization_process(
    result_dict_queue,
    queue,
    good_keyframes,
    keyframes_ref,
    good_points,
    fixed_points,
    verbose,
    rounds,
    mp_abort_flag,
):
    from .local_mapping import LocalMapping

    print = LocalMapping.print
    try:
        print("lba_optimization_process: starting...")

        abort_flag = g2o.Flag(False)

        # Create a thread for keeping abort_flag in sync with mp_abort_flag.
        # Why? The g2o-abort-flag passed (via pickling) to this launched parallel process (via multiprocessing module) is just a different instance that is not kept in sync
        # with its source instance in the main/parent process. This means we don't succeed to abort the LBA when a new keyframes is pushed to local mapping.
        sync_flag_thread = threading.Thread(
            target=sync_flag_fun, args=(abort_flag, mp_abort_flag, print)
        )
        sync_flag_thread.daemon = True  # Daemonize thread so it exits when the main thread does
        sync_flag_thread.start()

        # create g2o optimizer
        opt = g2o.SparseOptimizer()
        # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCSparseSE3())
        block_solver = g2o.BlockSolverSE3(g2o.LinearSolverEigenSE3())
        # block_solver = g2o.BlockSolverSE3(g2o.LinearSolverCholmodSE3())
        solver = g2o.OptimizationAlgorithmLevenberg(block_solver)
        opt.set_algorithm(solver)
        opt.set_force_stop_flag(abort_flag)

        graph_keyframes, graph_points, graph_edges = {}, {}, {}

        # add frame vertices to graph
        for kf in good_keyframes.values():
            # print('adding vertex frame ', f.id, ' to graph')
            kf_Tcw = kf.Tcw()
            se3 = g2o.SE3Quat(kf_Tcw[:3, :3], kf_Tcw[:3, 3])
            v_se3 = g2o.VertexSE3Expmap()
            v_se3.set_estimate(se3)
            v_se3.set_id(kf.kid * 2)  # even ids  (use f.kid here!)
            v_se3.set_fixed(kf.kid == 0 or kf in keyframes_ref)  # (use f.kid here!)
            opt.add_vertex(v_se3)
            graph_keyframes[kf] = v_se3

        thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS
        thHuberStereo = math.sqrt(7.815)  # chi-square 3 DOFS

        num_edges = 0
        num_bad_edges = 0

        eye2 = np.eye(2)
        eye3 = np.eye(3)

        inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2

        # Add point vertices to the graph
        for p in good_points:
            v_p = g2o.VertexSBAPointXYZ()
            v_p.set_id(p.id * 2 + 1)
            v_p.set_estimate(p.pt()[0:3].copy())
            v_p.set_marginalized(True)
            v_p.set_fixed(fixed_points)
            opt.add_vertex(v_p)
            graph_points[p] = v_p

            # add edges
            for kf, p_idx in p.observations():
                if kf.is_bad():
                    continue
                try:
                    v_se3_kf = graph_keyframes[kf]
                except KeyError:
                    continue

                if __debug__:
                    p_f = kf.get_point_match(p_idx)
                    if p_f != p:
                        print("frame: ", kf.id, " missing point ", p.id, " at index p_idx: ", p_idx)
                        if p_f is not None:
                            print("p_f:", p_f)
                        print("p:", p)
                assert kf.get_point_match(p_idx) is p

                # print('adding edge between point ', p.id,' and frame ', f.id)

                kf_kpsu_p_idx = kf.kpsu[p_idx]
                kf_kps_ur_p_idx = kf.kps_ur[p_idx] if kf.kps_ur is not None else -1

                is_stereo_obs = kf_kps_ur_p_idx >= 0
                invSigma2 = inv_level_sigmas2[kf.octaves[p_idx]]

                if Parameters.kUseSemanticsInOptimization and kf.kps_sem is not None:
                    invSigma2 *= SemanticMappingShared.get_semantic_weight(kf.kps_sem[p_idx])

                if is_stereo_obs:
                    edge = g2o.EdgeStereoSE3ProjectXYZ()
                    obs = [kf_kpsu_p_idx[0], kf_kpsu_p_idx[1], kf_kps_ur_p_idx]
                    edge.set_measurement(obs)

                    edge.set_information(eye3 * invSigma2)
                    edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberStereo))

                    edge.bf = kf.camera.bf
                else:
                    edge = g2o.EdgeSE3ProjectXYZ()
                    edge.set_measurement(kf_kpsu_p_idx)

                    edge.set_information(eye2 * invSigma2)
                    edge.set_robust_kernel(g2o.RobustKernelHuber(thHuberMono))

                edge.fx, edge.fy, edge.cx, edge.cy = (
                    kf.camera.fx,
                    kf.camera.fy,
                    kf.camera.cx,
                    kf.camera.cy,
                )

                edge.set_vertex(0, v_p)
                edge.set_vertex(1, v_se3_kf)
                opt.add_edge(edge)

                graph_edges[edge] = (p, kf, p_idx, is_stereo_obs)
                num_edges += 1

        if verbose:
            opt.set_verbose(True)

        print("lba_optimization_process: created opt...")

        result_dict = {"mean_squared_error": -1, "ratio_bad_observations": 0}

        if abort_flag.value:
            print("lba_optimization_process - aborting optimization")
            result_dict_queue.put(result_dict)
            queue.put("aborted")
            return

        # Initial optimization
        opt.initialize_optimization()
        opt.optimize(5)

        print("lba_optimization_process: done optimize(5)...")

        chi2Mono = 5.991  # chi-square 2 DOFs
        chi2Stereo = 7.815  # chi-square 3 DOFs

        if not abort_flag.value:

            # check inliers observation
            for edge, (p, kf, p_idx, is_stereo) in graph_edges.items():
                # if p.is_bad(): # redundant check since the considered points come from good_points
                #     continue

                edge_chi2 = edge.chi2()
                chi2_check_failure = (
                    (edge_chi2 > chi2Stereo) if is_stereo else (edge_chi2 > chi2Mono)
                )
                if chi2_check_failure or not edge.is_depth_positive():
                    edge.set_level(1)
                    num_bad_edges += 1
                edge.set_robust_kernel(None)

            # optimize again without outliers
            opt.initialize_optimization()
            opt.optimize(rounds)
            print("lba_optimization_process: done optimize(rounds)...")

        print("lba_optimization_process: searching for outliers...")

        # search for final outlier observations and clean map
        num_bad_observations = 0  # final bad observations
        outliers_edge_data = []

        chi2_limits = {True: chi2Stereo, False: chi2Mono}

        for edge, (p, kf, p_idx, is_stereo) in graph_edges.items():

            # if p.is_bad(): # redundant check since the considered points come from good_points
            #     continue

            assert kf.get_point_match(p_idx) is p

            if edge.chi2() > chi2_limits[is_stereo] or not edge.is_depth_positive():
                num_bad_observations += 1
                outliers_edge_data.append((p, kf, p_idx, is_stereo))

        num_active_edges = num_edges - num_bad_edges
        mean_squared_error = opt.active_chi2() / max(num_active_edges, 1)

        print("lba_optimization_process: preparing results ...")

        if sync_flag_thread.is_alive():
            abort_flag.value = True  # force the sync thread to exit
            sync_flag_thread.join(timeout=0.005)

        # Final results: keyframe poses and point positions
        keyframe_poses = {
            kf.kid: poseRt(
                graph_keyframes[kf].estimate().rotation().matrix(),
                graph_keyframes[kf].estimate().translation(),
            )
            for kf in graph_keyframes
        }
        point_positions = {p.id: graph_points[p].estimate() for p in graph_points}

        outliers_edge_data_out = [(p_idx, kf.kid) for p, kf, p_idx, is_stereo in outliers_edge_data]

        result_dict["keyframe_poses"] = keyframe_poses
        result_dict["point_positions"] = point_positions
        result_dict["outliers_edge_data_out"] = outliers_edge_data_out
        result_dict["mean_squared_error"] = mean_squared_error
        result_dict["ratio_bad_observations"] = num_bad_observations / max(num_edges, 1)

        result_dict_queue.put(result_dict)
        queue.put("finished")
        print("lba_optimization_process: completed")

    except Exception as e:

        Printer.red(f"lba_optimization_process: EXCEPTION: {e} !!!")
        print(f"lba_optimization_process: EXCEPTION: {e} !!!")
        traceback_details = traceback.format_exc()
        print(f"\t traceback details: {traceback_details}")
        result_dict_queue.put(result_dict)


def local_bundle_adjustment_parallel(
    keyframes,
    points,
    keyframes_ref=None,
    fixed_points=False,
    verbose=False,
    rounds=10,
    abort_flag=None,
    mp_abort_flag=None,
    map_lock=None,
):
    from .local_mapping import LocalMapping

    print = LocalMapping.print

    if keyframes_ref is None:
        keyframes_ref = []

    if abort_flag is None:
        abort_flag = g2o.Flag()

    # NOTE: we need a keyframe map (kf.id->kf) in order to be able retrieve and discard the outlier-edge keyframes after optimization
    good_keyframes = {kf.kid: kf for kf in keyframes if not kf.is_bad()}
    good_keyframes.update({kf.kid: kf for kf in keyframes_ref if not kf.is_bad()})

    good_points = [
        p for p in points if p is not None and not p.is_bad()
    ]  # and any(f in keyframes for f in p.keyframes())]

    # NOTE: We use the MultiprocessingManager to manage queues and avoid pickling problems with multiprocessing.
    mp_manager = MultiprocessingManager()
    queue = mp_manager.Queue()
    result_dict_queue = mp_manager.Queue()

    # Start the optimization process in parallel
    p = mp.Process(
        target=lba_optimization_process,
        args=(
            result_dict_queue,
            queue,
            good_keyframes,
            keyframes_ref,
            good_points,
            fixed_points,
            verbose,
            rounds,
            mp_abort_flag,
        ),
    )
    p.daemon = True

    p.start()
    print("local_bundle_adjustment_parallel - started")

    # HACK: for some reasons, p.join() hangs out once in a while (even after the last print). We terminate the process here once we receive an end-signal message.
    # p.join()
    message = queue.get()  # blocking call
    if p.is_alive():
        p.terminate()

    print("local_bundle_adjustment_parallel - joined")

    try:
        if result_dict_queue.qsize() == 0:
            print("local_bundle_adjustment_parallel - result_dict_queue is empty")
            result_dict = {"mean_squared_error": -1, "ratio_bad_observations": 0}
        else:
            result_dict = result_dict_queue.get()

        if result_dict["mean_squared_error"] != -1:
            # Extract the keyframe poses and point positions
            keyframe_poses = result_dict["keyframe_poses"]
            point_positions = result_dict["point_positions"]
            outliers_edge_data_out = result_dict["outliers_edge_data_out"]

            # Update the main process map with the new poses and point positions
            if map_lock is None:
                map_lock = threading.RLock()

            with map_lock:

                # remove outlier observations
                for p_idx, kf_kid in outliers_edge_data_out:
                    kf = good_keyframes[kf_kid]
                    p_f = kf.get_point_match(p_idx)
                    if p_f is not None:
                        assert p_f.id == p_idx
                        p_f.remove_observation(kf, p_idx, map_no_lock=True)
                        # the following instruction is now included in p.remove_observation()
                        # f.remove_point(p)   # it removes multiple point instances (if these are present)
                        # f.remove_point_match(p_idx) # this does not remove multiple point instances, but now there cannot be multiple instances any more

                # put frames back
                for kf in good_keyframes.values():
                    # if kf.kid in keyframe_poses:
                    #     kf.update_pose(keyframe_poses[kf.kid])
                    try:
                        kf.update_pose(keyframe_poses[kf.kid])
                        kf.lba_count += 1
                    except:
                        Printer.red(f"Missing pose for keyframe {kf.kid}")
                        pass  # kf.kid is not in keyframe_poses

                # put points back
                if not fixed_points:
                    for p in good_points:
                        if p is not None:
                            # if p.id in point_positions:
                            #     p.update_position(np.array(point_positions[p.id]))
                            #     p.update_normal_and_depth(force=True)
                            try:
                                p.update_position(np.array(point_positions[p.id]))
                                p.update_normal_and_depth(force=True)
                            except:
                                pass  # p.id is not in point_positions

            # Return success indicator
            return result_dict["mean_squared_error"], result_dict["ratio_bad_observations"]
        else:
            Printer.red(f"local_bundle_adjustment_parallel - error: {result_dict}")
            return -1, 0

    except Exception as e:
        Printer.red(f"local_bundle_adjustment_parallel - error: {result_dict}")
        print(f"local_bundle_adjustment_parallel - EXCEPTION: {e}")
        traceback_details = traceback.format_exc()
        print(f"\t traceback details: {traceback_details}")
        return -1, 0


# ------------------------------------------------------------------------------------------


# The goal of this function is to estimate the Sim(3) transformation (R12,t12,s12)
# between two keyframes (kf1 and kf2) and their 3D matched map points in a SLAM system.
# This optimization compute the Sim(3) transformation that minimizes the reprojection errors
# of the 3D matched map points of kf1 into kf2 and the reprojection errors of the
# 3D matched map points of kf2 into kf1.
# map_point_matches12[i] = map point of kf2 matched with i-th map point of kf1  (from 1 to 2)
# out: num_inliers, R12, t12, s12
def optimize_sim3(
    kf1: KeyFrame,
    kf2: KeyFrame,
    map_points1,
    map_point_matches12,
    R12,
    t12,
    s12,
    th2: float,
    fix_scale: bool,
    verbose: bool = False,
) -> int:

    from pyslam.loop_closing.loop_detector_base import LoopDetectorBase

    print = LoopDetectorBase.print

    # Calibration and Camera Poses
    cam1 = kf1.camera
    cam2 = kf2.camera

    kf1_Tcw = kf1.Tcw()
    kf2_Tcw = kf2.Tcw()
    R1w, t1w = kf1_Tcw[:3, :3], kf1_Tcw[:3, 3]
    R2w, t2w = kf2_Tcw[:3, :3], kf2_Tcw[:3, 3]

    optimizer = g2o.SparseOptimizer()
    solver = g2o.BlockSolverX(g2o.LinearSolverDenseX())
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    optimizer.set_algorithm(algorithm)

    sim3 = g2o.Sim3(R12.copy(), t12.ravel().copy(), s12)

    # Sim3 vertex
    sim3_vertex = g2o.VertexSim3Expmap()
    sim3_vertex.set_estimate(sim3)
    sim3_vertex.set_id(0)
    sim3_vertex.set_fixed(False)
    sim3_vertex._fix_scale = fix_scale
    sim3_vertex._principle_point1 = np.array([cam1.cx, cam1.cy])
    sim3_vertex._focal_length1 = np.array([cam1.fx, cam1.fy])
    sim3_vertex._principle_point2 = np.array([cam2.cx, cam2.cy])
    sim3_vertex._focal_length2 = np.array([cam2.fx, cam2.fy])
    optimizer.add_vertex(sim3_vertex)

    # MapPoint vertices and edges
    if map_points1 is None:
        map_points1 = kf1.get_points()
    num_matches = len(map_point_matches12)
    assert num_matches == len(map_points1)

    if verbose:
        print(f"optimize_sim3: num_matches = {num_matches}")

    edges_12 = []
    edges_21 = []
    vertex_indices = []

    delta_huber = np.sqrt(th2)

    inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2
    eye2 = np.eye(2)

    num_correspondences = 0
    for i in range(num_matches):
        mp1 = map_points1[i]
        if mp1 is None or mp1.is_bad():
            continue
        mp2 = map_point_matches12[i]  # map point of kf2 matched with i-th map point of kf1
        if mp2 is None or mp2.is_bad():
            continue

        vertex_id1 = 2 * i + 1
        vertex_id2 = 2 * (i + 1)
        index2 = mp2.get_observation_idx(kf2)
        if index2 >= 0:
            # Create and set vertex for map point 1 (fixed)
            v_mp1 = g2o.VertexSBAPointXYZ()
            v_mp1.set_estimate(R1w @ mp1.pt() + t1w)
            v_mp1.set_id(vertex_id1)
            v_mp1.set_fixed(True)
            optimizer.add_vertex(v_mp1)

            # Create and set vertex for map point 2 (fixed)
            v_mp2 = g2o.VertexSBAPointXYZ()
            v_mp2.set_estimate(R2w @ mp2.pt() + t2w)
            v_mp2.set_id(vertex_id2)
            v_mp2.set_fixed(True)
            optimizer.add_vertex(v_mp2)

            # Create and set edge 12 (project mp2_2 on camera 1 by using sim3(R12, t12, s12) to transform mp2_2 in mp2_1)
            edge_12 = g2o.EdgeSim3ProjectXYZ()
            edge_12.set_vertex(0, optimizer.vertex(vertex_id2))
            edge_12.set_vertex(1, optimizer.vertex(0))
            edge_12.set_measurement(kf1.kpsu[i])
            invSigma2_12 = inv_level_sigmas2[kf1.octaves[i]]

            if Parameters.kUseSemanticsInOptimization and kf1.kps_sem is not None:
                invSigma2_12 *= SemanticMappingShared.get_semantic_weight(kf1.kps_sem[i])

            edge_12.set_information(eye2 * invSigma2_12)
            edge_12.set_robust_kernel(g2o.RobustKernelHuber(delta_huber))
            optimizer.add_edge(edge_12)

            # Create and set edge 21 (project mp1_1 on camera 2 by using sim3(R21, t21, s21).inverse() to transform mp1_1 in mp1_2)
            edge_21 = g2o.EdgeInverseSim3ProjectXYZ()
            edge_21.set_vertex(0, optimizer.vertex(vertex_id1))
            edge_21.set_vertex(1, optimizer.vertex(0))
            edge_21.set_measurement(kf2.kpsu[index2])
            invSigma2_21 = inv_level_sigmas2[kf2.octaves[index2]]

            if Parameters.kUseSemanticsInOptimization and kf2.kps_sem is not None:
                invSigma2_21 *= SemanticMappingShared.get_semantic_weight(kf2.kps_sem[index2])

            edge_21.set_information(eye2 * invSigma2_21)
            edge_21.set_robust_kernel(g2o.RobustKernelHuber(delta_huber))
            optimizer.add_edge(edge_21)

            edges_12.append(edge_12)
            edges_21.append(edge_21)
            vertex_indices.append(i)

            num_correspondences += 1

    if verbose:
        print(f"optimize_sim3: num_correspondences = {num_correspondences}")

    # Optimize
    optimizer.initialize_optimization()
    if verbose:
        optimizer.set_verbose(True)
    optimizer.optimize(5)
    err = optimizer.active_chi2()

    # Check inliers
    num_bad = 0
    for i, edge_12 in enumerate(edges_12):
        edge_21 = edges_21[i]

        if (
            edge_12.chi2() > th2
            or not edge_12.is_depth_positive()
            or edge_21.chi2() > th2
            or not edge_21.is_depth_positive()
        ):
            index = vertex_indices[i]
            map_points1[index] = None
            optimizer.remove_edge(edge_12)
            optimizer.remove_edge(edge_21)
            edges_12[i] = None
            edges_21[i] = None
            num_bad += 1

    num_more_iterations = 10 if num_bad > 0 else 5

    if num_correspondences - num_bad < 10:
        print(
            f"optimize_sim3: Too few inliers, num_correspondences = {num_correspondences}, num_bad = {num_bad}"
        )
        return 0, None, None, None, 0  # num_inliers, R,t,scale, delta_err

    # Optimize again with inliers
    optimizer.initialize_optimization()
    optimizer.optimize(num_more_iterations)

    delta_err = (
        optimizer.active_chi2() - err
    )  # this must be negative to get a good optimization (optimizer.active_chi2() < err)

    num_inliers = 0
    for i, edge_12 in enumerate(edges_12):
        edge_21 = edges_21[i]

        if (
            edge_12 and edge_21
        ):  # we need to check for Nones potentially set by the first inlier check
            if (
                edge_12.chi2() > th2
                or not edge_12.is_depth_positive()
                or edge_21.chi2() > th2
                or not edge_21.is_depth_positive()
            ):
                index = vertex_indices[i]
                map_points1[index] = None
            else:
                num_inliers += 1

    # Recover optimized Sim3
    sim3_vertex_recov = optimizer.vertex(0)
    sim3 = sim3_vertex_recov.estimate()

    return num_inliers, sim3.rotation().matrix(), sim3.translation(), sim3.scale(), delta_err


# ------------------------------------------------------------------------------------------


def optimize_essential_graph(
    map_object,
    loop_keyframe: KeyFrame,
    current_keyframe: KeyFrame,
    non_corrected_sim3_map,
    corrected_sim3_map,
    loop_connections,
    fix_scale: bool,
    print_fun=print,
    verbose=False,
):

    # Setup optimizer
    optimizer = g2o.SparseOptimizer()
    optimizer.set_verbose(False)

    linear_solver = g2o.LinearSolverEigenSim3()
    solver = g2o.BlockSolverSim3(linear_solver)
    algorithm = g2o.OptimizationAlgorithmLevenberg(solver)
    algorithm.set_user_lambda_init(1e-16)
    optimizer.set_algorithm(algorithm)

    all_keyframes = map_object.get_keyframes()
    all_map_points = map_object.get_points()
    max_keyframe_id = map_object.max_keyframe_id

    vec_Scw = [None] * (max_keyframe_id + 1)  # use keyframe kid as index here
    vec_corrected_Swc = [None] * (max_keyframe_id + 1)  # use keyframe kid as index here
    # vertices = [None] * (max_keyframe_id + 1)            # use keyframe kid as index here

    min_number_features = 100

    # Set KeyFrame vertices
    for keyframe in all_keyframes:
        if keyframe.is_bad():
            continue
        vertex_sim3 = g2o.VertexSim3Expmap()

        keyframe_id = keyframe.kid

        try:  # if keyframe in corrected_sim3_map:
            corrected_sim3 = corrected_sim3_map[keyframe]
            Siw = Sim3Pose(R=corrected_sim3.R.copy(), t=corrected_sim3.t.copy(), s=corrected_sim3.s)
        except:
            kf_Tcw = keyframe.Tcw()
            Siw = Sim3Pose(kf_Tcw[:3, :3], kf_Tcw[:3, 3], 1.0)
        vec_Scw[keyframe_id] = Siw

        Siw_g2o = g2o.Sim3(Siw.R, Siw.t.ravel(), Siw.s)
        vertex_sim3.set_estimate(Siw_g2o)

        if keyframe == loop_keyframe:
            vertex_sim3.set_fixed(True)

        vertex_sim3.set_id(keyframe_id)
        vertex_sim3.set_marginalized(False)
        vertex_sim3._fix_scale = fix_scale

        optimizer.add_vertex(vertex_sim3)
        # vertices[keyframe_id] = vertex_sim3

    num_graph_edges = 0

    inserted_loop_edges = set()  # set of pairs (keyframe_id, connected_keyframe_id)
    mat_lambda = np.identity(7)

    # Set loop edges
    for keyframe, connections in loop_connections.items():
        keyframe_id = keyframe.kid
        Siw = vec_Scw[keyframe_id]
        if Siw is None:
            Printer.orange(f"[optimize_essential_graph] SiW for keyframe {keyframe_id} is None")
            continue
        Swi = Siw.inverse()

        for connected_keyframe in connections:
            connected_id = connected_keyframe.kid
            # accept (current_keyframe,loop_keyframe)
            # and all the other loop edges with weight >= min_number_features
            if (
                keyframe_id != current_keyframe.kid or connected_id != loop_keyframe.kid
            ) and keyframe.get_weight(connected_keyframe) < min_number_features:
                # print(f'skipping loop edge {keyframe_id} {connected_id}')
                continue

            Sjw = vec_Scw[connected_id]
            Sji = Sjw @ Swi

            Sji_g2o = g2o.Sim3(Sji.R, Sji.t.ravel(), Sji.s)

            edge = g2o.EdgeSim3()
            edge.set_vertex(1, optimizer.vertex(connected_id))
            edge.set_vertex(0, optimizer.vertex(keyframe_id))
            edge.set_measurement(Sji_g2o)
            edge.set_information(mat_lambda.copy())

            optimizer.add_edge(edge)
            num_graph_edges += 1
            inserted_loop_edges.add(
                (min(keyframe_id, connected_id), max(keyframe_id, connected_id))
            )

    # Set normal edges
    for keyframe in all_keyframes:
        keyframe_id = keyframe.kid
        parent_keyframe = keyframe.get_parent()

        try:
            Swi = non_corrected_sim3_map[keyframe].inverse()
        except:
            Swi = vec_Scw[keyframe_id].inverse()

        # Spanning tree edge
        if parent_keyframe:
            parent_id = parent_keyframe.kid

            try:
                Sjw = non_corrected_sim3_map[parent_keyframe].copy()
            except:
                Sjw = vec_Scw[parent_id]

            Sji = Sjw @ Swi

            Sji_g2o = g2o.Sim3(Sji.R, Sji.t.ravel(), Sji.s)

            edge = g2o.EdgeSim3()
            edge.set_vertex(1, optimizer.vertex(parent_id))
            edge.set_vertex(0, optimizer.vertex(keyframe_id))
            edge.set_measurement(Sji_g2o)
            edge.set_information(mat_lambda.copy())
            optimizer.add_edge(edge)
            num_graph_edges += 1

        # Loop edges
        for loop_edge in keyframe.get_loop_edges():
            if loop_edge.kid < keyframe_id:
                try:
                    Slw = non_corrected_sim3_map[loop_edge].copy()
                except:
                    Slw = vec_Scw[loop_edge.kid]  # already copy

                Sli = Slw @ Swi

                Sli_g2o = g2o.Sim3(Sli.R, Sli.t.ravel(), Sli.s)

                edge = g2o.EdgeSim3()
                edge.set_vertex(1, optimizer.vertex(loop_edge.kid))
                edge.set_vertex(0, optimizer.vertex(keyframe_id))
                edge.set_measurement(Sli_g2o)
                edge.set_information(mat_lambda.copy())
                optimizer.add_edge(edge)
                num_graph_edges += 1

        # Covisibility graph edges
        for connected_keyframe in keyframe.get_covisible_by_weight(min_number_features):
            if (
                connected_keyframe != parent_keyframe
                and not keyframe.has_child(connected_keyframe)
                and connected_keyframe.kid < keyframe_id
                and not connected_keyframe.is_bad()
                and (
                    min(keyframe_id, connected_keyframe.kid),
                    max(keyframe_id, connected_keyframe.kid),
                )
                not in inserted_loop_edges
            ):

                try:
                    Snw = non_corrected_sim3_map[connected_keyframe].copy()
                except:
                    Snw = vec_Scw[connected_keyframe.kid]  # already copy

                Sni = Snw @ Swi

                Sni_g2o = g2o.Sim3(Sni.R, Sni.t.ravel(), Sni.s)

                edge = g2o.EdgeSim3()
                edge.set_vertex(1, optimizer.vertex(connected_keyframe.kid))
                edge.set_vertex(0, optimizer.vertex(keyframe_id))
                edge.set_measurement(Sni_g2o)
                edge.set_information(mat_lambda.copy())
                optimizer.add_edge(edge)
                num_graph_edges += 1

    if verbose:
        print_fun(f"[optimize_essential_graph]: Total number of graph edges: {num_graph_edges}")

    # Optimize
    optimizer.initialize_optimization()
    optimizer.optimize(20)

    # SE3 Pose Recovering.  Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for keyframe in all_keyframes:
        keyframe_id = keyframe.kid

        vertex_sim3 = optimizer.vertex(keyframe_id)
        corrected_Siw = vertex_sim3.estimate()

        R = corrected_Siw.rotation().matrix()
        t = corrected_Siw.translation()
        s = corrected_Siw.scale()
        Siw = Sim3Pose(R, t, s)
        Swi = Siw.inverse()

        vec_corrected_Swc[keyframe_id] = Swi

        Tiw = poseRt(R, t / s)  # [R t/s; 0 1]
        keyframe.update_pose(Tiw)

    # Correct points: Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for map_point in all_map_points:
        if map_point.is_bad():
            continue

        if map_point.corrected_by_kf == current_keyframe.kid:
            reference_id = map_point.corrected_reference
        else:
            reference_keyframe = map_point.get_reference_keyframe()
            reference_id = reference_keyframe.kid

        Srw = vec_Scw[reference_id]
        corrected_Swr = vec_corrected_Swc[reference_id]

        P3Dw = map_point.pt()
        corrected_P3Dw = corrected_Swr.map(Srw.map(P3Dw)).ravel()
        map_point.update_position(corrected_P3Dw)

        map_point.update_normal_and_depth()

    mean_squared_error = optimizer.active_chi2() / max(num_graph_edges, 1)
    return mean_squared_error
