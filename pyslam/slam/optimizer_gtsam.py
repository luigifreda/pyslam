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

import g2o

from pyslam.config_parameters import Parameters
from pyslam.utilities.utils_sys import Printer
from pyslam.utilities.utils_geom import poseRt, inv_T, Sim3Pose
from pyslam.utilities.utils_mp import MultiprocessingManager

from .frame import FeatureTrackerShared
from .map_point import MapPoint
from .keyframe import KeyFrame


kMinDepth = 1e-3
kSigmaForFixed = 1e-6
kWeightForDisabledFactor = 1e-3


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
    local_window,
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
    if local_window is None:
        local_frames = keyframes
    else:
        local_frames = keyframes[-local_window:]

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
        # Why? The g2o-abort-flag passed (via pickling) to a launched parallel process (via multiprocessing module) is just a different instance that is not kept in sync
        # with its source instance in the parent process. This means we don't succeed to abort the BA when set the source instance.
        sync_flag_thread = threading.Thread(
            target=sync_flag_fun, args=(abort_flag, mp_abort_flag, print)
        )
        sync_flag_thread.daemon = True  # Daemonize thread so it exits when the main thread does
        sync_flag_thread.start()

    # Create GTSAM factor graph
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    sigma_for_fixed = kSigmaForFixed  # sigma used for fixed entities

    # Huber loss parameters for robust optimization
    th_huber_mono = np.sqrt(5.991)  # chi-square 2 DOFs
    th_huber_stereo = np.sqrt(7.815)  # chi-square 3 DOFs

    keyframe_keys = {}
    point_keys = {}
    graph_factors = {}

    # Add keyframes as pose variables
    for kf in (
        local_frames if fixed_points else keyframes
    ):  # if points are fixed then consider just the local frames, otherwise we need all frames or at least two frames for each point
        if kf.is_bad:
            continue
        pose_key = X(kf.kid)
        keyframe_keys[kf] = pose_key

        pose = gtsam.Pose3(gtsam.Rot3(kf.Rwc.copy()), gtsam.Point3(*kf.Ow.copy()))
        initial_estimates.insert(pose_key, pose)

        if kf.kid == 0 or kf not in local_frames:
            graph.add(
                gtsam.PriorFactorPose3(
                    pose_key, pose, gtsam.noiseModel.Isotropic.Sigma(6, sigma_for_fixed)
                )
            )

    num_edges = 0
    num_bad_edges = 0

    level_sigmas = FeatureTrackerShared.feature_manager.level_sigmas

    # Add points as graph vertices
    for p in points:
        if p is None or p.is_bad:  # do not consider bad points
            continue
        point_key = L(p.id)
        point_keys[p] = point_key
        point_position = gtsam.Point3(p.pt[0:3].copy())
        initial_estimates.insert(point_key, point_position)

        if fixed_points:
            graph.add(
                gtsam.PriorFactorPoint3(
                    point_key, point_position, gtsam.noiseModel.Isotropic.Sigma(3, sigma_for_fixed)
                )
            )

        # Add measurement factors
        for kf, idx in p.observations():
            # if kf.is_bad:  # redundant since we check kf is in graph_keyframes (selected as non-bad)
            #     continue
            if kf not in keyframe_keys:
                continue

            pose_key = keyframe_keys[kf]

            kf_kpsu_idx = kf.kpsu[idx]
            kf_kps_ur_idx = kf.kps_ur[idx] if kf.kps_ur is not None else -1

            is_stereo_obs = kf_kps_ur_idx >= 0
            # invSigma2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[kf.octaves[idx]]
            sigma = level_sigmas[kf.octaves[idx]]

            noise_model = gtsam_factors.SwitchableRobustNoiseModel(
                3 if is_stereo_obs else 2,
                sigma,
                th_huber_stereo if is_stereo_obs else th_huber_mono,
            )

            if is_stereo_obs:
                calib = gtsam.Cal3_S2Stereo(
                    kf.camera.fx, kf.camera.fy, 0, kf.camera.cx, kf.camera.cy, kf.camera.b
                )
                measurement = gtsam.StereoPoint2(
                    kf_kpsu_idx[0], kf_kps_ur_idx, kf_kpsu_idx[1]
                )  # uL, uR, v
                factor = gtsam_factors.WeightedGenericStereoProjectionFactor3D(
                    measurement, noise_model, pose_key, point_key, calib
                )
            else:
                calib = gtsam.Cal3_S2(kf.camera.fx, kf.camera.fy, 0, kf.camera.cx, kf.camera.cy)
                measurement = gtsam.Point2(kf_kpsu_idx[0], kf_kpsu_idx[1])
                factor = gtsam_factors.WeightedGenericProjectionFactorCal3_S2(
                    measurement, noise_model, pose_key, point_key, calib
                )

            graph.add(factor)
            graph_factors[factor, noise_model] = (
                p,
                kf,
                idx,
                is_stereo_obs,
            )  # one has kf.points[idx] == p
            num_edges += 1

    if abort_flag.value:
        return -1, result_dict

    chi2Mono = 5.991  # chi-square 2 DOFs
    chi2Stereo = 7.815  # chi-square 3 DOFs

    initial_mean_squared_error = graph.error(initial_estimates) / max(num_edges, 1)

    if robust_rounds > 0:
        params = gtsam.LevenbergMarquardtParams().CeresDefaults()
        params.setlambdaInitial(1e-5)  # Matches g2o’s _tau
        params.setlambdaLowerBound(1e-7)  # Prevent over-reduction
        params.setlambdaUpperBound(1e3)  # Prevent excessive increase
        params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
        params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
        params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

        params.setMaxIterations(robust_rounds)
        if verbose:
            params.setVerbosityLM("SUMMARY")

        # Optimize using Levenberg-Marquardt
        optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
        result = optimizer.optimize()

        # check inliers observation
        for (factor, noise_model), (p, kf, idx, is_stereo_obs) in graph_factors.items():

            factor.set_weight(1.0)  # reset the factor weight to 1.0 to compute a meaningful chi2
            chi2 = 2.0 * factor.error(
                result
            )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.

            chi2_check_failure = (chi2 > chi2Stereo) if is_stereo_obs else (chi2 > chi2Mono)

            if not chi2_check_failure:
                point_key = point_keys[p]
                pose_key = keyframe_keys[kf]
                point_position = np.asarray(result.atPoint3(point_key)).reshape(3, 1)
                pose_wc = np.asarray(result.atPose3(pose_key).matrix())  # Twc
                pose_cw = inv_T(pose_wc)
                Pc = (pose_cw[:3, :3] @ point_position + pose_cw[:3, 3].reshape(3, 1)).T
                # uv, depth = kf.camera.project(Pc)
                depth = Pc.ravel()[2]
                chi2_check_failure = depth <= kMinDepth

            if chi2_check_failure:
                num_bad_edges += 1
                factor.set_weight(kWeightForDisabledFactor)  # disable the weighted factor
            else:
                noise_model.set_robust_model_active(False)
                # inlier_factors.append(factor) # now we just use iso_factors
    else:
        result = initial_estimates

    if abort_flag.value:
        return -1, result_dict

    params = gtsam.LevenbergMarquardtParams().CeresDefaults()
    params.setlambdaInitial(1e-5)  # Matches g2o’s _tau
    params.setlambdaLowerBound(1e-7)  # Prevent over-reduction
    params.setlambdaUpperBound(1e3)  # Prevent excessive increase
    params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

    params.setMaxIterations(final_rounds)
    if verbose:
        params.setVerbosityLM("SUMMARY")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, result, params)
    new_result = optimizer.optimize()
    result = new_result

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
        for kf, pose_key in keyframe_keys.items():
            Twc = np.asarray(result.atPose3(pose_key).matrix())
            Tcw = inv_T(Twc)
            keyframe_updates[kf.id] = Tcw
    else:
        for kf, pose_key in keyframe_keys.items():
            Twc = np.asarray(result.atPose3(pose_key).matrix())
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
            for p, point_key in point_keys.items():
                point_updates[p.id] = np.asarray(result.atPoint3(point_key))
        else:
            if loop_kf_id == 0:
                for p, point_key in point_keys.items():
                    # direct update on map
                    p.update_position(np.asarray(result.atPoint3(point_key)))
                    p.update_normal_and_depth(force=True)
            else:
                for p, point_key in point_keys.items():
                    # update for loop closure
                    p.pt_GBA = np.asarray(result.atPoint3(point_key))
                    p.GBA_kf_id = loop_kf_id

    num_active_edges = num_edges - num_bad_edges
    mean_squared_error = graph.error(result) / num_active_edges

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
        local_window=None,
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


class ResectioningMonoFactor:
    def __init__(
        self,
        noise_model: gtsam.noiseModel.Base,
        pose_key: int,
        calib: gtsam.Cal3_S2,
        p: gtsam.Point2,
        P: gtsam.Point3,
        weight: float = 1.0,  # Initial weight
    ):
        self.weight = weight
        self.factor = gtsam.CustomFactor(noise_model, gtsam.KeyVector([pose_key]), self.error_func)
        self.pose_key = pose_key
        self.calib = calib
        self.p = p
        self.P = P

    def get_factor(self):
        """Returns the underlying GTSAM factor."""
        return self.factor

    def get_weight(self):
        return self.weight

    def set_weight(self, new_weight: float):
        self.weight = new_weight

    def error_func(
        self, this: gtsam.CustomFactor, v: gtsam.Values, H: list[np.ndarray]
    ) -> np.ndarray:
        try:
            pose = v.atPose3(self.pose_key)
            camera = gtsam.PinholeCameraCal3_S2(pose, self.calib)

            # Compute the reprojection error
            if H is None or len(H) == 0:
                return self.weight * (camera.project(self.P) - self.p)

            # Compute Jacobians if required
            Dpose = np.zeros((2, 6), order="F")
            Dpoint = np.zeros((2, 3), order="F")
            Dcal = np.zeros((2, 5), order="F")
            result = camera.project(self.P, Dpose, Dpoint, Dcal) - self.p

            # Apply weight to error and Jacobians
            H[0] = self.weight * Dpose
            if len(H) > 1:
                H[1] = self.weight * Dpoint

            return self.weight * result  # Scale the error
        except Exception as e:
            Printer.red(f"[resectioning_mono_factor]: Exception: {e}")
            result = np.zeros((2, 1), order="F")
            if H is not None and len(H) > 0:
                H[0] = np.zeros((2, 6), order="F")
                if len(H) > 1:
                    H[1] = np.zeros((2, 3), order="F")
        return result


class ResectioningStereoFactor:
    def __init__(
        self,
        noise_model: gtsam.noiseModel.Base,
        pose_key: int,
        calib: gtsam.Cal3_S2Stereo,
        p: gtsam.StereoPoint2,
        P: gtsam.Point3,
        weight: float = 1.0,  # Initial weight
    ):
        self.weight = weight
        self.factor = gtsam.CustomFactor(noise_model, gtsam.KeyVector([pose_key]), self.error_func)
        self.pose_key = pose_key
        self.calib = calib
        self.p = p
        self.P = P

    def get_factor(self):
        """Returns the underlying GTSAM factor."""
        return self.factor

    def get_weight(self):
        return self.weight

    def set_weight(self, new_weight: float):
        self.weight = new_weight

    def error_func(
        self, this: gtsam.CustomFactor, v: gtsam.Values, H: list[np.ndarray]
    ) -> np.ndarray:
        try:
            pose = v.atPose3(self.pose_key)
            camera = gtsam.StereoCamera(pose, self.calib)
            p_vec = self.p.vector()

            # Compute the reprojection error
            if H is None or len(H) == 0:
                return self.weight * (camera.project(self.P).vector() - p_vec)

            # Compute Jacobians if required
            Dpose = np.zeros((3, 6), order="F")
            Dpoint = np.zeros((3, 3), order="F")
            result = camera.project2(self.P, Dpose, Dpoint).vector() - p_vec

            # Apply weight to error and Jacobians
            H[0] = self.weight * Dpose
            if len(H) > 1:
                H[1] = self.weight * Dpoint

            return self.weight * result  # Scale the error
        except Exception as e:
            Printer.red(f"[resectioning_stereo_factor]: Exception: {e}")
            result = np.zeros((3, 1), order="F")
            if H is not None and len(H) > 0:
                H[0] = np.zeros((3, 6), order="F")
                if len(H) > 1:
                    H[1] = np.zeros((3, 3), order="F")
        return result


def resectioning_mono_factor_py(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    # host factor is the object that contains the actual gtsam factor
    host_factor = ResectioningMonoFactor(noise_model, pose_key, calib, p, P)
    factor = host_factor.get_factor()
    return factor, host_factor


def resectioning_stereo_factor_py(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2Stereo,
    p: gtsam.StereoPoint2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    # host factor is the object that contains the actual gtsam factor
    host_factor = ResectioningStereoFactor(noise_model, pose_key, calib, p, P)
    factor = host_factor.get_factor()
    return factor, host_factor


def resectioning_mono_factor(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    # host factor is the object that contains the actual gtsam factor
    # here host_factor and factor are the same
    host_factor = gtsam_factors.ResectioningFactor(noise_model, pose_key, calib, p, P)
    factor = host_factor
    return factor, host_factor


def resectioning_stereo_factor(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2Stereo,
    p: gtsam.StereoPoint2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    # host factor is the object that contains the actual gtsam factor
    # here host_factor and factor are the same
    host_factor = gtsam_factors.ResectioningFactorStereo(noise_model, pose_key, calib, p, P)
    factor = host_factor
    return factor, host_factor


def resectioning_mono_factor_Tcw(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    # host factor is the object that contains the actual gtsam factor
    # here host_factor and factor are the same
    host_factor = gtsam_factors.ResectioningFactorTcw(noise_model, pose_key, calib, p, P)
    factor = host_factor
    return factor, host_factor


def resectioning_stereo_factor_Tcw(
    noise_model: gtsam.noiseModel.Base,
    pose_key: int,
    calib: gtsam.Cal3_S2Stereo,
    p: gtsam.StereoPoint2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    # host factor is the object that contains the actual gtsam factor
    # here host_factor and factor are the same
    host_factor = gtsam_factors.ResectioningFactorStereoTcw(noise_model, pose_key, calib, p, P)
    factor = host_factor
    return factor, host_factor


# Here the values contain the Twc poses of the keyframes
class PoseOptimizerGTSAM:
    def __init__(self, frame, use_robust_factors=True):
        self.frame = frame
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.factor_tuples = {}
        self.num_factors = 0
        self.use_robust_factors = use_robust_factors

        self.K_mono = gtsam.Cal3_S2(
            frame.camera.fx, frame.camera.fy, 0, frame.camera.cx, frame.camera.cy
        )
        self.K_stereo = gtsam.Cal3_S2Stereo(
            frame.camera.fx, frame.camera.fy, 0, frame.camera.cx, frame.camera.cy, frame.camera.b
        )

        self.thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS
        self.thHuberStereo = math.sqrt(7.815)  # chi-square 3 DOFS

        self.add_mono_factor = resectioning_mono_factor
        self.add_stereo_factor = resectioning_stereo_factor
        if platform.system() == "Darwin":
            # NOTE: Under macOS I found some interface issues with the pybindings of the resectioning factors.
            self.add_mono_factor = resectioning_mono_factor_py
            self.add_stereo_factor = resectioning_stereo_factor_py

    def add_pose_node(self):
        pose_initial = gtsam.Pose3(
            gtsam.Rot3(self.frame.Rwc.copy()), gtsam.Point3(*self.frame.Ow.copy())
        )
        self.initial.insert(X(0), pose_initial)
        # NOTE: there is no need to set a prior here
        # noise_prior = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        # self.graph.add(gtsam.PriorFactorPose3(X(0), pose_initial, noise_prior))

    def add_observations(self):
        level_sigmas = FeatureTrackerShared.feature_manager.level_sigmas
        outliers = self.frame.outliers
        octaves = self.frame.octaves
        with MapPoint.global_lock:
            for idx, p in enumerate(self.frame.points):
                if p is None:
                    continue

                frame_kpsu_idx = self.frame.kpsu[idx]
                frame_kps_ur_idx = self.frame.kps_ur[idx] if self.frame.kps_ur is not None else -1

                # reset outlier flag
                outliers[idx] = False
                is_stereo_obs = frame_kps_ur_idx >= 0

                # invSigma2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[self.frame.octaves[idx]]
                sigma = level_sigmas[octaves[idx]]

                noise_model = gtsam_factors.SwitchableRobustNoiseModel(
                    3 if is_stereo_obs else 2,
                    sigma,
                    self.thHuberStereo if is_stereo_obs else self.thHuberMono,
                )

                # Add the observation factor
                if is_stereo_obs:
                    factor, host_factor = self.add_stereo_factor(
                        noise_model,
                        X(0),
                        self.K_stereo,
                        gtsam.StereoPoint2(frame_kpsu_idx[0], frame_kps_ur_idx, frame_kpsu_idx[1]),
                        gtsam.Point3(*p.pt),
                    )
                else:
                    factor, host_factor = self.add_mono_factor(
                        noise_model,
                        X(0),
                        self.K_mono,
                        gtsam.Point2(frame_kpsu_idx[0], frame_kpsu_idx[1]),
                        gtsam.Point3(*p.pt),
                    )

                if not self.use_robust_factors:
                    noise_model.set_robust_model_active(False)

                self.graph.add(factor)
                self.factor_tuples[p] = (factor, noise_model, idx, is_stereo_obs, host_factor)
                self.num_factors += 1

    def optimize(self, rounds=10, verbose=False):
        if self.num_factors < 3:
            Printer.red("pose_optimization: not enough correspondences!")
            return 0, False, 0

        chi2Mono = 5.991  # Chi-squared 2 DOFs
        chi2Stereo = 7.815  # chi-squared 3 DOFs

        is_ok = True

        # params = gtsam.LevenbergMarquardtParams().LegacyDefaults() # default ones
        params = gtsam.LevenbergMarquardtParams().CeresDefaults()
        params.setlambdaInitial(1e-5)  # Matches g2o’s _tau
        params.setlambdaLowerBound(1e-8)  # Prevent over-reduction
        params.setlambdaUpperBound(1e3)  # Prevent excessive increase
        params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
        params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
        params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

        params.setMaxIterations(rounds)

        if verbose:
            params.setVerbosityLM("SUMMARY")

        # initial_error = self.graph.error(self.initial)

        result, result_prev = None, None
        cost, cost_prev = None, float("inf")
        num_inliers = 0

        for it in range(4):
            # optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
            optimizer = gtsam_factors.LevenbergMarquardtOptimizerG2o(
                self.graph, self.initial, params
            )
            result_prev = result
            result = optimizer.optimize()

            # marginals = gtsam.Marginals(self.graph, result)

            cost = self.graph.error(result)  # Compute new cost
            cost_change = cost_prev - cost

            # if cost_change <= 0 or np.isinf(cost):
            #     Printer.orange(f"pose_optimization: Warning: Cost did not decrease or is not finite at iteration {it}! Previous: {cost_prev}, Current: {cost}")
            #     result = result_prev
            #     is_ok = False
            #     break

            cost_prev = cost

            num_bad_point_edges = 0
            total_inlier_error = 0.0
            num_inliers = 0
            num_edges = 0

            # pose_wc = np.array(result.atPose3(X(0)).matrix()) # Twc
            # pose_cw = inv_T(pose_wc)

            for p, (
                factor,
                noise_model,
                idx,
                is_stereo_obs,
                host_factor,
            ) in self.factor_tuples.items():
                host_factor.set_weight(
                    1.0
                )  # reset weight to enable back the factor and its correct error computation
                chi2 = 2.0 * factor.error(
                    result
                )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.

                chi2_check_failure = (chi2 > chi2Stereo) if is_stereo_obs else (chi2 > chi2Mono)
                if chi2_check_failure:
                    self.frame.outliers[idx] = True
                    host_factor.set_weight(kWeightForDisabledFactor)  # disable the factor
                    num_bad_point_edges += 1
                else:
                    self.frame.outliers[idx] = False
                    total_inlier_error += chi2  # Sum error only for inliers
                    num_inliers += 1

                if it == 2:
                    noise_model.set_robust_model_active(
                        False
                    )  # last iterations use isotropic factors

                num_edges += 1

            # if num_inliers < 10:
            if num_edges < 10:
                Printer.red("pose_optimization: stopped - not enough edges!")
                # result = result_prev
                # is_ok = False
                break

            self.initial = result  # restart from latest computations

        print(
            f"pose optimization: available {self.num_factors} points, found {num_bad_point_edges} bad points"
        )
        num_valid_points = self.num_factors - num_bad_point_edges
        if num_valid_points < 10:
            Printer.red("pose_optimization: not enough edges!")
            result = result_prev
            is_ok = False

        ratio_bad_points = num_bad_point_edges / max(self.num_factors, 1)
        if (
            num_valid_points > 15
            and ratio_bad_points > Parameters.kMaxOutliersRatioInPoseOptimization
        ):
            Printer.red(
                f"pose_optimization: percentage of bad points is too high: {ratio_bad_points*100:.2f}%"
            )
            is_ok = False

        # update pose estimation
        if is_ok and result is not None:
            is_ok = True
            pose_estimated = result.atPose3(X(0))
            Twc = pose_estimated.matrix()
            self.frame.update_pose(inv_T(Twc))
        else:
            is_ok = False

        mean_squared_error = total_inlier_error / max(num_inliers, 1) if num_inliers > 0 else -1
        return mean_squared_error, is_ok, num_valid_points

    def run(self, verbose=False, rounds=10):
        self.add_pose_node()
        self.add_observations()
        return self.optimize(rounds, verbose)


# Here the values contain the Tcw poses of the keyframes (instead of the default Twc)
class PoseOptimizerGTSAM_Tcw:
    def __init__(self, frame, use_robust_factors=True):
        self.frame = frame
        self.graph = gtsam.NonlinearFactorGraph()
        self.initial = gtsam.Values()
        self.factor_tuples = {}
        self.num_factors = 0
        self.use_robust_factors = use_robust_factors

        self.K_mono = gtsam.Cal3_S2(
            frame.camera.fx, frame.camera.fy, 0, frame.camera.cx, frame.camera.cy
        )
        self.K_stereo = gtsam.Cal3_S2Stereo(
            frame.camera.fx, frame.camera.fy, 0, frame.camera.cx, frame.camera.cy, frame.camera.b
        )

        self.thHuberMono = math.sqrt(5.991)  # chi-square 2 DOFS
        self.thHuberStereo = math.sqrt(7.815)  # chi-square 3 DOFS

        self.add_mono_factor = resectioning_mono_factor_Tcw
        self.add_stereo_factor = resectioning_stereo_factor_Tcw
        # if platform.system() == "Darwin":
        #     # NOTE: Under macOS I found some interface issues with the pybindings of the resectioning factors.
        #     self.add_mono_factor  = resectioning_mono_factor_py
        #     self.add_stereo_factor  = resectioning_stereo_factor_py

    def add_pose_node(self):
        pose_initial = gtsam.Pose3(
            gtsam.Rot3(self.frame.Rcw.copy()), gtsam.Point3(*self.frame.tcw.copy())
        )
        self.initial.insert(X(0), pose_initial)
        # NOTE: there is no need to set a prior here
        # noise_prior = gtsam.noiseModel.Isotropic.Sigma(6, 0.1)
        # self.graph.add(gtsam.PriorFactorPose3(X(0), pose_initial, noise_prior))

    def add_observations(self):
        level_sigmas = FeatureTrackerShared.feature_manager.level_sigmas
        outliers = self.frame.outliers
        octaves = self.frame.octaves
        with MapPoint.global_lock:
            for idx, p in enumerate(self.frame.points):
                if p is None:
                    continue

                frame_kpsu_idx = self.frame.kpsu[idx]
                frame_kps_ur_idx = self.frame.kps_ur[idx] if self.frame.kps_ur is not None else -1

                # reset outlier flag
                outliers[idx] = False
                is_stereo_obs = frame_kps_ur_idx >= 0

                # invSigma2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[self.frame.octaves[idx]]
                sigma = level_sigmas[octaves[idx]]

                noise_model = gtsam_factors.SwitchableRobustNoiseModel(
                    3 if is_stereo_obs else 2,
                    sigma,
                    self.thHuberStereo if is_stereo_obs else self.thHuberMono,
                )

                # Add the observation factor
                if is_stereo_obs:
                    factor, host_factor = self.add_stereo_factor(
                        noise_model,
                        X(0),
                        self.K_stereo,
                        gtsam.StereoPoint2(frame_kpsu_idx[0], frame_kps_ur_idx, frame_kpsu_idx[1]),
                        gtsam.Point3(*p.pt),
                    )
                else:
                    factor, host_factor = self.add_mono_factor(
                        noise_model,
                        X(0),
                        self.K_mono,
                        gtsam.Point2(frame_kpsu_idx[0], frame_kpsu_idx[1]),
                        gtsam.Point3(*p.pt),
                    )

                if not self.use_robust_factors:
                    noise_model.set_robust_model_active(False)

                self.graph.add(factor)
                self.factor_tuples[p] = (factor, noise_model, idx, is_stereo_obs, host_factor)
                self.num_factors += 1

    def optimize(self, rounds=10, verbose=False):
        if self.num_factors < 3:
            Printer.red("pose_optimization: not enough correspondences!")
            return 0, False, 0

        chi2Mono = 5.991  # Chi-squared 2 DOFs
        chi2Stereo = 7.815  # chi-squared 3 DOFs

        is_ok = True

        # params = gtsam.LevenbergMarquardtParams().LegacyDefaults() # default ones
        params = gtsam.LevenbergMarquardtParams().CeresDefaults()
        params.setlambdaInitial(1e-5)  # Matches g2o’s _tau
        params.setlambdaLowerBound(1e-10)  # Prevent over-reduction
        params.setlambdaUpperBound(1e3)  # Prevent excessive increase
        params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
        params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
        params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

        params.setMaxIterations(rounds)

        if verbose:
            params.setVerbosityLM("SUMMARY")

        # initial_error = self.graph.error(self.initial)

        result, result_prev = None, None
        cost, cost_prev = None, float("inf")
        num_inliers = 0

        for it in range(4):
            # optimizer = gtsam.LevenbergMarquardtOptimizer(self.graph, self.initial, params)
            optimizer = gtsam_factors.LevenbergMarquardtOptimizerG2o(
                self.graph, self.initial, params
            )
            result_prev = result
            result = optimizer.optimize()

            # marginals = gtsam.Marginals(self.graph, result)

            cost = self.graph.error(result)  # Compute new cost
            cost_change = cost_prev - cost

            # if cost_change <= 0 or np.isinf(cost):
            #     Printer.orange(f"pose_optimization: Warning: Cost did not decrease or is not finite at iteration {it}! Previous: {cost_prev}, Current: {cost}")
            #     result = result_prev
            #     is_ok = False
            #     break

            cost_prev = cost

            num_bad_point_edges = 0
            total_inlier_error = 0.0
            num_inliers = 0
            num_edges = 0

            # pose_wc = np.array(result.atPose3(X(0)).matrix()) # Twc
            # pose_cw = inv_T(pose_wc)

            for p, (
                factor,
                noise_model,
                idx,
                is_stereo_obs,
                host_factor,
            ) in self.factor_tuples.items():
                host_factor.set_weight(
                    1.0
                )  # reset weight to enable back the factor and its correct error computation
                chi2 = 2.0 * factor.error(
                    result
                )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.

                chi2_check_failure = (chi2 > chi2Stereo) if is_stereo_obs else (chi2 > chi2Mono)
                if chi2_check_failure:
                    self.frame.outliers[idx] = True
                    host_factor.set_weight(kWeightForDisabledFactor)  # disable the factor
                    num_bad_point_edges += 1
                else:
                    self.frame.outliers[idx] = False
                    total_inlier_error += chi2  # Sum error only for inliers
                    num_inliers += 1

                if it == 2:
                    noise_model.set_robust_model_active(
                        False
                    )  # last iterations use isotropic factors

                num_edges += 1

            # if num_inliers < 10:
            if num_edges < 10:
                Printer.red("pose_optimization: stopped - not enough edges!")
                # result = result_prev
                # is_ok = False
                break

            self.initial = result  # restart from latest computations

        print(
            f"pose optimization: available {self.num_factors} points, found {num_bad_point_edges} bad points"
        )
        num_valid_points = self.num_factors - num_bad_point_edges
        if num_valid_points < 10:
            Printer.red("pose_optimization: not enough edges!")
            result = result_prev
            is_ok = False

        ratio_bad_points = num_bad_point_edges / max(self.num_factors, 1)
        if (
            num_valid_points > 15
            and ratio_bad_points > Parameters.kMaxOutliersRatioInPoseOptimization
        ):
            Printer.red(
                f"pose_optimization: percentage of bad points is too high: {ratio_bad_points*100:.2f}%"
            )
            is_ok = False

        # update pose estimation
        if is_ok and result is not None:
            is_ok = True
            Tcw = result.atPose3(X(0)).matrix()  # we stored Tcw in the vertices
            self.frame.update_pose(Tcw)
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
    # optimizer = PoseOptimizerGTSAM_Tcw(frame)
    return optimizer.run(verbose, rounds)


# ------------------------------------------------------------------------------------------


# local bundle adjustment (optimize points reprojection error)
# - frames and points are optimized
# - frames_ref are fixed
def local_bundle_adjustment(
    keyframes,
    points,
    keyframes_ref=[],
    fixed_points=False,
    verbose=False,
    rounds=10,
    abort_flag=g2o.Flag(),
    mp_abort_flag=None,
    map_lock=None,
):
    graph = gtsam.NonlinearFactorGraph()
    initial_estimates = gtsam.Values()

    sigma_for_fixed = kSigmaForFixed  # sigma used for fixed points

    # Huber loss parameters for robust optimization
    th_huber_mono = np.sqrt(5.991)  # chi-square 2 DOFs
    th_huber_stereo = np.sqrt(7.815)  # chi-square 3 DOFs

    good_keyframes = [kf for kf in keyframes if not kf.is_bad] + [
        kf for kf in keyframes_ref if not kf.is_bad
    ]
    good_points = [
        p for p in points if p is not None and not p.is_bad
    ]  # and any(f in keyframes for f in p.keyframes())]

    keyframe_keys = {}
    point_keys = {}

    # Add keyframe (camera pose) vertices
    for kf in good_keyframes:
        pose_key = X(kf.kid)
        keyframe_keys[kf] = pose_key

        pose = gtsam.Pose3(gtsam.Rot3(kf.Rwc.copy()), gtsam.Point3(*kf.Ow.copy()))
        initial_estimates.insert(pose_key, pose)

        if kf.kid == 0 or kf in keyframes_ref:
            graph.add(
                gtsam.PriorFactorPose3(
                    pose_key, pose, gtsam.noiseModel.Isotropic.Sigma(6, sigma_for_fixed)
                )
            )

    num_edges = 0
    num_bad_edges = 0
    graph_factors = {}

    level_sigmas = FeatureTrackerShared.feature_manager.level_sigmas

    # Add 3D point vertices
    for p in good_points:
        point_key = L(p.id)
        point_keys[p] = point_key
        pt = p.pt[:3].copy()
        initial_estimates.insert(point_key, gtsam.Point3(pt))

        if fixed_points:
            graph.add(
                gtsam.PriorFactorPoint3(
                    point_key,
                    gtsam.Point3(pt),
                    gtsam.noiseModel.Isotropic.Sigma(3, sigma_for_fixed),
                )
            )

        # add edges
        good_observations = [
            (kf, p_idx) for kf, p_idx in p.observations() if not kf.is_bad and kf in keyframe_keys
        ]

        # Add reprojection factors
        for kf, p_idx in good_observations:
            pose_key = keyframe_keys[kf]

            assert kf.get_point_match(p_idx) is p

            kf_kpsu_p_idx = kf.kpsu[p_idx]
            kf_kps_ur_p_idx = kf.kps_ur[p_idx] if kf.kps_ur is not None else -1

            is_stereo_obs = kf_kps_ur_p_idx >= 0
            sigma = level_sigmas[kf.octaves[p_idx]]

            noise_model = gtsam_factors.SwitchableRobustNoiseModel(
                3 if is_stereo_obs else 2,
                sigma,
                th_huber_stereo if is_stereo_obs else th_huber_mono,
            )

            camera = kf.camera

            if is_stereo_obs:
                calib = gtsam.Cal3_S2Stereo(camera.fx, camera.fy, 0, camera.cx, camera.cy, camera.b)
                measurement = gtsam.StereoPoint2(
                    kf_kpsu_p_idx[0], kf_kps_ur_p_idx, kf_kpsu_p_idx[1]
                )  # uL, uR, v
                factor = gtsam_factors.WeightedGenericStereoProjectionFactor3D(
                    measurement, noise_model, pose_key, point_key, calib
                )
            else:
                calib = gtsam.Cal3_S2(camera.fx, camera.fy, 0, camera.cx, camera.cy)
                measurement = gtsam.Point2(kf_kpsu_p_idx[0], kf_kpsu_p_idx[1])
                factor = gtsam_factors.WeightedGenericProjectionFactorCal3_S2(
                    measurement, noise_model, pose_key, point_key, calib
                )

            graph.add(factor)
            graph_factors[(factor, noise_model)] = (
                p,
                kf,
                p_idx,
                is_stereo_obs,
            )  # one has kf.points[p_idx] == p
            num_edges += 1

    if abort_flag.value:
        return -1, 0

    chi2Mono = 5.991  # chi-square 2 DOFs
    chi2Stereo = 7.815  # chi-square 3 DOFs

    params = gtsam.LevenbergMarquardtParams().CeresDefaults()
    params.setlambdaInitial(1e-5)  # Matches g2o’s _tau
    params.setlambdaLowerBound(1e-7)  # Prevent over-reduction
    params.setlambdaUpperBound(1e3)  # Prevent excessive increase
    params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

    params.setMaxIterations(5)
    if verbose:
        params.setVerbosityLM("SUMMARY")

    # Optimize
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimates, params)
    result = optimizer.optimize()

    if not abort_flag.value:
        # check inliers observation
        for (factor, noise_model), (p, kf, p_idx, is_stereo) in graph_factors.items():

            # if p.is_bad: # redundant check since the considered points come from good_points
            #     continue

            factor.set_weight(1.0)  # reset the factor weight to 1.0 to compute a meaningful chi2
            chi2 = 2.0 * factor.error(
                result
            )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.

            chi2_check_failure = chi2 > (chi2Stereo if is_stereo else chi2Mono)
            if not chi2_check_failure:
                point_key = point_keys[p]
                pose_key = keyframe_keys[kf]
                point_position = np.asarray(result.atPoint3(point_key)).reshape(3, 1)
                pose_wc = np.asarray(result.atPose3(pose_key).matrix())  # Twc
                pose_cw = inv_T(pose_wc)
                Pc = (pose_cw[:3, :3] @ point_position + pose_cw[:3, 3].reshape(3, 1)).T
                # uv, depth = kf.camera.project(Pc)
                depth = Pc.ravel()[2]
                chi2_check_failure = depth <= kMinDepth

            if chi2_check_failure:
                num_bad_edges += 1
                factor.set_weight(kWeightForDisabledFactor)  # disable the factor
            else:
                noise_model.set_robust_model_active(False)  # we use the isotropic noise model

        # optimize again without outliers

        params = gtsam.LevenbergMarquardtParams().CeresDefaults()
        params.setlambdaInitial(1e-5)  # Matches g2o’s _tau
        params.setlambdaLowerBound(1e-7)  # Prevent over-reduction
        params.setlambdaUpperBound(1e3)  # Prevent excessive increase
        params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
        params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
        params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

        params.setMaxIterations(rounds)
        if verbose:
            params.setVerbosityLM("SUMMARY")

        new_optimizer = gtsam.LevenbergMarquardtOptimizer(graph, result, params)
        new_result = new_optimizer.optimize()
        result = new_result

    # search for final outlier observations and clean map
    num_bad_observations = 0  # final bad observations
    outliers_factors_data = []

    total_error = 0
    num_inlier_observations = 0

    for (factor, noise_model), (p, kf, p_idx, is_stereo) in graph_factors.items():

        # if p.is_bad: # redundant check since the considered points come from good_points
        #     continue

        assert kf.get_point_match(p_idx) is p

        factor.set_weight(1.0)  # reset the factor weight to 1.0 to compute a meaningful chi2
        chi2 = 2.0 * factor.error(
            result
        )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.

        chi2_check_failure = chi2 > (chi2Stereo if is_stereo else chi2Mono)
        if not chi2_check_failure:
            point_key = point_keys[p]
            pose_key = keyframe_keys[kf]
            point_position = np.asarray(result.atPoint3(point_key)).reshape(3, 1)
            pose_wc = np.asarray(result.atPose3(pose_key).matrix())  # Twc
            pose_cw = inv_T(pose_wc)
            Pc = (pose_cw[:3, :3] @ point_position + pose_cw[:3, 3].reshape(3, 1)).T
            # uv, depth = kf.camera.project(Pc)
            depth = Pc.ravel()[2]
            chi2_check_failure = depth <= kMinDepth

        if chi2_check_failure:
            num_bad_observations += 1
            outliers_factors_data.append((p, kf, p_idx, is_stereo))
        else:
            num_inlier_observations += 1
            total_error += chi2

    if map_lock is None:
        map_lock = threading.RLock()  # put a fake lock

    with map_lock:
        # remove outlier observations
        for p, kf, p_idx, is_stereo in outliers_factors_data:
            p_f = kf.get_point_match(p_idx)
            if p_f is not None:
                assert p_f is p
                p.remove_observation(kf, p_idx)
                # the following instruction is now included in p.remove_observation()
                # f.remove_point(p)   # it removes multiple point instances (if these are present)
                # f.remove_point_match(p_idx) # this does not remove multiple point instances, but now there cannot be multiple instances any more

        # put frames back
        for kf, pose_key in keyframe_keys.items():
            pose_estimated = result.atPose3(pose_key)
            Twc = pose_estimated.matrix()
            kf.update_pose(inv_T(Twc))
            kf.lba_count += 1

        # put points back
        if not fixed_points:
            for p, point_key in point_keys.items():
                p.update_position(np.asarray(result.atPoint3(point_key)))
                p.update_normal_and_depth(force=True)

    num_active_edges = num_inlier_observations  # num_edges-num_bad_edges
    mean_squared_error = total_error / max(
        num_active_edges, 1
    )  # opt.active_chi2()/max(num_active_edges,1)

    return mean_squared_error, num_bad_observations / max(num_edges, 1)


# ------------------------------------------------------------------------------------------


class SimResectioningFactor:
    def __init__(
        self,
        sim_pose_key: int,
        calib: gtsam.Cal3_S2,
        p: gtsam.Point2,
        P: gtsam.Point3,
        noise_model: gtsam.noiseModel.Base,
        weight: float = 1.0,  # Initial weight
    ):
        self.sim_pose_key = sim_pose_key
        self.calib = calib
        self.p = p
        self.P = P
        self.weight = weight
        # Create the CustomFactor with our error function
        self.factor = gtsam.CustomFactor(
            noise_model, gtsam.KeyVector([sim_pose_key]), self.error_func
        )

    def get_factor(self) -> gtsam.NonlinearFactor:
        """Returns the underlying GTSAM factor wrapped with weight management."""
        return self.factor

    def get_weight(self) -> float:
        return self.weight

    def set_weight(self, new_weight: float):
        self.weight = new_weight

    def error_func(
        self, this: gtsam.CustomFactor, values: gtsam.Values, H: list[np.ndarray]
    ) -> np.ndarray:
        # Retrieve similarity transform from the values using a helper function.
        sim3 = gtsam_factors.get_similarity3(values, self.sim_pose_key)

        def compute_error(sim: gtsam.Similarity3) -> np.ndarray:
            R = sim.rotation().matrix()  # 3x3 rotation matrix
            t = sim.translation()  # 3x1 translation vector
            s = sim.scale()  # Scalar scale factor
            # Correct transformation: P' = s * R * P + t
            transformed_P = s * (R @ self.P) + t
            projected = self.calib.K() @ transformed_P
            # Normalize projection and compute residual
            return projected[:2] / projected[2] - self.p

        # Multiply the residual by the weight
        error = self.weight * compute_error(sim3)

        # If Jacobian is requested, compute and weight it
        if H is not None:
            H[0] = self.weight * gtsam_factors.numerical_derivative11_v2_sim3(
                compute_error, sim3, 1e-5
            )

        return error


class SimInvResectioningFactor:
    def __init__(
        self,
        sim_pose_key: int,
        calib: gtsam.Cal3_S2,
        p: gtsam.Point2,
        P: gtsam.Point3,
        noise_model: gtsam.noiseModel.Base,
        weight: float = 1.0,  # Initial weight
    ):
        self.sim_pose_key = sim_pose_key
        self.calib = calib
        self.p = p
        self.P = P
        self.weight = weight
        self.factor = gtsam.CustomFactor(
            noise_model, gtsam.KeyVector([sim_pose_key]), self.error_func
        )

    def get_factor(self) -> gtsam.NonlinearFactor:
        """Returns the underlying GTSAM factor wrapped with weight management."""
        return self.factor

    def get_weight(self) -> float:
        return self.weight

    def set_weight(self, new_weight: float):
        self.weight = new_weight

    def error_func(
        self, this: gtsam.CustomFactor, values: gtsam.Values, H: list[np.ndarray]
    ) -> np.ndarray:
        # Retrieve similarity transform
        sim3 = gtsam_factors.get_similarity3(values, self.sim_pose_key)

        def compute_error(sim: gtsam.Similarity3) -> np.ndarray:
            R = sim.rotation().matrix()  # 3x3 rotation matrix
            t = sim.translation()  # 3x1 translation vector
            s = sim.scale()  # Scalar scale factor
            # Compute inverse transformation:
            R_inv = R.T / s  # Inverse rotation scaled by 1/s
            t_inv = -R_inv @ t  # Inverse translation
            transformed_P = R_inv @ self.P + t_inv
            projected = self.calib.K() @ transformed_P
            # Normalized projection
            return projected[:2] / projected[2] - self.p

        error = self.weight * compute_error(sim3)

        if H is not None:
            H[0] = self.weight * gtsam_factors.numerical_derivative11_v2_sim3(
                compute_error, sim3, 1e-5
            )
        return error


def sim_resectioning_factor_py(
    noise_model: gtsam.noiseModel.Base,
    sim_pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    host_factor = SimResectioningFactor(sim_pose_key, calib, p, P, noise_model)
    factor = host_factor.get_factor()
    return factor, host_factor


def sim_inv_resectioning_factor_py(
    noise_model: gtsam.noiseModel.Base,
    sim_pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    host_factor = SimInvResectioningFactor(sim_pose_key, calib, p, P, noise_model)
    factor = host_factor.get_factor()
    return factor, host_factor


def sim_resectioning_factor(
    noise_model: gtsam.noiseModel.Base,
    sim_pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    host_factor = gtsam_factors.SimResectioningFactor(sim_pose_key, calib, p, P, noise_model)
    factor = host_factor
    return factor, host_factor


def sim_inv_resectioning_factor(
    noise_model: gtsam.noiseModel.Base,
    sim_pose_key: int,
    calib: gtsam.Cal3_S2,
    p: gtsam.Point2,
    P: gtsam.Point3,
) -> gtsam.NonlinearFactor:
    host_factor = gtsam_factors.SimInvResectioningFactor(sim_pose_key, calib, p, P, noise_model)
    factor = host_factor
    return factor, host_factor


# [WIP] Not stable yet!
# The goal of the optimize_sim3 method is to estimate the Sim(3) transformation (R12,t12,s12)
# between two keyframes (kf1 and kf2) in a SLAM system. This optimization compute the Sim(3)
# transformation that minimizes the reprojection errors of the 3D matched map points of kf1 into kf2
# and the reprojection errors of the 3D matched map points of kf2 into kf1.
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

    sigma_for_fixed = kSigmaForFixed  # sigma used for fixed entities

    sim_resectioning_factor_fn = sim_resectioning_factor
    sim_inv_resectioning_factor_fn = sim_inv_resectioning_factor
    if platform.system() == "Darwin":
        # NOTE: Under macOS I found some interface issues with the pybindings of the resectioning factors.
        sim_resectioning_factor_fn = sim_resectioning_factor_py
        sim_inv_resectioning_factor_fn = sim_inv_resectioning_factor_py

    # Calibration and Camera Poses
    cam1 = kf1.camera
    K1_mono = gtsam.Cal3_S2(cam1.fx, cam1.fy, 0, cam1.cx, cam1.cy)

    cam2 = kf2.camera
    K2_mono = gtsam.Cal3_S2(cam2.fx, cam2.fy, 0, cam2.cx, cam2.cy)

    R1w, t1w = kf1.Rcw.copy(), kf1.tcw.copy().reshape(3, 1)
    R2w, t2w = kf2.Rcw.copy(), kf2.tcw.copy().reshape(3, 1)

    graph = gtsam.NonlinearFactorGraph()
    initial_estimate = gtsam.Values()

    # Initial Sim3 transformation
    sim3_init = gtsam.Similarity3(gtsam.Rot3(R12.copy()), gtsam.Point3(t12.ravel().copy()), s12)
    # initial_estimate.insert(X(0), sim3_init)
    gtsam_factors.insert_similarity3(initial_estimate, X(0), sim3_init)

    # print(f'Inserted sim3: R: {sim3_init.rotation()}, t: {sim3_init.translation()}, s: {sim3_init.scale()}')
    # sim3_init_back = gtsam_factors.get_similarity3(initial_estimate, X(0))
    # print(f'Getting back sim3: R: {sim3_init_back.rotation()}, t: {sim3_init_back.translation()}, s: {sim3_init_back.scale()}')

    if fix_scale:
        # Create a PriorFactor to fix the scale of the Sim3 transformation
        scale_prior = gtsam_factors.PriorFactorSimilarity3ScaleOnly(X(0), s12, sigma_for_fixed)
        graph.add(scale_prior)

    # MapPoint vertices and edges
    if map_points1 is None:
        map_points1 = kf1.get_points()
    num_matches = len(map_point_matches12)
    assert num_matches == len(map_points1)

    factors_12_data = []
    factors_21_data = []
    match_idxs = []

    delta_huber = np.sqrt(th2)

    level_sigmas = FeatureTrackerShared.feature_manager.level_sigmas

    num_correspondences = 0
    for i in range(num_matches):
        mp1 = map_points1[i]
        if mp1 is None or mp1.is_bad:
            continue
        mp2 = map_point_matches12[i]  # map point of kf2 matched with i-th map point of kf1
        if mp2 is None or mp2.is_bad:
            continue

        index2 = mp2.get_observation_idx(kf2)
        if index2 >= 0:

            sigma2_12 = level_sigmas[kf1.octaves[i]]
            robust_noise_12 = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(delta_huber),
                gtsam.noiseModel.Isotropic.Sigma(2, sigma2_12),
            )

            sigma2_21 = level_sigmas[kf2.octaves[index2]]
            robust_noise_21 = gtsam.noiseModel.Robust.Create(
                gtsam.noiseModel.mEstimator.Huber.Create(delta_huber),
                gtsam.noiseModel.Isotropic.Sigma(2, sigma2_21),
            )

            # Create a factor 12 (project mp2_2 on camera 1 by using sim3(R12, t12, s12) to transform mp2_2 in mp2_1)
            p2_c2 = R2w @ mp2.pt.reshape(3, 1) + t2w
            factor_12, host_factor_12 = sim_resectioning_factor_fn(
                robust_noise_12,
                X(0),
                K1_mono,
                gtsam.Point2(kf1.kpsu[i].ravel()),
                gtsam.Point3(p2_c2.ravel()),
            )
            graph.add(factor_12)

            # Create a factor 21 (project mp1_1 on camera 2 by using sim3(R21, t21, s21).inverse() to transform mp1_1 in mp1_2)
            p1_c1 = R1w @ mp1.pt.reshape(3, 1) + t1w
            factor_21, host_factor_21 = sim_inv_resectioning_factor_fn(
                robust_noise_21,
                X(0),
                K2_mono,
                gtsam.Point2(kf2.kpsu[index2].ravel()),
                gtsam.Point3(p1_c1.ravel()),
            )
            graph.add(factor_21)

            factors_12_data.append((factor_12, p2_c2, host_factor_12))
            factors_21_data.append((factor_21, p1_c1, host_factor_21))

            match_idxs.append(i)

            num_correspondences += 1

    if verbose:
        print(f"optimize_sim3: num_correspondences = {num_correspondences}")

    if num_correspondences < 10:
        print(f"optimize_sim3: Too few inliers, num_correspondences = {num_correspondences}")
        return 0, None, None, None, 0

    intial_error = graph.error(initial_estimate)

    # Optimizer
    params = gtsam.LevenbergMarquardtParams().CeresDefaults()
    params.setlambdaInitial(1e-5)  # Matches g2o’s _tau
    params.setlambdaLowerBound(1e-7)  # Prevent over-reduction
    params.setlambdaUpperBound(1e3)  # Prevent excessive increase
    params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

    params.setMaxIterations(5)
    if verbose:
        params.setVerbosityLM("SUMMARY")

    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    # sim3_optimized = result.at(gtsam.Similarity3, X(0))
    sim3_optimized = gtsam_factors.get_similarity3(result, X(0))
    R12_opt = sim3_optimized.rotation().matrix()
    t12_opt = sim3_optimized.translation().reshape(3, 1)
    s12_opt = sim3_optimized.scale()

    R21_opt = R12_opt.T / s12_opt
    t21_opt = (-R21_opt @ t12_opt).reshape(3, 1)

    assert len(factors_12_data) == len(factors_21_data)
    assert len(factors_12_data) == len(match_idxs)

    # Check inliers
    num_bad = 0
    for i, (factor_12, p2_c2, host_factor_12) in enumerate(factors_12_data):
        factor_21, p1_c1, host_factor_21 = factors_21_data[i]

        p2_c1 = s12_opt * R12_opt @ p2_c2.reshape(3, 1) + t12_opt
        # uv2, z2 = kf1.camera.project(p2_c1)

        p1_c2 = R21_opt @ p1_c1.reshape(3, 1) + t21_opt
        # uv1, z1 = kf2.camera.project(p1_c2)

        chi2_12 = 2.0 * factor_12.error(
            result
        )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
        chi2_21 = 2.0 * factor_21.error(
            result
        )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.

        if chi2_12 > th2 or not p2_c1[2] > 0 or chi2_21 > th2 or not p1_c2[2] > 0:
            index = match_idxs[i]
            map_points1[index] = None
            host_factor_12.set_weight(kWeightForDisabledFactor)
            host_factor_21.set_weight(kWeightForDisabledFactor)
            factors_12_data[i] = None
            factors_21_data[i] = None
            num_bad += 1

    if verbose:
        print(f"optimize_sim3: num_correspondences = {num_correspondences}")

    num_more_iterations = 10 if num_bad > 0 else 5

    if num_correspondences - num_bad < 10:
        print(
            f"optimize_sim3: Too few inliers, num_correspondences = {num_correspondences}, num_bad = {num_bad}"
        )
        return 0, None, None, None, 0  # num_inliers, R,t,scale, delta_err

    params = gtsam.LevenbergMarquardtParams().CeresDefaults()
    params.setlambdaInitial(1e-5)  # Matches g2o’s _tau
    params.setlambdaLowerBound(1e-7)  # Prevent over-reduction
    params.setlambdaUpperBound(1e3)  # Prevent excessive increase
    params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

    params.setMaxIterations(num_more_iterations)
    if verbose:
        params.setVerbosityLM("SUMMARY")

    initial_estimate = result
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_estimate, params)
    result = optimizer.optimize()

    delta_err = graph.error(result) - intial_error

    sim3_optimized = gtsam_factors.get_similarity3(result, X(0))
    R12_opt = sim3_optimized.rotation().matrix()
    t12_opt = sim3_optimized.translation().reshape(3, 1)
    s12_opt = sim3_optimized.scale()

    R21_opt = R12_opt.T / s12_opt
    t21_opt = (-R21_opt @ t12_opt).reshape(3, 1)

    num_inliers = 0
    for i, f12di in enumerate(factors_12_data):
        f21di = factors_21_data[i]
        if f12di and f21di:
            factor_12, p2_c2, host_factor_12 = f12di
            factor_21, p1_c1, host_factor_21 = f21di

            p2_c1 = s12_opt * R12_opt @ p2_c2.reshape(3, 1) + t12_opt.reshape(3, 1)
            # uv2, z2 = kf1.camera.project(p2_c1)

            p1_c2 = R21_opt @ p1_c1.reshape(3, 1) + t21_opt.reshape(3, 1)
            # uv1, z1 = kf2.camera.project(p1_c2)

            host_factor_12.set_weight(
                1.0
            )  # reset weight to 1.0 to activate the factor and correctly compute the error
            host_factor_21.set_weight(
                1.0
            )  # reset weight to 1.0 to activate the factor and correctly compute the error
            chi2_12 = 2.0 * factor_12.error(
                result
            )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
            chi2_21 = 2.0 * factor_21.error(
                result
            )  # from the gtsam code comments, error() is typically equal to log-likelihood, e.g. 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.

            if chi2_12 > th2 or not p2_c1[2] > 0 or chi2_21 > th2 or not p1_c2[2] > 0:
                index = match_idxs[i]
                map_points1[index] = None
            else:
                num_inliers += 1

    # Retrieve optimized Sim3
    # sim3_optimized = result.at(gtsam.Similarity3, X(0))
    sim3_optimized = gtsam_factors.get_similarity3(result, X(0))

    scale_out = sim3_optimized.scale() if not fix_scale else s12

    return (
        num_correspondences,
        sim3_optimized.rotation().matrix(),
        sim3_optimized.translation(),
        scale_out,
        delta_err,
    )


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

    sigma_for_fixed = kSigmaForFixed  # sigma used for fixed entities
    sigma_for_visual = 1  # sigma used for visual factors

    # Setup optimizer
    graph = gtsam.NonlinearFactorGraph()

    # Initial values and optimized poses
    initial_values = gtsam.Values()

    all_keyframes = map_object.get_keyframes()
    all_map_points = map_object.get_points()
    max_keyframe_id = map_object.max_keyframe_id

    vec_Scw = [None] * (max_keyframe_id + 1)  # use keyframe kid as index here
    vec_corrected_Swc = [None] * (max_keyframe_id + 1)  # use keyframe kid as index here
    # vertices = [None] * (max_keyframe_id + 1)          # use keyframe kid as index here

    min_number_features = 100

    # Set KeyFrame intial values
    for keyframe in all_keyframes:
        if keyframe.is_bad:
            continue

        keyframe_id = keyframe.kid

        try:  # if keyframe in corrected_sim3_map:
            corrected_sim3 = corrected_sim3_map[keyframe]
            Siw = Sim3Pose(R=corrected_sim3.R.copy(), t=corrected_sim3.t.copy(), s=corrected_sim3.s)
        except:
            Siw = Sim3Pose(keyframe.Rcw.copy(), keyframe.tcw.copy(), 1.0)
        vec_Scw[keyframe_id] = Siw

        Swi = Siw.inverse()
        Swi_gtsam = gtsam.Similarity3(gtsam.Rot3(Swi.R), gtsam.Point3(Swi.t.ravel()), Swi.s)
        # initial_values.insert(X(keyframe_id), Siw_gtsam)
        gtsam_factors.insert_similarity3(initial_values, X(keyframe_id), Swi_gtsam)

        if keyframe == loop_keyframe:
            # Create a PriorFactor to fix the Sim3 transformation
            fixed_sim3_prior = gtsam_factors.PriorFactorSimilarity3(
                X(keyframe_id), Swi_gtsam, gtsam.noiseModel.Isotropic.Sigma(7, sigma_for_fixed)
            )
            graph.add(fixed_sim3_prior)

        if fix_scale:
            # Create a PriorFactor to fix the scale of the Sim3 transformation
            fixed_scale_prior = gtsam_factors.PriorFactorSimilarity3ScaleOnly(
                X(keyframe_id), Swi.s, sigma_for_fixed
            )
            graph.add(fixed_scale_prior)

    num_graph_edges = 0

    inserted_loop_edges = set()  # set of pairs (keyframe_id, connected_keyframe_id)

    # Loop edges
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
                continue

            Sjw = vec_Scw[connected_id]
            Sji = Sjw @ Swi

            Sij = Sji.inverse()  # the measure is Swci.inverse() * Swcj = Scicj = Sij
            Sij_gtsam = gtsam.Similarity3(gtsam.Rot3(Sij.R), gtsam.Point3(Sij.t.ravel()), Sij.s)

            edge = gtsam_factors.BetweenFactorSimilarity3(
                X(keyframe_id),
                X(connected_id),
                Sij_gtsam,
                gtsam.noiseModel.Isotropic.Sigma(7, sigma_for_visual),
            )
            graph.add(edge)
            num_graph_edges += 1
            inserted_loop_edges.add(
                (min(keyframe_id, connected_id), max(keyframe_id, connected_id))
            )

    # Normal edges
    for keyframe in all_keyframes:
        keyframe_id = keyframe.kid
        parent_keyframe = keyframe.get_parent()

        # Spanning tree edge
        if parent_keyframe:
            parent_id = parent_keyframe.kid

            try:
                Swi = non_corrected_sim3_map[keyframe].inverse()
            except:
                Swi = vec_Scw[keyframe_id].inverse()

            try:
                Sjw = non_corrected_sim3_map[parent_keyframe].copy()
            except:
                Sjw = vec_Scw[parent_id]

            Sji = Sjw @ Swi

            Sij = Sji.inverse()  # the measure is Swci.inverse() * Swcj = Scicj = Sij
            Sij_gtsam = gtsam.Similarity3(gtsam.Rot3(Sij.R), gtsam.Point3(Sij.t.ravel()), Sij.s)

            edge = gtsam_factors.BetweenFactorSimilarity3(
                X(keyframe_id),
                X(parent_id),
                Sij_gtsam,
                gtsam.noiseModel.Isotropic.Sigma(7, sigma_for_visual),
            )
            graph.add(edge)
            num_graph_edges += 1

        # Loop edges
        for loop_edge in keyframe.get_loop_edges():
            if loop_edge.kid < keyframe_id:
                try:
                    Slw = non_corrected_sim3_map[loop_edge].copy()
                except:
                    Slw = vec_Scw[loop_edge.kid]  # already copy

                Sli = Slw @ Swi

                Sil = Sli.inverse()  # the measure is Swci.inverse() * Swcl = Scicl = Sil
                Sil_gtsam = gtsam.Similarity3(gtsam.Rot3(Sil.R), gtsam.Point3(Sil.t.ravel()), Sil.s)

                edge = gtsam.BetweenFactorSimilarity3(
                    X(keyframe_id),
                    X(loop_edge.kid),
                    Sil_gtsam,
                    gtsam.noiseModel.Isotropic.Sigma(7, sigma_for_visual),
                )
                graph.add(edge)
                num_graph_edges += 1

        # Covisibility graph edges
        for connected_keyframe in keyframe.get_covisible_by_weight(min_number_features):
            if (
                connected_keyframe != parent_keyframe
                and not keyframe.has_child(connected_keyframe)
                and connected_keyframe.kid < keyframe_id
                and not connected_keyframe.is_bad
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

                Sin = Sni.inverse()  # the measure is Swci.inverse() * Swcn = Scicn = Sin
                Sin_gtsam = gtsam.Similarity3(gtsam.Rot3(Sin.R), gtsam.Point3(Sin.t.ravel()), Sin.s)

                edge = gtsam_factors.BetweenFactorSimilarity3(
                    X(keyframe_id),
                    X(connected_keyframe.kid),
                    Sin_gtsam,
                    gtsam.noiseModel.Isotropic.Sigma(7, sigma_for_visual),
                )
                graph.add(edge)
                num_graph_edges += 1

    if verbose:
        print_fun(f"[optimize_essential_graph]: Total number of graph edges: {num_graph_edges}")

    # Optimize
    params = gtsam.LevenbergMarquardtParams().CeresDefaults()
    params.setlambdaInitial(1e-16)  # As in optimzer_g2o version of optimize_essential_graph()
    params.setlambdaLowerBound(1e-8)  # Prevent over-reduction
    params.setlambdaUpperBound(1e3)  # Prevent excessive increase
    params.setlambdaFactor(2.0)  # Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(True)  # Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(False)  # Mimics g2o’s ni

    params.setMaxIterations(20)
    if verbose:
        params.setVerbosityLM("SUMMARY")
    optimizer = gtsam.LevenbergMarquardtOptimizer(graph, initial_values, params)
    # optimizer = gtsam_factors.LevenbergMarquardtOptimizerG2o(graph, initial_values, params)
    result = optimizer.optimize()

    # SE3 Pose Recovering.  Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for keyframe in all_keyframes:
        keyframe_id = keyframe.kid

        # corrected_Swi = optimizer.values().atPose3(keyframe_id)
        corrected_Swi = gtsam_factors.get_similarity3(result, X(keyframe_id))

        R = corrected_Swi.rotation().matrix()
        t = corrected_Swi.translation()
        s = corrected_Swi.scale()
        Swi = Sim3Pose(R, t, s)
        Siw = Swi.inverse()
        vec_corrected_Swc[keyframe_id] = Swi

        Tiw = poseRt(Siw.R, Siw.t.ravel() / Siw.s)  # [R t/s; 0 1]
        keyframe.update_pose(Tiw)

    # Correct points: Transform to "non-optimized" reference keyframe pose and transform back with optimized pose
    for map_point in all_map_points:
        if map_point.is_bad:
            continue

        if map_point.corrected_by_kf == current_keyframe.kid:
            reference_id = map_point.corrected_reference
        else:
            reference_keyframe = map_point.get_reference_keyframe()
            reference_id = reference_keyframe.kid

        Srw = vec_Scw[reference_id]
        corrected_Swr = vec_corrected_Swc[reference_id]
        if Srw is None or corrected_Swr is None:
            continue

        P3Dw = map_point.pt
        corrected_P3Dw = corrected_Swr.map(Srw.map(P3Dw)).ravel()
        map_point.update_position(corrected_P3Dw)

        map_point.update_normal_and_depth()

    mean_squared_error = graph.error(result) / max(num_graph_edges, 1)
    return mean_squared_error
