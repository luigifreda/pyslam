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

from threading import Thread, Condition, RLock, Lock
import numpy as np
import cv2
from collections import defaultdict
import os
import time

from pyslam.utilities.logging import Printer
from pyslam.utilities.logging import Logging
from pyslam.utilities.multi_processing import MultiprocessingManager
from pyslam.utilities.img_management import LoopCandidateImgs
from pyslam.utilities.features import transform_float_to_binary_descriptor
from pyslam.utilities.data_management import empty_queue
from pyslam.utilities.drawing import draw_feature_matches
from pyslam.utilities.timer import TimerFps

from pyslam.viz.qimage_thread import QimageViewer

from pyslam.loop_closing.loop_detector_configs import LoopDetectorConfigs

from pyslam.slam import (
    Sim3Pose,
    KeyFrame,
    Frame,
    Map,
    optimizer_gtsam,
    optimizer_g2o,
    RotationHistogram,
    ProjectionMatcher,
)
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.slam.relocalizer import Relocalizer  # to avoid circular import
from pyslam.slam.global_bundle_adjustment import GlobalBundleAdjustment  # to avoid circular import

from pyslam.slam.frame import (
    compute_frame_matches,
    prepare_input_data_for_sim3solver,
    prepare_input_data_for_pnpsolver,
)

from pyslam.io.dataset_types import SensorType

from .loop_detecting_process import LoopDetectingProcess
from .loop_detector_base import LoopDetectorTask, LoopDetectorTaskType, LoopDetectorOutput
from pyslam.config_parameters import Parameters

import traceback
import platform
import pickle

import sim3solver
import pnpsolver

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pyslam.slam.slam import Slam  # Only imported when type checking, not at runtime
    from pyslam.slam.keyframe import KeyFrame
    from pyslam.slam.frame import Frame
    from pyslam.slam.map_point import MapPoint
    from pyslam.slam.map import Map
    from pyslam.slam.optimizer_g2o import optimizer_g2o
    from pyslam.slam.optimizer_gtsam import optimizer_gtsam
    from pyslam.slam.rotation_histogram import RotationHistogram


kVerbose = True
kTimerVerbose = False  # set this to True if you want to print timings
kPrintTrackebackDetails = True

kUseCv2ForDrawing = platform.system() != "Darwin"  # under mac we can't use cv2 imshow here

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


class ConsistencyGroup:
    def __init__(self, keyframes=None, consistency=None):
        self.keyframes = set() if keyframes is None else set(keyframes)
        self.consistency = 0 if consistency is None else consistency

    def __str__(self) -> str:
        return f"keyframes = {[kf.id for kf in self.keyframes]}, consistency = {self.consistency}"


# This checks the consistency of a candidates along loop detections.
class LoopGroupConsistencyChecker:
    def __init__(self, consistency_threshold=3):
        self.consistent_groups: list[ConsistencyGroup] = []  # type: list[ConsistencyGroup]
        self.consistency_threshold = consistency_threshold
        self.enough_consistent_candidates: list[KeyFrame] = (
            []
        )  # current set of enough consistent loop candidates

        self.timer = TimerFps("LoopGroupConsistencyChecker", is_verbose=kTimerVerbose)

    def clear_consistency_groups(self):
        self.consistent_groups = []

    def check_candidates(self, current_keyframe, candidate_keyframes):
        # For each loop candidate, check consistency with previous loop candidates.
        # Each candidate expands a covisibility group (keyframes connected to the loop candidate in the covisibility graph).
        # A group is consistent with a previous group if they share at least one keyframe.
        # We must detect a consistent loop in several consecutive keyframes to accept it.

        self.timer.start()

        self.enough_consistent_candidates = []

        # Recompute and update consistent groups
        current_consistent_groups = []
        is_consistent_group_updated = [False] * len(self.consistent_groups)

        for candidate_kf in candidate_keyframes:
            if candidate_kf.is_bad():
                continue
            # compute the expanded group of candidate keyframe
            candidate_kf_group = candidate_kf.get_connected_keyframes()
            candidate_kf_group.append(candidate_kf)

            is_candidate_kf_enough_consistent = False
            is_candidate_kf_consistent_for_some_group = False

            for consistent_group_idx, consistent_group in enumerate(self.consistent_groups):
                is_consistent = False

                for kf in candidate_kf_group:
                    if kf in consistent_group.keyframes:
                        is_consistent = True
                        is_candidate_kf_consistent_for_some_group = True
                        break

                if is_consistent:
                    current_consistency = consistent_group.consistency + 1
                    if not is_consistent_group_updated[consistent_group_idx]:
                        consistent_group = ConsistencyGroup(candidate_kf_group, current_consistency)
                        current_consistent_groups.append(consistent_group)
                        is_consistent_group_updated[consistent_group_idx] = (
                            True  # Avoid including the same group more than once
                        )

                    if (
                        current_consistency >= self.consistency_threshold
                        and not is_candidate_kf_enough_consistent
                    ):
                        self.enough_consistent_candidates.append(candidate_kf)
                        is_candidate_kf_enough_consistent = (
                            True  # Avoid inserting the same candidate more than once
                        )

            # If the group is not consistent with any previous group, insert with consistency counter set to zero
            if not is_candidate_kf_consistent_for_some_group:
                consistent_group = ConsistencyGroup(candidate_kf_group, 0)
                current_consistent_groups.append(consistent_group)

        # Update covisibility consistent groups
        self.consistent_groups = current_consistent_groups

        if False:
            LoopClosing.print(f"LoopGroupConsistencyChecker:")
            for i, g in enumerate(self.consistent_groups):
                LoopClosing.print(f"\tconsistent group[{i}]: {g}")

        self.timer.refresh()
        LoopClosing.print(
            f"LoopGroupConsistencyChecker: consistency check elapsed time: {self.timer.last_elapsed}"
        )

        if len(self.enough_consistent_candidates) == 0:
            return False
        else:
            return True


class LoopGeometryChecker:
    def __init__(self, is_monocular=False, map_frame_id_to_img=None):
        self.is_monocular = is_monocular
        self.success_loop_kf = None
        self.success_loop_kf_sim3_pose = None
        self.success_map_point_matches = None
        self.success_map_point_matches_idxs = None
        self.success_loop_map_points = set()
        self.map_frame_id_to_img = map_frame_id_to_img

        self.timer = TimerFps("LoopGeometryChecker", is_verbose=kTimerVerbose)

    def check_candidates(self, current_keyframe, candidate_keyframes):
        kp_match_idxs = defaultdict(
            lambda: (None, None)
        )  # dictionary of keypointmatches  (kf_i, kf_j) -> (idxs_i,idxs_j)
        try:
            self.timer.start()
            kp_match_idxs = compute_frame_matches(
                current_keyframe,
                candidate_keyframes,
                kp_match_idxs,
                do_parallel=Parameters.kLoopClosingParallelKpsMatching,
                max_workers=Parameters.kLoopClosingParallelKpsMatchingNumWorkers,
                ratio_test=Parameters.kLoopClosingFeatureMatchRatioTest,
                print_fun=print,
            )

            solvers = []
            considered_candidates = []
            mp_match_idxs = defaultdict(
                lambda: (None, None)
            )  # dictionary of map point matches  (kf_i, kf_j) -> (idxs_i,idxs_j)
            for i, kf in enumerate(candidate_keyframes):
                if kf is current_keyframe or kf.is_bad():
                    continue

                # extract matches from precomputed map
                idxs_kf_cur, idxs_kf = kp_match_idxs[(current_keyframe, kf)]
                assert len(idxs_kf_cur) == len(idxs_kf)

                # if features have descriptors with orientation then let's check the matches with a rotation histogram
                if FeatureTrackerShared.oriented_features:
                    # num_matches_before = len(idxs_kf_cur)
                    valid_match_idxs = RotationHistogram.filter_matches_with_histogram_orientation(
                        idxs_kf_cur, idxs_kf, current_keyframe.angles, kf.angles
                    )
                    if len(valid_match_idxs) > 0:
                        idxs_kf_cur = idxs_kf_cur[valid_match_idxs]
                        idxs_kf = idxs_kf[valid_match_idxs]
                    # print(f'LoopGeometryChecker: rotation histogram filter: #matches ({current_keyframe.id},{kf.id}): before {num_matches_before}, after {len(idxs_kf_cur)}')

                num_matches = len(idxs_kf_cur)
                LoopClosing.print(
                    f"LoopGeometryChecker: num_matches ({current_keyframe.id},{kf.id}): {num_matches}"
                )

                if num_matches < Parameters.kLoopClosingGeometryCheckerMinKpsMatches:
                    LoopClosing.print(
                        f"LoopGeometryChecker: skipping keyframe {kf.id} with too few matches ({num_matches}) (min: {Parameters.kLoopClosingGeometryCheckerMinKpsMatches})"
                    )
                    continue

                kf.set_not_erase()

                points_3d_w1, points_3d_w2, sigmas2_1, sigmas2_2, idxs1, idxs2 = (
                    prepare_input_data_for_sim3solver(current_keyframe, kf, idxs_kf_cur, idxs_kf)
                )
                # fill the dictionary of map point matches (its content needs to be cleaned up later with found inliers)
                mp_match_idxs[(current_keyframe, kf)] = (idxs1, idxs2)

                solver_input_data = sim3solver.Sim3SolverInput()
                solver_input_data.fix_scale = not self.is_monocular

                solver_input_data.K1 = current_keyframe.camera.K
                current_keyframe_Tcw = current_keyframe.Tcw()
                solver_input_data.Rcw1 = current_keyframe_Tcw[:3, :3]
                solver_input_data.tcw1 = current_keyframe_Tcw[:3, 3]

                solver_input_data.K2 = kf.camera.K
                kf_Tcw = kf.Tcw()
                solver_input_data.Rcw2 = kf_Tcw[:3, :3]
                solver_input_data.tcw2 = kf_Tcw[:3, 3]

                solver_input_data.points_3d_w1 = points_3d_w1
                solver_input_data.points_3d_w2 = points_3d_w2

                solver_input_data.sigmas2_1 = sigmas2_1
                solver_input_data.sigmas2_2 = sigmas2_2

                solver = sim3solver.Sim3Solver(solver_input_data)
                solver.set_ransac_parameters(0.99, 20, 300)

                solvers.append(solver)
                considered_candidates.append(kf)

            map_points1 = current_keyframe.get_points()
            n1 = len(map_points1)

            # check if candidates get a valid solution
            self.success_loop_kf = None
            for i, kf in enumerate(considered_candidates):
                # perform 5 ransac iterations on each solver
                transformation, is_no_more, inlier_flags, num_inliers, is_converged = solvers[
                    i
                ].iterate(5)
                inlier_flags = np.array(inlier_flags, dtype=bool)  # from from int8 to bool
                if (
                    is_converged and not is_no_more
                ):  # is_no_more becomes true when the all the iterations have been peformed in the attempt to converge
                    R12 = solvers[i].get_estimated_rotation()
                    t12 = solvers[i].get_estimated_translation()
                    scale12 = solvers[i].get_estimated_scale()
                    error3d = solvers[i].compute_3d_registration_error()
                    LoopClosing.print(
                        f"LoopGeometryChecker: candidate {kf.id} converged, num_inliers: {num_inliers}, error3d: {error3d},\n R12: {R12}, t12: {t12}, scale12: {scale12}"
                    )

                    idxs1, idxs2 = mp_match_idxs[(current_keyframe, kf)]
                    idxs1 = idxs1[inlier_flags]
                    idxs2 = idxs2[inlier_flags]
                    # Now, current_keyframe.points(idxs1[i]) is matched with kf.points(idxs2[i])

                    # Perform a guided matching and next optimize with all found correspondences
                    num_found_matches, matches12, matches21 = ProjectionMatcher.search_by_sim3(
                        current_keyframe, kf, idxs1, idxs2, scale12, R12, t12, print_fun=print
                    )

                    # NOTE:
                    # matches12: where kf2.points(matches12[i]) is matched to i-th map point in kf1 if matches12[i]>0    (from 1 to 2)
                    # matches21: where kf1.points(matches21[i]) is matched to i-th map point in kf2 if matches21[i]>0    (from 2 to 1)
                    LoopClosing.print(
                        f"LoopGeometryChecker: guided matching (ProjectionMatcher.search_by_sim3) - found map point matches ({current_keyframe.id},{kf.id}): {np.sum(matches12!=-1)}, starting from {len(idxs1)}"
                    )

                    assert len(matches12) == n1
                    map_points2 = kf.get_points()
                    map_point_matches12 = [
                        map_points2[idx] if idx > 0 else None for idx in matches12
                    ]  # from 1 to 2
                    assert len(map_point_matches12) == n1

                    if Parameters.kOptimizationLoopClosingUseGtsam:
                        optimize_sim3_fun = optimizer_gtsam.optimize_sim3
                    else:
                        optimize_sim3_fun = optimizer_g2o.optimize_sim3

                    # optimize with all the found corrispondences
                    num_inliers, R12, t12, scale12, delta_err = optimize_sim3_fun(
                        current_keyframe,
                        kf,
                        map_points1,
                        map_point_matches12,
                        R12,
                        t12,
                        scale12,
                        th2=Parameters.kLoopClosingTh2,
                        fix_scale=not self.is_monocular,
                    )

                    # TODO: add a more robust error check

                    if (
                        num_inliers > Parameters.kLoopClosingGeometryCheckerMinKpsMatches
                        and delta_err < 0
                    ):
                        self.success_loop_kf = kf
                        # compute the update the pose of the successful kf
                        self.success_loop_kf_sim3_pose = Sim3Pose(
                            R12, t12, scale12
                        ) @ Sim3Pose().from_se3_matrix(
                            kf.Tcw()
                        )  # Sc1w = Sc1c2 * Tc2w
                        self.success_map_point_matches = map_point_matches12  # success_map_point_matches[i] is the i-th map point matched in success_loop_kf or None
                        self.success_map_point_matches_idxs = matches12  # success_map_point_matches_idxs[i] is the index of the i-th map point matched in success_loop_kf or -1
                        LoopClosing.print(
                            f"LoopGeometryChecker: optimize_sim3 success - num_inliers: {num_inliers}, delta_err: {delta_err}"
                        )

                        # draw loop image matching for debug
                        if Parameters.kLoopClosingDebugShowLoopMatchedPoints and kUseCv2ForDrawing:
                            try:
                                cur_kf_img = (
                                    current_keyframe.img
                                    if current_keyframe.img is not None
                                    else self.map_frame_id_to_img[current_keyframe.id]
                                )
                                kf_img = (
                                    kf.img
                                    if kf.img is not None
                                    else self.map_frame_id_to_img[kf.id]
                                )
                                loop_img_matches = draw_feature_matches(
                                    cur_kf_img,
                                    kf_img,
                                    current_keyframe.kps[idxs1],
                                    kf.kps[idxs2],
                                    horizontal=False,
                                )
                                # cv2.namedWindow('loop_img_matches', cv2.WINDOW_NORMAL)
                                cv2.imshow("loop_img_matches", loop_img_matches)
                                cv2.waitKey(1)
                            except Exception as e:
                                LoopClosing.print(
                                    f"LoopGeometryChecker: failure while drawing loop image matching failed: {e}"
                                )

                        break  # got loop we can exit!
                    else:
                        LoopClosing.print(
                            f"LoopGeometryChecker: optimize_sim3 failure - num_inliers: {num_inliers}, delta_err: {delta_err}"
                        )

                else:
                    LoopClosing.print(
                        f"LoopGeometryChecker: candidate {kf.id} didnt converge, num_inliers: {num_inliers}"
                    )

            for kf in candidate_keyframes:
                if kf is not self.success_loop_kf:
                    kf.set_erase()

            if self.success_loop_kf is not None:

                # Retrieve map points seen in Loop Keyframe and its neighbors
                success_covisible_group = self.success_loop_kf.get_covisible_keyframes()
                success_covisible_group.append(self.success_loop_kf)
                self.success_loop_map_points = set().union(
                    *(kf.get_matched_good_points() for kf in success_covisible_group)
                )

                # Find more matches projecting the above found map points with the updated Sim3 pose
                num_new_found_points, self.success_map_point_matches = (
                    ProjectionMatcher.search_more_map_points_by_projection(
                        list(self.success_loop_map_points),  # Convert set to list for C++ binding
                        current_keyframe,
                        self.success_loop_kf_sim3_pose,
                        self.success_map_point_matches,
                        self.success_map_point_matches_idxs,
                        max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceMapSearch,
                        print_fun=print,
                    )
                )
                num_matched_map_points = sum(
                    match is not None for match in self.success_map_point_matches
                )

                LoopClosing.print(
                    f"LoopGeometryChecker: num_matched_map_points: {num_matched_map_points}, num_new_found_points by ProjectionMatcher.search_more_map_points_by_projection(): {num_new_found_points}"
                )

                if num_matched_map_points < Parameters.kLoopClosingMinNumMatchedMapPoints:
                    self.success_loop_kf = None

                # for safety repeating
                for kf in candidate_keyframes:
                    if kf is not self.success_loop_kf:
                        kf.set_erase()

            self.timer.refresh()
            LoopClosing.print(
                f"LoopGeometryChecker: geometry check elapsed time: {self.timer.last_elapsed}"
            )

            return self.success_loop_kf is not None

        except Exception as e:
            LoopClosing.print(f"LoopGeometryChecker: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                LoopClosing.print(f"\t traceback details: {traceback_details}")

        return False


class LoopCorrector:
    def __init__(
        self,
        slam: "Slam",
        is_monocular,
        loop_geometry_checker: LoopGeometryChecker,
        GBA: GlobalBundleAdjustment,
    ):
        self.slam = slam
        self.loop_geometry_checker: LoopGeometryChecker = loop_geometry_checker
        self.fix_scale = not is_monocular

        self.GBA = GBA
        self.is_GBA_running = False
        self.stop_GBA = False

        self.corrected_sim3_map = None  # keyframe -> sim3
        self.non_corrected_sim3_map = None  # keyframe -> sim3

        self.mean_graph_chi2_error = None

        self.timer = TimerFps("LoopCorrector", is_verbose=kTimerVerbose)

    @property
    def local_mapping(self):
        return self.slam.local_mapping

    @property
    def map(self):
        return self.slam.map

    def search_and_fuse(self):
        # Project MapPoints observed in the neighborhood of the loop keyframe
        # into the current keyframe and neighbors using corrected poses.
        # Fuse duplications.
        loop_map_points = np.array(list(self.loop_geometry_checker.success_loop_map_points))
        for keyframe, Scw in self.corrected_sim3_map.items():
            replace_points = [None] * len(loop_map_points)
            replace_points = ProjectionMatcher.search_and_fuse_for_loop_correction(
                keyframe, Scw, loop_map_points, replace_points
            )

            # Get map mutex
            with self.map.update_lock:
                # Replace map points
                for i, p_rep in enumerate(replace_points):
                    if p_rep is not None:
                        p_rep.replace_with(loop_map_points[i])

    def correct_loop(self, current_keyframe: KeyFrame):
        LoopClosing.print(f"LoopCorrector: starting...")
        try:
            self.timer.start()

            if self.GBA.is_running():
                LoopClosing.print("LoopCorrector: GBA is running, aborting...")
                self.GBA.abort()  # we'll actually quit it below

            # Send a stop signal to Local Mapping
            # Avoid new keyframes are inserted while correcting the loop
            self.local_mapping.request_stop()

            # wait till local mapping is idle
            self.local_mapping.wait_idle(print=print)

            # ensure current keyframe is updated
            current_keyframe.update_connections()

            # retrieve keyframes connected to the current keyframe and compute corrected Sim3 pose by propagation
            current_connected_keyframes = current_keyframe.get_connected_keyframes()
            current_connected_keyframes.append(current_keyframe)

            self.corrected_sim3_map = {}  # reset dictionary
            self.non_corrected_sim3_map = {}  # reset dictionary

            self.corrected_sim3_map[current_keyframe] = (
                self.loop_geometry_checker.success_loop_kf_sim3_pose
            )

            LoopClosing.print(f"LoopCorrector: updating the map...")
            Twc = current_keyframe.Twc()
            Scw = self.loop_geometry_checker.success_loop_kf_sim3_pose

            with self.map.update_lock:
                # Iterate over all current connected keyframes and propagate the sim3 correction obtained on current keyframe
                for connected_kfi in current_connected_keyframes:
                    Tiw = connected_kfi.Tcw()

                    if connected_kfi != current_keyframe:
                        Tic = Tiw @ Twc
                        Ric = Tic[:3, :3]
                        tic = Tic[:3, 3]
                        Sic = Sim3Pose(Ric, tic, 1.0)
                        corrected_Siw = Sic @ Scw
                        # Pose corrected with the Sim3 of the loop closure
                        self.corrected_sim3_map[connected_kfi] = corrected_Siw

                    Riw = Tiw[:3, :3]
                    tiw = Tiw[:3, 3]
                    Siw = Sim3Pose(Riw, tiw, 1.0)
                    # Pose without correction
                    self.non_corrected_sim3_map[connected_kfi] = Siw

                # Correct all map points observed by current keyframe and its neighbors,
                # so that they align with the other side of the loop
                for connected_kfi, corrected_Siw in self.corrected_sim3_map.items():
                    corrected_Swi = corrected_Siw.inverse()
                    Siw = self.non_corrected_sim3_map[connected_kfi]

                    correction_Sw = corrected_Swi @ Siw
                    correction_sRw = correction_Sw.R * correction_Sw.s
                    correction_tw = correction_Sw.t.reshape(3, 1)

                    # Correct MapPoints
                    map_points = connected_kfi.get_points()
                    for i, map_point in enumerate(map_points):
                        if (
                            not map_point
                            or map_point.is_bad()
                            or map_point.corrected_by_kf == current_keyframe.kid
                        ):  # use kid here
                            continue

                        # Project with non-corrected pose and project back with corrected pose
                        p3dw = map_point.pt().reshape(3, 1)
                        # corrected_p3dw = corrected_Swi @ Siw @ p3dw
                        corrected_p3dw = correction_sRw @ p3dw + correction_tw
                        map_point.update_position(corrected_p3dw.squeeze())
                        map_point.update_normal_and_depth()
                        map_point.corrected_by_kf = current_keyframe.kid  # use kid here
                        map_point.corrected_reference = connected_kfi.kid  # use kid here

                    # Update keyframe pose with corrected Sim3
                    corrected_Tiw = corrected_Siw.to_se3_matrix()  # [R t/s;0 1]
                    connected_kfi.update_pose(corrected_Tiw)
                    connected_kfi.update_connections()

            # Update matched map points and replace if duplicated (fusion loop)
            current_matched_points = self.loop_geometry_checker.success_map_point_matches
            for i, loop_map_point in enumerate(current_matched_points):
                if loop_map_point is not None:
                    cur_map_point = current_keyframe.get_point_match(i)
                    if cur_map_point is not None:
                        cur_map_point.replace_with(loop_map_point)
                    else:
                        loop_map_point.add_observation(current_keyframe, i)
                        loop_map_point.update_best_descriptor()

            LoopClosing.print(f"LoopCorrector: searching and fusing features...")
            # Project MapPoints observed in the neighborhood of the loop keyframe
            # into the current keyframe and neighbors using corrected poses.
            # Fuse duplications.
            self.search_and_fuse()

            LoopClosing.print(f"LoopCorrector: updating covisibility graph...")
            # After the map point fusion, new links in the covisibility graph will appear attaching both sides of the loop

            # Create a dictionary where each key is a KeyFrame and the value is a set of connected KeyFrames
            loop_connections = defaultdict(set)

            for kfi in current_connected_keyframes:
                # Get previous neighbors (covisible keyframes)
                previous_neighbors = kfi.get_covisible_keyframes()

                # Update connections and get the new ones
                kfi.update_connections()
                loop_connections[kfi] = set(kfi.get_connected_keyframes())

                # Remove previous neighbors from connections
                for previous_neighbor in previous_neighbors:
                    try:
                        loop_connections[kfi].remove(previous_neighbor)
                    except:
                        pass  # not found

                # Remove the current connected keyframes from the connection set
                for other_kf in current_connected_keyframes:
                    try:
                        loop_connections[kfi].remove(other_kf)
                    except:
                        pass  # not found

            # For C++ compatibility
            loop_connections = {
                kf: list(connections) for kf, connections in loop_connections.items()
            }

            LoopClosing.print(f"LoopCorrector: optimizing pose graph")
            loop_keyframe = self.loop_geometry_checker.success_loop_kf
            if Parameters.kOptimizationLoopClosingUseGtsam:
                optimize_essential_graph_fun = optimizer_gtsam.optimize_essential_graph
            else:
                optimize_essential_graph_fun = optimizer_g2o.optimize_essential_graph
            self.mean_graph_chi2_error = optimize_essential_graph_fun(
                self.map,
                loop_keyframe,
                current_keyframe,
                self.non_corrected_sim3_map,
                self.corrected_sim3_map,
                loop_connections,
                self.fix_scale,
                print_fun=print,
            )

            # Add loop edge
            loop_keyframe.add_loop_edge(current_keyframe)
            current_keyframe.add_loop_edge(loop_keyframe)

            # If GBA is disabled, rebuild volumetric integration right after pose-graph optimization
            if not Parameters.kUseGBA and self.slam.volumetric_integrator is not None:
                LoopClosing.print(
                    "LoopCorrector: rebuilding volumetric integration after pose-graph optimization (GBA disabled)"
                )
                self.slam.volumetric_integrator.rebuild(self.slam.map)

            # Start global bundle adjustment
            LoopClosing.print(
                f"LoopCorrector: starting global bundle adjustment with loop_keyframe {loop_keyframe.kid}..."
            )
            if Parameters.kUseGBA:
                self.GBA.quit()
                self.GBA.start(loop_keyframe.kid)

            # tell local mapping to restart in normal mode
            self.local_mapping.release()

            self.timer.refresh()
            LoopClosing.print(
                f"LoopCorrector: done - mean_graph_chi2_error: {self.mean_graph_chi2_error}, elapsed time: {self.timer.last_elapsed}"
            )

        except Exception as e:
            LoopClosing.print(f"LoopCorrector: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                LoopClosing.print(f"\t traceback details: {traceback_details}")


# LoopClosing is the main entry point for loop-closure.
# It does the full job of:
# (1) detecting candidates by using visual place recognition methods => LoopDetectingProcess
# (2) verifying them by checking their group consistency => LoopGroupConsistencyChecker
# (3) verifying them by checking their geometry consistency => LoopGeometryChecker
# (4) finally correcting the loop => LoopCorrector
class LoopClosing:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op

    def __init__(
        self, slam: "Slam", loop_detector_config=LoopDetectorConfigs.DBOW3, headless=False
    ):
        self.slam = slam
        self.sensor_type = slam.sensor_type
        self.is_monocular = self.sensor_type == SensorType.MONOCULAR

        self.timer = TimerFps("LoopClosing", is_verbose=kTimerVerbose)

        # self.keyframes_map = slam.map.keyframes_map
        self.last_loop_kf_id = 0

        self.headless = headless
        self.init_print()

        # to nicely visualize loop candidates in a single image
        self.loop_consistent_candidate_imgs = (
            LoopCandidateImgs()
            if Parameters.kLoopClosingDebugWithLoopConsistencyCheckImages
            else None
        )
        self.draw_loop_consistent_candidate_imgs_init = False
        self.draw_loop_detection_imgs_init = False
        self.draw_similarity_matrix_init = False

        self.store_kf_imgs = (
            Parameters.kLoopClosingDebugWithLoopConsistencyCheckImages
            or Parameters.kLoopClosingDebugShowLoopMatchedPoints
        )
        self.map_frame_id_to_img = {}
        self.map_frame_id_to_img_lock = Lock()

        self.loop_detecting_process = LoopDetectingProcess(
            slam, loop_detector_config
        )  # launched as a parallel process
        self.time_loop_detection = self.loop_detecting_process.time_loop_detection

        # NOTE: When using torch.multiprocessing with spawn start method, a lot of issues come with data pickling in the GBA object.
        #       In order to avoid that, we use a parallel thread (same CPU core set by GIL) instead of a parallel process (different CPU core).
        #       Unfortunately, this makes things a bit slower.
        use_multiprocessing = not MultiprocessingManager.is_start_method_spawn()
        self.GBA = GlobalBundleAdjustment(slam, use_multiprocessing=use_multiprocessing)

        self.loop_consistency_checker = LoopGroupConsistencyChecker()
        self.loop_geometry_checker = LoopGeometryChecker(
            self.is_monocular, self.map_frame_id_to_img
        )
        self.loop_corrector = LoopCorrector(
            slam, self.is_monocular, self.loop_geometry_checker, self.GBA
        )

        self.relocalizer = Relocalizer()

        self.mean_graph_chi2_error = None

        self.is_running = False
        self.stop = False
        self.work_thread = Thread(target=self.run)

        self.reset_mutex = RLock()
        self.reset_requested = False

        self._is_closing = False
        self.is_closing_condition = Condition()

    def init_print(self):
        if kVerbose:
            if Parameters.kLoopClosingDebugAndPrintToFile:
                # redirect the prints of loop closing to the file logs/loop_closing.log
                # you can watch the output in separate shell by running:
                # $ tail -f logs/loop_closing.log

                logging_file = Parameters.kLogsFolder + "/loop_closing.log"
                LoopClosing.local_logger = Logging.setup_file_logger(
                    "loop_closing_logger", logging_file, formatter=Logging.simple_log_formatter
                )

                def print_file(*args, **kwargs):
                    message = " ".join(
                        str(arg) for arg in args
                    )  # Convert all arguments to strings and join with spaces
                    return LoopClosing.local_logger.info(message, **kwargs)

            else:

                def print_file(*args, **kwargs):
                    message = " ".join(
                        str(arg) for arg in args
                    )  # Convert all arguments to strings and join with spaces
                    return print(message, **kwargs)

            LoopClosing.print = staticmethod(print_file)

    def is_ready(self):
        return self.is_running and self.loop_detecting_process.is_ready()

    def is_correcting(self):
        return self.GBA.is_correcting()

    @property
    def map(self):
        return self.slam.map

    @property
    def keyframes_map(self):
        return self.slam.map.keyframes_map

    def save(self, path):
        LoopClosing.print(f"LoopClosing: saving the loop closing state into {path}...")
        self.save_image_map_(path)
        self.loop_detecting_process.save(path)
        LoopClosing.print(f"LoopClosing: ...loop closing state successfully saved to: {path}")

    def load(self, path):
        LoopClosing.print(f"LoopClosing: loading the loop closing state from {path}...")
        self.load_image_map_(path)
        self.loop_detecting_process.load(path)
        LoopClosing.print(f"LoopClosing: ...loop closing state successfully loaded from: {path}")

    def save_image_map_(self, path):
        filepath = path + "/kf_image_map.pkl"
        LoopClosing.print(f"LoopClosing: saving KF image map to {filepath}...")
        # Save the dictionary to a file
        with self.map_frame_id_to_img_lock:
            with open(filepath, "wb") as file:
                pickle.dump(self.map_frame_id_to_img, file)

    def load_image_map_(self, path):
        filepath = path + "/kf_image_map.pkl"
        if not os.path.exists(filepath):
            LoopClosing.print(f"LoopClosing: KF image map does not exist: {filepath}")
            return
        LoopClosing.print(f"LoopClosing: loading KF image map from {filepath}...")
        # Load the dictionary from a file
        with self.map_frame_id_to_img_lock:
            with open(filepath, "rb") as file:
                self.map_frame_id_to_img = pickle.load(file)

    def request_reset(self):
        LoopClosing.print("LoopClosing: Requesting reset...")
        if self.reset_requested:
            LoopClosing.print("LoopClosing: reset already requested...")
            return
        self.GBA.quit()
        with self.reset_mutex:
            self.reset_requested = True
        while True:
            with (
                self.loop_detecting_process.q_in_condition
            ):  # to unblock self.loop_detecting_process.pop_output() in run() method
                self.loop_detecting_process.q_in_condition.notify_all()
            with self.reset_mutex:
                if not self.reset_requested:
                    break
            time.sleep(0.1)
        LoopClosing.print("LoopClosing: ...Reset done.")

    def reset_if_requested(self):
        with self.reset_mutex:
            if self.reset_requested:
                LoopClosing.print("LoopClosing: reset_if_requested()...")
                self.loop_detecting_process.request_reset()
                self.reset_requested = False

    def start(self):
        self.work_thread.start()

    def is_closing(self):
        with self.is_closing_condition:
            return self._is_closing

    def set_is_closing(self, flag):
        with self.is_closing_condition:
            self._is_closing = flag
            self.is_closing_condition.notify_all()

    def wait_if_closing(self):
        if self.is_running == False:
            return
        with self.is_closing_condition:
            while self._is_closing and self.is_running:
                Printer.cyan("LoopClosing: waiting for loop closing to finish...")
                self.is_closing_condition.wait()

    def quit(self):
        LoopClosing.print("LoopClosing: quitting...")
        if self.is_running:
            self.is_running = False
            if self.stop == False:
                self.stop = True
                if self.work_thread is not None and self.work_thread.is_alive():
                    self.work_thread.join(
                        timeout=Parameters.kMultithreadingThreadJoinDefaultTimeout
                    )
        self.loop_detecting_process.quit()
        self.GBA.quit()
        if (
            self.loop_consistent_candidate_imgs.candidates is not None
            or self.draw_similarity_matrix_init
        ):
            cv2.destroyAllWindows()
        if QimageViewer.is_running():
            QimageViewer.get_instance().quit()
        LoopClosing.print("LoopClosing: done")

    def add_keyframe(self, keyframe: KeyFrame, img, print=print):
        try:

            LoopClosing.print(
                f"LoopClosing: Adding keyframe with img id: {keyframe.id} (kid: {keyframe.kid})"
            )
            keyframe.set_not_erase()
            task_type = LoopDetectorTaskType.LOOP_CLOSURE
            # If the map contains less than 10 KF or less than 10 KF have passed from last loop detection
            if keyframe.kid < self.last_loop_kf_id + 10:
                task_type = (
                    LoopDetectorTaskType.COMPUTE_GLOBAL_DES
                )  # just compute the global descriptor for this keyframe

            if task_type == LoopDetectorTaskType.LOOP_CLOSURE:
                covisible_keyframes = keyframe.get_covisible_keyframes()
                connected_keyframes = keyframe.get_connected_keyframes()
            else:
                covisible_keyframes = []
                connected_keyframes = []
            task = LoopDetectorTask(
                keyframe,
                img,
                task_type,
                covisible_keyframes=covisible_keyframes,
                connected_keyframes=connected_keyframes,
            )

            self.loop_detecting_process.add_task(task)

        except Exception as e:
            LoopClosing.print(f"LoopClosing: add_keyframe: EXCEPTION: {e}!!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                LoopClosing.print(f"\t traceback details: {traceback_details}")

    # main loop in LoopClosing thread
    def run(self):
        # thread execution
        LoopClosing.print("LoopClosing: starting...")
        self.is_running = True
        while not self.stop:
            # Steps:
            # - Loop detection: get loop detection candidates from LoopDetectingProcess process (parallel process)
            # - Loop consistency verification: extract consistent candidates from LoopGroupConsistencyChecker (same thread)
            # - Loop geometry verification: extract geometry-verified candidates from LoopGeometryChecker (same thread)
            # - Loop correction: correct loop (same thread)
            try:

                self.reset_if_requested()

                # check if we have a new result from parallel GBA process and apply it
                if self.GBA.check_GBA_has_finished_and_correct_if_needed():
                    if self.slam.volumetric_integrator is not None:
                        self.slam.volumetric_integrator.rebuild(self.slam.map)

                # wait until we get loop-detection candidates from parallel LoopDetectingProcess
                LoopClosing.print("LoopClosing: waiting for loop-detection output...")
                detection_output = self.loop_detecting_process.pop_output()  # blocking call

                if detection_output is not None:
                    self.timer.start()

                    LoopClosing.print("..................................")
                    # retrieve the keyframe corresponding to the output img_id
                    keyframe = self.keyframes_map[detection_output.frame_id]

                    LoopClosing.print(
                        f"LoopClosing: processing KF: {keyframe.id}, detection: qin size: {self.loop_detecting_process.q_in.qsize()}, qout size: {self.loop_detecting_process.q_out.qsize()}"
                    )

                    # update the keyframe with the detection output
                    if keyframe.g_des is None:
                        keyframe.g_des = detection_output.g_des_vec

                    # for viz debugging
                    if self.store_kf_imgs:
                        with (
                            self.map_frame_id_to_img_lock
                        ):  # we lock in case we want to save the dictionary and we need to avoid increasing its size at the same time
                            self.map_frame_id_to_img[keyframe.id] = detection_output.img

                    # for viz debugging
                    if self.loop_consistent_candidate_imgs is not None:
                        self.loop_consistent_candidate_imgs.reset()

                    # If the map contains less than 10 KF or less than 10 KF have passed from last loop detection we skip the processing of this frame
                    if keyframe.kid < self.last_loop_kf_id + 10:
                        keyframe.set_erase()
                        LoopClosing.print(
                            f"LoopClosing: skipping keyframe (close to start or previous loop: {self.last_loop_kf_id})"
                        )
                        continue

                    # update covisible keyframes if needed
                    for i, cov_kf_id in enumerate(detection_output.covisible_ids):
                        if cov_kf_id in self.keyframes_map:
                            cov_kf = self.keyframes_map[cov_kf_id]
                            # update the cov keyframe with the detection output if needed
                            # if not cov_kf.is_bad() and cov_kf.g_des is None:
                            if cov_kf.g_des is None:
                                cov_kf.g_des = detection_output.covisible_gdes_vecs[i]

                    got_loop = False
                    if len(detection_output.candidate_idxs) == 0:
                        keyframe.set_erase()
                        self.loop_consistency_checker.clear_consistency_groups()
                        LoopClosing.print(
                            f"LoopClosing: KF: {keyframe.id}, no loop candidates detected"
                        )
                    else:
                        LoopClosing.print(
                            f"LoopClosing: KF: {keyframe.id}, detected loop candidates: {detection_output.candidate_idxs}"
                        )
                        loop_candidate_kfs = [
                            self.keyframes_map[idx]
                            for idx in detection_output.candidate_idxs
                            if idx in self.keyframes_map and not self.keyframes_map[idx].is_bad()
                        ]  # get back the keyframes from their ids

                        # verify group-consistency
                        got_consistent_candidates = self.loop_consistency_checker.check_candidates(
                            keyframe, loop_candidate_kfs
                        )

                        if got_consistent_candidates:
                            LoopClosing.print(
                                f"LoopClosing: KF: {keyframe.id}, got consistent loop candidates: {[kf.id for kf in self.loop_consistency_checker.enough_consistent_candidates]}"
                            )

                            consistent_candidates = [
                                kf
                                for kf in self.loop_consistency_checker.enough_consistent_candidates
                                if not kf.is_bad()
                            ]
                            for kf in consistent_candidates:
                                self.update_loop_consistent_candidate_imgs(kf.id)

                            # verify geometry consistency
                            got_loop = self.loop_geometry_checker.check_candidates(
                                keyframe, consistent_candidates
                            )
                            if got_loop:
                                LoopClosing.print("")
                                LoopClosing.print(
                                    f"[[[ LoopClosing: Got loop: {keyframe.id} with {self.loop_geometry_checker.success_loop_kf.id}!! ]]]"
                                )
                                LoopClosing.print("")
                            else:
                                LoopClosing.print(
                                    f"LoopClosing: KF: {keyframe.id}, geometry verification failed for loop candidates: {[kf.id for kf in consistent_candidates]}"
                                )

                    if got_loop:
                        # correct the loop
                        self.set_is_closing(True)  # communicate tracker to pause
                        self.loop_corrector.correct_loop(keyframe)
                        self.mean_graph_chi2_error = self.loop_corrector.mean_graph_chi2_error
                        self.last_loop_kf_id = keyframe.kid
                        self.set_is_closing(False)  # communicate tracker to restart
                    else:
                        keyframe.set_erase()

                    try:
                        self.draw_loop_detection_output_imgs(
                            keyframe.img, keyframe.id, detection_output
                        )
                    except Exception as e:
                        LoopClosing.print(
                            f"LoopClosing: draw_loop_detection_output_imgs EXCEPTION: {e} !!!"
                        )
                        if kPrintTrackebackDetails:
                            traceback_details = traceback.format_exc()
                            LoopClosing.print(f"\t traceback details: {traceback_details}")

                    self.timer.refresh()

                else:
                    LoopClosing.print(f"LoopClosing: No loop candidates detected")
                    time.sleep(0.01)

            except Exception as e:
                LoopClosing.print(f"LoopClosing: EXCEPTION: {e} !!!")
                if kPrintTrackebackDetails:
                    traceback_details = traceback.format_exc()
                    LoopClosing.print(f"\t traceback details: {traceback_details}")

            LoopClosing.print(
                f"LoopClosing: loop closing thread elapsed time: {self.timer.last_elapsed}"
            )

        # end of the while loop

        empty_queue(self.loop_detecting_process.q_out)  # empty the queue before exiting

        LoopClosing.print("LoopClosing: loop exit...")

    def update_loop_consistent_candidate_imgs(self, loop_img_id, loop_img_score=None):
        if self.loop_consistent_candidate_imgs is not None:
            assert self.store_kf_imgs
            if loop_img_id not in self.map_frame_id_to_img:
                LoopClosing.print(
                    f"LoopClosing: ERROR: loop image id {loop_img_id} not in map_frame_id_to_img. This is not expected !!!!!!!!"
                )
                return
            loop_img = self.map_frame_id_to_img[loop_img_id]
            if loop_img is None:
                LoopClosing.print(
                    f"LoopClosing: ERROR: loop image is None. This is not expected !!!!!!!!"
                )
            self.loop_consistent_candidate_imgs.add(loop_img.copy(), loop_img_id, loop_img_score)

    def draw_loop_detection_output_imgs(
        self, img_cur, img_id, detection_output: LoopDetectorOutput
    ):
        if self.headless:
            return
        draw = False
        if self.loop_consistent_candidate_imgs is not None:
            if not self.draw_loop_consistent_candidate_imgs_init:
                if kUseCv2ForDrawing:
                    cv2.namedWindow(
                        "loop closing: consistent candidates", cv2.WINDOW_NORMAL
                    )  # to get a resizable window
                self.draw_loop_consistent_candidate_imgs_init = True
            if self.loop_consistent_candidate_imgs.candidates is not None:
                draw = True
                if kUseCv2ForDrawing:
                    cv2.imshow(
                        "loop closing: consistent candidates",
                        self.loop_consistent_candidate_imgs.candidates,
                    )
                else:
                    QimageViewer.get_instance().draw(
                        self.loop_consistent_candidate_imgs.candidates,
                        "loop closing: consistent candidates",
                    )

        if detection_output.similarity_matrix is not None:
            if not self.draw_similarity_matrix_init:
                if kUseCv2ForDrawing:
                    cv2.namedWindow(
                        "loop closing: similarity matrix", cv2.WINDOW_NORMAL
                    )  # to get a resizable window
                self.draw_similarity_matrix_init = True
            draw = True
            if kUseCv2ForDrawing:
                cv2.imshow("loop closing: similarity matrix", detection_output.similarity_matrix)
            else:
                QimageViewer.get_instance().draw(
                    detection_output.similarity_matrix, "loop closing: similarity matrix"
                )

        if detection_output.loop_detection_img_candidates is not None:
            if not self.draw_loop_detection_imgs_init:
                if kUseCv2ForDrawing:
                    cv2.namedWindow(
                        "loop-detection: candidates", cv2.WINDOW_NORMAL
                    )  # to get a resizable window
                self.draw_loop_detection_imgs_init = True
            draw = True
            if kUseCv2ForDrawing:
                cv2.imshow(
                    "loop-detection: candidates", detection_output.loop_detection_img_candidates
                )
            else:
                QimageViewer.get_instance().draw(
                    detection_output.loop_detection_img_candidates, "loop-detection: candidates"
                )

        if draw:
            if kUseCv2ForDrawing:
                cv2.waitKey(1)

    def relocalize(self, frame: Frame, img):
        task_type = LoopDetectorTaskType.RELOCALIZATION
        task = LoopDetectorTask(
            frame, img, task_type, covisible_keyframes=[], connected_keyframes=[]
        )
        LoopClosing.print(f"Relocalization: Starting on frame id: {frame.id}...")
        detection_output = self.loop_detecting_process.relocalize(task)

        # process the candidates with the relocalizer (geometry verification and estimation)
        res = self.relocalizer.relocalize(frame, detection_output, self.keyframes_map)
        LoopClosing.print(
            f'Relocalization: {"Success" if res else "Failed"} on frame id: {frame.id}...'
        )
        return res
