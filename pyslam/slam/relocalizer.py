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
import multiprocessing as mp

from pyslam.slam import (
    Frame,
    optimizer_g2o,
    RotationHistogram,
)
from . import optimizer_gtsam
from .frame import compute_frame_matches, prepare_input_data_for_pnpsolver

from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.slam import ProjectionMatcher

from pyslam.utilities.logging import Printer, Logging, LoggerQueue
from pyslam.utilities.file_management import create_folder
from pyslam.loop_closing.loop_detector_base import LoopDetectorOutput

from pyslam.utilities.timer import TimerFps
from pyslam.config_parameters import Parameters

import logging

import traceback
import pnpsolver


# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .frame import Frame
    from . import optimizer_gtsam
    from . import optimizer_g2o
    from .rotation_histogram import RotationHistogram


kVerbose = True
kTimerVerbose = False  # set this to True if you want to print timings
kPrintTrackebackDetails = True

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


pose_optimization = (
    optimizer_gtsam.pose_optimization
    if Parameters.kOptimizationFrontEndUseGtsam
    else optimizer_g2o.pose_optimization
)


# Helper functions for relocalization
def _align_array_lengths(*arrays, print_func=None):
    """Align multiple arrays to the same length by truncating to minimum length.

    Args:
        *arrays: Variable number of arrays to align
        print_func: Optional print function for warnings

    Returns:
        Tuple of aligned arrays, all with the same length
    """
    if not arrays:
        return arrays

    min_len = min(len(arr) for arr in arrays if arr is not None)
    lengths = [len(arr) for arr in arrays]

    if all(l == min_len for l in lengths):
        return arrays

    if print_func:
        print_func(f"Relocalizer: WARNING - length mismatch: {lengths}. Truncating to {min_len}.")

    return tuple(arr[:min_len] if arr is not None else arr for arr in arrays)


def _convert_kf_points_indices_to_matched_indices(kf, kf_point_indices):
    """Convert indices from kf.points space to kf.get_matched_points() space.

    Args:
        kf: KeyFrame object
        kf_point_indices: Indices into kf.points array

    Returns:
        numpy array of indices into kf.get_matched_points(), or None if empty
    """
    if len(kf_point_indices) == 0:
        return None

    # Use Frame method to safely get matched point indices
    kf_matched_idxs = kf.get_matched_points_idxs()
    if len(kf_matched_idxs) == 0:
        return None

    # Create mapping from kf.points indices to get_matched_points() indices
    kf_to_matched_idx = {kf_idx: matched_idx for matched_idx, kf_idx in enumerate(kf_matched_idxs)}

    # Convert indices
    matched_points_indices = [
        kf_to_matched_idx[kf_idx] for kf_idx in kf_point_indices if kf_idx in kf_to_matched_idx
    ]

    return np.array(matched_points_indices, dtype=int) if matched_points_indices else None


def _validate_and_filter_indices(idxs_frame, idxs_kf, num_points, num_kf_points, print_func=None):
    """Validate indices are within bounds and create a boolean mask.

    Args:
        idxs_frame: Frame point indices
        idxs_kf: Keyframe point indices
        num_points: Number of frame points
        num_kf_points: Number of keyframe points
        print_func: Optional print function for warnings

    Returns:
        valid_mask: Boolean array indicating valid indices
    """
    valid_mask = (
        (idxs_frame >= 0) & (idxs_frame < num_points) & (idxs_kf >= 0) & (idxs_kf < num_kf_points)
    )

    if not np.all(valid_mask) and print_func:
        invalid_frame_count = np.sum((idxs_frame < 0) | (idxs_frame >= num_points))
        invalid_kf_count = np.sum((idxs_kf < 0) | (idxs_kf >= num_kf_points))
        if invalid_frame_count > 0:
            print_func(
                f"Relocalizer: WARNING - {invalid_frame_count} invalid frame indices "
                f"(max: {num_points-1}). Skipping."
            )
        if invalid_kf_count > 0:
            print_func(
                f"Relocalizer: WARNING - {invalid_kf_count} invalid keyframe indices "
                f"(max: {num_kf_points-1}). Skipping."
            )

    return valid_mask


# Relocalizer working on loop detection output
class Relocalizer:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op
    logging_manager, logger = None, None

    def __init__(self):
        self.timer = TimerFps("Relocalizer", is_verbose=kTimerVerbose)
        self.init_print()

    def init_print(self):
        if kVerbose:
            if Parameters.kRelocalizationDebugAndPrintToFile:
                # redirect the prints of local mapping to the file logs/relocalization.log
                # you can watch the output in separate shell by running:
                # $ tail -f logs/relocalization.log

                logging_file = Parameters.kLogsFolder + "/relocalization.log"
                create_folder(logging_file)
                if Relocalizer.logging_manager is None:
                    # Note: Each process has its own memory space, so singleton pattern works per-process
                    Relocalizer.logging_manager = LoggerQueue.get_instance(logging_file)
                    Relocalizer.logger = Relocalizer.logging_manager.get_logger(
                        "relocalization_logger"
                    )

                def print_file(*args, **kwargs):
                    try:
                        if Relocalizer.logger is not None:
                            message = " ".join(
                                str(arg) for arg in args
                            )  # Convert all arguments to strings and join with spaces
                            return Relocalizer.logger.info(message, **kwargs)
                    except:
                        print("Error printing: ", args, kwargs)

            else:

                def print_file(*args, **kwargs):
                    message = " ".join(
                        str(arg) for arg in args
                    )  # Convert all arguments to strings and join with spaces
                    return print(message, **kwargs)

            Relocalizer.print = staticmethod(print_file)

    def relocalize(self, frame: Frame, detection_output: LoopDetectorOutput, keyframes_map: dict):
        try:
            if detection_output is None or len(detection_output.candidate_idxs) == 0:
                msg = "None output" if detection_output is None else "No candidates"
                Relocalizer.print(f"Relocalizer: {msg} with frame {frame.id}")
                return False

            top_score = (
                detection_output.candidate_scores[0]
                if detection_output.candidate_scores is not None
                and len(detection_output.candidate_scores) > 0
                else None
            )
            Relocalizer.print(
                f"Relocalizer: Detected candidates for frame {frame.id}: {detection_output.candidate_idxs}"
                + (f", top score: {top_score}" if top_score is not None else "")
            )
            reloc_candidate_kfs = [
                keyframes_map[idx]
                for idx in detection_output.candidate_idxs
                if idx in keyframes_map
            ]  # get back the keyframes from their ids

            kp_match_idxs = defaultdict(
                lambda: (None, None)
            )  # dictionary of keypointmatches  (kf_i, kf_j) -> (idxs_i,idxs_j)

            self.timer.start()
            kp_match_idxs = compute_frame_matches(
                frame,
                reloc_candidate_kfs,
                kp_match_idxs,
                do_parallel=Parameters.kRelocalizationParallelKpsMatching,
                max_workers=Parameters.kRelocalizationParallelKpsMatchingNumWorkers,
                ratio_test=Parameters.kRelocalizationFeatureMatchRatioTest,
                print_fun=print,
            )

            solvers = []
            # solvers_input = []
            considered_candidates = []
            mp_match_idxs = defaultdict(
                lambda: (None, None)
            )  # dictionary of map point matches  (kf_i, kf_j) -> (idxs_i,idxs_j)
            for i, kf in enumerate(reloc_candidate_kfs):
                if kf.id == frame.id or kf.is_bad():
                    continue

                # extract matches from precomputed map
                idxs_frame, idxs_kf = kp_match_idxs[(frame, kf)]

                # Skip if no matches were found (None or empty)
                if (
                    idxs_frame is None
                    or idxs_kf is None
                    or len(idxs_frame) == 0
                    or len(idxs_kf) == 0
                ):
                    Relocalizer.print(
                        f"Relocalizer: no matches found between frame {frame.id} and keyframe {kf.id}"
                    )
                    continue

                assert len(idxs_frame) == len(idxs_kf)

                # if features have descriptors with orientation then let's check the matches with a rotation histogram
                if FeatureTrackerShared.oriented_features:
                    # num_matches_before = len(idxs_frame)
                    valid_match_idxs = RotationHistogram.filter_matches_with_histogram_orientation(
                        idxs_frame, idxs_kf, frame.angles, kf.angles
                    )
                    if len(valid_match_idxs) > 0:
                        idxs_frame = idxs_frame[valid_match_idxs]
                        idxs_kf = idxs_kf[valid_match_idxs]
                    # print(f'Relocalizer: rotation histogram filter: #matches ({frame.id},{kf.id}): before {num_matches_before}, after {len(idxs_frame)}')

                num_matches = len(idxs_frame)
                Relocalizer.print(f"Relocalizer: num_matches ({frame.id},{kf.id}): {num_matches}")

                if num_matches < Parameters.kRelocalizationMinKpsMatches:
                    Relocalizer.print(
                        f"Relocalizer: skipping keyframe {kf.id} with too few matches ({num_matches}) (min: {Parameters.kRelocalizationMinKpsMatches})"
                    )
                    continue

                points_3d_w, points_2d, sigmas2, idxs1, idxs2 = prepare_input_data_for_pnpsolver(
                    frame, kf, idxs_frame, idxs_kf, print=print
                )

                # fill the dictionary of map point matches (its content needs to be cleaned up later with found inliers)
                mp_match_idxs[(frame, kf)] = (idxs1, idxs2)

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
                    Relocalizer.print(
                        f"Relocalizer: skipping keyframe {kf.id} with too few correspondences ({num_correspondences}) (min: 4)"
                    )
                    continue

                # print(f'Relocalizer: initializing MLPnPsolver for keyframe {kf.id}, num correspondences: {num_correspondences}')
                solver = pnpsolver.MLPnPsolver(solver_input_data)
                solver.set_ransac_parameters(0.99, 10, 300, 6, 0.5, 5.991)

                solvers.append(solver)
                # solvers_input.append(solver_input_data)
                considered_candidates.append(kf)

            discarded = [False] * len(considered_candidates)
            success_relocalization_kf = None
            num_candidates = len(considered_candidates)
            num_matched_map_points = 0

            # check if candidates get a valid solution
            while num_candidates > 0 and success_relocalization_kf is None:
                for i, kf in enumerate(considered_candidates):
                    if discarded[i]:
                        continue

                    # perform 5 ransac iterations on each solver
                    Relocalizer.print(
                        f"Relocalizer: performing MLPnPsolver iterations for keyframe {kf.id}"
                    )
                    ok, Tcw, is_no_more, inlier_flags, num_inliers = solvers[i].iterate(5)

                    # We discard the candidate if the solver is no more able to find a solution
                    if is_no_more:
                        discarded[i] = True
                        num_candidates -= 1
                        continue

                    if not ok:
                        continue

                    inlier_flags = np.array(inlier_flags, dtype=bool)  # from from int8 to bool

                    # Capture the original pose before applying RANSAC solution
                    # This allows us to restore it if RANSAC or pose optimization fails
                    pose_original = frame.pose()

                    # we got a valid pose solution => let's optimize it
                    frame.update_pose(Tcw)

                    idxs_frame, idxs_kf = mp_match_idxs[(frame, kf)]
                    # Convert to numpy arrays for efficient vectorized operations
                    idxs_frame = np.asarray(idxs_frame)
                    idxs_kf = np.asarray(idxs_kf)

                    # Align array lengths to prevent indexing errors
                    idxs_frame, idxs_kf, inlier_flags = _align_array_lengths(
                        idxs_frame, idxs_kf, inlier_flags, print_func=Relocalizer.print
                    )

                    # Validate indices are within bounds and create mask
                    num_points = len(frame.points) if frame.points is not None else 0
                    num_kf_points = len(kf.points) if kf.points is not None else 0
                    valid_mask = _validate_and_filter_indices(
                        idxs_frame, idxs_kf, num_points, num_kf_points, print_func=Relocalizer.print
                    )

                    # Filter to valid indices
                    valid_idxs_frame = idxs_frame[valid_mask]
                    valid_idxs_kf = idxs_kf[valid_mask]
                    valid_inlier_flags = inlier_flags[valid_mask]

                    # Update frame points (need to iterate for assignment, but only valid ones)
                    for idx, kf_idx, is_inlier in zip(
                        valid_idxs_frame, valid_idxs_kf, valid_inlier_flags
                    ):
                        frame.points[idx] = kf.points[kf_idx] if is_inlier else None

                    idxs_kf_inliers = valid_idxs_kf[valid_inlier_flags]

                    mean_pose_opt_chi2_error, pose_is_ok, num_matched_map_points = (
                        pose_optimization(frame, verbose=False)
                    )
                    Relocalizer.print(
                        f"Relocalizer: pos opt1: error^2: {mean_pose_opt_chi2_error},  ok: {pose_is_ok}, #inliers: {num_matched_map_points}"
                    )

                    if not pose_is_ok:
                        # if current pose optimization failed, reset f_cur pose to original (before RANSAC)
                        frame.update_pose(pose_original)
                        continue

                    if num_matched_map_points < Parameters.kRelocalizationPoseOpt1MinMatches:
                        continue

                    # Safety check: ensure outliers array exists and has correct size
                    if frame.outliers is None or len(frame.outliers) != len(frame.points):
                        Relocalizer.print(
                            f"Relocalizer: WARNING - frame.outliers is None or size mismatch "
                            f"(outliers: {len(frame.outliers) if frame.outliers is not None else None}, "
                            f"points: {len(frame.points)}). Skipping outlier cleanup."
                        )
                    else:
                        # Use vectorized boolean indexing for efficient outlier cleanup
                        outliers_array = np.asarray(frame.outliers)
                        outlier_indices = np.flatnonzero(outliers_array)
                        for i in outlier_indices:
                            frame.points[i] = None

                    # if few inliers, search by projection in a coarse window and optimize again
                    if num_matched_map_points < Parameters.kRelocalizationDoPoseOpt2NumInliers:
                        # Convert idxs_kf_inliers from indices into kf.points to indices into kf.get_matched_points()
                        # because search_keyframe_by_projection expects indices into get_matched_points()
                        already_matched_ref_idxs = _convert_kf_points_indices_to_matched_indices(
                            kf, idxs_kf_inliers
                        )

                        idxs_kf, idxs_frame, num_new_found_map_points = (
                            ProjectionMatcher.search_keyframe_by_projection(
                                kf,
                                frame,
                                max_reproj_distance=Parameters.kRelocalizationMaxReprojectionDistanceMapSearchCoarse,
                                max_descriptor_distance=Parameters.kMaxDescriptorDistance,
                                ratio_test=Parameters.kRelocalizationFeatureMatchRatioTestLarge,
                                already_matched_ref_idxs=already_matched_ref_idxs,
                            )
                        )

                        if (
                            num_matched_map_points + num_new_found_map_points
                            >= Parameters.kRelocalizationDoPoseOpt2NumInliers
                        ):
                            pose_before = frame.pose()
                            mean_pose_opt_chi2_error, pose_is_ok, num_matched_map_points = (
                                pose_optimization(frame, verbose=False)
                            )
                            Relocalizer.print(
                                f"Relocalizer: pos opt2: error^2: {mean_pose_opt_chi2_error},  ok: {pose_is_ok}, #inliers: {num_matched_map_points}"
                            )

                            if not pose_is_ok:
                                # if current pose optimization failed, reset f_cur pose
                                frame.update_pose(pose_before)
                                continue

                            # if many inliers but still not enough, search by projection again in a narrower window
                            # the camera has been already optimized with many points
                            if (
                                num_matched_map_points > 30
                                and num_matched_map_points
                                < Parameters.kRelocalizationDoPoseOpt2NumInliers
                            ):
                                # Get matched point indices (safe for C++ MapPoint objects)
                                matched_ref_idxs = frame.get_matched_points_idxs()

                                idxs_kf, idxs_frame, num_new_found_map_points = (
                                    ProjectionMatcher.search_keyframe_by_projection(
                                        kf,
                                        frame,
                                        max_reproj_distance=Parameters.kRelocalizationMaxReprojectionDistanceMapSearchFine,
                                        max_descriptor_distance=0.7
                                        * Parameters.kMaxDescriptorDistance,
                                        ratio_test=Parameters.kRelocalizationFeatureMatchRatioTestLarge,
                                        already_matched_ref_idxs=matched_ref_idxs,
                                    )
                                )

                                # final optimization
                                if (
                                    num_matched_map_points + num_new_found_map_points
                                    >= Parameters.kRelocalizationDoPoseOpt2NumInliers
                                ):
                                    pose_before = frame.pose()
                                    mean_pose_opt_chi2_error, pose_is_ok, num_matched_map_points = (
                                        pose_optimization(frame, verbose=False)
                                    )
                                    Relocalizer.print(
                                        f"Relocalizer: pos opt3: error^2: {mean_pose_opt_chi2_error},  ok: {pose_is_ok}, #inliers: {num_matched_map_points}"
                                    )

                                    if not pose_is_ok:
                                        # if current pose optimization failed, reset f_cur pose
                                        frame.update_pose(pose_before)
                                        continue

                    if num_matched_map_points >= Parameters.kRelocalizationDoPoseOpt2NumInliers:
                        success_relocalization_kf = kf
                        break

            res = False
            if success_relocalization_kf is None:
                Relocalizer.print(
                    f"Relocalizer: failed, num_matched_map_points: {num_matched_map_points}"
                )
                res = False
            else:
                frame.kf_ref = success_relocalization_kf
                Relocalizer.print(
                    f"Relocalizer: success: connected frame id: {frame.id} to keyframe id: {frame.kf_ref.id}"
                )
                res = True

            self.timer.refresh()
            Relocalizer.print(f"Relocalizer: elapsed time: {self.timer.last_elapsed}")
            return res

        except Exception as e:
            Relocalizer.print(f"Relocalizer: EXCEPTION: {e} !!!")
            if kPrintTrackebackDetails:
                traceback_details = traceback.format_exc()
                Relocalizer.print(f"\t traceback details: {traceback_details}")

        return False
