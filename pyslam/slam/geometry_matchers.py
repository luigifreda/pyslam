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

import sys
import math
import numpy as np

from .frame import (
    FeatureTrackerShared,
    are_map_points_visible,
    are_map_points_visible_in_frame,
)

from .sim3_pose import Sim3Pose
from .keyframe import KeyFrame
from .frame import Frame
from .map_point import MapPoint
from .rotation_histogram import RotationHistogram

from pyslam.utilities.geom_2views import computeF12, check_dist_epipolar_line
from pyslam.utilities.logging import Printer
from pyslam.utilities.timer import Timer

from pyslam.config_parameters import Parameters

# Type hints for IDE navigation
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .frame import Frame
    from .keyframe import KeyFrame
    from .map_point import MapPoint
    from .rotation_histogram import RotationHistogram


kMinDistanceFromEpipole = Parameters.kMinDistanceFromEpipole
kMinDistanceFromEpipole2 = kMinDistanceFromEpipole * kMinDistanceFromEpipole
kCheckFeaturesOrientation = Parameters.kCheckFeaturesOrientation


class ProjectionMatcher:
    @staticmethod
    def search_frame_by_projection(*args, **kwargs):
        return _search_frame_by_projection(*args, **kwargs)

    @staticmethod
    def search_keyframe_by_projection(*args, **kwargs):
        return _search_keyframe_by_projection(*args, **kwargs)

    @staticmethod
    def search_map_by_projection(*args, **kwargs):
        return _search_map_by_projection(*args, **kwargs)

    @staticmethod
    def search_local_frames_by_projection(*args, **kwargs):
        return _search_local_frames_by_projection(*args, **kwargs)

    @staticmethod
    def search_all_map_by_projection(*args, **kwargs):
        return _search_all_map_by_projection(*args, **kwargs)

    @staticmethod
    def search_more_map_points_by_projection(*args, **kwargs):
        return _search_more_map_points_by_projection(*args, **kwargs)

    @staticmethod
    def search_and_fuse(*args, **kwargs):
        return _search_and_fuse(*args, **kwargs)

    @staticmethod
    def search_and_fuse_for_loop_correction(*args, **kwargs):
        return _search_and_fuse_for_loop_correction(*args, **kwargs)

    @staticmethod
    def search_by_sim3(*args, **kwargs):
        return _search_by_sim3(*args, **kwargs)


class EpipolarMatcher:
    @staticmethod
    def search_frame_for_triangulation(*args, **kwargs):
        return _search_frame_for_triangulation(*args, **kwargs)


# ===================================================================================
# Implementations
# ===================================================================================


# search by projection matches between {map points of f_ref} and {keypoints of f_cur},  (access frames from tracking thread, no need to lock)
def _search_frame_by_projection(
    f_ref: Frame,
    f_cur: Frame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceFrame,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    is_monocular=True,
    already_matched_ref_idxs=None,
):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance

    found_pts_count = 0
    idxs_ref = []
    idxs_cur = []

    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features

    check_forward_backward = not is_monocular

    trc = None
    forward = False
    backward = False
    if check_forward_backward:
        Tcw = f_cur.pose()
        Rcw = Tcw[:3, :3]
        tcw = Tcw[:3, 3]
        twc = -Rcw.T.dot(tcw)

        Trw = f_ref.pose()
        Rrw = Trw[:3, :3]
        trw = Trw[:3, 3]
        trc = Rrw.T.dot(twc) + trw

        forward = trc[2] > f_cur.camera.b
        backward = trc[2] < -f_cur.camera.b
        check_forward_backward = forward or backward

    # get all matched points of f_ref which are non-outlier
    matched_ref_idxs = np.array(
        [i for i, p in enumerate(f_ref.points) if p is not None and not f_ref.outliers[i]]
    )

    # if we have some already matched points in reference frame, remove them from the list
    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)

    matched_ref_points = f_ref.points[matched_ref_idxs]

    # project f_ref points on frame f_cur
    projs, depths = f_cur.project_map_points(matched_ref_points, f_cur.camera.is_stereo())

    # check if points lie on the image frame
    is_visible = f_cur.are_in_image(projs, depths)

    # # check if points are visible
    # is_visible, projs, depths, dists = f_cur.are_visible(matched_ref_points)

    kp_ref_octaves = f_ref.octaves[matched_ref_idxs]
    kp_ref_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[kp_ref_octaves]
    radiuses = max_reproj_distance * kp_ref_scale_factors
    kd_cur_idxs = f_cur.kd.query_ball_point(projs[:, :2], radiuses)

    do_check_stereo_reproj_err = f_cur.kps_ur is not None and len(f_cur.kps_ur) > 0

    scale_factors = FeatureTrackerShared.feature_manager.scale_factors

    cur_des = f_cur.des
    cur_angles = f_cur.angles
    cur_points = f_cur.points
    cur_octaves = f_cur.octaves
    cur_outliers = f_cur.outliers
    cur_kd = f_cur.kd
    cur_kpsu = f_cur.kpsu
    cur_kps_ur = f_cur.kps_ur

    ref_octaves = f_ref.octaves
    ref_angles = f_ref.angles

    # for ref_idx,p,j in zip(matched_ref_idxs, matched_ref_points, range(len(matched_ref_points))):
    for j, (ref_idx, p_ref) in enumerate(zip(matched_ref_idxs, matched_ref_points)):

        if not is_visible[j]:
            continue

        kp_ref_octave = ref_octaves[ref_idx]

        best_dist = np.inf  # math.inf
        # best_dist2 = math.inf
        # best_level = -1
        # best_level2 = -1
        best_k_idx = -1
        best_ref_idx = -1

        kd_cur_idxs_j = kd_cur_idxs[j]
        if do_check_stereo_reproj_err:
            check_stereo = cur_kps_ur[kd_cur_idxs_j] >= 0
            kp_cur_octaves = cur_octaves[kd_cur_idxs_j]
            kp_cur_scale_factors = scale_factors[kp_cur_octaves]
            errs_ur = np.fabs(projs[j, 2] - cur_kps_ur[kd_cur_idxs_j])
            ok_errs_ur = np.where(
                check_stereo, errs_ur < max_reproj_distance * kp_cur_scale_factors, True
            )

        for h, kd_idx in enumerate(kd_cur_idxs[j]):

            p_cur = cur_points[kd_idx]
            if p_cur is not None:
                if p_cur.num_observations() > 0:  # we already matched p_cur => discard it
                    continue

            kp_cur_octave = cur_octaves[kd_idx]

            # check if point is in the same octave as the reference point
            if check_forward_backward:
                if backward and kp_cur_octave > kp_ref_octave:
                    continue
                elif forward and kp_cur_octave < kp_ref_octave:
                    continue
                elif kp_cur_octave < (kp_ref_octave - 1) or kp_cur_octave > (kp_ref_octave + 1):
                    continue
            else:
                if kp_cur_octave < (kp_ref_octave - 1) or kp_cur_octave > (kp_ref_octave + 1):
                    continue

            if do_check_stereo_reproj_err and not ok_errs_ur[h]:
                continue

            descriptor_dist = p_ref.min_des_distance(cur_des[kd_idx])
            if descriptor_dist < best_dist:
                best_dist = descriptor_dist
                best_k_idx = kd_idx
                best_ref_idx = ref_idx

            # if descriptor_dist < best_dist:
            #     best_dist2 = best_dist
            #     best_level2 = best_level
            #     best_dist = descriptor_dist
            #     best_level = f_cur.octaves[kd_idx]
            #     best_k_idx = kd_idx
            #     best_ref_idx = i
            # else:
            #     if descriptor_dist < best_dist2:
            #         best_dist2 = descriptor_dist
            #         best_level2 = f_cur.octaves[kd_idx]

        # if best_k_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance:
            # apply match distance ratio test only if the best and second are in the same scale level
            # if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test):
            #    continue
            # print('b_dist : ', best_dist)
            if p_ref.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1
                idxs_ref.append(best_ref_idx)
                idxs_cur.append(best_k_idx)

                if check_orientation:
                    index_match = len(idxs_cur) - 1
                    rot = ref_angles[best_ref_idx] - cur_angles[best_k_idx]
                    rot_histo.push(rot, index_match)

            # print('best des distance: ', best_dist, ", max dist: ", max_descriptor_distance)
            # des_dists.append(best_dist)

    if check_orientation:
        valid_match_idxs = rot_histo.get_valid_idxs()
        print(
            "checking orientation consistency - valid matches % :",
            len(valid_match_idxs) / max(1, len(idxs_cur)) * 100,
            "% of ",
            len(idxs_cur),
            "matches",
        )
        # print('rotation histogram: ', rot_histo)
        idxs_ref = np.array(idxs_ref)[valid_match_idxs]
        idxs_cur = np.array(idxs_cur)[valid_match_idxs]
        found_pts_count = len(valid_match_idxs)

    return np.array(idxs_ref), np.array(idxs_cur), found_pts_count
    # return idxs_ref, idxs_cur, found_pts_count


# Search by projection between {keyframe map points} and {current frame keypoints}
def _search_keyframe_by_projection(
    kf_ref: KeyFrame,
    f_cur: Frame,
    max_reproj_distance,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    already_matched_ref_idxs=None,
):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance

    assert kf_ref.is_keyframe, "[search_keyframe_by_projection] kf_ref must be a KeyFrame"

    found_pts_count = 0
    idxs_ref = []
    idxs_cur = []

    rot_histo = RotationHistogram()
    check_orientation = kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features

    Tcw = f_cur.pose()
    Rcw = Tcw[:3, :3]
    tcw = Tcw[:3, 3]
    Ow = -Rcw.T @ tcw  # camera center in world coords

    ref_mps = kf_ref.get_matched_points()

    if len(ref_mps) == 0:
        return np.array([]), np.array([]), 0

    # Get valid map points (non-bad, non-outliers)
    matched_ref_idxs = np.array(
        [i for i, p in enumerate(ref_mps) if p is not None and not p.is_bad()]
    )

    # Remove already matched points if given
    if already_matched_ref_idxs is not None:
        matched_ref_idxs = np.setdiff1d(matched_ref_idxs, already_matched_ref_idxs)

    matched_ref_points = [ref_mps[i] for i in matched_ref_idxs]
    if len(matched_ref_points) == 0:
        return np.array([]), np.array([]), 0

    # points_w = np.array([p for p in matched_ref_points])

    # Project points
    # projs, depths = f_cur.project_map_points(points_w, f_cur.camera.is_stereo())
    # is_visible = f_cur.are_visible(projs, depths)
    is_visible, projs, depths, dists = f_cur.are_visible(matched_ref_points)

    # Predict detection levels
    # dists = np.linalg.norm(points_w - Ow, axis=1)
    # predicted_levels = np.array([p.predict_scale(dist, f_cur) for p, dist in zip(matched_ref_points, dists)])
    predicted_levels = MapPoint.predict_detection_levels(matched_ref_points, dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors
    kd_cur_idxs = f_cur.kd.query_ball_point(projs[:, :2], radiuses)

    for j, (ref_idx, mp) in enumerate(zip(matched_ref_idxs, matched_ref_points)):
        if not is_visible[j]:
            continue

        predicted_level = predicted_levels[j]
        kd_indices = kd_cur_idxs[j]

        best_dist = np.inf  # math.inf
        best_dist2 = np.inf  # math.inf
        best_level = -1
        best_level2 = -1
        best_k_idx = -1

        for idx2 in kd_indices:
            if f_cur.points[idx2] is not None:
                continue  # already matched

            kp_level = f_cur.octaves[idx2]
            if (kp_level < predicted_level - 1) or (kp_level > predicted_level + 1):
                continue

            descriptor_dist = mp.min_des_distance(f_cur.des[idx2])

            if descriptor_dist < best_dist:
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = idx2
            elif descriptor_dist < best_dist2:
                best_dist2 = descriptor_dist
                best_level2 = kp_level

        if best_dist < max_descriptor_distance:
            if (best_level == best_level2) and (best_dist > best_dist2 * ratio_test):
                continue

            if mp.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1
                idxs_ref.append(ref_idx)
                idxs_cur.append(best_k_idx)

                if check_orientation:
                    rot = kf_ref.angles[ref_idx] - f_cur.angles[best_k_idx]
                    rot_histo.push(rot, len(idxs_cur) - 1)

    if check_orientation:
        valid_match_idxs = rot_histo.get_valid_idxs()
        print(
            "checking orientation consistency - valid matches %:",
            len(valid_match_idxs) / max(1, len(idxs_cur)) * 100,
            "% of",
            len(idxs_cur),
            "matches",
        )

        idxs_ref = np.array(idxs_ref)[valid_match_idxs]
        idxs_cur = np.array(idxs_cur)[valid_match_idxs]
        found_pts_count = len(valid_match_idxs)

    return np.array(idxs_ref), np.array(idxs_cur), found_pts_count


# search by projection matches between {input map points} and {unmatched keypoints of frame f_cur}, (access frame from tracking thread, no need to lock)
def _search_map_by_projection(
    points: list[MapPoint],
    f_cur: Frame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceMap,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
    far_points_threshold=None,
):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance

    if len(points) == 0:
        return 0, []
    # check if points are visible
    visibility_flags, projs, depths, dists = f_cur.are_visible(points)

    predicted_levels = MapPoint.predict_detection_levels(points, dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors
    kd_cur_idxs = f_cur.kd.query_ball_point(projs, radiuses)

    # trick to filter out far points if required => we mark them as not visible
    if far_points_threshold is not None:
        # print(f'search_map_by_projection: using far points threshold: {far_points_threshold}')
        visibility_flags = np.logical_and(visibility_flags, depths < far_points_threshold)

    idxs_and_pts = [
        (i, p)
        for i, p in enumerate(points)
        if visibility_flags[i] and not p.is_bad() and p.last_frame_id_seen != f_cur.id
    ]

    found_pts_count = 0
    found_pts_fidxs = []  # idx of matched points in current frame

    cur_des = f_cur.des
    cur_octaves = f_cur.octaves
    cur_points = f_cur.points

    for i, p in idxs_and_pts:
        p.increase_visible()

        predicted_level = predicted_levels[i]

        best_dist = np.inf  # math.inf
        best_dist2 = np.inf  # math.inf
        best_level = -1
        best_level2 = -1
        best_k_idx = -1

        # find closest keypoints of f_cur
        for kd_idx in kd_cur_idxs[i]:

            p_f = cur_points[kd_idx]
            # check there is not already a match
            if p_f is not None and p_f.num_observations() > 0:
                continue

            # check detection level
            kp_level = cur_octaves[kd_idx]
            if (kp_level < predicted_level - 1) or (kp_level > predicted_level):
                continue

            descriptor_dist = p.min_des_distance(cur_des[kd_idx])

            if descriptor_dist < best_dist:
                best_dist2 = best_dist
                best_level2 = best_level
                best_dist = descriptor_dist
                best_level = kp_level
                best_k_idx = kd_idx
            elif descriptor_dist < best_dist2:
                best_dist2 = descriptor_dist
                best_level2 = kp_level

        # if best_k_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance:
            # apply match distance ratio test only if the best and second are in the same scale level
            if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test):
                continue
            # print('best des distance: ', best_dist, ", max dist: ", Parameters.kMaxDescriptorDistance)
            if p.add_frame_view(f_cur, best_k_idx):
                found_pts_count += 1
                found_pts_fidxs.append(best_k_idx)

            # reproj_dists.append(np.linalg.norm(projs[i] - f_cur.kpsu[best_k_idx]))

    # if len(reproj_dists) > 1:
    #     reproj_dist_sigma = 1.4826 * np.median(reproj_dists)

    return found_pts_count, found_pts_fidxs


# search by projection matches between {map points of last frames} and {unmatched keypoints of f_cur}, (access frame from tracking thread, no need to lock)
def _search_local_frames_by_projection(
    map, f_cur, local_window_size=Parameters.kLocalBAWindowSize, max_descriptor_distance=None
):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance

    # take the points in the last N frame
    frames = map.keyframes[-local_window_size:]
    frames_valid_points = set([p for f in frames for p in f.get_points() if (p is not None)])
    print("searching %d map points" % len(frames_valid_points))
    return _search_map_by_projection(
        list(frames_valid_points), f_cur, max_descriptor_distance=max_descriptor_distance
    )


# search by projection matches between {all map points} and {unmatched keypoints of f_cur}
def _search_all_map_by_projection(map, f_cur, max_descriptor_distance=None):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance

    return _search_map_by_projection(
        map.get_points(), f_cur, max_descriptor_distance=max_descriptor_distance
    )


# search by projection more matches between {input map points} and {unmatched keypoints of frame f_cur}
# in:
#   points: input map points
#   f_cur: current frame
#   Scw: suggested se3 or sim3 transformation
#   f_cur_matched_points: matched points in current frame  (f_cur_matched_points[i] is the i-th map point matched on f_cur or None)
# NOTE1: f_cur_matched_points is modified in place and passed by reference in output
# NOTE2: The suggested transformation Scw (in se3 or sim3) is used in the search (instead of using the current frame pose)
def _search_more_map_points_by_projection(
    points: set,
    f_cur: Frame,
    Scw,
    f_cur_matched_points: list,  # f_cur_matched_points[i] is the i-th map point matched in f_cur or None
    f_cur_matched_points_idxs: (
        list | None
    ) = None,  # f_cur_matched_points_idxs[i] is the index of the i-th map point matched in f_cur or -1
    max_reproj_distance=Parameters.kMaxReprojectionDistanceMap,
    max_descriptor_distance=None,
    print_fun=None,
):
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5 * Parameters.kMaxDescriptorDistance  # more conservative check

    found_pts_count = 0
    if len(points) == 0:
        return found_pts_count, f_cur_matched_points

    assert len(f_cur.points) == len(f_cur_matched_points)

    # extract from sim3 Scw=[s*Rcw, tcw; 0, 1] the corresponding se3 transformation Tcw=[Rcw, tcw/s]
    if isinstance(Scw, np.ndarray):
        sRcw = Scw[:3, :3]
        scw = math.sqrt(np.dot(sRcw[0, :3], sRcw[0, :3]))
        Rcw = sRcw / scw
        tcw = Scw[:3, 3] / scw
    elif isinstance(Scw, Sim3Pose):
        scw = Scw.s
        Rcw = Scw.R
        tcw = Scw.t / scw
    else:
        raise TypeError("Unsupported type '{}' for Scw".format(type(Scw)))

    # Build a set of matched map point ids
    if f_cur_matched_points_idxs is not None:
        f_cur_matched_points_idxs = {mp.id for mp in f_cur_matched_points if mp is not None}

    target_points = [
        p
        for p in points
        if (p is not None and not p.is_bad() and p.id not in f_cur_matched_points_idxs)
    ]

    if len(target_points) == 0:
        if print_fun is not None:
            print_fun(
                "search_more_map_points_by_projection: no target points available after difference"
            )
        return found_pts_count, f_cur_matched_points

    # check if points are visible
    visibility_flags, projs, depths, dists = are_map_points_visible_in_frame(
        target_points, f_cur, Rcw, tcw
    )

    if print_fun is not None:
        print_fun(f"search_more_map_points_by_projection: #visible points: {len(visibility_flags)}")

    predicted_levels = MapPoint.predict_detection_levels(target_points, dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors
    kd_cur_idxs = f_cur.kd.query_ball_point(projs, radiuses)

    # num_failures_vis_or_bad = 0
    # num_failures_pf_is_not_none = 0
    # num_failures_kp_level = 0
    # num_failures_max_des_distance = 0

    for i, p in enumerate(target_points):
        if not visibility_flags[i]:  # or p.is_bad():     # point not visible in frame or is bad
            # num_failures_vis_or_bad +=1
            continue

        predicted_level = predicted_levels[i]

        best_dist = np.inf  # math.inf
        best_k_idx = -1

        # find closest keypoints of f_cur
        for kd_idx in kd_cur_idxs[i]:

            p_f = f_cur_matched_points[kd_idx]
            # check there is not already a match in f_cur_matched_points
            if p_f is not None:
                # num_failures_pf_is_not_none +=1
                continue

            # check detection level
            kp_level = f_cur.octaves[kd_idx]
            if (kp_level < predicted_level - 1) or (kp_level > predicted_level):
                # if print_fun is not None:
                #     print_fun(f'search_more_map_points_by_projection: bad kp level: {kp_level},  predicted_level: {predicted_level}')
                # num_failures_kp_level += 1
                continue

            descriptor_dist = p.min_des_distance(f_cur.des[kd_idx])

            if descriptor_dist < best_dist:
                best_dist = descriptor_dist
                best_k_idx = kd_idx

        if best_dist < max_descriptor_distance:
            f_cur_matched_points[best_k_idx] = p
            found_pts_count += 1
        # else:
        #     if print_fun is not None:
        #         print_fun(f'search_more_map_points_by_projection: bad best_des_distance: {best_dist}, max_descriptor_distance: {max_descriptor_distance}')
        #     num_failures_max_des_distance += 1

    # if print_fun is not None:
    #     print_fun(f'search_more_map_points_by_projection: num_failures_vis_or_bad: {num_failures_vis_or_bad}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_pf_is_not_none: {num_failures_pf_is_not_none}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_kp_level: {num_failures_kp_level}')
    #     print_fun(f'search_more_map_points_by_projection: num_failures_max_des_distance: {num_failures_max_des_distance}')

    return found_pts_count, f_cur_matched_points


# search by projection matches between {input map points} and {keyframe points} and fuse them if they are close enough
def _search_and_fuse(
    points: list[MapPoint],
    keyframe: KeyFrame,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
    max_descriptor_distance=None,
    ratio_test=Parameters.kMatchRatioTestMap,
):
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5 * Parameters.kMaxDescriptorDistance  # more conservative check

    fused_pts_count = 0
    if len(points) == 0:
        Printer.red("search_and_fuse - no points")
        return fused_pts_count

    # good_pts_idxs = [i for i, p in enumerate(points) if p is not None and not p.is_bad() and not p.is_in_keyframe(keyframe)]
    good_pts_idxs = [
        i
        for i, p in enumerate(points)
        if p is not None and not p.is_bad_or_is_in_keyframe(keyframe)
    ]
    good_pts = [points[i] for i in good_pts_idxs]
    good_pts = np.asarray(good_pts)

    if len(good_pts_idxs) == 0:
        Printer.red("search_and_fuse - no matched points")
        return fused_pts_count

    # check if points are visible
    good_pts_visible, good_projs, good_depths, good_dists = keyframe.are_visible(
        good_pts, keyframe.camera.is_stereo()
    )

    if np.sum(good_pts_visible) == 0:
        Printer.orange("search_and_fuse - no visible points")
        return fused_pts_count

    predicted_levels = MapPoint.predict_detection_levels(good_pts, good_dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors

    inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2

    kd_idxs = keyframe.kd.query_ball_point(good_projs[:, :2], radiuses)

    do_stereo_check = keyframe.kps_ur is not None and len(keyframe.kps_ur) > 0

    if do_stereo_check:
        check_stereo_flags = keyframe.kps_ur >= 0
    else:
        check_stereo_flags = np.zeros(len(keyframe.kpsu), dtype=bool)

    octaves = keyframe.octaves
    kpsu = keyframe.kpsu
    kps_ur = keyframe.kps_ur if do_stereo_check else None
    des = keyframe.des

    # for j, (i, p) in enumerate(zip(good_pts_idxs, good_pts)):
    for j, p in enumerate(good_pts):
        if not good_pts_visible[
            j
        ]:  # or p.is_bad():     # point not visible in frame or point is bad
            # print('p[%d] visible: %d, bad: %d' % (i, int(good_pts_visible[j]), int(p.is_bad())))
            continue

        # if p.is_in_keyframe(keyframe):    # we already matched this map point to this keyframe
        #     #print('p[%d] already in keyframe' % (i))
        #     continue

        predicted_level = predicted_levels[j]

        best_dist = np.inf  # math.inf
        # best_dist2 = np.inf #math.inf
        # best_level = -1
        # best_level2 = -1
        best_kd_idx = -1

        # find closest keypoints of frame
        proj = good_projs[j]

        kd_idxs_j = kd_idxs[j]
        kd_idxs_j = np.array(kd_idxs_j, dtype=int)  # ensure it's an array for indexing

        len_kd_idxs_j = len(kd_idxs_j)
        if len_kd_idxs_j == 0:
            continue

        if len_kd_idxs_j > 0:
            valid_stereo_mask = check_stereo_flags[
                kd_idxs_j
            ]  # boolean mask of valid stereo matches
            errs_ur2 = np.zeros(len_kd_idxs_j, dtype=np.float32)

            if np.any(valid_stereo_mask):
                proj_ur = proj[2]  # scalar
                kps_ur_vals = kps_ur[kd_idxs_j[valid_stereo_mask]]  # only stereo-valid keypoints
                errs_ur = proj_ur - kps_ur_vals
                errs_ur2[valid_stereo_mask] = errs_ur**2

        for h, kd_idx in enumerate(kd_idxs_j):
            # check detection level
            kp_level = octaves[kd_idx]
            if (kp_level < predicted_level - 1) or (kp_level > predicted_level):
                # print('p[%d] wrong predicted level **********************************' % (i))
                continue

            # check the reprojection error
            invSigma2 = inv_level_sigmas2[kp_level]
            err = proj[:2] - kpsu[kd_idx]
            chi2 = np.dot(err, err) * invSigma2
            if check_stereo_flags[kd_idx]:
                chi2 += errs_ur2[h] * invSigma2
                if chi2 > Parameters.kChi2Stereo:  # chi-square 3 DOFs  (Hartley Zisserman pg 119)
                    # print('p[%d] big reproj err %f **********************************' % (i,chi2))
                    continue
            else:
                if chi2 > Parameters.kChi2Mono:  # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                    # print('p[%d] big reproj err %f **********************************' % (i,chi2))
                    continue

            descriptor_dist = p.min_des_distance(des[kd_idx])
            # print('p[%d] descriptor_dist %f **********************************' % (i,descriptor_dist))

            # if descriptor_dist < max_descriptor_distance and descriptor_dist < best_dist:
            if descriptor_dist < best_dist:
                # best_dist2 = best_dist
                # best_level2 = best_level
                best_dist = descriptor_dist
                # best_level = kp_level
                best_kd_idx = kd_idx
            # elif descriptor_dist < best_dist2:  # N.O.
            #     best_dist2 = descriptor_dist
            #     best_level2 = kp_level

        # if best_kd_idx > -1 and best_dist < max_descriptor_distance:
        if best_dist < max_descriptor_distance:
            # apply match distance ratio test only if the best and second are in the same scale level
            # if (best_level2 == best_level) and (best_dist > best_dist2 * ratio_test):  # N.O.
            #     #print('p[%d] best_dist > best_dist2 * ratio_test **********************************' % (i))
            #     continue
            p_keyframe = keyframe.get_point_match(best_kd_idx)
            # if there is already a map point replace it otherwise add a new point
            if p_keyframe is not None:
                # if not p_keyframe.is_bad():
                #     if p_keyframe.num_observations() > p.num_observations():
                #         p.replace_with(p_keyframe)
                #     else:
                #         p_keyframe.replace_with(p)
                p_keyframe_is_bad, p_keyframe_is_good_with_better_num_obs = (
                    p_keyframe.is_bad_and_is_good_with_min_obs(p.num_observations())
                )
                if not p_keyframe_is_bad:
                    if p_keyframe_is_good_with_better_num_obs:
                        p.replace_with(p_keyframe)
                    else:
                        p_keyframe.replace_with(p)
            else:
                p.add_observation(keyframe, best_kd_idx)
                # p.update_info()    # done outside!
            fused_pts_count += 1

    return fused_pts_count


# search by projection matches between {input map points} and {keyframe points} and fuse them if they are close enough
# use suggested Scw to project
def _search_and_fuse_for_loop_correction(
    keyframe: KeyFrame,
    Scw,
    points,
    replace_points,
    max_reproj_distance=Parameters.kLoopClosingMaxReprojectionDistanceFuse,
    max_descriptor_distance=None,
):
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5 * Parameters.kMaxDescriptorDistance  # more conservative check

    assert len(points) == len(replace_points)

    fused_pts_count = 0
    if len(points) == 0:
        Printer.red("search_and_fuse - no points")
        return replace_points

    # get all matched points of keyframe
    # good_pts_idxs = np.flatnonzero(points!=None)
    good_pts_idxs = [
        i
        for i, p in enumerate(points)
        if p is not None and not p.is_bad() and not p.is_in_keyframe(keyframe)
    ]
    good_pts = points[good_pts_idxs]

    if len(good_pts_idxs) == 0:
        Printer.red("search_and_fuse - no matched points")
        return replace_points

    # extract from sim3 Scw=[s*Rcw, tcw; 0, 1] the corresponding se3 transformation Tcw=[Rcw, tcw/s]
    if isinstance(Scw, np.ndarray):
        sRcw = Scw[:3, :3]
        scw = math.sqrt(np.dot(sRcw[0, :3], sRcw[0, :3]))
        Rcw = sRcw / scw
        tcw = Scw[:3, 3] / scw
    elif isinstance(Scw, Sim3Pose):
        scw = Scw.s
        Rcw = Scw.R
        tcw = Scw.t / scw
    else:
        raise TypeError("Unsupported type '{}' for Scw".format(type(Scw)))

    # check if points are visible
    good_pts_visible, good_projs, good_depths, good_dists = are_map_points_visible_in_frame(
        good_pts, keyframe, Rcw, tcw
    )

    if np.sum(good_pts_visible) == 0:
        Printer.orange("search_and_fuse - no visible points")
        return replace_points

    predicted_levels = MapPoint.predict_detection_levels(good_pts, good_dists)
    kp_scale_factors = FeatureTrackerShared.feature_manager.scale_factors[predicted_levels]
    radiuses = max_reproj_distance * kp_scale_factors

    kd_idxs = keyframe.kd.query_ball_point(good_projs[:, :2], radiuses)

    # for idx,p,j in zip(good_pts_idxs,good_pts,range(len(good_pts))):
    for j, (idx, p) in enumerate(zip(good_pts_idxs, good_pts)):
        if not good_pts_visible[
            j
        ]:  # or p.is_bad():     # point not visible in frame or point is bad
            # print('p[%d] visible: %d, bad: %d' % (i, int(good_pts_visible[j]), int(p.is_bad())))
            continue

        # if p.is_in_keyframe(keyframe):    # we already matched this map point to this keyframe
        #     #print('p[%d] already in keyframe' % (i))
        #     continue

        predicted_level = predicted_levels[j]

        best_dist = np.inf  # math.inf
        best_kd_idx = -1

        # find closest keypoints of frame
        proj = good_projs[j]

        kd_idxs_j = kd_idxs[j]

        for kd_idx in kd_idxs_j:

            # check detection level
            kp_level = keyframe.octaves[kd_idx]
            if (kp_level < predicted_level - 1) or (kp_level > predicted_level):
                # print('p[%d] wrong predicted level **********************************' % (i))
                continue

            descriptor_dist = p.min_des_distance(keyframe.des[kd_idx])
            # print('p[%d] descriptor_dist %f **********************************' % (i,descriptor_dist))

            if descriptor_dist < best_dist:
                best_dist = descriptor_dist
                best_kd_idx = kd_idx

        if best_dist < max_descriptor_distance:
            p_keyframe = keyframe.get_point_match(best_kd_idx)
            # if there is already a map point replace it
            if p_keyframe is not None:
                if not p_keyframe.is_bad():
                    replace_points[idx] = p_keyframe
            else:
                p.add_observation(keyframe, best_kd_idx)
                # p.update_info()    # done outside!
            fused_pts_count += 1

    return replace_points


# search new matches between unmatched map points of kf1 and kf2 by using a know sim3 transformation (guided matching)
# in:
#   kf1, kf2: keyframes
#   idxs1, idxs2:  kf1.points(idxs1[i]) is matched with kf2.points(idxs2[i])
#   s12, R12, t12: sim3 transformation that guides the matching
# out:
#   new_matches12: where kf2.points(new_matches12[i]) is matched to i-th map point in kf1 (includes the input matches) if new_matches12[i]>0
#   new_matches21: where kf1.points(new_matches21[i]) is matched to i-th map point in kf2 (includes the input matches) if new_matches21[i]>0
def _search_by_sim3(
    kf1: KeyFrame,
    kf2: KeyFrame,
    idxs1,
    idxs2,
    s12,
    R12,
    t12,
    max_reproj_distance=Parameters.kMaxReprojectionDistanceSim3,
    max_descriptor_distance=None,
    print_fun=None,
):
    if max_descriptor_distance is None:
        max_descriptor_distance = Parameters.kMaxDescriptorDistance

    assert len(idxs1) == len(idxs2)
    # Sim3 transformations between cameras
    sR12 = s12 * R12
    sR21 = (1.0 / s12) * R12.T
    t21 = -sR21 @ t12

    map_points1 = kf1.get_points()  # get all map points of kf1
    n1 = len(map_points1)
    new_matches12 = np.full(
        n1, -1, dtype=np.int32
    )  # kf2.points(new_matches12[i]) is matched to i-th map point in kf1 if new_matches12[i]>0 (from 1 to 2)
    good_points1 = np.array(
        [True if mp is not None and not mp.is_bad() else False for mp in map_points1]
    )

    map_points2 = kf2.get_points()  # get all map points of kf2
    n2 = len(map_points2)
    new_matches21 = np.full(
        n2, -1, dtype=np.int32
    )  # kf1.points(new_matches21[i]) is matched to i-th map point in kf2 if new_matches21[i]>0 (from 2 to 1)
    good_points2 = np.array(
        [True if mp is not None and not mp.is_bad() else False for mp in map_points2]
    )

    for idx1, idx2 in zip(idxs1, idxs2):
        # Integrate the matches we already have as input into the output
        if good_points1[idx1] and good_points2[idx2]:
            new_matches12[idx1] = idx2
            new_matches21[idx2] = idx1

    # if print_fun is not None:
    #     print_fun(f'search_by_sim3: starting num mp matches: {np.sum(new_matches12!=-1)}')

    map_points1_array = np.asarray(map_points1, dtype=object)
    map_points2_array = np.asarray(map_points2, dtype=object)

    # Find unmatched map points
    unmatched_idxs1 = np.array(
        [idx for idx in range(n1) if good_points1[idx] and new_matches12[idx] < 0], dtype=np.int32
    )
    unmatched_map_points1 = map_points1_array[unmatched_idxs1]

    unmatched_idxs2 = np.array(
        [idx for idx in range(n2) if good_points2[idx] and new_matches21[idx] < 0], dtype=np.int32
    )
    unmatched_map_points2 = map_points2_array[unmatched_idxs2]

    # if print_fun is not None:
    #     print_fun(f'search_by_sim3: found: {len(unmatched_idxs1)} unmatched map points of kf1 {kf1.id}, {len(unmatched_idxs2)} unmatched map points of kf2 {kf2.id}')

    scale_factors = FeatureTrackerShared.feature_manager.scale_factors

    # check which unmatched points of kf1 are visible on kf2
    visible_flags_21, projs_21, depths_21, dists_21 = are_map_points_visible(
        kf1, kf2, unmatched_map_points1, sR21, t21
    )

    num_visible_21 = np.sum(visible_flags_21)
    # if print_fun is not None:
    #     print_fun(f'search_by_sim3: {num_visible_21} map points of kf1 {kf1.id} are visible on kf2 {kf2.id}')

    if num_visible_21 > 0:
        predicted_levels = MapPoint.predict_detection_levels(unmatched_map_points1, dists_21)
        kp_scale_factors = scale_factors[predicted_levels]
        radiuses = max_reproj_distance * kp_scale_factors
        kd2_idxs = kf2.kd.query_ball_point(projs_21[:, :2], radiuses)  # search NN kps on kf2

        for i1, mp1 in enumerate(unmatched_map_points1):
            kd2_idxs_i = kd2_idxs[i1]
            predicted_level = predicted_levels[i1]

            best_dist = np.inf  # float('inf')
            best_idx = -1
            for kd2_idx in kd2_idxs_i:
                # check detection level
                kp_level = kf2.octaves[kd2_idx]
                if (kp_level < predicted_level - 1) or (kp_level > predicted_level):
                    continue

                dist = mp1.min_des_distance(kf2.des[kd2_idx])

                if dist < best_dist:
                    best_dist = dist
                    best_idx = kd2_idx

            if best_dist <= max_descriptor_distance:
                if new_matches21[best_idx] == -1:
                    new_matches12[unmatched_idxs1[i1]] = best_idx

    # check which unmatched points of kf2 are visible on kf1
    visible_flags_12, projs_12, depths_12, dists_12 = are_map_points_visible(
        kf2, kf1, unmatched_map_points2, sR12, t12
    )

    num_visible_12 = np.sum(visible_flags_12)
    # if print_fun is not None:
    #     print_fun(f'search_by_sim3: {num_visible_12} map points of kf2 {kf2.id} are visible on kf1 {kf1.id}')

    if num_visible_12 > 0:
        predicted_levels = MapPoint.predict_detection_levels(unmatched_map_points2, dists_12)
        kp_scale_factors = scale_factors[predicted_levels]
        radiuses = max_reproj_distance * kp_scale_factors
        kd1_idxs = kf1.kd.query_ball_point(projs_12[:, :2], radiuses)  # search NN kps on kf1

        for i2, mp2 in enumerate(unmatched_map_points2):
            kd1_idxs_i = kd1_idxs[i2]
            predicted_level = predicted_levels[i2]

            best_dist = np.inf  # float('inf')
            best_idx = -1
            for kd1_idx in kd1_idxs_i:
                # check detection level
                kp_level = kf1.octaves[kd1_idx]
                if (kp_level < predicted_level - 1) or (kp_level > predicted_level):
                    continue

                dist = mp2.min_des_distance(kf1.des[kd1_idx])

                if dist < best_dist:
                    best_dist = dist
                    best_idx = kd1_idx

            if best_dist <= max_descriptor_distance:
                if new_matches12[best_idx] == -1:
                    new_matches21[unmatched_idxs2[i2]] = best_idx

    # if print_fun is not None:
    #     print_fun(f'search_by_sim3: new matches before check: 1->2: {np.sum(new_matches12!=-1)}, 2->1: {np.sum(new_matches21!=-1)}')

    # Check agreement
    num_matches_found = 0
    for i1 in range(n1):
        idx2 = new_matches12[i1]  # index of kf2 point that matches with i1-th kf1 point
        if idx2 >= 0:
            idx1 = new_matches21[idx2]  # index of kf1 point that matches with idx2-th kf2 point
            if idx1 != i1:  # reset if mismatch
                new_matches12[i1] = -1
                new_matches21[idx2] = -1
            else:
                num_matches_found += 1

    # if print_fun is not None:
    #     print_fun(f'search_by_sim3: num matches found after final check: {num_matches_found}')
    #     print_fun(f'search_by_sim3: new matches after check: 1->2: {np.sum(new_matches12!=-1)}, 2->1: {np.sum(new_matches21!=-1)}')

    return num_matches_found, new_matches12, new_matches21


# search keypoint matches (for triangulations) between f1 and f2
# search for matches between unmatched keypoints (without a corresponding map point)
# in input we have already some pose estimates for f1 and f2
def _search_frame_for_triangulation(
    kf1: KeyFrame,
    kf2: KeyFrame,
    idxs1: list[int] | None = None,
    idxs2: list[int] | None = None,
    max_descriptor_distance: float | None = None,
    is_monocular: bool = True,
):
    if max_descriptor_distance is None:
        max_descriptor_distance = 0.5 * Parameters.kMaxDescriptorDistance  # more conservative check

    if __debug__:
        timer = Timer()
        timer.start()

    O1w = kf1.Ow()
    O2w = kf2.Ow()
    # compute epipoles
    e1, _ = kf1.project_point(O2w)  # in first frame
    e2, _ = kf2.project_point(O1w)  # in second frame

    baseline = np.linalg.norm(O1w - O2w)

    # if the translation is too small we cannot triangulate
    if not is_monocular:  # we assume the Inializer has been used for building the first map
        if baseline < kf2.camera.b:
            return [], [], 0  # EXIT
    else:
        median_depth = kf2.compute_points_median_depth()
        if median_depth == -1:
            Printer.orange("search for triangulation: f2 with negative median depth")
            median_depth = kf1.compute_points_median_depth()
        ratio_baseline_depth = baseline / median_depth
        if ratio_baseline_depth < Parameters.kMinRatioBaselineDepth:
            Printer.orange("search for triangulation: impossible with too low ratioBaselineDepth!")
            return [], [], 0  # EXIT

    # compute the fundamental matrix between the two frames by using their estimated poses
    F12, H21 = computeF12(kf1, kf2)

    if idxs1 is None or idxs2 is None:
        timerMatch = Timer()
        timerMatch.start()
        matching_result = FeatureTrackerShared.feature_matcher.match(
            kf1.img, kf2.img, kf1.des, kf2.des
        )
        idxs1, idxs2 = matching_result.idxs1, matching_result.idxs2
        if __debug__:
            print("search_frame_for_triangulation - matching - timer: ", timerMatch.elapsed())

    check_orientation = kCheckFeaturesOrientation and FeatureTrackerShared.oriented_features
    level_sigmas2 = FeatureTrackerShared.feature_manager.level_sigmas2
    scale_factors = FeatureTrackerShared.feature_manager.scale_factors

    # Convert to numpy arrays for vectorization
    idxs1 = np.array(idxs1)
    idxs2 = np.array(idxs2)

    # Vectorized filtering: check if points already have map points
    has_map_point1 = np.array([kf1.get_point_match(i) is not None for i in idxs1], dtype=bool)
    has_map_point2 = np.array([kf2.get_point_match(i) is not None for i in idxs2], dtype=bool)
    valid_matches = ~(has_map_point1 | has_map_point2)

    if not np.any(valid_matches):
        return [], [], 0

    # Filter valid matches
    valid_idxs1 = idxs1[valid_matches]
    valid_idxs2 = idxs2[valid_matches]

    # Vectorized descriptor distance computation
    des1_valid = kf1.des[valid_idxs1]
    des2_valid = kf2.des[valid_idxs2]
    descriptor_dists = FeatureTrackerShared.descriptor_distances(des1_valid, des2_valid)

    # Filter by descriptor distance
    good_descriptor = descriptor_dists <= max_descriptor_distance
    # Flatten in case descriptor_distances returns 2D array (e.g., l2_distances with keepdims=True)
    good_descriptor = good_descriptor.ravel()
    descriptor_dists = descriptor_dists.ravel()

    if not np.any(good_descriptor):
        return [], [], 0

    # Further filter by descriptor distance
    valid_idxs1 = valid_idxs1[good_descriptor]
    valid_idxs2 = valid_idxs2[good_descriptor]
    descriptor_dists = descriptor_dists[good_descriptor]

    # Vectorized keypoint extraction
    kps1_valid = kf1.kpsu[valid_idxs1]
    kps2_valid = kf2.kpsu[valid_idxs2]
    octaves2_valid = kf2.octaves[valid_idxs2]

    # Vectorized epipole distance check
    kp2_scale_factors = scale_factors[octaves2_valid]
    deltas = kps2_valid - e2
    epipole_distances_sq = np.sum(deltas**2, axis=1)
    min_epipole_distances_sq = kMinDistanceFromEpipole2 * kp2_scale_factors
    good_epipole_distance = epipole_distances_sq >= min_epipole_distances_sq

    if not np.any(good_epipole_distance):
        return [], [], 0

    # Further filter by epipole distance
    valid_idxs1 = valid_idxs1[good_epipole_distance]
    valid_idxs2 = valid_idxs2[good_epipole_distance]
    kps1_valid = kps1_valid[good_epipole_distance]
    kps2_valid = kps2_valid[good_epipole_distance]
    octaves2_valid = octaves2_valid[good_epipole_distance]

    # Vectorized epipolar constraint check
    sigma2_kps2 = level_sigmas2[octaves2_valid]

    # Vectorized epipolar line computation
    kps1_homogeneous = np.column_stack([kps1_valid, np.ones(len(kps1_valid))])
    epipolar_lines = (F12.T @ kps1_homogeneous.T).T  # [a, b, c] for each line

    # Vectorized distance computation
    numerators = (
        epipolar_lines[:, 0] * kps2_valid[:, 0]
        + epipolar_lines[:, 1] * kps2_valid[:, 1]
        + epipolar_lines[:, 2]
    )
    denominators = epipolar_lines[:, 0] ** 2 + epipolar_lines[:, 1] ** 2

    # Handle zero denominators
    valid_denominators = denominators > 1e-20
    if not np.any(valid_denominators):
        return [], [], 0

    # Compute distances only for valid denominators
    dists_sq = np.zeros(len(numerators))
    dists_sq[valid_denominators] = (numerators[valid_denominators] ** 2) / denominators[
        valid_denominators
    ]

    # Chi-square threshold check
    chi2_threshold = 3.84 * sigma2_kps2
    good_epipolar = dists_sq < chi2_threshold

    if not np.any(good_epipolar):
        return [], [], 0

    # Final filtering
    final_idxs1 = valid_idxs1[good_epipolar]
    final_idxs2 = valid_idxs2[good_epipolar]

    # Handle orientation consistency if needed
    if check_orientation:
        rot_histo = RotationHistogram()
        angles1_valid = kf1.angles[final_idxs1]
        angles2_valid = kf2.angles[final_idxs2]

        # for i, (angle1, angle2) in enumerate(zip(angles1_valid, angles2_valid)):
        #     rot = angle1 - angle2
        #     rot_histo.push(rot, i)

        rots = angles1_valid - angles2_valid
        rot_histo.push_entries(rots, [ii for ii in range(len(final_idxs1))])

        valid_match_idxs = rot_histo.get_valid_idxs()
        if len(valid_match_idxs) > 0:
            final_idxs1 = final_idxs1[valid_match_idxs]
            final_idxs2 = final_idxs2[valid_match_idxs]
        else:
            final_idxs1 = np.array([])
            final_idxs2 = np.array([])

    num_found_matches = len(final_idxs1)

    if __debug__:
        print("search_frame_for_triangulation - timer: ", timer.elapsed())

    return final_idxs1, final_idxs2, num_found_matches
