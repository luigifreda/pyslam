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
import os
import numpy as np
import time

# import json
import ujson as json

from collections import defaultdict, Counter
from itertools import chain

import cv2
import g2o

from pyslam.config_parameters import Parameters

from .frame import match_frames
from .feature_tracker_shared import FeatureTrackerShared

from pyslam.slam import (
    USE_CPP,
    Frame,
    MapPoint,
    KeyFrame,
    Map,
    optimizer_g2o,
    TrackingCore,
    RotationHistogram,
    ProjectionMatcher,
)

from . import optimizer_gtsam

from .slam_commons import SlamState

from .initializer import Initializer
from .slam_dynamic_config import SLAMDynamicConfig
from .motion_model import MotionModel, MotionModelDamping

from pyslam.io.dataset_types import SensorType

from pyslam.utilities.logging import Printer, Logging
from pyslam.utilities.drawing import draw_feature_matches
from pyslam.utilities.geometry import poseRt, inv_T

from pyslam.utilities.features import ImageGrid
from pyslam.utilities.timer import TimerFps
from pyslam.utilities.img_processing import detect_blur_laplacian


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .slam import Slam, SlamState
    from .frame import Frame
    from .keyframe import KeyFrame
    from .map_point import MapPoint
    from .map import Map
    from . import optimizer_g2o
    from . import optimizer_gtsam
    from .tracking_core import TrackingCore
    from .rotation_histogram import RotationHistogram
    from .geometry_matchers import ProjectionMatcher, EpipolarMatcher


kVerbose = True
kTimerVerbose = False

kShowFeatureMatches = (
    False  # this flag dominates over the following related ones kShowFeatureMatchesXXX
)
kShowFeatureMatchesPrevFrame = True
kShowFeatureMatchesRefFrame = True
kShowFeatureMatchesLocalMap = True

kLogKFinfoToFile = True

kUseDynamicDesDistanceTh = Parameters.kUseDynamicDesDistanceTh

kUseGroundTruthScale = False

kNumMinInliersPoseOptimizationTrackFrame = 10
kNumMinInliersPoseOptimizationTrackLocalMap = 20
kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle = (
    50  # 60  # defines bad/weak tracking condition
)


kUseMotionModel = Parameters.kUseMotionModel or Parameters.kUseSearchFrameByProjection
kUseSearchFrameByProjection = (
    Parameters.kUseSearchFrameByProjection and not Parameters.kUseEssentialMatrixFitting
)

kNumMinObsForKeyFrameDefault = 3

kMinDepth = Parameters.kMinDepth


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."


if not kVerbose:

    def print(*args, **kwargs):
        pass


class TrackingHistory(object):
    def __init__(self):
        self.relative_frame_poses = (
            []
        )  # list of relative frame poses w.r.t reference keyframes as g2o.Isometry3d() (see camera_pose.py)
        self.kf_references = []  # list of reference keyframes
        self.timestamps = []  # list of frame timestamps
        self.ids = []
        self.slam_states = []  # list of slam states

    def reset(self):
        self.relative_frame_poses.clear()
        self.kf_references.clear()
        self.timestamps.clear()
        self.ids.clear()
        self.slam_states.clear()


class Tracking:
    def __init__(self, slam: "Slam"):

        if kShowFeatureMatches:
            Frame.is_store_imgs = True

        self.slam = slam

        self.initializer = Initializer(self.sensor_type)

        self.motion_model = (
            MotionModel()
        )  # motion model for current frame pose prediction without damping
        # self.motion_model = MotionModelDamping()  # motion model for current frame pose prediction with damping

        self.dyn_config = SLAMDynamicConfig(
            self.feature_tracker.feature_manager.max_descriptor_distance
        )
        self.descriptor_distance_sigma: float = (
            self.feature_tracker.feature_manager.max_descriptor_distance
        )
        self.reproj_err_frame_map_sigma: float = Parameters.kMaxReprojectionDistanceMap
        if self.sensor_type == SensorType.RGBD:
            self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMapRgbd

        self.max_frames_between_kfs = int(slam.camera.fps) if slam.camera.fps is not None else 1
        self.max_frames_between_kfs_after_reloc = (
            self.max_frames_between_kfs
        )  # after relocalization, we need to insert a keyframe to avoid drift
        self.min_frames_between_kfs = 0

        # params read and set by Slam
        self.far_points_threshold = None
        self.use_fov_centers_based_kf_generation = False  # set by Slam.set_config_params
        self.max_fov_centers_distance = -1

        self.state = SlamState.NO_IMAGES_YET

        self.num_matched_kps = None  # current number of matched keypoints
        self.num_inliers = None  # current number of matched points
        self.num_matched_map_points = None  # current number of matched map points
        self.num_matched_map_points_in_last_pose_opt = None  # current number of matched map points (matched and found valid in last pose optimization)
        self.num_kf_ref_tracked_points = (
            None  # number of tracked points in k_ref (considering a minimum number of observations)
        )

        self.last_num_static_stereo_map_points = None
        self.total_num_static_stereo_map_points = 0

        self.last_reloc_frame_id = -float("inf")

        self.pose_is_ok = False
        self.mean_pose_opt_chi2_error = None
        self.predicted_pose = None
        self.velocity = None

        self.f_cur: Frame | None = None
        self.idxs_cur = None
        self.f_ref: Frame | None = None
        self.idxs_ref = None

        self.kf_ref = None  # reference keyframe (in general, different from last keyframe depending on the used approach)
        self.kf_last = None  # last keyframe
        self.kid_last_BA = -1  # last keyframe id when performed BA

        self.local_keyframes = []  # local keyframes
        self.local_points = []  # local points
        self.vo_points: list[MapPoint] = []  # visual odometry points

        self.tracking_history = TrackingHistory()

        self.timer_verbose = kTimerVerbose  # set this to True if you want to print timings
        self.timer_main_track = TimerFps("Track", is_verbose=self.timer_verbose)
        self.timer_pose_opt = TimerFps("Pose optimization", is_verbose=self.timer_verbose)
        self.timer_seach_frame_proj = TimerFps(
            "Search frame by proj", is_verbose=self.timer_verbose
        )
        self.timer_match = TimerFps("Match", is_verbose=self.timer_verbose)
        self.timer_pose_est = TimerFps("Ess mat pose estimation", is_verbose=self.timer_verbose)
        self.timer_frame = TimerFps("Frame", is_verbose=self.timer_verbose)
        self.timer_seach_map = TimerFps("Search map", is_verbose=self.timer_verbose)

        self.time_track = None

        self.init_history = True  # need to init history?
        self.poses = []  # history of poses
        self.pose_timestamps = []  # history of pose timestamps
        self.t0_est = None  # history of estimated translations
        self.t0_gt = None  # history of ground truth translations (if available)
        self.traj3d_est = []  # history of estimated translations centered w.r.t. first one
        self.traj3d_gt = (
            []
        )  # history of estimated ground truth translations centered w.r.t. first one

        self.cur_R = None  # current rotation Rwc w.r.t. world frame
        self.cur_t = None  # current translation twc w.r.t. world frame
        self.gt_x, self.gt_y, self.gt_z = None, None, None

        if kLogKFinfoToFile:
            self.kf_info_logger = Logging.setup_file_logger(
                "kf_info_logger",
                Parameters.kLogsFolder + "/kf_info.log",
                formatter=Logging.simple_log_formatter,
            )

    @property
    def feature_tracker(self):
        return self.slam.feature_tracker

    @property
    def map(self):
        return self.slam.map

    @property
    def camera(self):
        return self.slam.camera

    @property
    def sensor_type(self):
        return self.slam.sensor_type

    @property
    def local_mapping(self):
        return self.slam.local_mapping

    def reset(self):
        print("Tracking: reset...")

        self.initializer.reset()

        self.motion_model.reset()

        self.state = SlamState.NO_IMAGES_YET

        self.num_matched_kps = None  # current number of matched keypoints
        self.num_inliers = None  # current number of matched points
        self.num_matched_map_points = None  # current number of matched map points
        self.num_matched_map_points_in_last_pose_opt = None  # current number of matched map points (matched and found valid in current pose optimization)
        self.num_kf_ref_tracked_points = (
            None  # number of tracked points in k_ref (considering a minimum number of observations)
        )

        self.last_num_static_stereo_map_points = None
        self.total_num_static_stereo_map_points = 0

        self.pose_is_ok = False
        self.mean_pose_opt_chi2_error = None
        self.predicted_pose = None
        self.velocity = None

        self.f_cur = None
        self.idxs_cur = None
        self.f_ref = None
        self.idxs_ref = None

        self.kf_ref = None  # reference keyframe (in general, different from last keyframe depending on the used approach)
        self.kf_last = None  # last keyframe
        self.kid_last_BA = -1  # last keyframe id when performed BA

        self.local_keyframes.clear()
        self.local_points.clear()
        self.vo_points.clear()

        self.tracking_history.reset()

        self.init_history = True
        self.poses.clear()
        self.pose_timestamps.clear()
        self.t0_est = None  # history of estimated translations
        self.t0_gt = None  # history of ground truth translations (if available)
        self.traj3d_est.clear()
        self.traj3d_gt.clear()

        self.cur_R = None  # current rotation w.r.t. world frame
        self.cur_t = None  # current translation w.r.t. world frame
        self.gt_x, self.gt_y, self.gt_z = None, None, None

    def pose_optimization(self, f_cur, name=""):
        print("pose opt %s " % (name))
        pose_before = f_cur.pose()
        # f_cur pose optimization 1  (here we use f_cur pose as first guess and exploit the matched map points of f_ref )
        self.timer_pose_opt.start()
        if Parameters.kOptimizationFrontEndUseGtsam:
            pose_optimization_fun = optimizer_gtsam.pose_optimization  # [WIP] Not stable yet!
        else:
            pose_optimization_fun = optimizer_g2o.pose_optimization
        (
            self.mean_pose_opt_chi2_error,
            self.pose_is_ok,
            self.num_matched_map_points_in_last_pose_opt,
        ) = pose_optimization_fun(f_cur, verbose=False)
        self.timer_pose_opt.pause()
        print("     error^2: %f,  ok: %d" % (self.mean_pose_opt_chi2_error, int(self.pose_is_ok)))

        if not self.pose_is_ok:
            # if current pose optimization failed, reset f_cur pose
            f_cur.update_pose(
                pose_before
            )  # Note: This may be redundant since the pose is already reset in the calling functions. However, it is kept for consistency.

        return self.pose_is_ok, self.mean_pose_opt_chi2_error

    # track camera motion of f_cur w.r.t. f_ref
    def track_previous_frame(self, f_ref: Frame, f_cur: Frame):
        print(">>>> tracking previous frame ...")
        is_search_frame_by_projection_failure = False
        use_search_frame_by_projection = (
            self.motion_model.is_ok and kUseSearchFrameByProjection and kUseMotionModel
        )

        if use_search_frame_by_projection:
            # search frame by projection: match map points observed in f_ref with keypoints of f_cur
            print("search frame by projection")
            search_radius = Parameters.kMaxReprojectionDistanceFrame

            if self.sensor_type != SensorType.STEREO:  # [WIP]
                # if self.sensor_type == SensorType.RGBD:
                search_radius = Parameters.kMaxReprojectionDistanceFrameNonStereo

            f_cur.reset_points()
            self.timer_seach_frame_proj.start()
            idxs_ref, idxs_cur, num_found_map_pts = ProjectionMatcher.search_frame_by_projection(
                f_ref,
                f_cur,
                max_reproj_distance=search_radius,
                max_descriptor_distance=self.descriptor_distance_sigma,
                ratio_test=Parameters.kMatchRatioTestFrameByProjection,  # not used at the moment
                is_monocular=(self.sensor_type == SensorType.MONOCULAR),
            )
            self.timer_seach_frame_proj.refresh()
            self.num_matched_kps = len(idxs_cur)
            print("# matched map points in prev frame: %d " % self.num_matched_kps)

            # if not enough map point matches consider a larger search radius
            if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                f_cur.remove_frame_views(idxs_cur)
                f_cur.reset_points()
                idxs_ref, idxs_cur, num_found_map_pts = (
                    ProjectionMatcher.search_frame_by_projection(
                        f_ref,
                        f_cur,
                        max_reproj_distance=2 * search_radius,
                        max_descriptor_distance=self.descriptor_distance_sigma,
                        ratio_test=Parameters.kMatchRatioTestFrameByProjection,  # not used at the moment
                        is_monocular=(self.sensor_type == SensorType.MONOCULAR),
                    )
                )
                self.num_matched_kps = len(idxs_cur)
                Printer.orange(
                    "# matched map points in prev frame (wider search): %d " % self.num_matched_kps
                )

            if (
                (f_cur.is_blurry or f_ref.is_blurry)
                and self.num_matched_kps
                < Parameters.kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography
            ):
                # use homography RANSAC to find inliers and estimate the inter-frame transformation (assuming frames are very close in space)
                matching_is_ok, idxs_cur, idxs_ref, self.num_matched_kps, num_outliers = (
                    TrackingCore.find_homography_with_ransac(f_cur, f_ref, idxs_cur, idxs_ref)
                )
                if matching_is_ok:
                    Printer.orange(
                        f"# matched inter-frame map points with homography and RANSAC (blurry frames): {self.num_matched_kps}, percentage of inliers: {self.num_matched_kps/(num_outliers+self.num_matched_kps)*100:.2f}%"
                    )

            if kShowFeatureMatches and kShowFeatureMatchesPrevFrame:
                img_matches = draw_feature_matches(
                    f_ref.img,
                    f_cur.img,
                    f_ref.kps[idxs_ref],
                    f_cur.kps[idxs_cur],
                    f_ref.sizes[idxs_ref],
                    f_cur.sizes[idxs_cur],
                    horizontal=False,
                    show_kp_sizes=False,
                )
                cv2.imshow("tracking prev frame w/ projection - matches", img_matches)
                cv2.waitKey(1)

            if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchFrameByProjection:
                f_cur.remove_frame_views(idxs_cur)
                f_cur.reset_points()
                is_search_frame_by_projection_failure = True
                Printer.red(
                    "Not enough matches in search frame by projection: ", self.num_matched_kps
                )
            else:
                # # search frame by projection was successful => update descriptor distance sigma
                # if kUseDynamicDesDistanceTh:
                #     self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stats(
                #         f_ref, f_cur, idxs_ref, idxs_cur
                #     )

                # store tracking info (for possible reuse)
                self.idxs_ref = idxs_ref
                self.idxs_cur = idxs_cur

                pose_before_pos_opt = f_cur.pose()

                # f_cur pose optimization 1:
                # here, we use f_cur pose as first guess and exploit the matched map point of f_ref
                self.pose_optimization(f_cur, "proj-frame-frame")
                # update matched map points; discard outliers detected in last pose optimization
                self.num_matched_map_points = f_cur.clean_outlier_map_points()
                # print('     # num_matched_map_points_in_last_pose_opt: %d' % (self.num_matched_map_points_in_last_pose_opt) )
                # print('     # matched points: %d' % (self.num_matched_map_points) )

                if (
                    not self.pose_is_ok
                    or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame
                ):
                    # if not self.pose_is_ok or self.num_matched_map_points_in_last_pose_opt < kNumMinInliersPoseOptimizationTrackFrame:
                    Printer.red(
                        f"failure in tracking previous frame, # matched map points: {self.num_matched_map_points}, # matched map points in last pose opt: {self.num_matched_map_points_in_last_pose_opt}"
                    )
                    self.pose_is_ok = False
                    f_cur.update_pose(pose_before_pos_opt)
                    is_search_frame_by_projection_failure = True
                else:
                    # tracking was successful with enough inliers => update descriptor distance sigma
                    if kUseDynamicDesDistanceTh:
                        self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stats(
                            f_ref, f_cur, idxs_ref, idxs_cur
                        )

        if not use_search_frame_by_projection or is_search_frame_by_projection_failure:
            Printer.orange("using frame-frame matching")
            self.track_reference_frame(f_ref, f_cur, "match-frame-frame")

    # track camera motion of f_cur w.r.t. f_ref
    # estimate motion by matching keypoint descriptors
    def track_reference_frame(self, f_ref: Frame, f_cur: Frame, name=""):
        frame_str = "keyframe" if f_ref.is_keyframe else "frame"
        print(f">>>> tracking reference {frame_str} {f_ref.id} ...")
        if f_ref is None:
            self.pose_is_ok = False
            Printer.red(f"[track_reference_frame]: f_ref is None")
            return

        # find keypoint matches between f_cur and kf_ref
        print("matching keypoints with ", FeatureTrackerShared.feature_matcher.matcher_type.name)
        self.timer_match.start()

        # matching_result = match_frames(f_cur, f_ref) # original code that used all the reference keypoints

        # match only the reference keypoints in f_ref that correspond to map points and are not bad
        idxs_ref_map_points = np.asarray(
            f_ref.get_matched_good_points_idxs(), dtype=int
        )  # numpy array conversion for c++

        des_ref = f_ref.des[idxs_ref_map_points]
        kps_ref = f_ref.kps[idxs_ref_map_points]
        des_cur = f_cur.des
        kps_cur = f_cur.kps
        matching_result = FeatureTrackerShared.feature_matcher.match(
            f_cur.img, f_ref.img, des_cur, des_ref, kps1=kps_cur, kps2=kps_ref
        )
        matching_result.idxs2 = idxs_ref_map_points[
            matching_result.idxs2
        ]  # map back to the original reference keypoints

        self.timer_match.refresh()
        idxs_cur = (
            np.asarray(matching_result.idxs1, dtype=int)
            if matching_result.idxs1 is not None
            else np.array([], dtype=int)
        )
        idxs_ref = (
            np.asarray(matching_result.idxs2, dtype=int)
            if matching_result.idxs2 is not None
            else np.array([], dtype=int)
        )
        self.num_matched_kps = len(idxs_cur)
        if len(idxs_cur) == 0 or len(idxs_ref) == 0:
            print(
                f"[track_reference_frame]: # keypoint matches: idxs_cur: {len(idxs_cur)}, idxs_ref: {len(idxs_ref)}"
            )
        if FeatureTrackerShared.oriented_features and len(idxs_cur) > 0 and len(idxs_ref) > 0:
            valid_match_idxs = RotationHistogram.filter_matches_with_histogram_orientation(
                idxs_cur, idxs_ref, f_cur.angles, f_ref.angles
            )
            des_distances = FeatureTrackerShared.descriptor_distances(
                f_cur.des[idxs_cur], f_ref.des[idxs_ref]
            )
            valid_match_idxs = np.intersect1d(
                valid_match_idxs, np.where(des_distances <= 0.5 * self.descriptor_distance_sigma)[0]
            )
            if len(valid_match_idxs) > 0:
                idxs_cur = idxs_cur[valid_match_idxs]
                idxs_ref = idxs_ref[valid_match_idxs]
            else:
                idxs_cur = np.array([], dtype=int)
                idxs_ref = np.array([], dtype=int)
            self.num_matched_kps = len(idxs_cur)

        if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchReferenceFrame:
            self.pose_is_ok = False
            Printer.orange(
                "Not enough matches in search frame by projection: ", self.num_matched_kps
            )
            return

        # use homography RANSAC to find inliers and estimate the inter-frame transformation (assuming frames are very close in space)
        # only if both frames are not keyframes (so they are expected to be close in space) and at least one is blurry
        if (
            (f_cur.is_blurry or f_ref.is_blurry)
            and (not f_cur.is_keyframe and not f_ref.is_keyframe)
            and self.num_matched_kps
            < Parameters.kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography
        ):
            matching_is_ok, idxs_cur, idxs_ref, self.num_matched_kps, num_outliers = (
                TrackingCore.find_homography_with_ransac(f_cur, f_ref, idxs_cur, idxs_ref)
            )
            if matching_is_ok:
                Printer.orange(
                    f"# matched inter-frame map points with homography and RANSAC (blurry frames): {self.num_matched_kps}, percentage of inliers: {self.num_matched_kps/(num_outliers+self.num_matched_kps)*100:.2f}%"
                )

        print("# keypoints matched: %d " % self.num_matched_kps)

        if self.num_matched_kps < Parameters.kMinNumMatchedFeaturesSearchReferenceFrame:
            self.pose_is_ok = False
            Printer.orange(
                "Not enough matches in search frame by projection: ", self.num_matched_kps
            )
            return

        if Parameters.kUseEssentialMatrixFitting:
            self.timer_pose_est.start()
            # estimate camera orientation and inlier matches by fitting and essential matrix (see the limitations above)
            idxs_ref, idxs_cur, self.num_inliers = TrackingCore.estimate_pose_by_fitting_ess_mat(
                f_ref, f_cur, idxs_ref, idxs_cur
            )
            self.timer_pose_est.refresh()
            self.num_matched_kps = len(idxs_cur)

        if kUseDynamicDesDistanceTh:
            self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stats(
                f_ref, f_cur, idxs_ref, idxs_cur
            )

        # propagate map point matches from kf_ref to f_cur  (do not override idxs_ref, idxs_cur)
        max_descriptor_distance = (
            self.descriptor_distance_sigma
            if not f_ref.is_keyframe
            else 0.5 * self.descriptor_distance_sigma
        )

        num_found_map_pts_inter_frame, idx_ref_prop, idx_cur_prop = (
            TrackingCore.propagate_map_point_matches(
                f_ref, f_cur, idxs_ref, idxs_cur, max_descriptor_distance=max_descriptor_distance
            )
        )
        print("# matched map points in reference frame: %d " % num_found_map_pts_inter_frame)

        if kShowFeatureMatches and kShowFeatureMatchesRefFrame:
            img_matches = draw_feature_matches(
                f_ref.img,
                f_cur.img,
                f_ref.kps[idx_ref_prop],
                f_cur.kps[idx_cur_prop],
                f_ref.sizes[idx_ref_prop],
                f_cur.sizes[idx_cur_prop],
                horizontal=False,
                show_kp_sizes=False,
            )
            cv2.imshow("tracking ref frame w/o projection - matches", img_matches)
            cv2.waitKey(1)

        # store tracking info (for possible reuse)
        self.idxs_ref = idxs_ref
        self.idxs_cur = idxs_cur

        pose_before_pos_opt = f_cur.pose()

        # f_cur pose optimization using last matches with kf_ref:
        # here, we use first guess of f_cur pose and propated map point matches from f_ref (matched keypoints)
        self.pose_optimization(f_cur, name)
        # update matched map points; discard outliers detected in last pose optimization
        self.num_matched_map_points = f_cur.clean_outlier_map_points()
        # print('     # num_matched_map_points_in_last_pose_opt: %d' % (self.num_matched_map_points_in_last_pose_opt) )
        # print('     # matched points: %d' % (self.num_matched_map_points) )

        if (
            not self.pose_is_ok
            or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackFrame
        ):
            # if not self.pose_is_ok or self.num_matched_map_points_in_last_pose_opt < kNumMinInliersPoseOptimizationTrackFrame:
            f_cur.remove_frame_views(idxs_cur)
            f_cur.reset_points()
            Printer.red(
                f"failure in tracking reference {f_ref.id}, # matched map points: {self.num_matched_map_points}"
            )
            self.pose_is_ok = False
            f_cur.update_pose(pose_before_pos_opt)

    # track camera motion of f_cur w.r.t. given keyframe
    # estimate motion by matching keypoint descriptors
    def track_keyframe(self, keyframe: Frame, f_cur: Frame, name="match-frame-keyframe"):
        f_cur.update_pose(self.f_ref.pose())  # start pose optimization from last frame pose
        self.track_reference_frame(keyframe, f_cur, name)

    def update_local_map(self):
        self.f_cur.clean_bad_map_points()
        # self.local_points = self.map.local_map.get_points()
        self.kf_ref, self.local_keyframes, self.local_points = (
            self.map.local_map.get_frame_covisibles(self.f_cur)
        )
        if self.kf_ref is not None:
            self.f_cur.kf_ref = self.kf_ref

        if False:
            # one-time diagnostic
            if any(not isinstance(p, MapPoint) for p in self.local_points):
                bad = [type(p) for p in self.local_points if not isinstance(p, MapPoint)]
                print(
                    f"[update_local_map] local_points contains non-MapPoint: {bad[:5]}{'...' if len(bad)>5 else ''}"
                )

    # track camera motion of f_cur w.r.t. the built local map
    # find matches between {local map points} (points in the built local map) and {unmatched keypoints of f_cur}
    def track_local_map(self, f_cur: Frame):
        if self.map.local_map.is_empty():
            return
        print(">>>> tracking local map...")
        self.timer_seach_map.start()

        self.update_local_map()

        if self.local_points is None or len(self.local_points) == 0:
            self.pose_is_ok = False
            return

        self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMap
        if self.sensor_type == SensorType.RGBD:
            self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMapRgbd
        if f_cur.id < self.last_reloc_frame_id + 2:
            self.reproj_err_frame_map_sigma = Parameters.kMaxReprojectionDistanceMapReloc

        # use the updated local map to search for matches between {local map points} and {unmatched keypoints of f_cur}
        num_found_map_pts, matched_points_frame_idxs = ProjectionMatcher.search_map_by_projection(
            self.local_points,
            f_cur,
            max_reproj_distance=self.reproj_err_frame_map_sigma,
            max_descriptor_distance=self.descriptor_distance_sigma,
            ratio_test=Parameters.kMatchRatioTestMap,
            far_points_threshold=self.far_points_threshold,
        )
        self.timer_seach_map.refresh()
        # print('reproj_err_sigma: ', reproj_err_frame_map_sigma, ' used: ', self.reproj_err_frame_map_sigma)
        print(
            f"# matched map points in local map: {num_found_map_pts}, perc%: {100*num_found_map_pts/len(self.local_points):.2f}"
        )
        # print("# local map points ", self.map.local_map.num_points())

        if kShowFeatureMatches and kShowFeatureMatchesLocalMap:
            img_matched_trails = f_cur.draw_feature_trails(
                f_cur.img.copy(), matched_points_frame_idxs, trail_max_length=3
            )
            cv2.imshow("tracking local map - matched trails", img_matched_trails)
            cv2.waitKey(1)

        pose_before_pos_opt = f_cur.pose()

        # f_cur pose optimization 2 with all the matched local map points
        self.pose_optimization(f_cur, "proj-map-frame")

        # here we reset outliers only in the case of STEREO; in other cases,
        # we let them reach the keyframe generation and then bundle adjustment will possible decide if remove them or not;
        # only after keyframe generation the outliers are cleaned!
        self.num_matched_map_points = f_cur.update_map_points_statistics(self.sensor_type)

        # print('     # num_matched_points: %d' % (self.num_matched_map_points) )
        if (
            not self.pose_is_ok
            or self.num_matched_map_points < kNumMinInliersPoseOptimizationTrackLocalMap
        ):
            Printer.red(
                f"failure in tracking local map, # matched map points: {self.num_matched_map_points}"
            )
            self.pose_is_ok = False
            f_cur.update_pose(pose_before_pos_opt)

        # if kUseDynamicDesDistanceTh:
        #    self.reproj_err_frame_map_sigma = self.dyn_config.update_reproj_err_map_stats(reproj_err_frame_map_sigma)

    # store frame history in order to retrieve the complete camera trajectory
    def update_tracking_history(self):
        if self.state == SlamState.OK:
            isometry3d_Tcr = (
                self.f_cur.isometry3d() * self.f_cur.kf_ref.isometry3d().inverse()
            )  # pose of current frame w.r.t. current reference keyframe kf_ref
            self.tracking_history.relative_frame_poses.append(isometry3d_Tcr)
            self.tracking_history.kf_references.append(self.kf_ref)
            self.tracking_history.timestamps.append(self.f_cur.timestamp)
            self.tracking_history.ids.append(self.f_cur.id)
        else:
            if len(self.tracking_history.relative_frame_poses) > 0:
                self.tracking_history.relative_frame_poses.append(
                    self.tracking_history.relative_frame_poses[-1]
                )
                self.tracking_history.kf_references.append(self.tracking_history.kf_references[-1])
                self.tracking_history.timestamps.append(self.tracking_history.timestamps[-1])
                self.tracking_history.ids.append(self.tracking_history.ids[-1])
        self.tracking_history.slam_states.append(self.state)

    def clean_vo_points(self):
        for p in self.vo_points:
            p.set_bad()
            p.delete()
        self.vo_points.clear()

    def need_new_keyframe(self, f_cur: Frame):

        # If Local Mapping is freezed by a Loop Closure do not insert keyframes
        if self.local_mapping.is_stopped() or self.local_mapping.is_stop_requested():
            return False

        num_keyframes = self.map.num_keyframes()

        # Do not insert keyframes if not enough frames have passed from last relocalisation
        if (
            f_cur.id < self.last_reloc_frame_id + self.max_frames_between_kfs_after_reloc
            and num_keyframes > self.max_frames_between_kfs
        ):
            print(
                f"Not inserting keyframe {f_cur.id} because it is too close to the last reloc frame {self.last_reloc_frame_id}, max_frames_between_kfs: {self.max_frames_between_kfs}"
            )
            return False

        nMinObs = kNumMinObsForKeyFrameDefault
        if num_keyframes <= 2:
            nMinObs = 2  # if just two keyframes then we can have just two observations
        num_kf_ref_tracked_points = self.kf_ref.num_tracked_points(
            nMinObs
        )  # number of tracked points in k_ref
        num_f_cur_tracked_points = (
            f_cur.num_matched_inlier_map_points()
        )  # number of inliers map points in f_cur
        # num_f_cur_tracked_points = self.num_matched_map_points if self.num_matched_map_points is not None else 0 # updated in the last self.track_local_map()
        tracking_info_message = f"F({f_cur.id}) #matched points: {num_f_cur_tracked_points}, KF({self.kf_ref.id}) #matched points: {num_kf_ref_tracked_points}"
        Printer.green(tracking_info_message)

        if kLogKFinfoToFile:
            self.kf_info_logger.info(tracking_info_message)

        self.num_kf_ref_tracked_points = num_kf_ref_tracked_points

        is_local_mapping_idle = self.local_mapping.is_idle()
        local_mapping_queue_size = self.local_mapping.queue_size()
        print(
            "is_local_mapping_idle: ",
            is_local_mapping_idle,
            ", local_mapping_queue_size: ",
            local_mapping_queue_size,
        )

        # Check how many "close" points are being tracked and how many could be potentially created.
        num_non_tracked_close = 0
        num_tracked_close = 0
        tracked_mask = (
            None  # if needed, create a mask for tracked points (not None and not an outlier)
        )
        is_need_to_insert_close = False
        if self.sensor_type != SensorType.MONOCULAR:
            num_tracked_close, num_non_tracked_close, tracked_mask = (
                TrackingCore.count_tracked_and_non_tracked_close_points(f_cur, self.sensor_type)
            )

            is_need_to_insert_close = (
                num_tracked_close < Parameters.kNumMinTrackedClosePointsForNewKfNonMonocular
            ) and (
                num_non_tracked_close > Parameters.kNumMaxNonTrackedClosePointsForNewKfNonMonocular
            )

        #  Thresholds
        thRefRatio = Parameters.kThNewKfRefRatioStereo
        if num_keyframes < 2:
            thRefRatio = 0.4
        if self.sensor_type == SensorType.MONOCULAR:
            thRefRatio = Parameters.kThNewKfRefRatioMonocular

        if not Parameters.kLocalMappingOnSeparateThread:
            if self.sensor_type != SensorType.MONOCULAR:
                # NOTE: in single-threaded mode, is_local_mapping_idle is always True => cond1b always Trye => too many keyframes
                self.min_frames_between_kfs = 3

        # condition 1a: more than "max_frames_between_kfs" have passed from last keyframe insertion
        cond1a = f_cur.id >= (self.kf_last.id + self.max_frames_between_kfs)

        # condition 1b: more than "min_frames_between_kfs" have passed and local mapping is idle
        cond1b = (
            f_cur.id >= (self.kf_last.id + self.min_frames_between_kfs)
        ) and is_local_mapping_idle

        # condition 1c: tracking is weak 1 with non-monocular sensors
        cond1c = (self.sensor_type != SensorType.MONOCULAR) and (
            num_f_cur_tracked_points
            < num_kf_ref_tracked_points * Parameters.kThNewKfRefRatioNonMonocular
            or is_need_to_insert_close
        )

        # condition 1d: tracking image coverage is weak
        # we divide the image in 3x2 cells and check that each cell is filled by at least one point (the partition is assumed to be gross in order not to generate too many KFs)
        cond1d = False
        if Parameters.kUseFeatureCoverageControlForNewKf:
            image_grid = ImageGrid(self.camera.width, self.camera.height, num_div_x=3, num_div_y=2)
            if tracked_mask is None:
                tracked_mask = f_cur.get_tracked_mask()
            image_grid.add_points(f_cur.kps[tracked_mask])
            num_uncovered_cells = image_grid.num_cells_uncovered(num_min_points=1)
            cond1d = num_uncovered_cells > 1
            if True:
                cv2.namedWindow("grid_img", cv2.WINDOW_NORMAL)
                cv2.imshow("grid_img", image_grid.get_grid_img())
                cv2.waitKey(1)

        # condition 2: few tracked features compared to reference keyframe
        cond2 = (
            num_f_cur_tracked_points < num_kf_ref_tracked_points * thRefRatio
            or is_need_to_insert_close
        ) and (num_f_cur_tracked_points > Parameters.kNumMinPointsForNewKf)

        # condition 3: distance to closest fov center is too big
        cond3 = False
        if self.use_fov_centers_based_kf_generation:
            if num_f_cur_tracked_points > Parameters.kNumMinPointsForNewKf:
                # compute distance to closest fov center
                close_kfs = self.local_keyframes
                if not self.kf_last in close_kfs:
                    close_kfs.append(self.kf_last)
                if len(close_kfs) > 0:
                    close_fov_centers_w = np.array(
                        [
                            kf.fov_center_w.flatten()
                            for kf in close_kfs
                            if kf.fov_center_w is not None
                        ]
                    )
                    if close_fov_centers_w.shape[0] > 0:
                        dists = np.linalg.norm(
                            close_fov_centers_w - f_cur.fov_center_w.flatten(), axis=1
                        )
                        min_dist = np.min(dists)
                        cond3 = min_dist > self.max_fov_centers_distance

        # print(f'KF conditions: 1a: {cond1a}, 1b: {cond1b}, 1c: {cond1c}, 1d: {cond1d}, 2: {cond2}, 3: {cond3}')
        condition_checks = ((cond1a or cond1b or cond1c or cond1d) and cond2) or cond3
        if condition_checks:
            print(
                f"KF conditions: ( (1a:{cond1a} or 1b:{cond1b} or 1c:{cond1c} or 1d:{cond1d}) and 2: {cond2} ) or 3: {cond3}"
            )

        if condition_checks:
            if is_local_mapping_idle:
                return True
            else:
                if Parameters.kUseInterruptLocalMapping or USE_CPP:
                    self.local_mapping.interrupt_optimization()
                if self.sensor_type == SensorType.MONOCULAR:
                    if local_mapping_queue_size <= 3:
                        return True
                    else:
                        return False
                else:
                    return False
        else:
            return False

    def create_new_keyframe(self, f_cur: Frame, img, img_right=None, depth=None):
        if not self.local_mapping.set_do_not_stop(True):
            return

        kf_new = KeyFrame(f_cur, img, img_right, depth)
        self.kf_last = kf_new
        self.kf_ref = kf_new
        f_cur.kf_ref = kf_new

        Printer.green(
            f"Adding new KF with id {kf_new.id}, img shape: {img.shape if img is not None else None}, img_right shape: {img_right.shape if img_right is not None else None}, depth shape: {depth.shape if depth is not None else None}"
        )
        if kLogKFinfoToFile:
            self.kf_info_logger.info("adding new KF with frame id % d: " % (f_cur.id))

        self.map.add_keyframe(
            kf_new
        )  # add kf_cur to map (moved from local_mapping.py to this point)
        # NOTE: This is done here since a new keyframe-id (kid) is assigned when adding to map

        if self.sensor_type != SensorType.MONOCULAR:
            self.create_and_add_stereo_map_points_on_new_kf(f_cur, kf_new, img)

        self.local_mapping.push_keyframe(kf_new, img, img_right, depth)

        self.local_mapping.set_do_not_stop(False)

    def relocalize(self, f_cur: Frame, img):
        Printer.green(f"Relocalizing frame id: {f_cur.id}...")
        if self.slam.loop_closing is not None:
            return self.slam.loop_closing.relocalize(f_cur, img)
        else:
            Printer.yellow(
                f"[Tracking]: WARNING you did not set any loop closing / relocalize method!"
            )
            return False

    def create_vo_points_on_last_frame(self):
        # Validate required objects exist
        if (
            self.sensor_type == SensorType.MONOCULAR
            or self.kf_last is None
            or self.f_ref is None
            or self.kf_last.id == self.f_ref.id
            or self.f_ref.depths is None
        ):
            return

        print("Creating VO points...")

        # Create VO points using the new frame method
        created_points = TrackingCore.create_vo_points(
            self.f_ref, max_num_points=Parameters.kMaxNumVisualOdometryPoints
        )

        # Add to VO points list
        self.vo_points.extend(created_points)

        num_added_points = len(created_points)
        print(
            f"Added #new VO points: {num_added_points}, current #VO points: {len(self.vo_points)}"
        )

    # kf is a newly created keyframe starting from frame f
    def create_and_add_stereo_map_points_on_new_kf(self, f: Frame, kf: KeyFrame, img):
        if self.sensor_type != SensorType.MONOCULAR and kf.depths is not None:
            num_added_points = TrackingCore.create_and_add_stereo_map_points_on_new_kf(
                f, kf, self.map, img
            )
            self.last_num_static_stereo_map_points = num_added_points
            self.total_num_static_stereo_map_points += num_added_points

    # Since we do not have real-time performances, we can slow-down things and make tracking wait till local mapping gets idle
    # N.B.: this function must be called outside 'with self.map.update_lock' blocks,
    #       since both self.track() and the local-mapping optimization use the RLock 'map.update_lock'
    #       => they cannot wait for each other once map.update_lock is locked (deadlock)
    def wait_for_local_mapping(
        self, timeout=Parameters.kWaitForLocalMappingTimeout, check_on_exit=False
    ):
        if not Parameters.kLocalMappingOnSeparateThread:
            return

        if Parameters.kTrackingWaitForLocalMappingToGetIdle:
            # If there are still keyframes in the queue, wait for local mapping to get idle
            if not self.local_mapping.is_idle():
                while self.local_mapping.queue_size() > 0 and not self.local_mapping.is_idle():
                    print(">>>> waiting for local mapping...")
                    self.local_mapping.wait_idle(
                        print=print, timeout=None
                    )  # NOTE: no timeout here, just wait until local mapping is idle!
        else:
            # if we are close to bad tracking give local mapping more time
            if (
                self.num_matched_map_points
                and self.num_matched_map_points
                < kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle
            ):
                # if self.local_mapping.queue_size() > 0: # less restrictive
                if not self.local_mapping.is_idle() or self.local_mapping.queue_size() > 0:
                    Printer.orange(
                        ">>>> close to bad tracking: forcing waiting for local mapping..."
                    )
                    self.local_mapping.wait_idle(print=print, timeout=timeout)

                if Parameters.kUseParallelProcessLBA and not self.local_mapping.is_idle():
                    timeout = Parameters.kParallelLBAWaitIdleTimeout
                    Printer.orange(
                        f">>>> close to bad tracking: forcing waiting for local mapping (parallel LBA, KF queue size: {self.local_mapping.queue_size()})..."
                    )
                    self.local_mapping.wait_idle(print=print, timeout=timeout)

            local_mapping_queue_size = self.local_mapping.queue_size()
            local_mapping_is_idle = self.local_mapping.is_idle()
            # if local_mapping_queue_size > 0 or not local_mapping_is_idle:  # more restrictive
            if local_mapping_queue_size > 0:  # before
                Printer.orange(
                    f">>>> waiting for local mapping idle (queue_size={local_mapping_queue_size}, idle={local_mapping_is_idle})..."
                )
                self.local_mapping.wait_idle(print=print, timeout=timeout)

        # check again for debug
        if check_on_exit:
            is_local_mapping_idle = self.local_mapping.is_idle()
            local_mapping_queue_size = self.local_mapping.queue_size()
            Printer.green(
                "wait_for_local_mapping - is_local_mapping_idle: ",
                is_local_mapping_idle,
                ", local_mapping_queue_size: ",
                local_mapping_queue_size,
            )

    # Here, pose estimates are saved online: At each frame, the current pose estimate is saved.
    # Note that in other frameworks, pose estimates may be saved at the end of the dataset playback
    # so that each pose estimate is refined multiple times by LBA and BA over the multiple window optimizations that cover it.
    def update_history(self):
        f_cur = self.map.get_frame(-1)
        f_cur_pose = f_cur.pose()
        self.cur_R = f_cur_pose[:3, :3].T
        self.cur_t = np.dot(-self.cur_R, f_cur_pose[:3, 3])
        if self.init_history is True:
            self.t0_est = np.array(
                [self.cur_t[0], self.cur_t[1], self.cur_t[2]]
            )  # starting translation
            if self.gt_x is not None:
                self.t0_gt = np.array([self.gt_x, self.gt_y, self.gt_z])  # starting translation
        if self.t0_est is not None:
            p = [
                self.cur_t[0] - self.t0_est[0],
                self.cur_t[1] - self.t0_est[1],
                self.cur_t[2] - self.t0_est[2],
            ]  # the estimated traj starts at 0
            self.traj3d_est.append(p)
            if self.t0_gt is not None:
                self.traj3d_gt.append(
                    [
                        self.gt_x - self.t0_gt[0],
                        self.gt_y - self.t0_gt[1],
                        self.gt_z - self.t0_gt[2],
                    ]
                )
            self.poses.append(poseRt(self.cur_R, p))
            self.pose_timestamps.append(f_cur.timestamp)

    # @ main track method @
    def track(self, img, img_right, depth, img_id, timestamp=None):
        Printer.cyan(
            f"@tracking {self.sensor_type.name}, img id: {img_id}, frame id: {Frame.next_id()}, state: {self.state.name}"
        )
        time_start = time.time()

        # check image size is coherent with camera params
        print(f"img.shape: {img.shape}, camera: {self.camera.height}x{self.camera.width}")
        if depth is not None:
            print("depth.shape: ", depth.shape)
        if img_right is not None:
            print("img_right.shape: ", img_right.shape)
        assert img.shape[0:2] == (self.camera.height, self.camera.width)
        if timestamp is not None:
            print("timestamp: ", timestamp)

        self.timer_main_track.start()

        # at initialization time is better to use more extracted features
        if self.state != SlamState.OK:
            FeatureTrackerShared.feature_tracker.set_double_num_features()
        else:
            FeatureTrackerShared.feature_tracker.set_normal_num_features()

        # preprocessing
        f_cur_is_blurry = False
        if Parameters.kUseMotionBlurDection:
            f_cur_is_blurry, f_cur_laplacian_var = detect_blur_laplacian(
                img, threshold=Parameters.kMotionBlurDetectionLalacianVarianceThreshold
            )
            if f_cur_is_blurry:
                Printer.purple(f"img {img_id} is blurry, laplacian_var: {f_cur_laplacian_var}")
                # img = cv2.GaussianBlur(img, (5, 5), 0) # NOTE: just testing prefiltering to push feature tracking on higher levels of the pyramid

        # build current frame
        self.timer_frame.start()
        f_cur = Frame(
            self.camera, img, img_right=img_right, depth=depth, timestamp=timestamp, img_id=img_id
        )
        self.f_cur = f_cur
        if f_cur_is_blurry:
            f_cur.is_blurry = True
            f_cur.laplacian_var = f_cur_laplacian_var

        # print("frame: ", f_cur.id)
        self.timer_frame.refresh()

        # reset indexes of matches
        self.idxs_ref = []
        self.idxs_cur = []

        if self.state == SlamState.NO_IMAGES_YET:
            # push first frame in the inizializer
            self.initializer.init(f_cur, img, img_right, depth)
            self.state = SlamState.NOT_INITIALIZED
            return  # EXIT (jump to second frame)

        if self.state == SlamState.NOT_INITIALIZED:
            # try to inizialize
            initializer_output, intializer_is_ok = self.initializer.initialize(
                f_cur, img, img_right, depth
            )
            if intializer_is_ok:
                kf_ref = initializer_output.kf_ref
                kf_cur = initializer_output.kf_cur

                # add the two initialized frames in the map
                self.map.keyframe_origins.add(kf_ref)
                self.map.add_frame(kf_ref)  # add first frame in map and update its frame id
                self.map.add_frame(kf_cur)  # add second frame in map and update its frame id
                # add the two initialized frames as keyframes in the map
                self.map.add_keyframe(kf_ref)  # add first keyframe in map and update its kid
                self.map.add_keyframe(kf_cur)  # add second keyframe in map and update its kid
                kf_ref.init_observations()
                kf_cur.init_observations()

                # add points in map
                new_pts_count, _, _ = self.map.add_points(
                    initializer_output.pts,
                    None,
                    kf_cur,
                    kf_ref,
                    initializer_output.idxs_cur,
                    initializer_output.idxs_ref,
                    img,
                    do_check=False,
                )
                # new_pts_count = self.map.add_stereo_points(initializer_output.pts, None, f_cur, kf_cur, np.arange(len(initializer_output.pts), dtype=np.int), img)

                Printer.green(
                    f"map: initialized with kfs {kf_ref.id}, {kf_cur.id} and {new_pts_count} new map points"
                )

                # update covisibility graph connections
                kf_ref.update_connections()
                kf_cur.update_connections()

                # update tracking info
                self.f_cur = kf_cur
                self.f_cur.kf_ref = kf_ref
                self.kf_ref = kf_cur  # set reference keyframe
                self.kf_last = kf_cur  # set last added keyframe
                self.map.local_map.update(self.kf_ref)
                self.state = SlamState.OK

                self.update_tracking_history()
                self.motion_model.update_pose(
                    kf_cur.timestamp, kf_cur.position(), kf_cur.quaternion()
                )
                self.motion_model.is_ok = False  # after initialization you cannot use motion model for next frame pose prediction (time ids of initialized poses may not be consecutive)

                self.initializer.reset()

                if kUseDynamicDesDistanceTh:
                    self.descriptor_distance_sigma = self.dyn_config.update_descriptor_stats(
                        kf_ref, kf_cur, initializer_output.idxs_ref, initializer_output.idxs_cur
                    )

                # push new keyframes to loop closing to create and store global descriptors into loop detection database
                if self.slam.loop_closing is not None:
                    print(f"pushing new keyframe {kf_ref.id} to loop closing...")
                    self.slam.loop_closing.add_keyframe(kf_ref, kf_ref.img)

                    print(f"pushing new keyframe {kf_cur.id} to loop closing...")
                    self.slam.loop_closing.add_keyframe(kf_cur, kf_cur.img)

                if self.slam.volumetric_integrator is not None:
                    print(f"pushing new keyframe {kf_ref.id} to volumetric integrator...")
                    self.slam.volumetric_integrator.add_keyframe(
                        kf_ref, kf_ref.img, kf_ref.img_right, kf_ref.depth_img
                    )

                    print(f"pushing new keyframe {kf_cur.id} to volumetric integrator...")
                    self.slam.volumetric_integrator.add_keyframe(
                        kf_cur, kf_cur.img, kf_cur.img_right, kf_cur.depth_img
                    )

                if self.slam.semantic_mapping is not None:
                    print(f"pushing new keyframe {kf_ref.id} to semantic mapping...")
                    self.slam.semantic_mapping.push_keyframe(
                        kf_ref, kf_ref.img, kf_ref.img_right, kf_ref.depth_img
                    )

                    print(f"pushing new keyframe {kf_cur.id} to semantic mapping...")
                    self.slam.semantic_mapping.push_keyframe(
                        kf_cur, kf_cur.img, kf_cur.img_right, kf_cur.depth_img
                    )

            return  # EXIT (jump to next frame)

        # get previous frame in map as reference
        f_ref = self.map.get_frame(-1)
        # f_ref_2 = self.map.get_frame(-2)
        self.f_ref = f_ref

        assert f_ref.img_id == f_cur.img_id - 1

        # add current frame f_cur to map
        self.map.add_frame(f_cur)
        self.f_cur.kf_ref = self.kf_ref

        # reset pose state flag
        self.pose_is_ok = False

        # HACK: Since loop closing may be not fast enough (when adjusting the loop) in python (and tracking is not in real-time) => give loop closing more time to process stuff
        if self.slam.loop_closing is not None:
            if self.slam.loop_closing.is_closing():
                self.local_mapping.wait_idle(print=print)
                self.slam.loop_closing.wait_if_closing()
                print("...loop closing done")

        # HACK: Since local mapping may be not fast enough in python (and tracking is not in real-time) => give local mapping more time to process stuff
        self.wait_for_local_mapping()  # N.B.: this must be outside the `with self.map.update_lock:` block

        with self.map.update_lock:

            # DEBUG:
            # if img_id == 50:
            # # if self.map.num_keyframes() > 50: # for ibow
            #     self.state = SlamState.LOST # force to lost state for testing relocalization

            if self.state == SlamState.OK:
                # SLAM is OK
                # check for map point replacements in previous frame f_ref (some points might have been replaced by local mapping during point fusion)
                self.f_ref.check_replaced_map_points()

                # set intial guess for current pose optimization
                if kUseMotionModel and self.motion_model.is_ok:
                    print("using motion model for next pose prediction")
                    # update f_ref pose according to its reference keyframe (the pose of the reference keyframe could have been updated by local mapping)
                    self.f_ref.update_pose(
                        self.tracking_history.relative_frame_poses[-1]
                        * self.f_ref.kf_ref.isometry3d()
                    )
                    if Parameters.kUseVisualOdometryPoints:
                        self.create_vo_points_on_last_frame()

                    # udpdate (velocity) old motion model                                              # c1=ref_ref, c2=ref, c3=cur;  c=cur, r=ref
                    # self.velocity = np.dot(f_ref.pose, inv_T(self.f_ref_2.pose))                     # Tc2c1 = Tc2w * Twc1   (predicted Tcr)
                    # self.predicted_pose = g2o.Isometry3d(np.dot(self.velocity, f_ref.pose))          # Tc3w = Tc2c1 * Tc2w   (predicted Tcw)

                    # predict pose by using motion model
                    self.predicted_pose, _ = self.motion_model.predict_pose(
                        timestamp, self.f_ref.position(), self.f_ref.orientation()
                    )
                    f_cur.update_pose(self.predicted_pose.matrix().copy())
                else:
                    print("setting f_cur.pose <-- f_ref.pose")
                    # use reference frame pose as initial guess
                    f_cur.update_pose(f_ref.pose())

                if not self.motion_model.is_ok or f_cur.id < self.last_reloc_frame_id + 2:
                    assert self.kf_ref is not None
                    # track the camera motion from kf_ref to f_cur
                    self.track_keyframe(self.kf_ref, f_cur)
                else:
                    # track camera motion from f_ref to f_cur
                    self.track_previous_frame(f_ref, f_cur)

                    if not self.pose_is_ok:
                        # if previous track didn't go well then track the camera motion from kf_ref to f_cur
                        self.track_keyframe(self.kf_ref, f_cur)

            else:
                # SLAM is NOT OK
                if self.state != SlamState.INIT_RELOCALIZE:
                    self.state = SlamState.RELOCALIZE
                if self.relocalize(f_cur, img):
                    # Always set last_reloc_frame_id after successful relocalization
                    self.last_reloc_frame_id = f_cur.id
                    if self.state != SlamState.INIT_RELOCALIZE:
                        pass  # Already set above
                    self.state = SlamState.OK
                    self.pose_is_ok = True
                    self.kf_ref = f_cur.kf_ref  # right side updated by self.relocalize()
                    self.kf_last = self.kf_ref
                    Printer.green(
                        f"Relocalization successful, img id: {img_id}, last reloc frame id: {f_cur.id} reconnected to keyframe id {f_cur.kf_ref.id}"
                    )
                    # Update local map immediately after relocalization to ensure it's based on the new reference keyframe
                    # This ensures that track_local_map can find the correct local map points
                    self.map.local_map.update(self.kf_ref)
                    # Reset motion model after relocalization
                    # After relocalization, the coordinate frame changes completely, so we need to reset
                    # the motion model (not just update it) to clear old velocity/delta information
                    # The motion model's delta_position and delta_orientation are based on the old trajectory
                    # and will be completely wrong after relocalization
                    self.motion_model.reset()
                    self.motion_model.update_pose(timestamp, f_cur.position(), f_cur.quaternion())
                    # After relocalization, motion model is not reliable for next frame (it has no history),
                    # so it will remain disabled (is_ok=False from reset) until we have at least 2 frames
                else:
                    self.pose_is_ok = False
                    Printer.red("Relocalization failed")
                    if self.slam.loop_closing is None:
                        Printer.yellow(
                            "WARNING: you did not set any loop closing / relocalize method!"
                        )

            # now, having a better estimate of f_cur pose, we can find more map point matches:
            # find matches between {local map points} (points in the local map) and {unmatched keypoints of f_cur}
            if self.pose_is_ok:
                self.track_local_map(f_cur)

            # update slam state
            if self.pose_is_ok:
                self.state = SlamState.OK
            else:
                if self.state == SlamState.OK:
                    self.state = SlamState.LOST
                    Printer.red("tracking failure")

            # update motion model state
            # Don't re-enable motion model if we just relocalized (no history yet)
            # After reset, motion model needs at least 2 frames to compute delta/velocity
            if f_cur.id <= self.last_reloc_frame_id + 1:
                self.motion_model.is_ok = False
            else:
                self.motion_model.is_ok = self.pose_is_ok

            if self.pose_is_ok:  # if tracking was successful

                # update motion model
                self.motion_model.update_pose(timestamp, f_cur.position(), f_cur.quaternion())

                f_cur.clean_vo_matches()  # clean VO matches
                self.clean_vo_points()  # clean VO points

                # do we need a new KeyFrame?
                need_new_kf = self.need_new_keyframe(f_cur)

                if need_new_kf:
                    Printer.bold_cyan("NEW KF")
                    self.create_new_keyframe(f_cur, img, img_right, depth)
                    print(
                        f"New keyframe created: {f_cur.id}, local_mapping_queue_size: {self.local_mapping.queue_size()}"
                    )
                # else:
                #     Printer.yellow('NOT KF')

                # Similar to ORBSLAM2:
                # Clean outliers once keyframe generation has been managed:
                # we allow points with high innovation (considered outliers by the Huber Function)
                # pass to the new keyframe, so that bundle adjustment will finally decide
                # if they are outliers or not. We don't want next frame to estimate its position
                # with those points so we discard them in the frame.
                f_cur.clean_outlier_map_points()

                if need_new_kf:
                    if not Parameters.kLocalMappingOnSeparateThread:
                        self.local_mapping.is_running = True
                        while self.local_mapping.queue_size() > 0:
                            self.local_mapping.step()
                            for kf in self.map.local_map.get_keyframes():
                                kf.update_connections()
                            # if self.kf_ref is not None:
                            #     self.map.local_map.update(self.kf_ref)

        # end block {with self.map.update_lock:}

        # NOTE: this reset must be outside the block {with self.map.update_lock:}
        need_reset = self.slam.reset_requested or (
            self.state == SlamState.LOST and self.map.num_keyframes_session() <= 5
        )
        if need_reset:
            Printer.yellow("\nTracking: SLAM resetting...\n")
            state_before_reset = self.state
            self.slam.reset_session()
            if state_before_reset == SlamState.LOST and self.map.is_reloaded():
                self.state = SlamState.INIT_RELOCALIZE
                self.motion_model.is_ok = False
            Printer.yellow("\nTracking: SLAM reset done\n")
            return

        if self.f_cur.kf_ref is None:
            self.f_cur.kf_ref = self.kf_ref

        self.update_tracking_history()  # must stay after having updated slam state (self.state)
        self.update_history()
        Printer.green(
            "map: %d points, %d keyframes" % (self.map.num_points(), self.map.num_keyframes())
        )
        # self.update_history()

        self.timer_main_track.refresh()
        elapsed_time = time.time() - time_start
        self.time_track = elapsed_time
        print("Tracking: elapsed_time: ", elapsed_time)
