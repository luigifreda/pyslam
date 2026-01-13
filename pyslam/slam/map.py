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

import time
import numpy as np
import math
import cv2
import ujson as json

from collections import Counter, deque
from ordered_set import OrderedSet  # from https://pypi.org/project/ordered-set/
from threading import RLock, Thread

from pyslam.utilities.geometry import poseRt, add_ones, add_ones_1D

from .frame import Frame, FrameBase
from .feature_tracker_shared import FeatureTrackerShared
from .keyframe import KeyFrame
from .map_point import MapPoint, MapPointBase

from pyslam.utilities.logging import Printer

from pyslam.config_parameters import Parameters

import traceback

import g2o
from . import optimizer_g2o
from . import optimizer_gtsam

import pyslam_utils

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from pyslam.viz.viewer3D import Viewer3DMapInput


kVerbose = True
kMaxLenFrameDeque = 20


if not kVerbose:

    def print(*args, **kwargs):
        pass


class ReloadedSessionMapInfo:
    def __init__(self, num_keyframes, num_points, max_point_id, max_frame_id, max_keyframe_id):
        self.num_keyframes = num_keyframes
        self.num_points = num_points
        self.max_point_id = max_point_id
        self.max_frame_id = max_frame_id
        self.max_keyframe_id = max_keyframe_id


class MapStateData:
    """
    Data class for the map state data.
    It is used to store the map state in the form of a set of data arrays for the viewer.
    """

    def __init__(self):
        self.poses = []
        self.pose_timestamps = []
        self.fov_centers = []
        self.fov_centers_colors = []
        self.points = []
        self.colors = []
        self.semantic_colors = []
        self.covisibility_graph = []
        self.spanning_tree = []
        self.loops = []


class Map(object):
    def __init__(self):
        self._lock = RLock()
        self._update_lock = RLock()

        self.frames: deque[Frame] = deque(
            maxlen=kMaxLenFrameDeque
        )  # deque with max length, it is thread-safe
        self.keyframes: OrderedSet[KeyFrame] = OrderedSet()
        self.points: set[MapPoint] = set()

        self.keyframe_origins: OrderedSet[KeyFrame] = (
            OrderedSet()
        )  # first keyframe(s) where the map is rooted

        self.keyframes_map: dict[int, KeyFrame] = (
            {}
        )  # map: frame id -> keyframe  (for fast retrieving keyframe from img_id/frame_id)

        self.max_point_id = 0  # 0 is the first point id
        self.max_frame_id = 0  # 0 is the first frame id
        self.max_keyframe_id = 0  # 0 is the first keyframe id (kid)

        self.reloaded_session_map_info: ReloadedSessionMapInfo | None = None

        # local map
        # self.local_map = LocalWindowMap(map=self)
        self.local_map = LocalCovisibilityMap(map=self)

        self.viewer_scale = -1

    def is_reloaded(self):
        return self.reloaded_session_map_info is not None

    def reset(self):
        print("Map: reset...")
        with self._lock:
            with self._update_lock:
                self.frames.clear()
                self.keyframes.clear()
                self.points.clear()

                self.keyframe_origins.clear()
                self.keyframes_map.clear()

                self.local_map.reset()

    def reset_session(self):
        print("Map: reset_session...")
        with self._lock:
            with self._update_lock:
                if self.reloaded_session_map_info is None:
                    self.reset()
                else:
                    # First, collect keyframes to remove
                    keyframes_to_remove = [
                        kf
                        for kf in self.keyframes
                        if kf.kid >= self.reloaded_session_map_info.max_keyframe_id
                    ]
                    for kf in keyframes_to_remove:
                        kf.set_bad()
                        self.keyframes.discard(kf)  # Discard instead of remove to avoid KeyError
                        self.keyframe_origins.discard(kf)  # Safe discard
                        self.keyframes_map.pop(kf.id, None)  # Use pop() to avoid KeyError

                    # Similarly for points
                    points_to_remove = [
                        p
                        for p in self.points
                        if p.id >= self.reloaded_session_map_info.max_point_id
                    ]
                    for p in points_to_remove:
                        p.set_bad()
                        self.points.discard(p)  # Safe discard

                    # Similarly for frames
                    frames_to_remove = [
                        f
                        for f in self.frames
                        if f.id >= self.reloaded_session_map_info.max_frame_id
                    ]
                    for f in frames_to_remove:
                        self.frames.remove(f)  # Since deque is not a set, use remove here

                    # Reset the session of the local map
                    self.local_map.reset_session(keyframes_to_remove, points_to_remove)

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the RLock from the state (don't pickle it)
        if "_lock" in state:
            del state["_lock"]
        if "_update_lock" in state:
            del state["_update_lock"]
        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        # Recreate the RLock after unpickling
        self._lock = RLock()
        self._update_lock = RLock()

    @property
    def lock(self):
        return self._lock

    @property
    def update_lock(self):
        return self._update_lock

    def get_points(self):
        with self._lock:
            return self.points.copy()

    def num_points(self):
        with self._lock:
            return len(self.points)

    def get_frame(self, idx):
        with self._lock:
            try:
                return self.frames[idx]
            except:
                return None

    def get_frames(self):
        with self._lock:
            return self.frames.copy()

    def num_frames(self):
        with self._lock:
            return len(self.frames)

    def get_keyframes(self):
        with self._lock:
            return self.keyframes.copy()

    def get_first_keyframe(self):
        with self._lock:
            return self.keyframes[0]

    def get_last_keyframe(self):
        with self._lock:
            return self.keyframes[-1]

    # get the last N=local_window_size map keyframes
    def get_last_keyframes(self, local_window_size=Parameters.kLocalBAWindowSize):
        with self._lock:
            return OrderedSet(self.keyframes.copy()[-local_window_size:])

    # return the total number of keyframes
    def num_keyframes(self):
        with self._lock:
            return len(self.keyframes)

    # return the number of keyframes of this session
    def num_keyframes_session(self):
        with self._lock:
            if self.reloaded_session_map_info is not None:
                return len(self.keyframes) - self.reloaded_session_map_info.num_keyframes
            else:
                return len(self.keyframes)

    def delete(self):
        with self._lock:
            for f in self.frames:
                f.reset_points()
            for kf in self.keyframes:
                kf.reset_points()

    def add_point(self, point):
        with self._lock:
            ret = self.max_point_id  # override original id
            point.id = ret
            point.map = self
            self.max_point_id += 1
            # self.points.append(point)
            self.points.add(point)
            return ret

    def remove_point(self, point: MapPoint):
        with self._lock:
            try:
                self.points.remove(point)
            except:
                pass
            point.delete()

    def remove_point_no_lock(self, point: MapPoint):
        try:
            self.points.remove(point)
        except:
            pass
        point.delete()

    def add_frame(self, frame: Frame, override_id=False):
        with self._lock:
            ret = frame.id
            if override_id:
                ret = self.max_frame_id
                frame.id = ret  # override original id
                self.max_frame_id += 1
            else:
                self.max_frame_id = max(self.max_frame_id, frame.id + 1)
            self.frames.append(frame)
            return ret

    def remove_frame(self, frame: Frame):
        with self._lock:
            try:
                self.frames.remove(frame)
            except:
                pass

    def add_keyframe(self, keyframe: KeyFrame):
        with self._lock:
            assert keyframe.is_keyframe
            ret = self.max_keyframe_id
            keyframe.kid = ret  # override original keyframe kid
            keyframe.is_keyframe = True
            keyframe.map = self
            self.keyframes.add(keyframe)
            self.keyframes_map[keyframe.id] = keyframe
            self.max_keyframe_id += 1
            return ret

    def remove_keyframe(self, keyframe: KeyFrame):
        with self._lock:
            assert keyframe.is_keyframe
            try:
                self.keyframes.remove(keyframe)
                del self.keyframes_map[keyframe.id]
            except:
                pass

    def draw_feature_trails(
        self, img: np.ndarray, with_level_radius: bool = False, trail_max_length: int = 16
    ):
        if len(self.frames) > 0:
            img_draw = self.frames[-1].draw_all_feature_trails(
                img, with_level_radius, trail_max_length
            )
            return img_draw
        return img

    def get_data_arrays_for_drawing(
        self,
        max_points_to_visualize=Parameters.kMaxSparseMapPointsToVisualize,
        min_weight_for_drawing_covisibility_edge=Parameters.kMinWeightForDrawingCovisibilityEdge,
    ):
        """
        Returns:
            - map_state: filled map state with arrays of
                * poses, poses timestamps,
                * fov centers, fov centers colors,
                * points, colors,
                * semantic colors
                * covisibility graph, spanning tree, loops
        """
        from pyslam.semantics.semantic_mapping_shared import SemanticMappingShared

        map_state = MapStateData()

        keyframes = self.get_keyframes()
        num_map_keyframes = len(keyframes)
        if num_map_keyframes > 0:
            # Twc() and timestamp collection
            map_state.poses = np.array([kf.Twc() for kf in keyframes], dtype=float)
            map_state.pose_timestamps = np.array(
                [kf.timestamp for kf in keyframes], dtype=np.float64
            )

            # Only gather fov centers that exist
            fov_centers = [kf.fov_center_w.T for kf in keyframes if kf.fov_center_w is not None]
            if fov_centers:
                map_state.fov_centers = np.asarray(fov_centers, dtype=float).reshape(-1, 3)
                map_state.fov_centers_colors = np.tile(
                    np.array([1.0, 0.0, 0.0], dtype=float), (len(fov_centers), 1)
                )

        # map points
        map_points = list(self.get_points())
        num_map_points = len(map_points)
        if num_map_points > 0:
            # Downsampling for very large clouds to reduce queue bandwidth and GL load
            if num_map_points > max_points_to_visualize:
                Printer.orange(
                    f"Viewer3D: draw_slam_map - downsampling map points from {num_map_points} to {max_points_to_visualize}"
                )
                idx = np.random.choice(num_map_points, max_points_to_visualize, replace=False)
                sel_points = [map_points[i] for i in idx]
            else:
                sel_points = map_points

            N = len(sel_points)
            pts = np.empty((N, 3), dtype=np.float32)
            cols_rgb = np.empty((N, 3), dtype=np.float32)
            sem_colors = np.zeros((N, 3), dtype=np.float32)

            is_semantic_mapping_active = (
                SemanticMappingShared.semantic_feature_type is not None
            )  # SemanticMappingShared.sem_des_to_rgb is not None
            try:
                if is_semantic_mapping_active:
                    for i, p in enumerate(sel_points):
                        pts[i] = p.pt()
                        cols_rgb[i] = p.color
                        sem_colors[i] = p.semantic_color
                else:
                    for i, p in enumerate(sel_points):
                        pts[i] = p.pt()
                        cols_rgb[i] = p.color
            except Exception as e:
                Printer.red(f"Viewer3D: draw_slam_map - error: {e}")

            map_state.points = pts
            map_state.colors = cols_rgb[:, ::-1] / 255.0  # BGR -> RGB and normalize
            map_state.semantic_colors = (
                sem_colors / 255.0 if is_semantic_mapping_active else sem_colors
            )

        # graphs
        if keyframes:
            cov_lines = []
            span_lines = []
            loop_lines = []
            for kf in keyframes:
                Ow = kf.Ow()
                for kf_cov in kf.get_covisible_by_weight(min_weight_for_drawing_covisibility_edge):
                    if kf_cov.kid > kf.kid:
                        cov_lines.append([*Ow, *kf_cov.Ow()])
                if kf.parent is not None:
                    span_lines.append([*Ow, *kf.parent.Ow()])
                for kf_loop in kf.get_loop_edges():
                    if kf_loop.kid > kf.kid:
                        loop_lines.append([*Ow, *kf_loop.Ow()])
            map_state.covisibility_graph = (
                np.asarray(cov_lines, dtype=float) if cov_lines else np.empty((0, 6), dtype=float)
            )
            map_state.spanning_tree = (
                np.asarray(span_lines, dtype=float) if span_lines else np.empty((0, 6), dtype=float)
            )
            map_state.loops = (
                np.asarray(loop_lines, dtype=float) if loop_lines else np.empty((0, 6), dtype=float)
            )

        return map_state

    # add new points to the map from 3D point estimations, frames and pairwise matches
    # points3d is [Nx3]
    def add_points(
        self,
        points3d,
        mask_pts3d,
        kf1: KeyFrame,
        kf2: KeyFrame,
        idxs1,
        idxs2,
        img1,
        do_check=True,
        cos_max_parallax=Parameters.kCosMaxParallax,
        far_points_threshold=None,
    ):
        """
        Add new points to the map from 3D point estimations, frames and pairwise matches.
        Args:
            points3d: [Nx3] 3D points
            mask_pts3d: [N] mask of points to add
            kf1: KeyFrame
            kf2: KeyFrame
            idxs1: [N] indices of points in kf1
            idxs2: [N] indices of points in kf2
            img1: Image
            do_check: bool, if True, check if points are valid
            cos_max_parallax: float, max parallax angle
            far_points_threshold: float, threshold for far points
        """
        # with self._lock:
        assert kf1.is_keyframe and kf2.is_keyframe  # kf1 and kf2 must be keyframes
        assert points3d.shape[0] == len(idxs1)
        assert len(idxs2) == len(idxs1)

        idxs1 = np.array(idxs1)
        idxs2 = np.array(idxs2)

        added_map_points = []
        out_mask_pts3d = np.full(points3d.shape[0], False, dtype=bool)
        if mask_pts3d is None:
            mask_pts3d = np.full(points3d.shape[0], True, dtype=bool)

        if do_check:

            # project points
            uvs1, proj_depths1 = kf1.project_points(points3d)
            bad_depths1 = proj_depths1 <= 0
            uvs2, proj_depths2 = kf2.project_points(points3d)
            bad_depths2 = proj_depths2 <= 0

            if far_points_threshold is not None:
                # print(f'Map: adding points: far_points_threshold: {far_points_threshold}')
                far_depths1 = proj_depths1 > far_points_threshold
                bad_depths1 = bad_depths1 | far_depths1
                far_depths2 = proj_depths2 > far_points_threshold
                bad_depths2 = bad_depths2 | far_depths2

            is_stereo1 = (
                np.zeros(len(idxs1), dtype=bool) if kf1.kps_ur is None else kf1.kps_ur[idxs1] >= 0
            )
            is_mono1 = np.logical_not(is_stereo1)
            is_stereo2 = (
                np.zeros(len(idxs2), dtype=bool) if kf2.kps_ur is None else kf2.kps_ur[idxs2] >= 0
            )
            is_mono2 = np.logical_not(is_stereo2)

            # compute back-projected rays (unit vectors)
            rays1 = np.dot(kf1.Rwc(), add_ones(kf1.kpsn[idxs1]).T).T
            norm_rays1 = np.linalg.norm(rays1, axis=-1, keepdims=True)
            rays1 /= norm_rays1

            rays2 = np.dot(kf2.Rwc(), add_ones(kf2.kpsn[idxs2]).T).T
            norm_rays2 = np.linalg.norm(rays2, axis=-1, keepdims=True)
            rays2 /= norm_rays2

            # compute dot products of rays
            cos_parallaxs = np.sum(rays1 * rays2, axis=1)

            # if we have depths check if we can use depths in case of bad parallax
            if kf1.depths is not None and kf2.depths is not None:
                # NOTE: 2.0 is certainly higher than any cos_parallax value
                cos_parallax_stereo1 = (
                    np.where(
                        is_stereo1,
                        np.cos(2.0 * np.arctan2(kf1.camera.b / 2, kf1.depths[idxs1])),
                        2.0,
                    )
                    if kf1.depths is not None
                    else [2.0] * len(idxs1)
                )
                cos_parallax_stereo2 = (
                    np.where(
                        is_stereo2,
                        np.cos(2.0 * np.arctan2(kf2.camera.b / 2, kf2.depths[idxs2])),
                        2.0,
                    )
                    if kf2.depths is not None
                    else [2.0] * len(idxs2)
                )
                cos_parallax_stereo = np.minimum(cos_parallax_stereo1, cos_parallax_stereo2)

                # check if we can recover bad-parallx points from stereo/rgbd data
                try_recover3d_from_stereo = np.logical_or(
                    cos_parallaxs < 0,
                    np.logical_or(
                        cos_parallaxs > cos_parallax_stereo, cos_parallaxs > cos_max_parallax
                    ),
                )
                recover3d_from_stereo1 = np.logical_and(
                    try_recover3d_from_stereo,
                    np.logical_and(is_stereo1, cos_parallax_stereo1 < cos_parallax_stereo2),
                )
                recover3d_from_stereo2 = np.logical_and(
                    np.logical_and(
                        try_recover3d_from_stereo, np.logical_not(recover3d_from_stereo1)
                    ),
                    np.logical_and(is_stereo2, cos_parallax_stereo2 < cos_parallax_stereo1),
                )
                recovered3d_from_stereo = np.logical_or(
                    recover3d_from_stereo1, recover3d_from_stereo2
                )

                if np.any(recover3d_from_stereo1):
                    points3d[recover3d_from_stereo1, :], _ = kf1.unproject_points_3d(
                        idxs1[recover3d_from_stereo1], transform_in_world=True
                    )
                if np.any(recover3d_from_stereo2):
                    points3d[recover3d_from_stereo2, :], _ = kf2.unproject_points_3d(
                        idxs2[recover3d_from_stereo2], transform_in_world=True
                    )
            else:
                recovered3d_from_stereo = np.zeros(len(idxs1), dtype=bool)

            # we don't have bad parallax where we recovered from stereo
            bad_cos_parallaxs = np.logical_and(
                np.logical_or(cos_parallaxs < 0, cos_parallaxs > cos_max_parallax),
                np.logical_not(recovered3d_from_stereo),
            )

            # compute reprojection errors and check chi2
            bad_chis2_1 = None
            bad_chis2_2 = None

            # compute mono reproj errors on kf1
            errs1_mono_vec = uvs1 - kf1.kpsu[idxs1]
            errs1 = np.where(
                is_mono1[:, np.newaxis], errs1_mono_vec, np.zeros_like(errs1_mono_vec)
            )  # mono errors
            errs1_sqr = np.sum(errs1 * errs1, axis=1)  # squared reprojection errors
            kps1_levels = kf1.octaves[idxs1]
            invSigmas2_1 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[kps1_levels]
            chis2_1_mono = errs1_sqr * invSigmas2_1  # chi-squared

            # stereo reprojection error
            #     u   = fx*x*invz+cx
            #     u_r = u - camera.bf*invz
            #     v   = fy*y*invz+cy
            #     errX   = u - kp.pt.x
            #     errY   = v - kp.pt.y
            #     errX_r = u_r - kp_ur

            # compute stereo reproj errors on kf1
            if kf1.kps_ur is not None:
                kp1_ur = kf1.kps_ur[idxs1]  # kp right coords if available
                depths1 = kf1.depths[idxs1]
                safe_depths1 = np.where(
                    depths1 == 0, np.inf, depths1
                )  # to prevent division by zero
                errs1_stereo_vec = np.concatenate(
                    (
                        errs1_mono_vec,
                        (uvs1[:, 0] - kf1.camera.bf / safe_depths1 - kp1_ur)[:, np.newaxis],
                    ),
                    axis=1,
                )  # stereo errors
                errs1_stereo = np.where(
                    is_stereo1[:, np.newaxis], errs1_stereo_vec, np.zeros_like(errs1_stereo_vec)
                )
                errs1_stereo_sqr = np.sum(
                    errs1_stereo * errs1_stereo, axis=1
                )  # squared reprojection errors
                chis2_1_stereo = errs1_stereo_sqr * invSigmas2_1  # chi-squared
                # bad_chis2_1 = np.logical_or(chis2_1_mono > Parameters.kChi2Mono, chis2_1_stereo > Parameters.kChi2Stereo)
                bad_chis2_1 = np.where(
                    is_stereo1,
                    chis2_1_stereo > Parameters.kChi2Stereo,
                    chis2_1_mono > Parameters.kChi2Mono,
                )
            else:
                bad_chis2_1 = chis2_1_mono > Parameters.kChi2Mono

            # compute mono reproj errors on kf1
            errs2_mono_vec = uvs2 - kf2.kpsu[idxs2]  # mono errors
            errs2 = np.where(is_mono2[:, np.newaxis], errs2_mono_vec, np.zeros(2))  # mono errors
            errs2_sqr = np.sum(errs2 * errs2, axis=1)  # squared reprojection errors
            kps2_levels = kf2.octaves[idxs2]
            invSigmas2_2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2[kps2_levels]
            chis2_2_mono = errs2_sqr * invSigmas2_2  # chi-squared

            if kf2.kps_ur is not None:
                kp2_ur = (
                    kf2.kps_ur[idxs2] if kf2.kps_ur is not None else [-1] * len(idxs2)
                )  # kp right coords if available
                depths2 = kf2.depths[idxs2]
                safe_depths2 = np.where(
                    depths2 == 0, np.inf, depths2
                )  # to prevent division by zero
                errs2_stereo_vec = np.concatenate(
                    (
                        errs2_mono_vec,
                        (uvs2[:, 0] - kf2.camera.bf / safe_depths2 - kp2_ur)[:, np.newaxis],
                    ),
                    axis=1,
                )  # stereo errors
                errs2_stereo = np.where(
                    is_stereo2[:, np.newaxis], errs2_stereo_vec, np.zeros_like(errs2_stereo_vec)
                )
                errs2_stereo_sqr = np.sum(
                    errs2_stereo * errs2_stereo, axis=1
                )  # squared reprojection errors
                chis2_2_stereo = errs2_stereo_sqr * invSigmas2_2  # chi-squared
                # bad_chis2_2 = np.logical_or(chis2_2_mono > Parameters.kChi2Mono, chis2_2_stereo > Parameters.kChi2Stereo)
                bad_chis2_2 = np.where(
                    is_stereo2,
                    chis2_2_stereo > Parameters.kChi2Stereo,
                    chis2_2_mono > Parameters.kChi2Mono,
                )
            else:
                bad_chis2_2 = (
                    chis2_2_mono > Parameters.kChi2Mono
                )  # chi-square 2 DOFs  (Hartley Zisserman pg 119)

            # scale consistency check
            ratio_scale_consistency = (
                Parameters.kScaleConsistencyFactor
                * FeatureTrackerShared.feature_manager.scale_factor
            )
            scale_factors_x_depths1 = (
                FeatureTrackerShared.feature_manager.scale_factors[kps1_levels] * proj_depths1
            )
            scale_factors_x_depths1_x_ratio_scale_consistency = (
                scale_factors_x_depths1 * ratio_scale_consistency
            )
            scale_factors_x_depths2 = (
                FeatureTrackerShared.feature_manager.scale_factors[kps2_levels] * proj_depths2
            )
            scale_factors_x_depths2_x_ratio_scale_consistency = (
                scale_factors_x_depths2 * ratio_scale_consistency
            )
            bad_scale_consistency = np.logical_or(
                (scale_factors_x_depths1 > scale_factors_x_depths2_x_ratio_scale_consistency),
                (scale_factors_x_depths2 > scale_factors_x_depths1_x_ratio_scale_consistency),
            )

            # combine all checks
            bad_points = (
                bad_cos_parallaxs
                | bad_depths1
                | bad_depths2
                | bad_chis2_1
                | bad_chis2_2
                | bad_scale_consistency
            )
            if False:  # for debugging
                print(f"[add_points] bad_points = {np.sum(bad_points)} of {len(idxs1)}")
                print(f"\t bad_depths1 = {np.sum(bad_depths1)}")
                print(f"\t bad_depths2 = {np.sum(bad_depths2)}")
                print(f"\t bad_chis2_1 = {np.sum(bad_chis2_1)}")
                print(f"\t bad_chis2_2 = {np.sum(bad_chis2_2)}")
                print(f"\t bad_scale_consistency = {np.sum(bad_scale_consistency)}")

        # end if do_check

        # get color patches
        # Q(@luigifreda): this gets img_coords from kf1 but kf_ref in MapPoint is kf2
        img_coords = np.floor(kf1.kps[idxs1]).astype(np.intp)  # image keypoints coordinates
        delta = Parameters.kSparseImageColorPatchDelta
        default_color = np.array([255, 0, 0], dtype=np.float32)
        img1 = np.ascontiguousarray(img1)  # Ensure contiguous for Numba

        mean_colors = pyslam_utils.extract_mean_colors(img1, img_coords, delta, default_color)
        # mean_colors = np.full((len(idxs1), 3), [255, 0, 0], dtype=np.float32) # to get all blue

        for i, p in enumerate(points3d):
            if not mask_pts3d[i]:
                # print('p[%d] not good' % i)
                continue

            # perform different required checks before adding the point
            if do_check and bad_points[i]:
                continue

            # add the point to this map
            idx1_i = idxs1[i]
            idx2_i = idxs2[i]

            # get the color of the point
            try:
                color = mean_colors[i]
            except IndexError:
                Printer.orange("color out of range")
                color = (255, 0, 0)

            # add the point to this map
            mp = MapPoint(p[0:3], color, kf2, idx2_i)
            self.add_point(mp)  # add point to this map
            mp.add_observation(kf1, idx1_i)
            mp.add_observation(kf2, idx2_i)
            mp.update_info()
            out_mask_pts3d[i] = True
            added_map_points.append(mp)
        return len(added_map_points), out_mask_pts3d, added_map_points

    # add new points to the map from 3D point stereo-back-projection
    # points3d is [Nx3]
    def add_stereo_points(self, points3d, mask_pts3d, f: Frame, kf: KeyFrame, idxs, img):
        # with self._lock:
        assert kf.is_keyframe

        if mask_pts3d is None:
            mask_pts3d = np.full(points3d.shape[0], True, dtype=bool)

        # get color patches
        img_coords = np.floor(kf.kps[idxs]).astype(np.intp)  # image keypoints coordinates
        delta = Parameters.kSparseImageColorPatchDelta
        default_color = np.array([255, 0, 0], dtype=np.float32)
        img = np.ascontiguousarray(img)
        mean_colors = pyslam_utils.extract_mean_colors(img, img_coords, delta, default_color)

        num_added_points = 0
        for i, p in enumerate(points3d):
            if not mask_pts3d[i]:
                # print('p[%d] not good' % i)
                continue

            # get the color of the point
            try:
                color = mean_colors[i]
            except IndexError:
                Printer.orange("color out of range")
                color = (255, 0, 0)

            # add the point to this map
            mp = MapPoint(p[0:3], color, kf, idxs[i])

            # we need to add the point both the originary frame and the newly created keyframe
            f.points[idxs[i]] = mp  # add point to the frame
            self.add_point(mp)  # add point to this map
            mp.add_observation(kf, idxs[i])
            mp.update_info()
            num_added_points += 1
        return num_added_points

    # remove points which have a big reprojection error
    def remove_points_with_big_reproj_err(self, points):
        inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2

        if points is None:
            with self._lock:
                points = list(self.get_points())
        with self.update_lock:
            # print('map points: ', sorted([p.id for p in self.points]))
            # print('points: ', sorted([p.id for p in points]))
            culled_pt_count = 0
            for p in points:
                # compute reprojection error
                chi2s = []
                for f, idx in p.observations():
                    uv = f.kpsu[idx]
                    proj, z = f.project_map_point(p)
                    invSigma2 = inv_level_sigmas2[f.octaves[idx]]
                    err = proj - uv
                    chi2s.append(np.inner(err, err) * invSigma2)
                # cull
                # mean_chi2 = np.mean(chi2s)
                if (
                    np.mean(chi2s) > Parameters.kChi2Mono
                ):  # chi-square 2 DOFs  (Hartley Zisserman pg 119)
                    culled_pt_count += 1
                    # print('removing point: ',p.id, 'from frames: ', [f.id for f in p.keyframes])
                    self.remove_point(p)
            Printer.blue("# culled map points: ", culled_pt_count)

    def compute_mean_reproj_error(self, points=None):
        chi2 = 0
        num_obs = 0
        inv_level_sigmas2 = FeatureTrackerShared.feature_manager.inv_level_sigmas2
        # with self._lock:
        with self.update_lock:
            if points is None:
                points = self.points
            for p in points:
                # compute reprojection error
                for f, idx in p.observations():
                    uv = f.kpsu[idx]
                    proj, _ = f.project_map_point(p)
                    invSigma2 = inv_level_sigmas2[f.octaves[idx]]
                    err = proj - uv
                    chi2 += np.inner(err, err) * invSigma2
                    num_obs += 1
        return chi2 / max(num_obs, 1)

    # BA considering all keyframes:
    # - local keyframes are adjusted,
    # - other keyframes are fixed
    # - all points are adjusted
    def optimize(
        self,
        local_window_size=Parameters.kLargeBAWindowSize,
        verbose=False,
        rounds=10,
        use_robust_kernel=False,
        do_cull_points=False,
        abort_flag=None,
    ):
        """
        Optimize pixel reprojection error, bundle adjustment.
        Returns:
        - mean_squared_error
        - result_dict: filled dictionary with the updates of the keyframes and points if provided in the input
        """
        if abort_flag is None:
            abort_flag = g2o.Flag()

        if Parameters.kOptimizationBundleAdjustUseGtsam:
            bundle_adjustment_fun = optimizer_gtsam.bundle_adjustment
        else:
            bundle_adjustment_fun = optimizer_g2o.bundle_adjustment
        res = bundle_adjustment_fun(
            self.get_keyframes(),
            self.get_points(),
            local_window_size=local_window_size,
            rounds=rounds,
            loop_kf_id=0,
            use_robust_kernel=use_robust_kernel,
            abort_flag=abort_flag,
            verbose=verbose,
        )
        if do_cull_points:
            self.remove_points_with_big_reproj_err(self.get_points())
        return res

    # local BA: only local keyframes and local points are adjusted
    def locally_optimize(
        self, kf_ref, verbose=False, rounds=10, abort_flag=None, mp_abort_flag=None
    ):
        """
        Local bundle adjustment (optimize points reprojection error)
        Returns:
        - mean_squared_error
        """
        from .local_mapping import LocalMapping

        print = LocalMapping.print

        if abort_flag is None:
            abort_flag = g2o.Flag()

        try:
            keyframes, points, ref_keyframes = self.local_map.update(kf_ref)
            print("local optimization window: ", sorted([kf.id for kf in keyframes]))
            print("                     refs: ", sorted([kf.id for kf in ref_keyframes]))
            print("                   #points: ", len(points))
            # print('                   points: ', sorted([p.id for p in points]))
            # err = optimizer_g2o.optimize(frames, points, None, False, verbose, rounds)
            # NOTE: Why do we want to use parallel multi-processing instead of multi-threading for local BA?
            #       Unfortunately, the GIL does use a SINGLE CPU-core under multi-threading.
            #       On the other hand, multi-processing allows to distribute computation over multiple CPU-cores.
            if Parameters.kOptimizationBundleAdjustUseGtsam:
                ba_function = (
                    optimizer_gtsam.local_bundle_adjustment
                )  # [WIP] testing gtsam, override
            else:
                ba_function = (
                    optimizer_g2o.local_bundle_adjustment_parallel
                    if Parameters.kUseParallelProcessLBA
                    else optimizer_g2o.local_bundle_adjustment
                )
            mean_squared_error, ratio_bad_observations = ba_function(
                keyframes,
                points,
                ref_keyframes,
                False,
                verbose,
                rounds,
                abort_flag=abort_flag,
                mp_abort_flag=mp_abort_flag,
                map_lock=self.update_lock,
            )
            Printer.green(
                "local optimization - perc bad observations: %.2f %%"
                % (ratio_bad_observations * 100)
            )
            return mean_squared_error
        except Exception as e:
            print(f"locally_optimize: EXCEPTION: {e} !!!")
            traceback_details = traceback.format_exc()
            print(f"\t traceback details: {traceback_details}")
            return -1

    def to_json(self, out_json=None):
        if out_json is None:
            out_json = {}
        with self._lock:
            with self.update_lock:
                # static stuff
                out_json["FrameBase._id"] = FrameBase.next_id()
                out_json["MapPointBase._id"] = MapPointBase.next_id()

                # non-static stuff
                out_json["frames"] = [f.to_json() for f in self.frames]
                out_json["keyframes"] = [kf.to_json() for kf in self.keyframes if not kf.is_bad()]
                out_json["points"] = [p.to_json() for p in self.points if not p.is_bad()]
                out_json["keyframe_origins"] = [kf.to_json() for kf in self.keyframe_origins]

                out_json["max_frame_id"] = self.max_frame_id
                out_json["max_point_id"] = self.max_point_id
                out_json["max_keyframe_id"] = self.max_keyframe_id

                out_json["viewer_scale"] = self.viewer_scale
        return out_json

    # NOTE: keep this updated according to new data structure changes
    def serialize(self):
        ret_json = self.to_json()
        return json.dumps(ret_json)

    # NOTE: keep this updated according to new data structure changes
    def from_json(self, loaded_json):
        # Handle both string (C++ serialization) and dict (Python serialization) inputs
        if isinstance(loaded_json, str):
            # C++ serialization: JSON string needs to be parsed
            try:
                loaded_json = json.loads(loaded_json)
            except (json.JSONDecodeError, ValueError) as e:
                raise ValueError(f"Failed to parse JSON string for map: {e}")
        elif not isinstance(loaded_json, dict):
            raise TypeError(f"Map.from_json expects str or dict, got {type(loaded_json)}")

        # static stuff
        FrameBase._id = loaded_json["FrameBase._id"]
        MapPointBase._id = loaded_json["MapPointBase._id"]

        with self._lock:
            with self.update_lock:
                # non-static stuff
                print("\tLoading frames...")
                self.frames = [
                    KeyFrame.from_json(f) if bool(f["is_keyframe"]) else Frame.from_json(f)
                    for f in loaded_json["frames"]
                ]
                print("\tLoading keyframes...")
                self.keyframes = [KeyFrame.from_json(kf) for kf in loaded_json["keyframes"]]
                print("\tLoading points...")
                self.points = [MapPoint.from_json(p) for p in loaded_json["points"]]

                self.max_frame_id = loaded_json["max_frame_id"]
                self.max_point_id = loaded_json["max_point_id"]
                self.max_keyframe_id = loaded_json["max_keyframe_id"]

                self.viewer_scale = loaded_json["viewer_scale"]

                # now replace ids with actual objects in the map assets
                print("\tReplacing ids with actual objects in frames...")
                for f in self.frames:
                    f.replace_ids_with_objects(self.points, self.frames, self.keyframes)
                print("\tReplacing ids with actual objects in keyframes...")
                for kf in self.keyframes:
                    kf.replace_ids_with_objects(self.points, self.frames, self.keyframes)
                    kf.map = self  # set the map
                print("\tReplacing ids with actual objects in points...")
                for p in self.points:
                    p.replace_ids_with_objects(self.points, self.frames, self.keyframes)
                    p.map = self  # set the map

                # reconstruct the keyframes_map
                self.keyframes_map = {}
                for kf in self.keyframes:
                    self.keyframes_map[kf.id] = kf

                # recover keyframe origins from keyframe map
                self.keyframe_origins = [
                    self.keyframes_map[kfjson["id"]]
                    for kfjson in loaded_json["keyframe_origins"]
                    if kfjson["id"] in self.keyframes_map
                ]

                self.frames = deque(self.frames, maxlen=kMaxLenFrameDeque)
                self.keyframes = OrderedSet(self.keyframes)
                self.points = set(self.points)
                self.keyframe_origins = OrderedSet(self.keyframe_origins)

                self.reloaded_session_map_info = ReloadedSessionMapInfo(
                    len(self.keyframes),
                    len(self.points),
                    self.max_point_id,
                    self.max_frame_id,
                    self.max_keyframe_id,
                )

        print(
            f"Map: from_json - FrameBase._id: {FrameBase._id} - max_frame_id: {self.max_frame_id}"
        )
        print(
            f"Map: from_json - MapPointBase._id: {MapPointBase._id} - max_point_id: {self.max_point_id}"
        )

        # Sync static ID counters with max IDs so new objects continue from the correct ID
        FrameBase.set_id(self.max_frame_id)
        MapPointBase.set_id(self.max_point_id)

    def deserialize(self, s):
        ret = json.loads(s)
        self.from_json(ret)

    def save(self, filename):
        with open(filename, "w") as f:
            f.write(self.serialize())
        Printer.green("\t ...map saved to: ", filename)

    def load(self, filename):
        with open(filename, "rb") as f:
            self.deserialize(f.read())
        Printer.green("\t ...map loaded from: ", filename)


# Local map base class
class LocalMapBase(object):
    def __init__(self, map: "Map" = None):
        self._lock = RLock()
        self.map = map
        self.keyframes = OrderedSet()  # collection of local keyframes
        self.points = set()  # points visible in 'keyframes'
        self.ref_keyframes = (
            set()
        )  # collection of 'covisible' keyframes not in self.keyframes that see at least one point in self.points

    def reset(self):
        with self._lock:
            self.keyframes.clear()
            self.points.clear()
            self.ref_keyframes.clear()

    def reset_session(self, keyframes_to_remove=None, points_to_remove=None):
        with self._lock:
            if keyframes_to_remove is None and points_to_remove is None:
                self.reset()
            else:
                for kf in keyframes_to_remove:
                    self.keyframes.discard(kf)
                    self.ref_keyframes.discard(kf)
                for p in points_to_remove:
                    self.points.discard(p)

    @property
    def lock(self):
        return self._lock

    def is_empty(self):
        with self._lock:
            return len(self.keyframes) == 0

    def get_points(self):
        with self._lock:
            return self.points.copy()

    def num_points(self):
        with self._lock:
            return len(self.points)

    def get_keyframes(self):
        with self._lock:
            return self.keyframes.copy()

    def num_keyframes(self):
        with self._lock:
            return len(self.keyframes)

    # given some input local keyframes, get all the viewed points and all the reference keyframes (that see the viewed points but are not in the local keyframes)
    def update_from_keyframes(self, local_keyframes):
        # remove possible bad keyframes
        local_keyframes = {kf for kf in local_keyframes if not kf.is_bad()}

        # all good points in local_keyframes (only one instance per point)
        viewed_good_points = {p for kf in local_keyframes for p in kf.get_matched_good_points()}

        # reference keyframes: keyframes not in local_keyframes that see points observed in local_keyframes
        # get the keyframes viewing p but not in local_keyframes
        ref_keyframes = {
            kf_viewing
            for p in viewed_good_points
            for kf_viewing in p.keyframes()
            if not kf_viewing.is_bad() and kf_viewing not in local_keyframes
        }

        with self.lock:
            self.keyframes = local_keyframes
            self.points = viewed_good_points
            self.ref_keyframes = ref_keyframes

        return local_keyframes, viewed_good_points, ref_keyframes

    # from a given input frame compute:
    # - the reference keyframe (the keyframe that sees most map points of the frame)
    # - the local keyframes
    # - the local points
    def get_frame_covisibles(self, frame: Frame):
        points = frame.get_matched_good_points()
        if len(points) == 0:
            Printer.red("LocalMapBase: get_frame_covisibles - frame without points")

        # for all map points in frame check in which other keyframes are they seen
        # increase counter for those keyframes
        viewing_keyframes = [
            kf for p in points for kf in p.keyframes() if not kf.is_bad()
        ]  # if kf in keyframes]
        viewing_keyframes = Counter(viewing_keyframes)
        if len(viewing_keyframes) == 0:
            Printer.red("LocalMapBase: get_frame_covisibles - no viewing keyframes")
            return None, None, None

        # get the keyframe that sees most points
        kf_ref = viewing_keyframes.most_common(1)[0][0]

        # include also some not-already-included keyframes that are neighbors to already-included keyframes
        # Create a list that grows during iteration (like C++)
        local_keyframes_list = list(viewing_keyframes.keys())

        for kf in local_keyframes_list:
            # Limit the number of keyframes
            if len(local_keyframes_list) >= Parameters.kMaxNumOfKeyframesInLocalMap:
                break

            second_neighbors = kf.get_best_covisible_keyframes(
                Parameters.kNumBestCovisibilityKeyFrames
            )
            # viewing_keyframes.update([kf for kf in second_neighbors if not kf.is_bad()]) # more aggressive but slower
            for kf_ in second_neighbors:
                if not kf_.is_bad() and kf_ not in viewing_keyframes:
                    viewing_keyframes.update([kf_])
                    local_keyframes_list.append(kf_)  # Add to list so it gets processed
                    break  # only one second neighbor per kf is needed

            children = kf.get_children()
            # viewing_keyframes.update([kf for kf in children if not kf.is_bad()]) # more aggressive but slower
            for kf_ in children:
                if not kf_.is_bad() and kf_ not in viewing_keyframes:
                    viewing_keyframes.update([kf_])
                    local_keyframes_list.append(kf_)  # Add to list so it gets processed
                    break  # only one child is needed per kf is needed

            parent = kf.get_parent()
            if parent and not parent.is_bad() and parent not in viewing_keyframes:
                viewing_keyframes.update([parent])
                local_keyframes_list.append(parent)  # Add to list so it gets processed

        # select the top N keyframes by count
        local_keyframes_counts = viewing_keyframes.most_common(
            Parameters.kMaxNumOfKeyframesInLocalMap
        )

        local_points = set()
        local_keyframes = []
        for kf, c in local_keyframes_counts:
            local_points.update(kf.get_matched_points())
            local_keyframes.append(kf)

        return kf_ref, local_keyframes, list(local_points)


# Local window map (last N keyframes)
class LocalWindowMap(LocalMapBase):
    def __init__(self, map: "Map" = None, local_window_size=Parameters.kLocalBAWindowSize):
        super().__init__(map)
        self.local_window_size = local_window_size  # length of the local window

    def update_keyframes(self, kf_ref=None):
        with self._lock:
            # get the last N=local_window_size keyframes
            self.keyframes = self.map.get_last_keyframes(self.local_window_size)
            return self.keyframes

    def get_best_neighbors(self, kf_ref=None, N=20):
        return self.map.get_last_keyframes(N)

    # update the local keyframes, the viewed points and the reference keyframes (that see the viewed points but are not in the local keyframes)
    def update(self, kf_ref=None):
        self.update_keyframes(kf_ref)
        return self.update_from_keyframes(self.keyframes)


# Local map from covisibility graph
class LocalCovisibilityMap(LocalMapBase):
    def __init__(self, map: "Map" = None):
        super().__init__(map)

    def __getstate__(self):
        # Create a copy of the instance's __dict__
        state = self.__dict__.copy()
        # Remove the RLock from the state (don't pickle it)
        if "_lock" in state:
            del state["_lock"]
        return state

    def __setstate__(self, state):
        # Restore the state (without 'lock' initially)
        self.__dict__.update(state)
        # Recreate the RLock after unpickling
        self._lock = RLock()

    def update_keyframes(self, kf_ref):
        with self._lock:
            assert kf_ref is not None
            self.keyframes = OrderedSet()
            self.keyframes.add(kf_ref)
            neighbor_kfs = [kf for kf in kf_ref.get_covisible_keyframes() if not kf.is_bad()]
            self.keyframes.update(neighbor_kfs)
            return self.keyframes

    def get_best_neighbors(self, kf_ref, N=20):
        return kf_ref.get_best_covisible_keyframes(N)

    # update the local keyframes, the viewed points and the reference keyframes (that see the viewed points but are not in the local keyframes)
    def update(self, kf_ref):
        self.update_keyframes(kf_ref)
        return self.update_from_keyframes(self.keyframes)
