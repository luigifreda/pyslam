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

from __future__ import print_function  # This must be the first statement before other statements

import os
import time
import numpy as np
import g2o
from enum import Enum

from collections import defaultdict

from threading import RLock, Thread, Condition
from concurrent.futures import ThreadPoolExecutor
from queue import Queue

from pyslam.config_parameters import Parameters
from pyslam.utilities.logging import Printer

from .frame import compute_frame_matches
from .map import Map
from .keyframe import KeyFrame
from .map_point import MapPoint
from .geometry_matchers import ProjectionMatcher


from pyslam.io.dataset_types import SensorType

import multiprocessing as mp
from scipy.spatial import cKDTree


kNumMinObsForKeyFrameTrackedPoints = 3


from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Only imported when type checking, not at runtime
    from .slam import Slam  #
    from .keyframe import KeyFrame
    from .geometry_matchers import ProjectionMatcher


class LocalMappingCore:
    print = staticmethod(lambda *args, **kwargs: None)  # Default: no-op

    def __init__(self, map: Map, sensor_type):
        self.map = map
        self.sensor_type = sensor_type

        self.recently_added_points = set()

        self.kf_cur: KeyFrame | None = None  # current processed keyframe
        self.kid_last_BA = -1  # last keyframe id when performed BA

        # use self.set_opt_abort_flag() to manage the following two guys
        self.opt_abort_flag = g2o.Flag(False)  # for multi-threading
        self.mp_opt_abort_flag = mp.Value("i", False)  # for multi-processing (when used)

    def reset(self):
        self.recently_added_points.clear()

    def add_points(self, points: list[MapPoint]):
        self.recently_added_points.update(points)

    def remove_points(self, points: list[MapPoint]):
        self.recently_added_points.difference_update(points)

    def set_opt_abort_flag(self, value):
        if self.opt_abort_flag.value != value:
            self.opt_abort_flag.value = value  # for multi-threading
            self.mp_opt_abort_flag.value = value  # for multi-processing  (when used)

    def local_BA(self):
        # local optimization
        err = self.map.locally_optimize(
            kf_ref=self.kf_cur, abort_flag=self.opt_abort_flag, mp_abort_flag=self.mp_opt_abort_flag
        )
        num_kf_ref_tracked_points = self.kf_cur.num_tracked_points(
            kNumMinObsForKeyFrameTrackedPoints
        )  # number of tracked points in k_ref
        return err, num_kf_ref_tracked_points

    def large_window_BA(self):
        # large window optimization of the map
        self.kid_last_BA = self.kf_cur.kid
        err, _ = self.map.optimize(
            local_window_size=Parameters.kLargeBAWindowSize, abort_flag=self.opt_abort_flag
        )  # verbose=True)
        return err

    def process_new_keyframe(self):
        # associate map points to keyframe observations (only good points)
        # and update normal and descriptor
        LocalMappingCore.print(f">>>> updating map points ...")
        good_points_and_idxs = self.kf_cur.get_matched_good_points_and_idxs()
        for p, idx in good_points_and_idxs:
            # Try to add observation
            added = p.add_observation(self.kf_cur, idx)
            if added:
                p.update_info()
            else:
                # this happens for new stereo points inserted by Tracking
                self.recently_added_points.add(p)

        LocalMappingCore.print(">>>> updating connections ...")
        self.kf_cur.update_connections()
        # self.map.add_keyframe(self.kf_cur)   # add kf_cur to map   (moved to tracking.py into create_new_keyframe())

    def cull_map_points(self):
        # LocalMappingCore.print(">>>> culling map points...")
        th_num_observations = 2
        if self.sensor_type != SensorType.MONOCULAR:
            th_num_observations = 3
        min_found_ratio = 0.25
        current_kid = self.kf_cur.kid
        remove_set = set()
        for p in self.recently_added_points:
            if p.is_bad():
                remove_set.add(p)
            elif p.get_found_ratio() < min_found_ratio:
                p.set_bad()
                self.map.remove_point(p)
                remove_set.add(p)
            elif (current_kid - p.first_kid) >= 2 and p.num_observations() <= th_num_observations:
                p.set_bad()
                self.map.remove_point(p)
                remove_set.add(p)
            elif (
                current_kid - p.first_kid
            ) >= 3:  # after three keyframes we do not consider the point a recent one
                remove_set.add(p)
        self.recently_added_points = self.recently_added_points - remove_set
        num_culled_points = len(remove_set)
        return num_culled_points

    # check if once we remove "kf_to_remove" from covisible_kfs we still have that the max distance among fov centers is less than D
    @staticmethod
    def check_remaining_fov_centers_max_distance(
        covisible_kfs: list["KeyFrame"], kf_to_remove: "KeyFrame", dist: float
    ):
        # fov centers that remain if we remove kf_to_remove
        remaining_fov_centers = [kf.fov_center_w for kf in covisible_kfs if kf != kf_to_remove]
        if len(remaining_fov_centers) == 0:
            return False
        remaining_fov_centers = np.hstack(remaining_fov_centers).T
        tree = cKDTree(remaining_fov_centers)
        # Check the distance to the nearest neighbor for each remaining point
        distances, _ = tree.query(
            remaining_fov_centers, k=2
        )  # k=2 because the closest point is itself
        # Check the second nearest neighbor distance (ignoring the self-match at k=1)
        return np.all(distances[:, 1] < dist)

    def cull_keyframes(self, use_fov_centers_based_kf_generation, max_fov_centers_distance):
        LocalMappingCore.print(">>>> culling keyframes...")
        # check redundant keyframes in local keyframes: a keyframe is considered redundant if the 90% of the MapPoints it sees,
        # are seen in at least other 3 keyframes (in the same or finer scale)
        num_culled_keyframes = 0
        th_num_observations = 3
        covisible_kfs = self.kf_cur.get_covisible_keyframes()
        LocalMappingCore.print(f"covisible keyframes: {len(covisible_kfs)}")
        for kf in covisible_kfs:
            if kf.kid == 0:
                continue
            kf_num_points = 0  # num good points for kf
            kf_num_redundant_observations = 0  # num redundant observations for kf
            idxs_and_kf_points = [
                (i, p) for i, p in enumerate(kf.get_points()) if p is not None and not p.is_bad()
            ]
            # for i,p in enumerate(kf.get_points()):
            #     if p is not None and not p.is_bad():
            for i, p in idxs_and_kf_points:
                # Only check depth for non-monocular sensors
                if self.sensor_type != SensorType.MONOCULAR:
                    if kf.depths is not None and (
                        kf.depths[i] > kf.camera.depth_threshold or kf.depths[i] < 0.0
                    ):
                        continue
                kf_num_points += 1
                if p.num_observations() > th_num_observations:
                    scale_level = kf.octaves[i]  # scale level of observation in kf
                    p_num_observations = 0
                    for kf_j, idx in p.observations():
                        if kf_j is kf:
                            continue
                        assert not kf_j.is_bad()
                        scale_level_i = kf_j.octaves[idx]  # scale level of observation in kfi
                        if (
                            scale_level_i <= scale_level + 1
                        ):  # N.B.1 <- more aggressive culling  (expecially when scale_factor=2)
                            # if scale_level_i <= scale_level:     # N.B.2 <- only same scale or finer
                            p_num_observations += 1
                            if p_num_observations >= th_num_observations:
                                break
                    if p_num_observations >= th_num_observations:
                        kf_num_redundant_observations += 1
            remove_kf = (
                kf_num_redundant_observations
                > Parameters.kKeyframeCullingRedundantObsRatio * kf_num_points
            ) and (kf_num_points > Parameters.kKeyframeCullingMinNumPoints)

            if remove_kf:
                # check if the keyframe is too close in time to its parent
                delta_time_parent = abs(kf.timestamp - kf.parent.timestamp)
                LocalMappingCore.print(
                    f"kf {kf.id} to remove: delta time parent: {delta_time_parent}"
                )
                if delta_time_parent < Parameters.kKeyframeMaxTimeDistanceInSecForCulling:
                    remove_kf = False
                # check if the keyframe is too far from the FOV centers of the covisible keyframes
                if use_fov_centers_based_kf_generation:
                    if not LocalMappingCore.check_remaining_fov_centers_max_distance(
                        covisible_kfs, kf, max_fov_centers_distance
                    ):
                        remove_kf = False

            if remove_kf:
                kf.set_bad()
                num_culled_keyframes += 1
                LocalMappingCore.print(
                    "culling keyframe ",
                    kf.id,
                    " (set it bad) - redundant observations: ",
                    kf_num_redundant_observations / max(kf_num_points, 1),
                    "%",
                )
        return num_culled_keyframes

    # fuse close map points of local keyframes
    def fuse_map_points(self, descriptor_distance_sigma):
        # LocalMappingCore.print(">>>> fusing map points")
        total_fused_pts = 0

        num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesStereo
        if self.sensor_type == SensorType.MONOCULAR:
            num_neighbors = Parameters.kLocalMappingNumNeighborKeyFramesMonocular

        # 1. Get direct neighbors
        local_keyframes = self.map.local_map.get_best_neighbors(self.kf_cur, N=num_neighbors)
        target_kfs = set()
        for kf in local_keyframes:
            if kf is self.kf_cur or kf.is_bad():
                continue
            target_kfs.add(kf)
            # 2. Add second neighbors
            for kf2 in self.map.local_map.get_best_neighbors(kf, N=5):
                if kf2 is self.kf_cur or kf2.is_bad():
                    continue
                target_kfs.add(kf2)

        LocalMappingCore.print(
            "local map keyframes: ", [kf.id for kf in target_kfs], " + ", self.kf_cur.id, "..."
        )

        # 3. Fuse current keyframe's points into all target keyframes
        for kf in target_kfs:
            kf_cur_points = self.kf_cur.get_points()
            num_fused_pts = ProjectionMatcher.search_and_fuse(
                kf_cur_points,
                kf,
                max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
                max_descriptor_distance=0.5 * descriptor_distance_sigma,
            )
            LocalMappingCore.print(
                f"\t #fused map points: {num_fused_pts} for KFs ({self.kf_cur.id}, {kf.id})"
            )
            total_fused_pts += num_fused_pts

        # 4. Fuse all target keyframes' points into current keyframe
        fuse_candidates = {
            p for kf in target_kfs for p in kf.get_points() if p is not None and not p.is_bad()
        }  # Remove duplicates via set
        fuse_candidates = np.array(list(fuse_candidates))

        num_fused_pts = ProjectionMatcher.search_and_fuse(
            fuse_candidates,
            self.kf_cur,
            max_reproj_distance=Parameters.kMaxReprojectionDistanceFuse,
            max_descriptor_distance=0.5 * descriptor_distance_sigma,
        )
        LocalMappingCore.print(
            f"\t #fused map points: {num_fused_pts} for local map into KF {self.kf_cur.id}"
        )
        total_fused_pts += num_fused_pts

        # 5. Update all map points in current keyframe
        # points_to_update = [p for p in self.kf_cur.get_points() if p and not p.is_bad()]
        points_to_update = self.kf_cur.get_matched_good_points()
        with ThreadPoolExecutor(
            max_workers=Parameters.kLocalMappingParallelFusePointsNumWorkers
        ) as executor:
            executor.map(lambda p: p.update_info(), points_to_update)

        # 6. Update connections in covisibility graph
        self.kf_cur.update_connections()

        return total_fused_pts
