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
from copy import deepcopy
from threading import Condition, RLock
from types import SimpleNamespace

import cv2
import numpy as np
from scipy.spatial import cKDTree

from pyslam.io.dataset_types import SensorType
from pyslam.config_parameters import Parameters
from pyslam.local_features.feature_manager import feature_manager_factory
from pyslam.local_features.feature_tracker import FeatureTrackerTypes, feature_tracker_factory
from pyslam.local_features.feature_tracker_configs import FeatureTrackerConfigs
from pyslam.loop_closing.loop_closing import LoopClosing
from pyslam.loop_closing.loop_detector_configs import (
    LoopDetectorConfigs,
    loop_detector_config_check,
)
from pyslam.slam import CameraPose, EpipolarMatcher, Frame, KeyFrame, Map, MapPoint, TrackingCore
from pyslam.slam.frame import compute_frame_matches
from pyslam.utilities.geom_triangulation import triangulate_normalized_points
from pyslam.slam.feature_tracker_shared import FeatureTrackerShared
from pyslam.utilities.logging import Printer
from pyslam.utilities.waiting import wait_for_ready


# Direct landmarks are not shared ORB points, so update_connections() would leave
# the graph empty. These temporal links keep the consistency check and spanning
# tree connected. Weight 100 matches the essential-graph covisibility threshold.
kRecentCovisibility = 10
kCovisibilityWeight = 100
kPoseChangeTolerance = 1e-8
# Monocular keyframes have no depth for descriptor keypoints. A keypoint within this many
# pixels of a direct landmark observation takes that landmark's 3D position.
kMonoAssociationRadiusPx = 3.0
# A monocular correction scale needs this many keyframe points seen before and after it.
kMinScaleSupport = 5
kMonoPointColor = np.array([0, 200, 255], dtype=np.uint8)
# Monocular initialization stores its reference frame as a keyframe only once the two-view
# geometry succeeds, so the frames seen before the first keyframe are kept for it.
kMaxPendingInitFrames = 100
# Monocular keyframes also triangulate descriptor matches with this many previous keyframes
# (ORB-SLAM local mapping). Landmark association alone covers too few keypoints for Sim(3)
# verification, which needs a 3D point on both sides of a match.
kMonoTriangulationNeighbors = 3


class DirectLoopGate:
    """Idle/stop protocol LoopClosing and global BA already use for local mapping.

    Tracking calls acquire_tracking() before mutating the direct map and
    finish_tracking() when it is idle again. release() publishes the corrected
    poses before clearing the stop, so the next track() cannot resume early.
    """

    def __init__(self, publish):
        self._publish = publish
        self._lock = RLock()
        self._idle = Condition(self._lock)
        self._resume = Condition(self._lock)
        self.is_running = True
        self._is_idle = True
        self.stop_requested = False
        self.stopped = False

    def acquire_tracking(self):
        with self._lock:
            while self.is_running and self.stop_requested:
                self._is_idle = True
                self.stopped = True
                self._idle.notify_all()
                self._resume.wait()
            if not self.is_running:
                return False
            self._is_idle = False
            self.stopped = False
            return True

    def finish_tracking(self):
        with self._lock:
            self._is_idle = True
            self._idle.notify_all()

    def request_stop(self):
        with self._lock:
            self.stop_requested = True

    def is_stop_requested(self):
        with self._lock:
            return self.stop_requested

    def is_idle(self):
        with self._lock:
            return self._is_idle

    def wait_idle(self, print=print, timeout=None):
        """Block until tracking is idle; `timeout` only paces a progress warning.

        Callers publish poses into DirectSlam right after this returns, so it
        must never return while track() is running. A pending stop request makes
        the next acquire_tracking() park, so tracking always becomes idle.
        """
        if not self.is_running:
            return
        with self._lock:
            while not self._is_idle and self.is_running:
                notified = self._idle.wait(timeout=timeout)
                if not notified:
                    print("DirectLoopGate: still waiting for tracking to become idle")
            if self._is_idle:
                self.stopped = True

    def queue_size(self):
        return 0

    def release(self):
        if not self.is_running:
            return
        try:
            self._publish()
        finally:
            with self._lock:
                self.stopped = False
                self.stop_requested = False
                self._is_idle = True
                self._resume.notify_all()
                self._idle.notify_all()

    def quit(self):
        with self._lock:
            self.is_running = False
            self.stop_requested = False
            self._is_idle = True
            self._resume.notify_all()
            self._idle.notify_all()


class DirectLoopClosing:
    """Run the standard LoopClosing thread on descriptor keyframes of a DirectSlam map."""

    def __init__(self, direct_slam, camera, sensor_type, config, headless=False):
        self.direct_slam = direct_slam
        self.camera = camera
        self.sensor_type = sensor_type
        self.headless = headless
        self.volumetric_integrator = None
        self.map = Map()
        self._graph_revision = 0
        self._clear_state()
        self._has_quit = False

        feature_config = self._feature_tracker_config(config)
        self._feature_config = feature_config
        loop_config = self._loop_detector_config(config)
        self.feature_tracker = feature_tracker_factory(**feature_config)
        if self.feature_tracker.tracker_type == FeatureTrackerTypes.LK:
            raise ValueError(
                "Direct loop closing cannot use a Lucas-Kanade feature tracker. "
                "Select a descriptor tracker such as ORB2."
            )
        FeatureTrackerShared.set_feature_tracker(self.feature_tracker, force=True)
        # The right image only feeds descriptor extraction and stereo matching,
        # which both work on grayscale unless the descriptor needs color.
        self._send_gray_right = not getattr(
            self.feature_tracker.feature_manager, "need_color_image", True)
        self._check_loop_detector(loop_config)

        self.local_mapping = DirectLoopGate(self._publish_poses)
        self.loop_closing = None
        try:
            self.loop_closing = LoopClosing(self, loop_config, headless=headless)
            self.loop_closing.start()
            wait_for_ready(self.loop_closing.is_ready, "LoopClosing")
        except Exception:
            self.quit()
            raise
        Printer.green("Direct SLAM: loop closing initialized and ready")

    def _clear_state(self):
        """Per-run bookkeeping, shared by __init__ and reset(). Caller holds map.update_lock (or owns self)."""
        self._keyframes = []
        self._keyframe_index = {}  # keyframe id -> index in _keyframes
        self._prepared_keyframes = set()
        # Links from earlier keyframes to a new one are deferred until the new
        # keyframe has features, so verification never sees a featureless neighbor.
        self._pending_links = {}
        self._next_unprepared = 0
        # Map points each keyframe created, so a pose change only moves its own
        # points instead of scanning the whole descriptor map.
        self._keyframe_points = {}
        # Last direct Twc seen per keyframe (same order as _keyframes); finds the
        # poses local BA moved with one vectorized comparison per frame.
        self._synced_twc = np.empty((0, 4, 4), dtype=np.float64)
        # Monocular only: direct observations captured when each keyframe was added
        # (ids, pixels, camera-frame points), the shared descriptor point of each direct
        # landmark, and each keyframe's (keypoint index, landmark id) associations.
        self._direct_observations = {}
        self._pending_init_frames = {}  # img_id -> (image, image_right, depth, timestamp)
        self._landmark_points = {}
        self._landmark_matches = {}
        # Viewer graph as index pairs into _keyframes. Edges are appended when a
        # keyframe is added; a published correction bumps _graph_revision
        # (on the LoopClosing thread) to force a full rebuild.
        self._graph_edges = {name: [] for name in ("covisibility_graph", "spanning_tree", "loops")}
        self._graph_arrays = None
        self._graph_built_revision = getattr(self, "_graph_revision", 0)

    @staticmethod
    def _feature_tracker_config(config):
        selected = FeatureTrackerConfigs.ORB2
        if config is not None and config.feature_tracker_config_name is not None:
            selected = FeatureTrackerConfigs.get_config_from_name(config.feature_tracker_config_name)
            if selected is None:
                raise ValueError(
                    f"Unknown FeatureTrackerConfig.name: {config.feature_tracker_config_name}"
                )
        selected = deepcopy(selected)
        if config is not None and config.num_features_to_extract > 0:
            Printer.yellow(
                "Direct loop closing: setting num_features from settings: ",
                config.num_features_to_extract,
            )
            selected["num_features"] = config.num_features_to_extract
        return selected

    @staticmethod
    def _loop_detector_config(config):
        selected = LoopDetectorConfigs.DBOW3
        if config is not None and config.loop_detection_config_name is not None:
            selected = LoopDetectorConfigs.get_config_from_name(config.loop_detection_config_name)
            if selected is None:
                raise ValueError(
                    f"Unknown LoopDetectionConfig.name: {config.loop_detection_config_name}"
                )
        return selected

    def _check_loop_detector(self, loop_config):
        local_config = loop_config.get("local_feature_manager_config")
        local_manager = feature_manager_factory(**local_config) if local_config is not None else None
        probe = SimpleNamespace(
            local_feature_manager=local_manager,
            global_descriptor_type=loop_config["global_descriptor_type"],
            vocabulary_data=loop_config.get("vocabulary_data"),
        )
        loop_detector_config_check(probe, self.feature_tracker.feature_manager.descriptor_type)

    def acquire_tracking(self):
        return self.local_mapping.acquire_tracking()

    def finish_tracking(self):
        self.local_mapping.finish_tracking()

    def note_tracked_frame(self, image, image_right, depth, img_id, timestamp):
        poses, timestamps = self.direct_slam.keyframes()
        with self.map.update_lock:
            self._sync_poses(poses[: len(self._keyframes)])
            added = len(poses) - len(self._keyframes)
            if added == 0:
                if not self._keyframes:
                    self._remember_init_frame(image, image_right, depth, img_id, timestamp)
                return
            if added == 2 and not self._keyframes and self.sensor_type == SensorType.MONOCULAR:
                # Monocular two-view initialization: the reference frame and this one.
                ref_img_id = int(self.direct_slam.keyframe_img_ids()[0])
                ref = self._pending_init_frames.get(ref_img_id)
                if ref is None:
                    Printer.yellow(
                        f"Direct loop closing: initialization frame {ref_img_id} is no longer cached; "
                        "using the current image for its descriptors")
                    ref = (image, image_right, depth, timestamps[0])
                self._pending_init_frames.clear()
                self._add_keyframe(ref[0], ref[1], ref[2], ref_img_id, ref[3], poses[0])
                self._add_keyframe(image, image_right, depth, img_id, timestamp, poses[1])
                return
            if added != 1:
                raise RuntimeError(
                    f"Direct loop closing expected one new keyframe after tracking, got {added}"
                )
            self._pending_init_frames.clear()
            self._add_keyframe(image, image_right, depth, img_id, timestamp, poses[-1])

    def _remember_init_frame(self, image, image_right, depth, img_id, timestamp):
        """Caller holds map.update_lock. Bounded cache of frames before the first keyframe."""
        if self.sensor_type != SensorType.MONOCULAR:
            return
        self._pending_init_frames[int(img_id)] = (
            np.array(image, copy=True),
            None if image_right is None else np.array(image_right, copy=True),
            None if depth is None else np.array(depth, copy=True),
            timestamp,
        )
        while len(self._pending_init_frames) > kMaxPendingInitFrames:
            self._pending_init_frames.pop(next(iter(self._pending_init_frames)))

    def _sync_poses(self, twc_poses):
        """Caller holds map.update_lock. Move keyframes (and their points) that local/manual BA changed."""
        count = len(self._keyframes)
        if len(twc_poses) != count or count == 0:
            return
        twc_all = np.asarray(twc_poses, dtype=np.float64).reshape(count, 4, 4)
        diff = np.abs(twc_all - self._synced_twc).reshape(count, -1).max(axis=1)
        for index in np.flatnonzero(diff >= kPoseChangeTolerance):
            keyframe = self._keyframes[index]
            twc = twc_all[index]
            updated = np.linalg.inv(twc)
            current = np.asarray(keyframe.Tcw(), dtype=np.float64)
            self._synced_twc[index] = twc
            if np.max(np.abs(updated - current)) < kPoseChangeTolerance:
                continue
            delta = twc @ current
            rotation = delta[:3, :3]
            translation = delta[:3, 3]
            keyframe.update_pose(updated)
            for point in self._keyframe_points.get(keyframe.id, ()):
                if point.is_bad():
                    continue
                reference = point.get_reference_keyframe()
                if reference is None or reference.id != keyframe.id:
                    continue
                position = np.asarray(point.pt(), dtype=np.float64).reshape(3)
                point.update_position(rotation @ position + translation)
                point.update_normal_and_depth(True)

    def _add_keyframe(self, image, image_right, depth, img_id, timestamp, twc):
        image = np.ascontiguousarray(image)
        if image_right is not None:
            if image_right.ndim == 3 and self._send_gray_right:
                # Same conversion the feature manager and stereo matcher apply in
                # the worker, so features are unchanged; the payload is 3x smaller.
                image_right = cv2.cvtColor(image_right, cv2.COLOR_RGB2GRAY)
            image_right = np.ascontiguousarray(image_right)
        if depth is not None:
            depth = np.ascontiguousarray(depth)
        pose = CameraPose(np.linalg.inv(np.asarray(twc, dtype=np.float64)))
        frame = Frame(
            self.camera,
            None,  # Feature extraction is deferred to LoopDetectingProcess.
            pose=pose,
            timestamp=timestamp,
            img_id=int(img_id),
        )
        keyframe = KeyFrame(frame, image, image_right, depth)
        # The spanning tree is the temporal chain set below. Clear the flag so a
        # later update_connections() (loop correction/fusion) cannot re-parent
        # this keyframe to kf_max, which may be a newer descendant: that leaves
        # the old parent's child link in place and forms a cycle, and the GBA
        # spanning-tree walk then never terminates while holding map.update_lock.
        keyframe.is_first_connection = False
        self.map.add_keyframe(keyframe)
        if not self._keyframes:
            origins = set(self.map.keyframe_origins)
            origins.add(keyframe)
            self.map.keyframe_origins = origins
        else:
            keyframe.set_parent(self._keyframes[-1])
            neighbors = self._keyframes[-kRecentCovisibility:]
            # Only the forward link now: the new keyframe has no features yet.
            # The reverse links are added in _install_features().
            for neighbor in neighbors:
                keyframe.add_connection(neighbor, kCovisibilityWeight)
            self._pending_links[keyframe.id] = list(neighbors)
        if self.sensor_type == SensorType.MONOCULAR:
            # Captured on the tracking thread: the loop thread must not read the direct map
            # while tracking runs. Camera-frame points survive later pose updates.
            self._direct_observations[keyframe.id] = self.direct_slam.keyframe_observations(
                len(self._keyframes))
        self._keyframe_index[keyframe.id] = len(self._keyframes)
        self._keyframes.append(keyframe)
        self._synced_twc = np.concatenate(
            (self._synced_twc, np.asarray(twc, dtype=np.float64).reshape(1, 4, 4)))
        self._append_graph_edges(keyframe)
        self.loop_closing.add_keyframe(keyframe, image, feature_extraction=dict(
            config=self._feature_config, camera=self.camera,
            image_right=image_right, depth=depth))

    def prepare_loop_keyframe(self, keyframe, features):
        """Install worker features before geometry verification, at the latest pose.

        Called on the LoopClosing thread. Depth and features were computed in
        its detector process; only descriptor-map point creation stays here.
        """
        with self.map.update_lock:
            if keyframe.id not in self.map.keyframes_map:
                return False
            if keyframe.id in self._prepared_keyframes:
                return True
            index = self._keyframe_index.get(keyframe.id)
            if index is not None:
                # The single detector process returns results in queue order. An
                # earlier keyframe still unprepared now never got a result (its
                # task failed), so give it valid empty features instead of None.
                empty = {name: value[:0] if isinstance(value, np.ndarray) else None
                         for name, value in features.items()}
                while self._next_unprepared < index:
                    older = self._keyframes[self._next_unprepared]
                    self._next_unprepared += 1
                    if older.id not in self._prepared_keyframes:
                        self._install_features(older, empty)
                self._next_unprepared = max(self._next_unprepared, index + 1)
            self._install_features(keyframe, features)
            return True

    def _install_features(self, keyframe, features):
        """Caller holds map.update_lock."""
        for name, values in features.items():
            setattr(keyframe, name, values)
        count = 0 if keyframe.kps is None else len(keyframe.kps)
        keyframe.reset_points()
        if count and self.sensor_type != SensorType.MONOCULAR:
            TrackingCore.create_and_add_stereo_map_points_on_new_kf(
                keyframe, keyframe, self.map, keyframe.img)
            self._keyframe_points[keyframe.id] = [
                point for point in keyframe.get_points() if point is not None]
        elif count:
            self._create_monocular_points(keyframe)
            self._triangulate_monocular_points(keyframe)
        self._direct_observations.pop(keyframe.id, None)
        self._prepared_keyframes.add(keyframe.id)
        for neighbor in self._pending_links.pop(keyframe.id, ()):
            neighbor.add_connection(keyframe, kCovisibilityWeight)

    def _create_monocular_points(self, keyframe):
        """Caller holds map.update_lock. Give descriptor keypoints the 3D position of the
        direct landmark observed at the same pixel. One descriptor point per landmark is
        shared by every keyframe that observes it, so Sim(3) verification and covisibility
        see real point matches."""
        observations = self._direct_observations.get(keyframe.id)
        if observations is None:
            return
        ids, pixels, points_c = observations
        pixels = np.asarray(pixels, dtype=np.float64).reshape(-1, 2)
        points_c = np.asarray(points_c, dtype=np.float64).reshape(-1, 3)
        if len(ids) == 0:
            return
        kps = np.asarray(keyframe.kps, dtype=np.float64).reshape(-1, 2)
        distances, nearest = cKDTree(kps).query(pixels, distance_upper_bound=kMonoAssociationRadiusPx)
        # Each keypoint keeps its closest landmark.
        best = {}
        for j, (distance, k) in enumerate(zip(distances, nearest)):
            if np.isfinite(distance) and (k not in best or distance < best[k][0]):
                best[k] = (distance, j)
        twc = np.asarray(keyframe.Twc(), dtype=np.float64)
        created = []
        matches = []
        for k, (_distance, j) in best.items():
            k = int(k)
            landmark_id = int(ids[j])
            point = self._landmark_points.get(landmark_id)
            if point is None or point.is_bad():
                position = twc[:3, :3] @ points_c[j] + twc[:3, 3]
                point = MapPoint(position, kMonoPointColor, keyframe, k)
                self.map.add_point(point)
                self._landmark_points[landmark_id] = point
                created.append(point)
            if point.add_observation(keyframe, k):
                point.update_info()
                matches.append((k, landmark_id))
        self._keyframe_points[keyframe.id] = created
        self._landmark_matches[keyframe.id] = matches

    def _triangulate_monocular_points(self, keyframe):
        """Caller holds map.update_lock. Triangulate epipolar descriptor matches between
        keyframe and its previous prepared keyframes, at the direct poses."""
        index = self._keyframe_index.get(keyframe.id)
        if not index:
            return
        neighbors = [
            other for other in self._keyframes[max(0, index - kMonoTriangulationNeighbors):index]
            if other.id in self._prepared_keyframes and other.kps is not None and len(other.kps) > 0
        ]
        if not neighbors:
            return
        matches = compute_frame_matches(keyframe, neighbors, defaultdict(lambda: (None, None)),
                                        do_parallel=False, is_monocular=True)
        max_distance = 0.5 * self.feature_tracker.feature_manager.max_descriptor_distance
        created = self._keyframe_points.setdefault(keyframe.id, [])
        for other in neighbors:
            idxs_cur, idxs_other = matches[(keyframe, other)]
            if idxs_cur is None or len(idxs_cur) == 0:
                continue
            # Only keypoints without a map point on either side are returned.
            idxs_cur, idxs_other, found = EpipolarMatcher.search_frame_for_triangulation(
                keyframe, other, idxs_cur, idxs_other,
                max_descriptor_distance=max_distance, is_monocular=True)
            if found == 0 or len(idxs_cur) == 0:
                continue
            points, mask = triangulate_normalized_points(
                keyframe.pose(), other.pose(), keyframe.kpsn[idxs_cur], other.kpsn[idxs_other])
            _count, _mask, added = self.map.add_points(
                points, mask, keyframe, other, idxs_cur, idxs_other, keyframe.img, do_check=True)
            created.extend(added)

    def _monocular_scales(self, poses):
        """Per-keyframe scale of a Sim(3) correction, or None when no keyframe was rescaled.

        Runs while tracking is parked, so the direct map still holds the geometry from
        before the correction. Loop correction and global BA both move descriptor points
        with their keyframe's similarity, so a point seen from a corrected keyframe at
        distance d before is at d / s after. The median ratio over the keyframe's points
        recovers s; keyframes without support take the nearest estimate.
        """
        direct_twc, _ = self.direct_slam.keyframes()
        count = len(self._keyframes)
        if len(direct_twc) != count:
            return None
        scales = np.full(count, np.nan)
        for i, keyframe in enumerate(self._keyframes):
            tcw_new = poses[i]
            if np.max(np.abs(tcw_new @ np.asarray(direct_twc[i]) - np.eye(4))) < kPoseChangeTolerance:
                scales[i] = 1.0  # Not part of the correction.
                continue
            matches = self._landmark_matches.get(keyframe.id)
            if not matches:
                continue
            ids, _pixels, points_c = self.direct_slam.keyframe_observations(i)
            before = dict(zip(ids, np.asarray(points_c, dtype=np.float64).reshape(-1, 3)))
            ratios = []
            for k, landmark_id in matches:
                old = before.get(landmark_id)
                point = keyframe.get_point_match(k)
                if old is None or point is None or point.is_bad():
                    continue
                new = tcw_new[:3, :3] @ np.asarray(point.pt(), dtype=np.float64).reshape(3) + tcw_new[:3, 3]
                if old[2] > 0.0 and new[2] > 0.0:
                    ratios.append(np.linalg.norm(old) / np.linalg.norm(new))
            if len(ratios) >= kMinScaleSupport:
                scales[i] = float(np.median(ratios))
        valid = np.flatnonzero(np.isfinite(scales) & (scales > 0.0))
        if len(valid) == 0:
            return None
        missing = np.flatnonzero(~(np.isfinite(scales) & (scales > 0.0)))
        if len(missing):
            nearest = valid[np.abs(missing[:, None] - valid[None, :]).argmin(axis=1)]
            scales[missing] = scales[nearest]
        if np.max(np.abs(scales - 1.0)) < 1e-9:
            return None
        return scales.tolist()

    def _publish_poses(self):
        if not self._keyframes:
            return
        poses = []
        for keyframe in self._keyframes:
            pose = np.asarray(keyframe.Tcw(), dtype=np.float64)
            if pose.shape != (4, 4) or not np.isfinite(pose).all():
                raise ValueError("Loop-closing keyframe pose is not a finite 4x4 matrix")
            poses.append(pose)
        if self.sensor_type == SensorType.MONOCULAR:
            with self.map.update_lock:
                scales = self._monocular_scales(poses)
            if scales is not None:
                self.direct_slam.apply_keyframe_poses(poses, scales)
            else:
                self.direct_slam.apply_keyframe_poses(poses)
        else:
            self.direct_slam.apply_keyframe_poses(poses)
        # Runs while tracking is parked. Re-baseline the change detector on the
        # published poses so _sync_poses does not re-apply this correction, and
        # rebuild the viewer graph since loop correction may have rewired it.
        twc_poses, _ = self.direct_slam.keyframes()
        if len(twc_poses) == len(self._keyframes):
            self._synced_twc = np.asarray(twc_poses, dtype=np.float64).reshape(-1, 4, 4).copy()
        self._graph_revision += 1

    def _keyframe_graph_edges(self, keyframe, edges):
        """Append keyframe's drawable edges as index pairs. Caller holds map.update_lock."""
        index = self._keyframe_index
        own = index.get(keyframe.id)
        if own is None:
            return
        # Draw from the newer side: older keyframes link to a new one only after
        # its features arrive (see _install_features), while the new keyframe's
        # own links to older ones exist as soon as it is added.
        for other in keyframe.get_covisible_by_weight(
            int(Parameters.kMinWeightForDrawingCovisibilityEdge)
        ):
            if other.kid < keyframe.kid and other.id in index:
                edges["covisibility_graph"].append((index[other.id], own))
        parent = keyframe.get_parent()
        if parent is not None and parent.id in index:
            edges["spanning_tree"].append((own, index[parent.id]))
        for other in keyframe.get_loop_edges():
            if other.kid > keyframe.kid and other.id in index:
                edges["loops"].append((own, index[other.id]))

    def _append_graph_edges(self, keyframe):
        """Caller holds map.update_lock (tracking thread, in _add_keyframe)."""
        self._keyframe_graph_edges(keyframe, self._graph_edges)
        self._graph_arrays = None

    def graph_segments(self, twc_poses):
        """Use descriptor topology with the displayed direct camera centers.

        The caller owns the tracking gate, excluding pose correction/publication.
        Descriptor poses can differ after manual/local BA and are never used here.
        Topology is cached as index pairs: a new keyframe only appends its own
        edges, and only a published correction triggers a full rebuild.
        """
        revision = self._graph_revision
        if revision != self._graph_built_revision:
            edges = {name: [] for name in self._graph_edges}
            with self.map.update_lock:
                for keyframe in self._keyframes:
                    self._keyframe_graph_edges(keyframe, edges)
            self._graph_edges = edges
            self._graph_arrays = None
            self._graph_built_revision = revision
        if self._graph_arrays is None:
            self._graph_arrays = {
                name: np.asarray(pairs, dtype=np.int64).reshape(-1, 2)
                for name, pairs in self._graph_edges.items()}
        centers = np.asarray(twc_poses, dtype=np.float64).reshape(-1, 4, 4)[:, :3, 3]
        segments = {}
        for name, pairs in self._graph_arrays.items():
            pairs = pairs[(pairs < len(centers)).all(axis=1)]
            segments[name] = np.hstack((centers[pairs[:, 0]], centers[pairs[:, 1]]))
        return segments

    def reset(self):
        """Reset loop closing, the descriptor map and DirectSlam together.

        request_reset() waits for the LoopClosing thread, which may itself be
        waiting for tracking to go idle, so it runs before taking the gate. The
        map clears then run inside the gate, so no correction can publish poses
        into DirectSlam while it is being reset.
        """
        if self.loop_closing is not None:
            self.loop_closing.request_reset()
        if not self.acquire_tracking():
            return False
        try:
            with self.map.update_lock:
                self.map.reset()
                self._clear_state()
            self.direct_slam.reset()
        finally:
            self.finish_tracking()
        return True

    def quit(self):
        if self._has_quit:
            return
        self._has_quit = True
        self.local_mapping.quit()
        if self.loop_closing is not None:
            self.loop_closing.quit()
            self.loop_closing = None
