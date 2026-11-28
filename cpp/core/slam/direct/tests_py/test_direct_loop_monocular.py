"""Monocular loop-closing bridge: landmark association, late observations, fusion."""

from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyslam.config import Config  # Configure the compiled dependency paths.
from pyslam.io.dataset_types import SensorType
import pyslam.slam.direct_loop_closing as direct_loop_closing
from pyslam.slam.direct_loop_closing import DirectLoopClosing


class FakeKeyframe:
    """Descriptor keyframe with one map-point slot per keypoint."""

    def __init__(self, kid, kps, points=None):
        self.id = kid
        self.kps = np.asarray(kps, dtype=np.float64)
        self.slots = {}
        self.bad = False
        self.points = list(points or [])

    def Twc(self):
        return np.eye(4)

    def get_point_match(self, k):
        return self.slots.get(k)

    def is_bad(self):
        return self.bad

    def get_points(self):
        return self.points

    def get_matched_good_points(self):
        return [p for p in self.points if p is not None]


class FakePoint:
    """MapPoint stand-in: add_observation fills the keyframe slot, as KeyFrame does."""

    created = []

    def __init__(self, position, color, keyframe, k):
        self.position = np.asarray(position, dtype=np.float64)
        self.bad = False
        self.replacement = None
        self.observations = []
        FakePoint.created.append(self)

    def add_observation(self, keyframe, k):
        if keyframe.slots.get(k) is not None or any(kf is keyframe for kf, _ in self.observations):
            return False
        keyframe.slots[k] = self
        self.observations.append((keyframe, k))
        return True

    def update_info(self):
        pass

    def is_bad(self):
        return self.bad

    def get_replacement(self):
        return self.replacement


def observations(ids, pixels):
    """(ids, pixels, camera-frame points) as DirectSlam.keyframe_observations returns them."""
    points_c = [[0.0, 0.0, 10.0 + i] for i in ids]
    return list(ids), np.asarray(pixels, dtype=np.float64), np.asarray(points_c, dtype=np.float64)


class DirectLoopMonocularTest(unittest.TestCase):
    def setUp(self):
        FakePoint.created = []
        patcher = patch.object(direct_loop_closing, "MapPoint", FakePoint)
        patcher.start()
        self.addCleanup(patcher.stop)

    def bridge(self):
        bridge = DirectLoopClosing.__new__(DirectLoopClosing)
        bridge.camera = object()
        bridge.sensor_type = SensorType.MONOCULAR
        bridge.map = SimpleNamespace(update_lock=RLock(), keyframes_map={}, keyframe_origins=set(),
                                     add_keyframe=Mock(), add_point=Mock())
        bridge._graph_revision = 0
        bridge._clear_state()
        bridge._send_gray_right = True
        bridge._feature_config = {"num_features": 200}
        bridge.loop_closing = SimpleNamespace(add_keyframe=Mock())
        bridge.feature_tracker = SimpleNamespace(
            feature_manager=SimpleNamespace(max_descriptor_distance=100.0))
        return bridge

    def test_landmarks_attach_to_nearest_free_keypoint_and_refresh_adds_only_new_ones(self):
        bridge = self.bridge()
        keyframe = FakeKeyframe(5, [[10, 10], [50, 10], [90, 10], [130, 10]])
        # Landmark 1 is near keypoint 0, landmark 2 near keypoint 1, landmark 9 near none.
        first = observations([1, 2, 9], [[11, 10], [50, 12], [300, 300]])
        self.assertEqual(bridge._associate_landmarks(keyframe, first), 2)
        self.assertEqual(bridge._landmark_matches[5], [(0, 1), (1, 2)])
        self.assertEqual(len(FakePoint.created), 2)
        np.testing.assert_allclose(keyframe.slots[0].position, [0.0, 0.0, 11.0])

        # Keypoint 2 meanwhile got a triangulated point: a landmark near it must not replace it.
        triangulated = object()
        keyframe.slots[2] = triangulated
        # The refresh repeats landmarks 1 and 2 and adds 3 (near keypoint 2) and 4 (near 3).
        later = observations([1, 2, 3, 4], [[11, 10], [50, 12], [91, 10], [129, 11]])
        self.assertEqual(bridge._associate_landmarks(keyframe, later), 1)
        self.assertEqual(bridge._landmark_matches[5], [(0, 1), (1, 2), (3, 4)])
        self.assertIs(keyframe.slots[2], triangulated)
        self.assertEqual(len(FakePoint.created), 3)
        self.assertEqual(bridge._keyframe_points[5], FakePoint.created)

    def test_landmark_point_is_shared_across_keyframes_and_follows_replacement(self):
        bridge = self.bridge()
        first = FakeKeyframe(1, [[10, 10]])
        bridge._associate_landmarks(first, observations([7], [[10, 10]]))
        point = first.slots[0]
        # Fusion merged the landmark's point into another one.
        merged = FakePoint([0, 0, 1], None, first, 0)
        point.bad = True
        point.replacement = merged
        second = FakeKeyframe(2, [[20, 20]])
        bridge._associate_landmarks(second, observations([7], [[20, 21]]))
        self.assertIs(second.slots[0], merged)
        self.assertEqual(bridge._keyframe_points[2], [])  # Nothing new was created.

    def add_keyframes(self, bridge, count):
        image = np.zeros((40, 60), dtype=np.uint8)
        keyframes = []
        for kid in range(count):
            keyframe = Mock(id=kid)
            keyframe.get_covisible_by_weight.return_value = []
            keyframe.get_loop_edges.return_value = []
            keyframe.get_parent.return_value = None
            keyframe.is_bad.return_value = False
            keyframes.append(keyframe)
        with patch("pyslam.slam.direct_loop_closing.Frame"), \
             patch("pyslam.slam.direct_loop_closing.KeyFrame", side_effect=keyframes):
            for kid in range(count):
                bridge._add_keyframe(image, None, None, kid, 0.1 * kid, np.eye(4))
                bridge._submit_due_keyframes()
        return keyframes

    def test_monocular_keyframes_are_submitted_late_with_their_observations_at_that_time(self):
        bridge = self.bridge()
        captured = []

        def keyframe_observations(index):
            # One landmark whose id records when it was read; camera-frame point at z = 5.
            count = len(bridge._keyframes)
            captured.append((index, count))
            return [100 * index + count], np.zeros((1, 2)), np.array([[0.0, 0.0, 5.0]])

        bridge.direct_slam = SimpleNamespace(keyframe_observations=keyframe_observations)
        delay = direct_loop_closing.kMonoSubmitDelay
        keyframes = self.add_keyframes(bridge, delay + 3)
        submitted = [call.args[0] for call in bridge.loop_closing.add_keyframe.call_args_list]
        # In insertion order, each once kMonoSubmitDelay newer keyframes exist.
        self.assertEqual(submitted, keyframes[:3])
        self.assertEqual(captured, [(i, i + delay + 1) for i in range(3)])
        ids, _pixels, points_w = bridge._direct_observations[0]
        self.assertEqual(ids, [delay + 1])
        np.testing.assert_allclose(points_w, [[0.0, 0.0, 5.0]])  # Keyframe at the identity.
        self.assertEqual([entry[0] for entry in bridge._pending_submissions],
                         list(range(3, delay + 3)))

    def test_observations_are_captured_in_world_coordinates_at_the_current_pose(self):
        # Review finding 2: a camera-frame snapshot later combined with a newer keyframe pose
        # would misplace the point. The capture converts with the pose of the same instant.
        bridge = self.bridge()
        twc = np.eye(4)
        twc[:3, 3] = [1.0, 2.0, 3.0]
        bridge._synced_twc = twc.reshape(1, 4, 4).copy()
        bridge.direct_slam = SimpleNamespace(
            keyframe_observations=lambda i: ([7], np.zeros((1, 2)), np.array([[0.0, 0.0, 4.0]])))
        ids, _pixels, points_w = bridge._capture_observations(0)
        self.assertEqual(ids, [7])
        np.testing.assert_allclose(points_w, [[1.0, 2.0, 7.0]])

    def test_local_ba_moves_descriptor_points_onto_the_direct_landmarks(self):
        bridge = self.bridge()
        keyframe = FakeKeyframe(5, [[10, 10], [50, 10]])
        bridge._keyframes = [keyframe]
        bridge._associate_landmarks(keyframe, observations([1, 2], [[10, 10], [50, 10]]))
        moved = {1: np.array([0.5, 0.0, 9.0]), 2: np.array([-1.0, 0.2, 13.0])}
        bridge._synced_twc = np.eye(4).reshape(1, 4, 4).copy()
        bridge.direct_slam = SimpleNamespace(keyframe_observations=lambda i: (
            [1, 2], np.zeros((2, 2)), np.array([moved[1], moved[2]])))
        updates = {}
        for k, landmark_id in ((0, 1), (1, 2)):
            point = keyframe.slots[k]
            point.update_position = lambda p, landmark_id=landmark_id: updates.__setitem__(landmark_id, p)
            point.update_normal_and_depth = lambda *_args: None
        bridge._sync_landmark_points([0])
        np.testing.assert_allclose(updates[1], moved[1])
        np.testing.assert_allclose(updates[2], moved[2])

    def test_local_ba_refreshes_observations_waiting_for_worker_features(self):
        bridge = self.bridge()
        bridge._keyframes = [FakeKeyframe(5, [[10, 10]])]
        bridge._synced_twc = np.eye(4).reshape(1, 4, 4)
        bridge._direct_observations[5] = observations([7], [[10, 10]])
        moved = np.array([[0.5, 0.2, 9.0]])
        bridge.direct_slam = SimpleNamespace(keyframe_observations=lambda i: (
            [7], np.array([[10.0, 10.0]]), moved))
        bridge._sync_landmark_points([0])
        bridge._create_monocular_points(bridge._keyframes[0])
        np.testing.assert_allclose(bridge._landmark_points[7].position, moved[0])

    def test_local_ba_syncs_landmarks_even_when_their_observer_pose_is_fixed(self):
        bridge = self.bridge()
        keyframe = FakeKeyframe(5, [[10, 10]])
        bridge._keyframes = [keyframe]
        bridge._keyframe_index = {5: 0}
        bridge._synced_twc = np.eye(4).reshape(1, 4, 4)
        bridge._associate_landmarks(keyframe, observations([7], [[10, 10]]))
        point = bridge._landmark_points[7]
        point.update_position = lambda p: setattr(point, "position", p)
        point.update_normal_and_depth = lambda *_: None
        moved = np.array([[0.5, 0.2, 9.0]])
        bridge.direct_slam = SimpleNamespace(
            keyframes=lambda: ([np.eye(4)], [0.0]),
            last_ba=lambda: SimpleNamespace(applied=True, keyframes=1),
            keyframe_observations=lambda i: ([7], np.array([[10.0, 10.0]]), moved))
        bridge.note_tracked_frame(np.zeros((40, 60), np.uint8), None, None, 1, 0.1)
        np.testing.assert_allclose(point.position, moved[0])

    def test_late_refresh_waits_until_due_and_is_kept_for_unprepared_keyframes(self):
        bridge = self.bridge()
        bridge.direct_slam = SimpleNamespace(
            keyframe_observations=lambda i: ([i], np.zeros((1, 2)), np.array([[0.0, 0.0, 2.0]])))
        prepared, unprepared = Mock(id=10), Mock(id=11)
        prepared.is_bad.return_value = unprepared.is_bad.return_value = False
        bridge._keyframes = [prepared, unprepared]
        bridge._synced_twc = np.stack([np.eye(4), np.eye(4)])
        bridge._synced_twc[1, 0, 3] = 1.0
        bridge._prepared_keyframes = {10}
        bridge._observation_refreshes = [(0, 2), (1, 2), (1, 3)]
        with patch.object(DirectLoopClosing, "_associate_landmarks") as associate:
            bridge._refresh_landmark_associations()
        associate.assert_called_once()
        self.assertIs(associate.call_args.args[0], prepared)
        self.assertEqual(associate.call_args.args[1][0], [0])
        # Snapshot taken at the due count; _install_features() associates it later.
        ids, _pixels, points_w = bridge._direct_observations[11]
        self.assertEqual(ids, [1])
        np.testing.assert_allclose(points_w, [[1.0, 0.0, 2.0]])
        self.assertEqual(bridge._observation_refreshes, [(1, 3)])

    def relocalization_bridge(self, local_ok, global_ok):
        bridge = self.bridge()
        keyframes = [Mock(id=kid) for kid in range(14)]
        for keyframe in keyframes:
            keyframe.is_bad.return_value = False
        bridge._keyframes = keyframes
        bridge._prepared_keyframes = {kf.id for kf in keyframes if kf.id != 13}  # 13 not installed yet.
        landmark_point = FakePoint([1.0, 2.0, 9.0], None, keyframes[0], 0)
        landmark_point.id = 501
        triangulated = FakePoint([0.0, 1.0, 7.0], None, keyframes[0], 1)
        triangulated.id = 502
        bad = FakePoint([0.0, 0.0, 5.0], None, keyframes[0], 2)
        bad.id = 503
        bad.bad = True
        outlier = FakePoint([0.0, 0.0, 6.0], None, keyframes[0], 3)
        outlier.id = 504
        for point in (landmark_point, triangulated, bad, outlier):
            point.pt = lambda p=point: p.position
        bridge._landmark_points = {42: landmark_point}
        tcw = np.eye(4)
        tcw[0, 3] = 0.5
        frame = SimpleNamespace(
            get_points=lambda: [landmark_point, None, triangulated, bad, outlier],
            outliers=[False, False, False, False, True],
            kps=np.array([[10.0, 20.0], [0.0, 0.0], [30.0, 40.0], [1.0, 1.0], [2.0, 2.0]]),
            Tcw=lambda: tcw)
        calls = []
        relocalizer = SimpleNamespace(relocalize=lambda f, output, kf_map: calls.append(
            ("local", list(output.candidate_idxs))) or local_ok)
        bridge.loop_closing = SimpleNamespace(
            relocalizer=relocalizer, keyframes_map={},
            relocalize=lambda f, image: calls.append(("global",)) or global_ok)
        restarted = []
        bridge.direct_slam = SimpleNamespace(relocalize=lambda *args: restarted.append(args) or True)
        return bridge, frame, calls, restarted

    def test_relocalization_tries_recent_keyframes_then_the_database(self):
        bridge, frame, calls, restarted = self.relocalization_bridge(local_ok=False, global_ok=True)
        with patch.object(direct_loop_closing, "Frame", return_value=frame):
            self.assertTrue(bridge.relocalize(np.zeros((8, 8, 3), np.uint8), None, None, 77, 7.7))
        # Newest prepared keyframes first (13 has no features yet), at most kLocalRelocalizationKeyframes.
        expected_local = list(range(12, -1, -1))[:direct_loop_closing.kLocalRelocalizationKeyframes]
        self.assertEqual(calls, [("local", expected_local), ("global",)])
        _image, _right, _depth, img_id, timestamp, tcw, pixels, points_w, landmark_ids = restarted[0]
        self.assertEqual((img_id, timestamp), (77, 7.7))
        np.testing.assert_allclose(tcw[0, 3], 0.5)
        # The landmark's descriptor point re-acquires landmark 42; the triangulated point is new
        # (-1); the bad point and the outlier are left out.
        self.assertEqual(landmark_ids, [42, -1])
        np.testing.assert_allclose(pixels, [[10.0, 20.0], [30.0, 40.0]])
        np.testing.assert_allclose(points_w, [[1.0, 2.0, 9.0], [0.0, 1.0, 7.0]])

    def test_relocalization_skips_the_database_when_recent_keyframes_suffice(self):
        bridge, frame, calls, restarted = self.relocalization_bridge(local_ok=True, global_ok=False)
        with patch.object(direct_loop_closing, "Frame", return_value=frame):
            self.assertTrue(bridge.relocalize(np.zeros((8, 8, 3), np.uint8), None, None, 5, 0.5))
        self.assertEqual([call[0] for call in calls], ["local"])
        self.assertEqual(len(restarted), 1)

    def test_failed_relocalization_leaves_direct_slam_untouched(self):
        bridge, frame, calls, restarted = self.relocalization_bridge(local_ok=False, global_ok=False)
        with patch.object(direct_loop_closing, "Frame", return_value=frame):
            self.assertFalse(bridge.relocalize(np.zeros((8, 8, 3), np.uint8), None, None, 5, 0.5))
        self.assertEqual(restarted, [])

    def test_fusion_uses_recent_prepared_neighbors_in_both_directions(self):
        bridge = self.bridge()

        def point(bad=False):
            mock = Mock()  # Hashable, like MapPoint.
            mock.is_bad.return_value = bad
            return mock

        shared, own_a, own_b, culled = point(), point(), point(), point(bad=True)
        neighbors = [FakeKeyframe(kid, [[0, 0]], points=[shared, own_a if kid % 2 else own_b, None, culled])
                     for kid in range(direct_loop_closing.kMonoFuseNeighbors + 2)]
        current_point = point()
        current = FakeKeyframe(100, [[0, 0]], points=[current_point])
        bridge._keyframes = neighbors + [current]
        bridge._keyframe_index = {kf.id: i for i, kf in enumerate(bridge._keyframes)}
        neighbors[-1].bad = True
        unprepared = neighbors[-2]
        bridge._prepared_keyframes = {kf.id for kf in neighbors if kf is not unprepared}
        with patch.object(direct_loop_closing, "ProjectionMatcher") as matcher:
            bridge._fuse_monocular_points(current)
        calls = matcher.search_and_fuse.call_args_list
        recent = neighbors[-direct_loop_closing.kMonoFuseNeighbors:]
        expected = [kf for kf in recent if kf is not unprepared and not kf.bad]
        # Current points into each neighbor, then the neighbors' points into the current keyframe.
        self.assertEqual([call.args[1] for call in calls[:-1]], expected)
        self.assertTrue(all(call.args[0] == [current_point] for call in calls[:-1]))
        self.assertIs(calls[-1].args[1], current)
        self.assertEqual(set(calls[-1].args[0]), {shared, own_a, own_b})
        self.assertEqual(calls[-1].kwargs["max_descriptor_distance"], 50.0)

    def test_first_keyframe_has_no_neighbors_to_fuse(self):
        bridge = self.bridge()
        current = FakeKeyframe(0, [[0, 0]])
        bridge._keyframes = [current]
        bridge._keyframe_index = {0: 0}
        with patch.object(direct_loop_closing, "ProjectionMatcher") as matcher:
            bridge._fuse_monocular_points(current)
        matcher.search_and_fuse.assert_not_called()


if __name__ == "__main__":
    unittest.main()
