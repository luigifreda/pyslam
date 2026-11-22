"""Regression checks for deferred direct-keyframe feature preparation."""

from threading import RLock
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from pyslam.config import Config  # Configure the compiled dependency paths.
from pyslam.io.dataset_types import SensorType
from pyslam.slam.direct_loop_closing import DirectLoopClosing


class DirectLoopFeaturesTest(unittest.TestCase):
    def bridge(self, keyframe=None):
        bridge = DirectLoopClosing.__new__(DirectLoopClosing)
        bridge.camera = object()
        bridge.sensor_type = SensorType.RGBD
        bridge.map = SimpleNamespace(update_lock=RLock(), keyframes_map={},
                                     keyframe_origins=set(), add_keyframe=Mock())
        bridge._graph_revision = 0
        bridge._clear_state()
        bridge._send_gray_right = True
        bridge._feature_config = {"num_features": 200}
        bridge.loop_closing = SimpleNamespace(add_keyframe=Mock())
        if keyframe is not None:
            bridge.map.keyframes_map[keyframe.id] = keyframe
        return bridge

    def test_monocular_scale_is_recovered_from_descriptor_points(self):
        # Keyframe 0 is not corrected (scale 1). Keyframe 1 is corrected with s = 2: its
        # points end up at half their camera distance. Keyframe 2 changed but has no point
        # support, so it takes the nearest estimate.
        bridge = self.bridge()
        bridge.sensor_type = SensorType.MONOCULAR
        rng = np.random.default_rng(3)
        old_points_c = rng.uniform([-3, -2, 4], [3, 2, 20], size=(12, 3))
        ids = list(range(100, 112))
        old_twc = [np.eye(4) for _ in range(3)]
        for i, twc in enumerate(old_twc):
            twc[2, 3] = float(i)
        new_tcw = [np.linalg.inv(twc) for twc in old_twc]
        new_tcw[1][:3, 3] += [0.3, 0.0, -0.2]
        new_tcw[2][:3, 3] += [0.0, 0.5, 0.0]

        keyframes = []
        for i in range(3):
            matches, points = [], {}
            if i < 2:
                scale = 2.0 if i == 1 else 1.0
                new_twc = np.linalg.inv(new_tcw[i])
                for k, (landmark_id, x_c) in enumerate(zip(ids, old_points_c)):
                    world = new_twc[:3, :3] @ (x_c / scale) + new_twc[:3, 3]
                    points[k] = SimpleNamespace(pt=lambda w=world: w, is_bad=lambda: False)
                    matches.append((k, landmark_id))
            keyframe = SimpleNamespace(id=10 + i, get_point_match=points.get)
            keyframes.append(keyframe)
            bridge._landmark_matches[keyframe.id] = matches
        bridge._keyframes = keyframes
        bridge.direct_slam = SimpleNamespace(
            keyframes=lambda: (old_twc, [0.0, 0.1, 0.2]),
            keyframe_observations=lambda i: (ids, np.zeros((12, 2)), old_points_c))
        # Keyframe 0 keeps its pose exactly.
        new_tcw[0] = np.linalg.inv(old_twc[0])
        scales = bridge._monocular_scales(new_tcw)
        # Keyframe 2 has no support and is outside every supported span: it keeps scale 1
        # instead of copying its neighbor's (review finding 2).
        np.testing.assert_allclose(scales, [1.0, 2.0, 1.0], rtol=1e-9)

        # Nothing rescaled: plain SE3 publication.
        unchanged = [np.linalg.inv(twc) for twc in old_twc]
        self.assertIsNone(bridge._monocular_scales(unchanged))

    def test_unsupported_keyframe_scale_is_interpolated_between_supported_ones(self):
        # Keyframe 0 is unchanged (scale 1), keyframe 2 is corrected with s = 3 and has point
        # support, keyframe 1 changed without support: it takes the interpolated scale 2.
        bridge = self.bridge()
        bridge.sensor_type = SensorType.MONOCULAR
        rng = np.random.default_rng(5)
        old_points_c = rng.uniform([-3, -2, 4], [3, 2, 20], size=(12, 3))
        ids = list(range(200, 212))
        old_twc = [np.eye(4) for _ in range(3)]
        for i, twc in enumerate(old_twc):
            twc[2, 3] = float(i)
        new_tcw = [np.linalg.inv(twc) for twc in old_twc]
        new_tcw[1][:3, 3] += [0.2, 0.0, 0.0]
        new_tcw[2][:3, 3] += [0.0, 0.4, 0.0]
        keyframes = []
        for i in range(3):
            matches, points = [], {}
            if i == 2:
                new_twc = np.linalg.inv(new_tcw[i])
                for k, (landmark_id, x_c) in enumerate(zip(ids, old_points_c)):
                    world = new_twc[:3, :3] @ (x_c / 3.0) + new_twc[:3, 3]
                    points[k] = SimpleNamespace(pt=lambda w=world: w, is_bad=lambda: False)
                    matches.append((k, landmark_id))
            keyframe = SimpleNamespace(id=20 + i, get_point_match=points.get)
            keyframes.append(keyframe)
            bridge._landmark_matches[keyframe.id] = matches
        bridge._keyframes = keyframes
        bridge.direct_slam = SimpleNamespace(
            keyframes=lambda: (old_twc, [0.0, 0.1, 0.2]),
            keyframe_observations=lambda i: (ids, np.zeros((12, 2)), old_points_c))
        np.testing.assert_allclose(bridge._monocular_scales(new_tcw), [1.0, 2.0, 3.0], rtol=1e-9)

    def test_initialization_keeps_the_reference_frame_however_long_it_takes(self):
        # Review finding 5: the reference image must never be evicted, and another image must
        # never be paired with the reference pose.
        bridge = self.bridge()
        bridge.sensor_type = SensorType.MONOCULAR
        reference = 3
        bridge.direct_slam = SimpleNamespace(init_reference_img_id=lambda: reference)
        frames = {}
        for img_id in range(250):
            frames[img_id] = np.full((4, 6, 3), img_id % 256, dtype=np.uint8)
            if img_id == 200:
                reference = 200  # The initializer replaced its reference with this frame.
            bridge._remember_init_frame(frames[img_id], None, None, img_id, 0.1 * img_id)
            if img_id == 150:
                self.assertEqual(list(bridge._pending_init_frames), [3])
        self.assertEqual(list(bridge._pending_init_frames), [200])
        np.testing.assert_array_equal(bridge._pending_init_frames[200][0], frames[200])

    def test_a_missing_reference_image_gives_a_blank_keyframe_not_the_current_image(self):
        bridge = self.bridge()
        bridge.sensor_type = SensorType.MONOCULAR
        bridge.direct_slam = SimpleNamespace(
            keyframes=lambda: ([np.eye(4), np.eye(4)], [0.0, 0.5]),
            keyframe_img_ids=lambda: [0, 5])
        current = np.full((4, 6, 3), 99, dtype=np.uint8)
        added = []
        with patch.object(DirectLoopClosing, "_add_keyframe",
                          side_effect=lambda image, *args: added.append((image, args[2]))):
            bridge.note_tracked_frame(current, None, None, 5, 0.5)
        self.assertEqual([img_id for _image, img_id in added], [0, 5])
        self.assertFalse(np.any(added[0][0]))  # Blank: no descriptors for the reference pose.
        self.assertIs(added[1][0], current)

    def test_registration_queues_raw_input_without_extracting_features(self):
        bridge = self.bridge()
        image = np.zeros((40, 60), dtype=np.uint8)
        depth = np.ones(image.shape, dtype=np.float32)
        keyframe = Mock(id=7)
        keyframe.get_covisible_by_weight.return_value = []
        keyframe.get_loop_edges.return_value = []
        with patch("pyslam.slam.direct_loop_closing.Frame") as frame, \
             patch("pyslam.slam.direct_loop_closing.KeyFrame", return_value=keyframe), \
             patch("pyslam.slam.direct_loop_closing.TrackingCore") as core:
            bridge._add_keyframe(image, None, depth, 3, 0.1, np.eye(4))
            self.assertIsNone(frame.call_args.args[1])
            self.assertNotIn("depth", frame.call_args.kwargs)
            core.create_and_add_stereo_map_points_on_new_kf.assert_not_called()
        request = bridge.loop_closing.add_keyframe.call_args.kwargs["feature_extraction"]
        self.assertIs(request["depth"], depth)
        self.assertIs(request["camera"], bridge.camera)
        self.assertEqual(bridge._keyframes, [keyframe])

    def test_spanning_tree_parent_is_fixed_at_registration(self):
        # update_connections() would otherwise re-parent a keyframe to kf_max (often a
        # newer descendant) during loop correction, creating a spanning-tree cycle.
        bridge = self.bridge()
        image = np.zeros((40, 60), dtype=np.uint8)
        keyframes = []
        for kid in (1, 2):
            keyframe = Mock(id=kid, is_first_connection=True)
            keyframe.get_covisible_by_weight.return_value = []
            keyframe.get_loop_edges.return_value = []
            keyframe.get_parent.return_value = None
            keyframes.append(keyframe)
        with patch("pyslam.slam.direct_loop_closing.Frame"), \
             patch("pyslam.slam.direct_loop_closing.KeyFrame", side_effect=keyframes):
            for kid in (1, 2):
                bridge._add_keyframe(image, None, None, kid, 0.1 * kid, np.eye(4))
        for keyframe in keyframes:
            self.assertIs(keyframe.is_first_connection, False)
        keyframes[1].set_parent.assert_called_once_with(keyframes[0])

    def test_result_uses_latest_pose_and_creates_points_only_once(self):
        keyframe = SimpleNamespace(id=7, img=np.zeros((40, 60)), pose="corrected",
                                   reset_points=Mock(), get_points=lambda: [None])
        bridge = self.bridge(keyframe)
        features = dict(kps=np.array([[10., 20.]]), depths=np.array([4.]))
        with patch("pyslam.slam.direct_loop_closing.TrackingCore") as core:
            def create(frame, kf, map_, image):
                self.assertIs(frame, keyframe)
                self.assertIs(kf, keyframe)
                self.assertEqual(kf.pose, "corrected")
                np.testing.assert_array_equal(kf.depths, [4.])
            core.create_and_add_stereo_map_points_on_new_kf.side_effect = create
            self.assertTrue(bridge.prepare_loop_keyframe(keyframe, features))
            self.assertTrue(bridge.prepare_loop_keyframe(keyframe, features))
            self.assertEqual(core.create_and_add_stereo_map_points_on_new_kf.call_count, 1)
            self.assertEqual(keyframe.reset_points.call_count, 1)

    def test_sync_moves_only_own_points_of_changed_keyframes_and_never_reapplies(self):
        def make_keyframe(kid):
            kf = SimpleNamespace(id=kid, tcw=np.eye(4))
            kf.Tcw = lambda: kf.tcw
            kf.update_pose = lambda pose: setattr(kf, "tcw", np.array(pose, dtype=float))
            return kf

        def make_point(reference):
            point = SimpleNamespace(xyz=np.array([1., 2., 3.]))
            point.is_bad = lambda: False
            point.get_reference_keyframe = lambda: reference
            point.pt = lambda: point.xyz
            point.update_position = lambda xyz: setattr(point, "xyz", np.array(xyz))
            point.update_normal_and_depth = lambda force: None
            return point

        bridge = self.bridge()
        moved, fixed = make_keyframe(1), make_keyframe(2)
        bridge._keyframes = [moved, fixed]
        bridge._keyframe_index = {1: 0, 2: 1}
        bridge._synced_twc = np.stack([np.eye(4), np.eye(4)])
        moved_point, fixed_point = make_point(moved), make_point(fixed)
        bridge._keyframe_points = {1: [moved_point], 2: [fixed_point]}

        twc = np.eye(4)
        twc[:3, 3] = [0.5, 0., 0.]
        bridge._sync_poses([twc, np.eye(4)])
        np.testing.assert_allclose(moved.tcw, np.linalg.inv(twc))
        np.testing.assert_allclose(moved_point.xyz, [1.5, 2., 3.])
        np.testing.assert_allclose(fixed_point.xyz, [1., 2., 3.])
        bridge._sync_poses([twc, np.eye(4)])
        np.testing.assert_allclose(moved_point.xyz, [1.5, 2., 3.])

        # LoopClosing corrects descriptor poses and points itself, then publishes.
        corrected = np.eye(4)
        corrected[:3, 3] = [0., 2., 0.]
        moved.update_pose(np.linalg.inv(corrected))
        moved_point.xyz = np.array([1., 4., 3.])
        direct = {}
        bridge.direct_slam = SimpleNamespace(
            apply_keyframe_poses=lambda poses: direct.update(
                twc=[np.linalg.inv(pose) for pose in poses]),
            keyframes=lambda: (direct["twc"], [0., 0.]))
        bridge._publish_poses()
        bridge._sync_poses(direct["twc"])
        np.testing.assert_allclose(moved_point.xyz, [1., 4., 3.])
        np.testing.assert_allclose(moved.tcw, np.linalg.inv(corrected))

    def test_task_sends_neighbor_images_only_when_the_worker_may_read_them(self):
        from pyslam.loop_closing.loop_detector_base import LoopDetectorTask

        def make(kid, g_des):
            return SimpleNamespace(id=kid, kps=[], angles=[], sizes=[], octaves=[], des=[],
                                   img=np.zeros((2, 2), np.uint8), g_des=g_des,
                                   is_bad=lambda: False)

        current = make(9, None)
        known, missing = make(1, [0.5]), make(2, None)
        standard = LoopDetectorTask(current, current.img, covisible_keyframes=[known, missing])
        self.assertIs(standard.keyframe_data.img, current.img)
        self.assertEqual([kf.img is None for kf in standard.covisible_keyframes_data], [True, False])

        # The direct worker caches neighbor global descriptors itself.
        direct = LoopDetectorTask(current, current.img, covisible_keyframes=[known, missing],
                                  feature_extraction={})
        self.assertIs(direct.keyframe_data.img, current.img)
        self.assertTrue(all(kf.img is None for kf in direct.covisible_keyframes_data))

    def test_reset_discards_inflight_result_and_empty_features_are_supported(self):
        keyframe = SimpleNamespace(id=7, reset_points=Mock())
        bridge = self.bridge()
        with patch("pyslam.slam.direct_loop_closing.TrackingCore") as core:
            self.assertFalse(bridge.prepare_loop_keyframe(keyframe, dict(kps=None)))
            bridge.map.keyframes_map[7] = keyframe
            self.assertTrue(bridge.prepare_loop_keyframe(keyframe, dict(kps=None)))
            core.create_and_add_stereo_map_points_on_new_kf.assert_not_called()

    def test_neighbors_link_back_only_after_features_and_failed_tasks_get_empty_features(self):
        bridge = self.bridge()

        def make(kid):
            keyframe = Mock(id=kid, kid=kid, kps=None, img=None)
            keyframe.get_covisible_by_weight.return_value = []
            keyframe.get_loop_edges.return_value = []
            keyframe.get_points.return_value = []
            bridge.map.keyframes_map[kid] = keyframe
            return keyframe

        keyframes = [make(kid) for kid in range(3)]
        with patch("pyslam.slam.direct_loop_closing.Frame"), \
             patch("pyslam.slam.direct_loop_closing.KeyFrame", side_effect=keyframes), \
             patch("pyslam.slam.direct_loop_closing.CameraPose"):
            for _ in keyframes:
                bridge._add_keyframe(np.zeros((4, 4), np.uint8), None, None, 0, 0.0, np.eye(4))
        first, failed, current = keyframes
        # Newer -> older links exist immediately; older -> newer wait for features.
        current.add_connection.assert_any_call(first, 100)
        current.add_connection.assert_any_call(failed, 100)
        first.add_connection.assert_not_called()

        features = dict(kps=np.ones((2, 2), np.float32), des=np.ones((2, 32), np.uint8))
        with patch("pyslam.slam.direct_loop_closing.TrackingCore"):
            self.assertTrue(bridge.prepare_loop_keyframe(first, features))
            # The detector never returned a result for `failed`.
            self.assertTrue(bridge.prepare_loop_keyframe(current, features))
        self.assertEqual(failed.kps.shape, (0, 2))
        self.assertEqual(failed.des.shape, (0, 32))
        self.assertEqual(failed.des.dtype, np.uint8)
        self.assertEqual(bridge._prepared_keyframes, {0, 1, 2})
        first.add_connection.assert_any_call(failed, 100)
        first.add_connection.assert_any_call(current, 100)
        failed.add_connection.assert_any_call(current, 100)

    def test_reset_requests_loop_reset_first_then_clears_everything_inside_the_gate(self):
        from pyslam.slam.direct_loop_closing import DirectLoopGate

        bridge = self.bridge()
        events = []
        bridge.local_mapping = DirectLoopGate(lambda: events.append("publish"))
        bridge.loop_closing = SimpleNamespace(
            request_reset=lambda: events.append(("request_reset", bridge.local_mapping.is_idle())))
        bridge.map.reset = lambda: events.append(("map", bridge.local_mapping.is_idle()))
        bridge.direct_slam = SimpleNamespace(
            reset=lambda: events.append(("slam", bridge.local_mapping.is_idle())))
        bridge._keyframes.append(Mock(id=1))
        bridge._keyframe_index[1] = 0
        bridge._prepared_keyframes.add(1)
        bridge._pending_links[1] = [Mock()]
        bridge._next_unprepared = 1

        self.assertTrue(bridge.reset())
        # request_reset waits for LoopClosing, so it must run with the gate free.
        self.assertEqual(events, [("request_reset", True), ("map", False), ("slam", False)])
        self.assertTrue(bridge.local_mapping.is_idle())
        self.assertEqual((bridge._keyframes, bridge._keyframe_index, bridge._prepared_keyframes,
                          bridge._pending_links, bridge._next_unprepared), ([], {}, set(), {}, 0))

    def test_reset_waits_for_a_running_correction_and_skips_after_quit(self):
        import threading
        from pyslam.slam.direct_loop_closing import DirectLoopGate

        bridge = self.bridge()
        events = []
        bridge.local_mapping = DirectLoopGate(lambda: events.append("publish"))
        bridge.loop_closing = SimpleNamespace(request_reset=lambda: None)
        bridge.map.reset = lambda: None
        bridge.direct_slam = SimpleNamespace(reset=lambda: events.append("slam reset"))

        # A loop correction holds the stop; reset must not clear the map under it.
        bridge.local_mapping.request_stop()
        bridge.local_mapping.wait_idle()
        resetter = threading.Thread(target=bridge.reset)
        resetter.start()
        resetter.join(0.2)
        self.assertTrue(resetter.is_alive())
        self.assertEqual(events, [])
        bridge.local_mapping.release()
        resetter.join(2.0)
        self.assertFalse(resetter.is_alive())
        self.assertEqual(events, ["publish", "slam reset"])

        bridge.local_mapping.quit()
        self.assertFalse(bridge.reset())
        self.assertEqual(events, ["publish", "slam reset"])


if __name__ == "__main__":
    unittest.main()
