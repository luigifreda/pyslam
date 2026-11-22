"""main_slam_direct.py: the viewers stay open after the last frame until the user closes them."""

import os
import sys
import unittest
from unittest.mock import Mock, patch

from pyslam.config import Config  # Configure the compiled dependency paths.

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")))
import main_slam_direct  # noqa: E402


class FakeDisplay:
    """Closed after `open_polls` checks, or when 'q' is queued in the image window."""

    def __init__(self, open_polls, keys=()):
        self.polls = 0
        self.open_polls = open_polls
        self.keys = list(keys)
        self.viewer = Mock()
        self.viewer.is_closed.side_effect = self.closed
        self.images = Mock()
        self.images.get_key.side_effect = lambda: self.keys.pop(0) if self.keys else None
        self.capture = Mock(return_value="snapshot")
        self.draw = Mock()

    def closed(self):
        self.polls += 1
        return self.polls > self.open_polls


class MainSlamDirectViewerTest(unittest.TestCase):
    def test_final_report_and_export_follow_loop_closing_shutdown(self):
        from types import SimpleNamespace
        from pyslam.io.dataset_types import SensorType

        events = []
        dataset = SimpleNamespace(sensor_type=SensorType.RGBD, is_ok=False,
                                  close=lambda: events.append("dataset closed"))
        config = SimpleNamespace(cam_settings={}, dataset_settings={})
        camera = SimpleNamespace(is_distorted=False)
        slam = Mock()
        loop = Mock()
        loop.quit.side_effect = lambda: events.append("final loop correction")
        with patch.object(sys, "argv", ["main_slam_direct.py", "--headless",
                                       "--trajectory-file", "/tmp/unused-trajectory.txt"]), \
             patch.object(main_slam_direct, "Config", return_value=config), \
             patch.object(main_slam_direct, "dataset_factory", return_value=dataset), \
             patch.object(main_slam_direct, "camera_from_config", return_value=camera), \
             patch.object(main_slam_direct, "USE_CPP", True), \
             patch.object(main_slam_direct.Parameters, "kUseLoopClosing", True), \
             patch("pyslam.slam.cpp.DirectSlam", return_value=slam), \
             patch("pyslam.slam.direct_loop_closing.DirectLoopClosing", return_value=loop), \
             patch("pyslam.io.ground_truth.groundtruth_factory", return_value=None), \
             patch("pyslam.io.ground_truth.is_valid_groundtruth", return_value=False), \
             patch.object(main_slam_direct, "save_trajectory", side_effect=lambda *_: events.append("export")), \
             patch.object(main_slam_direct, "report", side_effect=lambda *_: events.append("report")):
            main_slam_direct.main()
        self.assertEqual(events, ["final loop correction", "dataset closed", "export", "report"])

    def test_monocular_viewer_scale_follows_the_direct_initial_map_scale(self):
        from types import SimpleNamespace
        from pyslam.config_parameters import Parameters
        from pyslam.io.dataset_types import SensorType

        slam = SimpleNamespace(monocular_init_median_depth=6.0)
        mono = SimpleNamespace(scale_viewer_3d=0.1, sensor_type=SensorType.MONOCULAR)
        self.assertAlmostEqual(main_slam_direct.viewer_scale(mono, slam),
                               0.1 * 6.0 / Parameters.kInitializerDesiredMedianDepth)
        # Metric maps keep the dataset scale.
        for sensor_type in (SensorType.STEREO, SensorType.RGBD):
            dataset = SimpleNamespace(scale_viewer_3d=0.1, sensor_type=sensor_type)
            self.assertEqual(main_slam_direct.viewer_scale(dataset, slam), 0.1)

    def test_quit_is_requested_by_closing_the_viewer_or_pressing_q(self):
        self.assertFalse(main_slam_direct.display_quit_requested(FakeDisplay(open_polls=1)))
        self.assertTrue(main_slam_direct.display_quit_requested(FakeDisplay(open_polls=0)))
        self.assertTrue(main_slam_direct.display_quit_requested(FakeDisplay(open_polls=5, keys=["q"])))
        self.assertFalse(main_slam_direct.display_quit_requested(FakeDisplay(open_polls=5, keys=["x"])))

    def test_waits_until_closed_and_redraws_inside_the_tracking_gate(self):
        display = FakeDisplay(open_polls=3)
        events = []
        loop_closing = Mock()
        loop_closing.acquire_tracking.side_effect = lambda: events.append("acquire") or True
        loop_closing.finish_tracking.side_effect = lambda: events.append("finish")
        display.capture.side_effect = lambda *args: events.append("capture") or "snapshot"
        slam = object()
        main_slam_direct.wait_for_viewer_close(
            display, slam, loop_closing, ("image", 41, 4.1), period=0.0, redraw_every=0.0)
        self.assertEqual(display.polls, 4)  # Returned as soon as the viewer reported closed.
        self.assertEqual(events, ["acquire", "capture", "finish"] * 3)
        display.capture.assert_called_with(slam, "image", 41, 4.1, loop_closing)
        display.draw.assert_called_with("snapshot", 41, refresh_fps=False)

    def test_q_in_the_image_window_ends_the_wait(self):
        display = FakeDisplay(open_polls=100, keys=[None, "q"])
        main_slam_direct.wait_for_viewer_close(display, object(), None, ("image", 1, 0.1),
                                               period=0.0, redraw_every=0.0)
        self.assertEqual(display.draw.call_count, 1)

    def test_without_a_tracked_frame_or_after_quit_nothing_is_captured(self):
        display = FakeDisplay(open_polls=2)
        main_slam_direct.wait_for_viewer_close(display, object(), None, None, period=0.0, redraw_every=0.0)
        display.capture.assert_not_called()
        # The loop-closing gate refuses once it has quit: stop waiting instead of drawing.
        display = FakeDisplay(open_polls=100)
        loop_closing = Mock()
        loop_closing.acquire_tracking.return_value = False
        main_slam_direct.wait_for_viewer_close(display, object(), loop_closing, ("image", 1, 0.1),
                                               period=0.0, redraw_every=0.0)
        display.capture.assert_not_called()
        loop_closing.finish_tracking.assert_not_called()



class ImageUndistorterTest(unittest.TestCase):
    @staticmethod
    def camera(distorted=True, raw=False):
        import numpy as np
        from types import SimpleNamespace

        return SimpleNamespace(
            K=np.array([[100.0, 0.0, 32.0], [0.0, 100.0, 24.0], [0.0, 0.0, 1.0]]),
            D=np.array([0.2, -0.1, 0.001, 0.001, 0.0]) if distorted else np.zeros(5),
            width=64, height=48, is_distorted=distorted, projects_raw_keypoints=lambda: raw,
            u_min=-3.0, u_max=67.0, v_min=-2.0, v_max=50.0)

    def test_only_a_distorted_pinhole_is_undistorted(self):
        from pyslam.io.dataset_types import SensorType

        self.assertIsNone(main_slam_direct.ImageUndistorter.for_camera(self.camera(False), SensorType.RGBD))
        # Raw-pixel models (e.g. Kannala-Brandt) are tracked on distorted images directly.
        self.assertIsNone(main_slam_direct.ImageUndistorter.for_camera(self.camera(True, raw=True), SensorType.RGBD))
        with self.assertRaises(SystemExit):
            main_slam_direct.ImageUndistorter.for_camera(self.camera(), SensorType.STEREO)
        camera = self.camera()
        undistorter = main_slam_direct.ImageUndistorter.for_camera(camera, SensorType.MONOCULAR)
        self.assertIsNotNone(undistorter)
        self.assertFalse(camera.is_distorted)
        self.assertFalse(any(camera.D))
        self.assertEqual((camera.u_min, camera.u_max, camera.v_min, camera.v_max), (0.0, 64.0, 0.0, 48.0))

    def test_images_match_opencv_and_depth_is_never_blended(self):
        import cv2
        import numpy as np
        from pyslam.io.dataset_types import SensorType

        camera = self.camera()
        K, D = camera.K.copy(), camera.D.copy()
        undistorter = main_slam_direct.ImageUndistorter.for_camera(camera, SensorType.RGBD)
        rng = np.random.default_rng(1)
        image = rng.integers(0, 256, (48, 64, 3), dtype=np.uint8)
        expected = cv2.undistort(image, K, D, None, K)
        diff = np.abs(undistorter.image(image).astype(int) - expected.astype(int))
        self.assertLessEqual(np.percentile(diff, 99), 1)
        depth = np.where(np.arange(64)[None, :] < 32, 1.0, 5.0).astype(np.float32).repeat(48, axis=0)
        values = set(np.unique(undistorter.depth(depth)).tolist())
        self.assertTrue(values <= {0.0, 1.0, 5.0}, values)  # 0 only outside the source image.
        self.assertIsNone(undistorter.depth(None))

if __name__ == "__main__":
    unittest.main()
