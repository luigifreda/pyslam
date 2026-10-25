"""Regression checks for isolated viewer startup and failure reporting (no display required)."""
import subprocess
import os
import signal
from pathlib import Path
import sys
import time
import unittest
from unittest.mock import patch

import numpy as np
import torch.multiprocessing as mp

from pyslam.config_parameters import Parameters
from pyslam.viz.viewer3D import Viewer3D


class QueueViewer(Viewer3D):
    def viewer_run(self, *args):
        self.qvo.put((mp.get_start_method(), "tensorflow" in sys.modules,
                      self.slam_drawer_thread is None))
        self._is_running.value = 1
        self._is_looping.value = 1
        self.qvo.put(self.qmap.get(timeout=10))
        while self._is_running.value:
            time.sleep(0.01)


class ParameterViewer(Viewer3D):
    def viewer_run(self, *args):
        import pyslam.viz.viewer3D as module

        self.qvo.put(
            (Parameters.kDenseMappingDtypeVertices, module.kMaxSparseMapPointsToVisualize)
        )
        self._is_running.value = 1
        self._is_looping.value = 1
        while self._is_running.value:
            time.sleep(0.01)


class FailedViewer(Viewer3D):
    def viewer_run(self, *args):
        raise SystemExit(7)


class SignalIgnoringViewer(Viewer3D):
    def viewer_run(self, *args):
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        self.qvo.put("handler ready")
        while True:
            time.sleep(1)


class StalledViewer(Viewer3D):
    def viewer_run(self, *args):
        time.sleep(60)


class ViewerStartupTests(unittest.TestCase):
    def test_main_import_does_not_load_optional_native_libraries(self):
        subprocess.run(
            [sys.executable, "-c", "import main_slam, sys; "
             "assert 'tensorflow' not in sys.modules; "
             "assert 'pyslam.depth_estimation.depth_estimator_factory' not in sys.modules"],
            check=True, timeout=30,
        )

    def test_spawn_with_draw_thread_and_map_transfer(self):
        original_start_method = mp.get_start_method()
        with patch.object(Parameters, "kViewerDrawSlamMapOnSeparateThread", True):
            viewer = QueueViewer()
        try:
            points = np.arange(3000, dtype=np.float64).reshape(-1, 3)
            viewer.qmap.put(points)
            viewer.wait_for_ready(timeout=30)
            self.assertEqual(viewer.qvo.get(timeout=5), ("spawn", False, True))
            np.testing.assert_array_equal(viewer.qvo.get(timeout=5), points)
            self.assertEqual(mp.get_start_method(), original_start_method)
        finally:
            viewer.quit()

    def test_spawn_preserves_runtime_viewer_parameters(self):
        with (
            patch.object(Parameters, "kDenseMappingDtypeVertices", "float64"),
            patch.object(Parameters, "kMaxSparseMapPointsToVisualize", 12345),
        ):
            viewer = ParameterViewer()
        try:
            viewer.wait_for_ready(timeout=30)
            self.assertEqual(viewer.qvo.get(timeout=5), ("float64", 12345))
        finally:
            viewer.quit()

    def test_shutdown_with_pending_visualization_data_exits(self):
        subprocess.run(
            [sys.executable, str(Path(__file__).with_name("viewer_queue_shutdown_probe.py"))],
            check=True, timeout=30,
        )

    @unittest.skipUnless(os.name == "posix", "POSIX signal behavior")
    def test_timeout_kills_child_that_ignores_termination(self):
        viewer = SignalIgnoringViewer()
        try:
            self.assertEqual(viewer.qvo.get(timeout=30), "handler ready")
            with self.assertRaises(TimeoutError):
                viewer.wait_for_ready(timeout=0.1)
            self.assertEqual(viewer.vp.exitcode, -signal.SIGKILL)
            if viewer.slam_drawer_thread is not None:
                self.assertFalse(viewer.slam_drawer_thread.draw_thread.is_alive())
        finally:
            viewer.quit()

    def test_failed_child_is_reported_without_waiting_for_timeout(self):
        viewer = FailedViewer()
        try:
            with self.assertRaisesRegex(RuntimeError, "exit code 7"):
                viewer.wait_for_ready(timeout=30)
        finally:
            viewer.quit()

    def test_stalled_child_is_terminated_on_timeout(self):
        viewer = StalledViewer()
        try:
            with self.assertRaises(TimeoutError):
                viewer.wait_for_ready(timeout=0.1)
            self.assertFalse(viewer.vp.is_alive())
        finally:
            viewer.quit()

    def test_startup_failure_stops_ground_truth_aligner(self):
        for viewer_class, error, timeout in (
            (FailedViewer, RuntimeError, 30),
            (StalledViewer, TimeoutError, 0.1),
        ):
            with self.subTest(viewer=viewer_class.__name__):
                viewer = viewer_class()
                try:
                    viewer.set_gt_trajectory(np.zeros((20, 3)), np.arange(20))
                    aligner = viewer.trajectory_aligner
                    deadline = time.monotonic() + 30
                    while not viewer.is_aligner_running.value and time.monotonic() < deadline:
                        time.sleep(0.01)
                    self.assertTrue(aligner.is_alive())
                    self.assertEqual(viewer.is_aligner_running.value, 1)

                    with self.assertRaises(error):
                        viewer.wait_for_ready(timeout=timeout)

                    self.assertFalse(aligner.is_alive())
                    self.assertIsNotNone(aligner.exitcode)
                    self.assertEqual(viewer.is_aligner_running.value, 0)
                finally:
                    viewer.quit()


if __name__ == "__main__":
    unittest.main()
