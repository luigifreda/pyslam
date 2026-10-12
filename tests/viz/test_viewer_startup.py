"""Regression checks for isolated viewer startup and failure reporting (no display required)."""
import subprocess
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


class FailedViewer(Viewer3D):
    def viewer_run(self, *args):
        raise SystemExit(7)


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


if __name__ == "__main__":
    unittest.main()
