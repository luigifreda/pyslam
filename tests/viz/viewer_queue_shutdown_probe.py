"""Verify interpreter shutdown with visualization data still queued."""
import time

import numpy as np

from pyslam.viz.viewer3D import Viewer3D


class NonConsumingViewer(Viewer3D):
    def viewer_run(self, *args):
        self._is_running.value = 1
        self._is_looping.value = 1
        while self._is_running.value:
            time.sleep(0.01)


if __name__ == "__main__":
    viewer = NonConsumingViewer()
    try:
        viewer.wait_for_ready(timeout=30)
        for _ in range(8):
            viewer.qmap.put(np.zeros((65536, 3), np.float64))
    finally:
        viewer.quit()
    print("QUIT_RETURNED_WITH_PENDING_DATA", flush=True)
