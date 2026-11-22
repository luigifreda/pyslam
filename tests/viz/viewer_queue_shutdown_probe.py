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
*
* Verify interpreter shutdown with visualization data still queued.
"""
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
