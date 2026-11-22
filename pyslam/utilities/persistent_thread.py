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

Run GPU inference on one long-lived thread per model.

PyTorch keeps part of its CUDA state per thread (e.g. cuDNN execution-plan caches). A
call from a new thread rebuilds it for every convolution. The SLAM front end extracts
stereo features on threads created per frame (Frame / C++ core), so a network pays that
setup on every frame: measured at KITTI resolution, DeDoDe 175 ms -> ~2.8 s and ALIKED
20 ms -> ~45 ms per call from a fresh thread. Forwarding the call to a persistent
worker keeps those caches warm; the hand-off costs microseconds.
"""

from concurrent.futures import ThreadPoolExecutor
import threading


class PersistentThread:
    def __init__(self, name="pyslam-inference"):
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix=name)
        self._worker_ident = None
        # Start the worker now and record its identity, to run re-entrant calls inline.
        self._executor.submit(self._register).result()

    def _register(self):
        self._worker_ident = threading.get_ident()

    def run(self, fn, *args, **kwargs):
        """Run fn on the persistent worker and return its result (exceptions propagate)."""
        if threading.get_ident() == self._worker_ident:
            return fn(*args, **kwargs)  # already on the worker: avoid a self-deadlock
        return self._executor.submit(fn, *args, **kwargs).result()

    def shutdown(self):
        self._executor.shutdown(wait=False)
