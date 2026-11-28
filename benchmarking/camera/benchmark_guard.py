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
* Serialize camera benchmarks and stop replays before exhausting host memory.
"""

import fcntl
import math
import os
from pathlib import Path
import signal
import sys
import tempfile
import threading


def memory_bytes():
    """Read resident process memory and host headroom on Linux."""
    status = dict(line.split(':', 1) for line in Path('/proc/self/status').read_text().splitlines()
                  if ':' in line)
    meminfo = dict(line.split(':', 1) for line in Path('/proc/meminfo').read_text().splitlines())
    return int(status['VmRSS'].split()[0]) * 1024, int(meminfo['MemAvailable'].split()[0]) * 1024


class BenchmarkGuard:
    def __init__(self, max_rss_gib=8., min_available_gib=8., lock_path=None, interval=0.5):
        if (not all(math.isfinite(v) for v in (max_rss_gib, min_available_gib, interval))
                or max_rss_gib <= 0 or min_available_gib < 0 or interval <= 0):
            raise ValueError('Invalid benchmark memory limits')
        self.max_rss = max_rss_gib * 1024**3
        self.min_available = min_available_gib * 1024**3
        self.lock_path = Path(lock_path) if lock_path else Path(tempfile.gettempdir()) / (
            f'pyslam-camera-benchmark-{os.getuid()}.lock')
        self.interval = interval
        self.peak_rss = 0
        self._stop = threading.Event()
        self._file = None
        self._thread = None

    def check(self):
        rss, available = memory_bytes()
        self.peak_rss = max(self.peak_rss, rss)
        if rss > self.max_rss:
            raise MemoryError(f'Benchmark RSS {rss / 1024**3:.2f} GiB exceeds '
                              f'{self.max_rss / 1024**3:.2f} GiB budget')
        if available < self.min_available:
            raise MemoryError(f'Host memory headroom {available / 1024**3:.2f} GiB is below '
                              f'{self.min_available / 1024**3:.2f} GiB reserve')

    def _watch(self):
        while not self._stop.wait(self.interval):
            try:
                self.check()
            except (MemoryError, OSError) as error:
                print(f'BENCHMARK_MEMORY_STOP: {error}', file=sys.stderr, flush=True)
                os.kill(os.getpid(), signal.SIGINT)
                # A long native call may delay Python exception handling. Do not
                # leave the process allocating indefinitely during shutdown.
                if not self._stop.wait(10.):
                    os._exit(137)
                return

    def __enter__(self):
        self._file = self.lock_path.open('a+')
        try:
            fcntl.flock(self._file, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._file.close()
            self._file = None
            raise RuntimeError('Another camera benchmark/replay is running; run tests sequentially') from None
        try:
            self.check()
        except BaseException:
            self.__exit__(None, None, None)
            raise
        self._thread = threading.Thread(target=self._watch, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, *args):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=1.)
        if self._file:
            fcntl.flock(self._file, fcntl.LOCK_UN)
            self._file.close()
            self._file = None
