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
* Small subprocess tests; no SLAM imports or large memory allocations.
"""

import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from benchmark_guard import BenchmarkGuard


class TestBenchmarkGuard(unittest.TestCase):
    def test_invalid_limits_cannot_disable_guard(self):
        for value in (float('nan'), float('inf'), -1., 0.):
            with self.subTest(value=value), self.assertRaises(ValueError):
                BenchmarkGuard(max_rss_gib=value)

    def test_second_process_cannot_overlap_and_lock_is_released(self):
        with tempfile.TemporaryDirectory() as tmp:
            lock = Path(tmp) / 'guard.lock'
            code = ("from benchmark_guard import BenchmarkGuard; "
                    f"guard=BenchmarkGuard(lock_path={str(lock)!r}, min_available_gib=0); "
                    "guard.__enter__(); guard.__exit__(None,None,None)")
            with BenchmarkGuard(lock_path=lock, min_available_gib=0):
                result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                                        text=True, capture_output=True, timeout=5)
                self.assertNotEqual(result.returncode, 0)
                self.assertIn('run tests sequentially', result.stderr)
            result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                                    text=True, capture_output=True, timeout=5)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_watchdog_interrupts_small_over_budget_child(self):
        with tempfile.TemporaryDirectory() as tmp:
            code = f"""
from benchmark_guard import BenchmarkGuard
import time
with BenchmarkGuard(max_rss_gib=0.032, min_available_gib=0,
                    lock_path={str(Path(tmp) / 'guard.lock')!r}, interval=0.02):
    allocation = bytearray(40 * 1024**2)
    time.sleep(3)
"""
            result = subprocess.run([sys.executable, '-c', code], cwd=Path(__file__).parent,
                                    text=True, capture_output=True, timeout=5)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('BENCHMARK_MEMORY_STOP', result.stderr)


if __name__ == '__main__':
    unittest.main()
