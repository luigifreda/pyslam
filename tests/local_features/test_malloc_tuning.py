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
* glibc heap tuning applied by pyslam.config (see pyslam/utilities/malloc_tuning.py).
"""

import os
from pathlib import Path
import platform
import subprocess
import sys
import unittest

ROOT = Path(__file__).resolve().parents[2]

SIFT_FAULTS = """
import resource
import pyslam.config  # applies the tuning
from pyslam.utilities.malloc_tuning import configure_malloc
import numpy as np, cv2
cv2.setNumThreads(1)
rng = np.random.default_rng(17)
img = cv2.GaussianBlur(rng.integers(0, 256, (480, 640), dtype=np.uint8), (5, 5), 0.8)
sift = cv2.SIFT_create(2000)
for _ in range(2):
    sift.detectAndCompute(img, None)
before = resource.getrusage(resource.RUSAGE_SELF).ru_minflt
for _ in range(5):
    sift.detectAndCompute(img, None)
faults = (resource.getrusage(resource.RUSAGE_SELF).ru_minflt - before) / 5
print(configure_malloc(), faults)
"""


def run(code, **env):
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, text=True, capture_output=True,
                            timeout=120, env=dict(os.environ, CUDA_VISIBLE_DEVICES="", **env))
    if result.returncode != 0:
        raise AssertionError(result.stderr[-4000:])
    return result.stdout.strip().splitlines()[-1]


@unittest.skipUnless(platform.system() == "Linux", "glibc mallopt tuning is Linux-only")
class MallocTuningTest(unittest.TestCase):
    def test_applied_idempotent_and_configurable(self):
        code = ("from pyslam.utilities.malloc_tuning import configure_malloc\n"
                "first = configure_malloc(); assert configure_malloc() == first; print(first)")
        self.assertEqual(run(code), "{'trim_threshold_mb': 128, 'mmap_threshold_mb': 32}")
        self.assertEqual(run(code, PYSLAM_MALLOC_TRIM_MB="256", PYSLAM_MALLOC_MMAP_MB="16"),
                         "{'trim_threshold_mb': 256, 'mmap_threshold_mb': 16}")
        self.assertEqual(run(code, PYSLAM_MALLOC_TUNING="0"), "None")

    def test_per_frame_image_buffers_are_not_refaulted(self):
        # Without tuning, each SIFT call at 640x480 trims ~70 MB of heap and re-faults it
        # (~17k minor faults per call). With pyslam.config imported it must stay ~0.
        settings, faults = run(SIFT_FAULTS).rsplit(" ", 1)
        self.assertIn("trim_threshold_mb", settings)
        self.assertLess(float(faults), 200, "per-frame buffers are re-faulted: heap tuning lost")


if __name__ == "__main__":
    unittest.main()
