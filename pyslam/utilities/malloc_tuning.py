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

glibc heap tuning for per-frame image processing.

OpenCV/numpy allocate and free tens of MB of image buffers per frame (e.g. SIFT's
up-sampled float pyramid releases ~70 MB per call at 640x480). With glibc's defaults the
heap top is trimmed back to the kernel after every call (brk churn) and re-faulted on the
next one: ~17k minor page faults per SIFT call, ~10% of its time. glibc's adaptive mmap
threshold used to hide this only by accident, when some unrelated eagerly imported module
had freed a large block. These settings make the fast state explicit and deterministic.

Environment:
  PYSLAM_MALLOC_TUNING=0        disable
  PYSLAM_MALLOC_TRIM_MB=<int>   heap-top bytes kept before trimming (default 128)
  PYSLAM_MALLOC_MMAP_MB=<int>   allocations >= this use mmap (default 32, glibc's adaptive max)
"""

import ctypes
import os
import platform

_M_TRIM_THRESHOLD = -1
_M_MMAP_THRESHOLD = -3

kDefaultTrimThresholdMB = 128
kDefaultMmapThresholdMB = 32

_applied = None


def _env_int(name, default):
    try:
        return int(os.environ.get(name, default))
    except ValueError:
        return default


def configure_malloc():
    """Apply once per process. Returns the applied settings, or None if not applied."""
    global _applied
    if _applied is not None:
        return _applied or None
    _applied = {}
    if os.environ.get("PYSLAM_MALLOC_TUNING", "1").strip().lower() in {"0", "false", "no", "off"}:
        return None
    if platform.system() != "Linux":
        return None
    try:
        libc = ctypes.CDLL("libc.so.6")
        mallopt = libc.mallopt
    except (OSError, AttributeError):
        return None  # not glibc (e.g. musl)
    trim_mb = max(0, _env_int("PYSLAM_MALLOC_TRIM_MB", kDefaultTrimThresholdMB))
    mmap_mb = max(0, _env_int("PYSLAM_MALLOC_MMAP_MB", kDefaultMmapThresholdMB))
    # mallopt returns 1 on success. Setting M_MMAP_THRESHOLD disables glibc's adaptive
    # threshold, which is the intent: the behavior no longer depends on import history.
    ok_mmap = mallopt(_M_MMAP_THRESHOLD, mmap_mb << 20) == 1
    ok_trim = mallopt(_M_TRIM_THRESHOLD, trim_mb << 20) == 1
    if ok_mmap and ok_trim:
        _applied = {"trim_threshold_mb": trim_mb, "mmap_threshold_mb": mmap_mb}
    return _applied or None
