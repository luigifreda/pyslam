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
"""

import os
import sys

import cv2
import numpy as np


def _load_module():
    try:
        import data_thread_reader as module
    except ImportError:
        lib_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "cpp", "lib"))
        if lib_dir not in sys.path:
            sys.path.insert(0, lib_dir)
        import data_thread_reader as module
    return module


def make_stream(
    name,
    paths,
    imread_flag=cv2.IMREAD_COLOR,
    loader="imread",
    ops=None,
    expected_size=None,
):
    """Describe one prefetch stream. `paths` is the absolute file list in frame order."""
    spec = {
        "name": name,
        "paths": [os.path.abspath(os.fspath(path)) if path else "" for path in paths],
        "imread_flag": int(imread_flag),
        "loader": loader,
        "ops": list(ops or []),
    }
    if expected_size is not None:
        spec["expected_size"] = (int(expected_size[0]), int(expected_size[1]))
    return spec


class ImagePrefetcher:
    """Bounded background decoder. Each `get` returns a NumPy view of the decoded buffer."""

    def __init__(self, streams, capacity=6):
        module = _load_module()
        self._impl = module.ImagePrefetcher(int(capacity))
        self._streams = set()
        for spec in streams:
            name = spec["name"]
            self._streams.add(name)
            self._impl.add_stream(
                name,
                list(spec["paths"]),
                int(spec.get("imread_flag", cv2.IMREAD_COLOR)),
                spec.get("loader", "imread") == "npy",
            )
            if spec.get("expected_size") is not None:
                width, height = spec["expected_size"]
                self._impl.set_expected_size(name, int(width), int(height))
            for op in spec.get("ops", []):
                kind = op["type"]
                if kind == "clahe":
                    self._impl.add_clahe(name, float(op.get("clip", 3.0)), int(op.get("tile", 8)))
                elif kind == "remap":
                    map2 = op.get("map2")
                    if map2 is None:
                        map2 = np.empty((0, 0), dtype=np.float32)
                    self._impl.add_remap(
                        name, np.ascontiguousarray(op["map1"]), np.ascontiguousarray(map2)
                    )
                elif kind == "resize":
                    self._impl.add_resize(name, int(op["width"]), int(op["height"]))
                elif kind == "gray_to_rgb":
                    self._impl.add_gray_to_rgb(name)
                elif kind == "scale_float":
                    self._impl.add_scale_float(name, float(op["factor"]))
                else:
                    raise ValueError(f"Unknown prefetch op: {kind}")
        self._impl.start()

    def has_stream(self, name):
        return name in self._streams

    def get(self, name, frame_id):
        return self._impl.get(name, int(frame_id))

    def stream_length(self, name):
        return int(self._impl.stream_length(name))

    def unpublished_count(self):
        return int(self._impl.unpublished_count())

    def close(self):
        impl = self._impl
        self._impl = None
        if impl is not None:
            impl.stop()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
