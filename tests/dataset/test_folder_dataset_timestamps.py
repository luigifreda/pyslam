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

"""Tests for FolderDataset filename timestamp parsing."""

import contextlib
import io
import os
import tempfile
import unittest
from unittest import TestCase

import cv2
import numpy as np

from pyslam.io.dataset import (
    FolderDataset,
    folder_image_timestamps,
    timestamp_from_image_filename,
)
from pyslam.io.dataset_types import DatasetType, SensorType


class TestTimestampFromImageFilename(TestCase):
    def test_integer_stem(self):
        self.assertEqual(timestamp_from_image_filename("/data/000123.png"), 123.0)

    def test_fractional_tum_stem(self):
        self.assertAlmostEqual(
            timestamp_from_image_filename("1305031102.175304.png"),
            1305031102.175304,
        )

    def test_non_numeric_stem(self):
        self.assertIsNone(timestamp_from_image_filename("frame_001.png"))

    def test_uses_basename_not_parent_digits(self):
        self.assertIsNone(timestamp_from_image_filename("/run/123/image.png"))


class TestFolderImageTimestamps(TestCase):
    def test_integer_sequence_and_last_frame(self):
        ts, nxt = folder_image_timestamps("000.png", "001.png", 0.1, 0.0)
        self.assertEqual((ts, nxt), (0.0, 1.0))
        ts, nxt = folder_image_timestamps("002.png", None, 0.1, 1.0)
        self.assertEqual((ts, nxt), (2.0, 2.1))

    def test_fractional_stems(self):
        ts, nxt = folder_image_timestamps(
            "1305031102.175304.png", "1305031102.211214.png", 0.1, 0.0
        )
        self.assertAlmostEqual(ts, 1305031102.175304)
        self.assertAlmostEqual(nxt, 1305031102.211214)

    def test_non_numeric_falls_back_to_fps(self):
        ts, nxt = folder_image_timestamps("frame_001.png", "frame_002.png", 0.2, 0.0)
        self.assertAlmostEqual(ts, 0.2)
        self.assertAlmostEqual(nxt, 0.4)

    def test_next_name_not_numeric(self):
        ts, nxt = folder_image_timestamps("10.png", "frame_11.png", 0.1, 0.0)
        self.assertAlmostEqual(ts, 10.0)
        self.assertAlmostEqual(nxt, 10.1)


class TestFolderDatasetGetImageTimestamps(TestCase):
    def _write_images(self, folder, names):
        img = np.zeros((8, 8, 3), dtype=np.uint8)
        for name in names:
            self.assertTrue(cv2.imwrite(os.path.join(folder, name), img), name)

    def _read_all(self, folder, timestamps=None, fps=10):
        with contextlib.redirect_stdout(io.StringIO()):
            dataset = FolderDataset(
                folder,
                "*.png",
                sensor_type=SensorType.MONOCULAR,
                fps=fps,
                timestamps=timestamps,
                type=DatasetType.FOLDER,
            )
        out = []
        frame_id = 0
        while True:
            image = dataset.getImage(frame_id)
            if image is None:
                break
            out.append((dataset.getTimestamp(), dataset.getNextTimestamp()))
            frame_id += 1
        return out

    def _assert_timestamp_pairs(self, got, expected):
        self.assertEqual(len(got), len(expected))
        for (ts, nxt), (exp_ts, exp_nxt) in zip(got, expected):
            self.assertAlmostEqual(ts, exp_ts)
            self.assertAlmostEqual(nxt, exp_nxt)

    def test_last_integer_frame_does_not_raise(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write_images(folder, ["000.png", "001.png", "002.png"])
            got = self._read_all(folder)
        self._assert_timestamp_pairs(got, [(0.0, 1.0), (1.0, 2.0), (2.0, 2.1)])

    def test_fractional_filename_stems(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write_images(
                folder, ["1305031102.175304.png", "1305031102.211214.png"]
            )
            got = self._read_all(folder)
        self.assertEqual(len(got), 2)
        self.assertAlmostEqual(got[0][0], 1305031102.175304)
        self.assertAlmostEqual(got[0][1], 1305031102.211214)
        self.assertAlmostEqual(got[1][0], 1305031102.211214)
        self.assertAlmostEqual(got[1][1], 1305031102.211214 + 0.1)

    def test_timestamps_file_overrides_filenames(self):
        with tempfile.TemporaryDirectory() as folder:
            self._write_images(folder, ["000.png", "001.png"])
            stamps_name = "times.txt"
            with open(os.path.join(folder, stamps_name), "w") as f:
                f.write("1.5\n2.7\n")
            got = self._read_all(folder, timestamps=stamps_name)
        self._assert_timestamp_pairs(got, [(1.5, 2.7), (2.7, 2.8)])


if __name__ == "__main__":
    unittest.main()
