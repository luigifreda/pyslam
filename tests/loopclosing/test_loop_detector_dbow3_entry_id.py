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

"""Tests for LoopDetectorDBoW3 entry-id mapping.

The detector must key map_entry_id_to_frame_id on the id returned by the
database, resync the Python counter on mismatch, and skip unmapped query hits
instead of raising KeyError.
"""

import unittest
from types import SimpleNamespace
from unittest import TestCase

import numpy as np

from pyslam.config import Config

_config = Config()

from pyslam.loop_closing.loop_detector_base import (
    LoopDetectorBase,
    LoopDetectorOutput,
    LoopDetectorTask,
    LoopDetectorTaskType,
)
from pyslam.loop_closing.loop_detector_dbow3 import LoopDetectorDBoW3


class _FakeBow:
    def toVec(self):
        return np.array([1.0, 0.0], dtype=np.float32)


class _FakeResult:
    def __init__(self, entry_id, score=0.9):
        self.id = entry_id
        self.score = score


class _FakeDb:
    def __init__(self, next_id=0):
        self.next_id = next_id
        self.query_results = []

    def addBowVector(self, vec):
        entry_id = self.next_id
        self.next_id += 1
        return entry_id

    def query(self, vec, max_results=5, max_id=-1):
        return list(self.query_results)


class _StubLoopDetectorDBoW3(LoopDetectorDBoW3):
    def __init__(self, db=None):
        LoopDetectorBase.__init__(self)
        self.db = db if db is not None else _FakeDb()
        self.voc = None

    def compute_global_des(self, local_des, img):
        return _FakeBow()


def _make_task(frame_id, task_type):
    dummy = SimpleNamespace(
        id=frame_id,
        kps=[],
        angles=[],
        sizes=[],
        octaves=[],
        des=[],
        img=None,
        g_des=None,
    )
    task = LoopDetectorTask(dummy, None, task_type=task_type)
    task.keyframe_data.id = frame_id
    task.keyframe_data.g_des = None
    task.keyframe_data.des = []
    task.keyframe_data.img = None
    return task


class TestLoopDetectorDBoW3EntryId(TestCase):
    def test_maps_on_database_assigned_id(self):
        detector = _StubLoopDetectorDBoW3(_FakeDb(next_id=0))
        out = detector.run_task(_make_task(10, LoopDetectorTaskType.COMPUTE_GLOBAL_DES))
        self.assertIsInstance(out, LoopDetectorOutput)
        self.assertEqual(detector.map_entry_id_to_frame_id[0], 10)
        self.assertEqual(detector.entry_id, 1)

    def test_resyncs_when_database_id_diverges(self):
        detector = _StubLoopDetectorDBoW3(_FakeDb(next_id=7))
        detector.entry_id = 2
        detector.run_task(_make_task(42, LoopDetectorTaskType.COMPUTE_GLOBAL_DES))
        self.assertEqual(detector.map_entry_id_to_frame_id[7], 42)
        self.assertNotIn(2, detector.map_entry_id_to_frame_id)
        self.assertEqual(detector.entry_id, 8)

    def test_reloc_skips_unmapped_query_hits(self):
        detector = _StubLoopDetectorDBoW3(_FakeDb(next_id=0))
        detector.run_task(_make_task(1, LoopDetectorTaskType.COMPUTE_GLOBAL_DES))
        detector.run_task(_make_task(2, LoopDetectorTaskType.COMPUTE_GLOBAL_DES))
        detector.db.query_results = [_FakeResult(0, 0.8), _FakeResult(99, 0.7)]

        out = detector.run_task(_make_task(100, LoopDetectorTaskType.RELOCALIZATION))
        self.assertEqual(out.candidate_idxs, [1])
        self.assertEqual(out.candidate_scores, [0.8])


if __name__ == "__main__":
    unittest.main()
