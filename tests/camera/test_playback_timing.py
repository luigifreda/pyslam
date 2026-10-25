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

import json
from pathlib import Path
import tempfile
import unittest

from pyslam.utilities.playback_timing import PlaybackTiming


class TestPlaybackTiming(unittest.TestCase):
    def test_stage_totals_and_fps_exclude_idle_between_frames(self):
        times = iter([0., .01, .03, .035, 10., 10.02, 10.03])
        timing = PlaybackTiming("unused.json", warmup_frames=1, clock=lambda: next(times))
        timing.begin(0)
        timing.mark("loading")
        timing.mark("tracking")
        timing.finish()
        timing.begin(1)
        timing.mark("tracking")
        timing.finish()
        summary = timing.summarize(timing.samples)
        self.assertAlmostEqual(summary["stages_ms"]["total"]["mean"], 32.5)
        self.assertAlmostEqual(summary["stages_ms"]["loading"]["mean"], 5.)
        self.assertAlmostEqual(summary["fps"], 1000 / 32.5)
        for row in timing.samples:
            stages = sum(v for k, v in row.items() if k not in ("frame_id", "total"))
            self.assertAlmostEqual(stages, row["total"])

    def test_discarded_or_paused_iterations_are_not_frames(self):
        timing = PlaybackTiming("unused.json", clock=lambda: 1.)
        timing.begin(3)
        timing.discard()
        timing.mark("pacing")
        timing.finish()
        self.assertEqual(timing.summarize(timing.samples), {"frames": 0})

    def test_report_persists_metadata_and_warmup_selection(self):
        times = iter([0., .1, 1., 1.02])
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "nested/report.json"
            timing = PlaybackTiming(path, {"headless": False}, warmup_frames=1,
                                    clock=lambda: next(times))
            for frame in range(2):
                timing.begin(frame)
                timing.finish()
            timing.write()
            report = json.loads(path.read_text())
            self.assertFalse(path.with_name("report.json.tmp").exists())
        self.assertEqual(report["metadata"], {"headless": False})
        self.assertEqual(report["all_frames"]["frames"], 2)
        self.assertEqual(report["after_warmup"]["frames"], 1)
        self.assertAlmostEqual(report["after_warmup"]["fps"], 50.)


if __name__ == "__main__":
    unittest.main()
