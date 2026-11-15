"""GBA correction must terminate on a corrupted (cyclic) spanning tree.

Regression: the spanning-tree walk in correct_after_GBA() had no visited set, so a
cycle (e.g. K-1 -> K -> K-1) made it loop forever while holding map.update_lock,
freezing tracking.
"""

import threading
import unittest
from threading import RLock
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np

from pyslam.config import Config  # Configure the compiled dependency paths.
from pyslam.slam.global_bundle_adjustment import GlobalBundleAdjustment


class FakeKeyFrame:
    """Hashable stand-in exposing only what the spanning-tree correction uses."""

    def __init__(self, kid):
        self.id = kid
        self.children = set()
        self.tcw = np.eye(4)
        self.num_updates = 0
        self.Tcw_GBA = None
        self.is_Tcw_GBA_valid = False
        self.GBA_kf_id = -1

    def get_children(self):
        return set(self.children)

    def Tcw(self):
        return self.tcw

    def Twc(self):
        return np.linalg.inv(self.tcw)

    def is_bad(self):
        return False

    def update_pose(self, pose):
        self.tcw = np.array(pose, dtype=float)
        self.num_updates += 1


make_keyframe = FakeKeyFrame


class GBASpanningTreeCycleTest(unittest.TestCase):
    def test_correction_terminates_and_updates_each_keyframe_once(self):
        keyframes = [make_keyframe(i) for i in range(4)]
        for parent, child in zip(keyframes, keyframes[1:]):
            parent.children.add(child)
        # Cycle: kf3 also lists kf2 as a child (kf2 re-parented to a descendant).
        keyframes[3].children.add(keyframes[2])

        loop_kf_id = 3
        gba = GlobalBundleAdjustment.__new__(GlobalBundleAdjustment)
        gba.use_multiprocessing = False
        gba.loop_kf_id = loop_kf_id
        gba._is_correcting = SimpleNamespace(value=0)
        gba.local_mapping = Mock()
        gba.local_mapping.queue_size.return_value = 0
        gba.map = SimpleNamespace(update_lock=RLock(),
                                  keyframe_origins=[keyframes[0]],
                                  get_keyframes=lambda: list(keyframes),
                                  get_points=lambda: [])
        # Only the origin was optimized; the rest get the propagated correction.
        corrected = np.eye(4)
        corrected[0, 3] = 1.0
        gba.result_dict_queue = [dict(keyframe_updates={0: corrected}, point_updates={})]

        result = {}
        worker = threading.Thread(
            target=lambda: result.setdefault("ok", gba.correct_after_GBA()), daemon=True)
        worker.start()
        worker.join(5.0)

        self.assertFalse(worker.is_alive(), "GBA spanning-tree walk did not terminate")
        self.assertTrue(result["ok"])
        self.assertEqual([kf.num_updates for kf in keyframes], [1, 1, 1, 1])
        for kf in keyframes:
            np.testing.assert_allclose(kf.tcw, corrected)
        gba.local_mapping.release.assert_called_once()
        self.assertEqual(gba._is_correcting.value, 0)


if __name__ == "__main__":
    unittest.main()
