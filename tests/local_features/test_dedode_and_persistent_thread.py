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
* PersistentThread (GPU inference on one long-lived thread) and the DeDoDe wrapper contract.
"""

import os
import threading
import unittest

import numpy as np

import pyslam.config  # noqa: F401  (native library paths)
from pyslam.utilities.persistent_thread import PersistentThread


class PersistentThreadTest(unittest.TestCase):
    def test_calls_from_any_thread_run_on_one_worker(self):
        worker = PersistentThread("test")
        self.addCleanup(worker.shutdown)
        idents = [worker.run(threading.get_ident)]
        for _ in range(3):
            thread = threading.Thread(target=lambda: idents.append(worker.run(threading.get_ident)))
            thread.start()
            thread.join()
        self.assertEqual(len(set(idents)), 1)
        self.assertNotEqual(idents[0], threading.get_ident())

    def test_exceptions_propagate_and_reentrant_calls_do_not_deadlock(self):
        worker = PersistentThread("test")
        self.addCleanup(worker.shutdown)
        with self.assertRaisesRegex(ValueError, "boom"):
            worker.run(lambda: (_ for _ in ()).throw(ValueError("boom")))
        self.assertEqual(worker.run(lambda: worker.run(lambda: 42)), 42)


class PersistentThreadAdaptorTest(unittest.TestCase):
    def test_detector_calls_run_on_one_worker_and_attributes_pass_through(self):
        from pyslam.local_features.feature_manager_adaptors import PersistentThreadAdaptor

        class Detector:
            num_features = 7

            def __init__(self):
                self.threads = []

            def detectAndCompute(self, frame, mask=None):
                self.threads.append(threading.get_ident())
                return [frame], mask

            def detect(self, frame, mask=None):
                return self.detectAndCompute(frame, mask)[0]

            def compute(self, frame, kps=None, mask=None):
                return kps, frame

            def setMaxFeatures(self, n):
                self.num_features = n

        inner = Detector()
        adaptor = PersistentThreadAdaptor(inner, "test")
        self.addCleanup(adaptor._worker.shutdown)
        self.assertEqual(adaptor.detectAndCompute(1, mask=2), ([1], 2))
        thread = threading.Thread(target=lambda: adaptor.detect(3))
        thread.start()
        thread.join()
        self.assertEqual(adaptor.compute(4, kps=[5]), ([5], 4))
        self.assertEqual(len(set(inner.threads)), 1)
        adaptor.setMaxFeatures(11)
        self.assertEqual(adaptor.num_features, 11)

    def test_flag_controls_wrapping_of_learned_detectors(self):
        from pyslam.config_parameters import Parameters
        from pyslam.local_features import feature_manager as fm
        from pyslam.local_features.feature_types import FeatureDetectorTypes

        self.assertIn(FeatureDetectorTypes.ALIKED, fm.kPersistentThreadDetectorTypes)
        self.assertNotIn(FeatureDetectorTypes.ORB2, fm.kPersistentThreadDetectorTypes)
        self.assertNotIn(FeatureDetectorTypes.DEDODE, fm.kPersistentThreadDetectorTypes)  # has its own worker
        self.assertNotIn(FeatureDetectorTypes.RDD, fm.kPersistentThreadDetectorTypes)  # has its own worker
        self.assertIsInstance(Parameters.kFeatureExtractionPersistentThread, bool)


def _dedode_weights_cached():
    import torch

    folder = os.path.join(torch.hub.get_dir(), "checkpoints")
    return all(os.path.isfile(os.path.join(folder, name))
               for name in ("dedode_detector_L_v2.pth", "dedode_descriptor_B.pth"))


@unittest.skipUnless(_dedode_weights_cached(), "DeDoDe checkpoints not cached (no download in tests)")
class DeDoDeContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import cv2
        from pyslam.local_features.feature_dedode import DeDoDeFeature2D

        cls.extractor = DeDoDeFeature2D(num_features=300)
        rng = np.random.default_rng(1)
        cls.image = cv2.GaussianBlur(rng.integers(0, 256, (120, 200, 3), dtype=np.uint8), (5, 5), 1.0)

    def test_output_contract(self):
        kps, des = self.extractor.detectAndCompute(self.image)
        self.assertEqual(len(kps), 300)
        self.assertEqual(des.shape, (300, 256))
        self.assertEqual(des.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(des, axis=1), 1.0, rtol=1e-4)
        pts = np.float32([k.pt for k in kps])
        self.assertTrue(np.all((pts >= 0) & (pts < [200, 120])))
        gray_kps, gray_des = self.extractor.detectAndCompute(self.image[:, :, 0])  # grayscale accepted
        self.assertEqual(gray_des.shape[1], 256)

    def test_deterministic_and_identical_from_a_fresh_thread(self):
        kps, des = self.extractor.detectAndCompute(self.image)
        ref = (np.float32([k.pt for k in kps]), des.copy())
        out = {}

        def run():
            k, d = self.extractor.detectAndCompute(self.image)
            out["result"] = (np.float32([x.pt for x in k]), d)

        thread = threading.Thread(target=run)
        thread.start()
        thread.join()
        np.testing.assert_array_equal(out["result"][0], ref[0])
        np.testing.assert_array_equal(out["result"][1], ref[1])

    def test_feature_budget_is_respected(self):
        self.extractor.setMaxFeatures(120)
        self.addCleanup(self.extractor.setMaxFeatures, 300)
        self.assertEqual(len(self.extractor.detectAndCompute(self.image)[0]), 120)



def _rdd_weights_cached():
    import torch

    return os.path.isfile(os.path.join(torch.hub.get_dir(), "checkpoints", "rdd.pth"))


@unittest.skipUnless(_rdd_weights_cached(), "RDD weights not cached (no download in tests)")
class RddContractTest(unittest.TestCase):
    def test_output_contract_and_fresh_thread_determinism(self):
        import cv2
        from pyslam.local_features.feature_rdd import RddFeature2D

        os.environ.setdefault("PYSLAM_OFFLINE", "1")
        extractor = RddFeature2D(num_features=200)
        rng = np.random.default_rng(2)
        image = cv2.GaussianBlur(rng.integers(0, 256, (128, 192, 3), dtype=np.uint8), (5, 5), 1.0)
        kps, des = extractor.detectAndCompute(image)
        self.assertLessEqual(len(kps), 200)
        self.assertEqual(des.shape, (len(kps), 256))
        self.assertEqual(des.dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(des, axis=1), 1.0, rtol=1e-4)
        out = {}
        thread = threading.Thread(target=lambda: out.update(r=extractor.detectAndCompute(image)))
        thread.start()
        thread.join()
        np.testing.assert_array_equal(np.float32([k.pt for k in out["r"][0]]), np.float32([k.pt for k in kps]))
        np.testing.assert_array_equal(out["r"][1], des)


if __name__ == "__main__":
    unittest.main()
