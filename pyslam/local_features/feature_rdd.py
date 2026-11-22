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

RDD: Robust Feature Detector and Descriptor using Deformable Transformer (Chen et al.,
CVPR 2025), https://github.com/xtcpete/rdd (Apache-2.0).

The code is a pinned source snapshot, not pip-installed. Its top-level package is named
`src`, so it is mounted under the private name `pyslam_vendor_rdd`: no generic `src`
module is registered and no sys.path entry is added. Multi-scale deformable attention
uses the upstream pure-PyTorch path when the optional CUDA extension is not compiled.
Weights: `weights/rdd.pth` from the authors' Google Drive folder (the release retrained
after their data-leakage fix), stored in torch.hub's checkpoint cache and checked
against a recorded SHA-256 before loading.
"""

import os
import sys
import types
import warnings
from threading import RLock

import cv2
import numpy as np
import torch

from pyslam.utilities.persistent_thread import PersistentThread

from .feature_base import BaseFeature2D

kRddRepo = "xtcpete/rdd:02ff3724279332eca395befd8a8ba5198f3484f0"
kRddWeightsDriveId = "18Xyzwtrk7dqppZzPz01Tpo6cdMBk_h6J"  # weights/rdd.pth
kRddWeightsSha256 = "b9f72267db638d7daa3a896d2c688d5fe02ea0a2acf39a1d4218305b4153b1d6"
kRddPackage = "pyslam_vendor_rdd"


def _import_rdd():
    from pyslam.loop_closing.global_feature_torchhub import github_snapshot

    if kRddPackage not in sys.modules:
        package = types.ModuleType(kRddPackage)
        package.__path__ = [os.path.join(github_snapshot(kRddRepo), "src")]
        sys.modules[kRddPackage] = package
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message=".*MultiScaleDeformableAttention CUDA extension.*")
        import importlib

        return importlib.import_module(kRddPackage + ".RDD")


def rdd_weights_path():
    """Cached weights, downloaded once (unless offline) and SHA-256 verified."""
    from pyslam.visual_models.weights import offline_requested, verify

    path = os.path.join(torch.hub.get_dir(), "checkpoints", "rdd.pth")
    if not os.path.isfile(path):
        if offline_requested():
            raise RuntimeError(f"RDD weights not cached at {path} (offline mode)")
        import gdown

        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        gdown.download(id=kRddWeightsDriveId, output=tmp, quiet=True)
        verify(tmp, kRddWeightsSha256)
        os.replace(tmp, path)
    return verify(path, kRddWeightsSha256)


# One model and one inference thread per device, shared by all RddFeature2D instances
# (e.g. the left/right stereo extractors): halves GPU memory (the PyTorch deformable
# attention is memory hungry; two instances ran out of memory at KITTI resolution on a
# 12 GB GPU) and serializes calls, which RDD's GPU load does anyway.
_shared = {}
_shared_lock = RLock()


def _shared_model(device):
    with _shared_lock:
        key = str(device)
        if key not in _shared:
            rdd = _import_rdd()
            model = rdd.build(weights=str(rdd_weights_path()))
            model = model.to(device) or model  # upstream to()/eval() overrides do not all return self
            model.eval()
            _shared[key] = (model, PersistentThread("rdd"))
        return _shared[key]


class RddFeature2D(BaseFeature2D):
    def __init__(self, num_features=2000, device=None, keypoint_size=1.0):
        self.lock = RLock()
        self.num_features = num_features
        self.keypoint_size = keypoint_size
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        # see pyslam/utilities/persistent_thread.py and _shared_model() above
        self.model, self.worker = _shared_model(self.device)
        self.frame = None
        self.kps = []
        self.des = None
        print(f"RddFeature2D: device {self.device}")

    def setMaxFeatures(self, num_features):
        self.num_features = num_features

    def extract(self, image):
        return self.worker.run(self._extract, image, self.num_features)

    @torch.inference_mode()
    def _extract(self, image, num_features):
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        # The shared model's top-k is set per call, on the worker thread (calls are serialized).
        if self.model.softdetect is None or self.model.softdetect.top_k != num_features:
            self.model.set_softdetect(top_k=num_features, scores_th=self.model.detection_threshold)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(self.device, non_blocking=True)
        tensor = tensor.permute(2, 0, 1)[None].float().div_(255.0)
        out = self.model.extract(tensor)[0]
        return out["keypoints"].float().cpu().numpy(), out["descriptors"].float().cpu().numpy()

    def detectAndCompute(self, frame, mask=None):  # mask is not supported (as for ALIKED)
        with self.lock:
            self.frame = frame
            pts, self.des = self.extract(frame)
            self.kps = list(cv2.KeyPoint.convert(np.ascontiguousarray(pts, dtype=np.float32),
                                                 size=self.keypoint_size))
            return self.kps, self.des

    def detect(self, frame, mask=None):
        with self.lock:
            self.detectAndCompute(frame)
            return self.kps

    def compute(self, frame, kps=None, mask=None):
        with self.lock:
            if self.frame is not frame:
                self.detectAndCompute(frame)
            return self.kps, self.des
