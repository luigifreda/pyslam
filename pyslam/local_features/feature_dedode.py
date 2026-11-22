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

DeDoDe v2 detector + DeDoDe descriptor, through kornia (Apache-2.0). Method and weights:
Edstedt et al., "DeDoDe: Detect, Don't Describe - Describe, Don't Detect" (3DV 2024) and
"DeDoDe v2" (CVPRW 2024), https://github.com/Parskatt/DeDoDe (MIT).

Detector "L-C4-v2" (DeDoDe v2: better under rotation, less keypoint clustering) and the
VGG-based "B-upright" 256-D descriptor (the DINOv2 "G" descriptor is heavier). Keypoints
are the deterministic top-k of the detector score map. Descriptors are L2-normalized
float32, so pyslam's L2 matchers and descriptor-distance thresholds apply.
"""

from threading import RLock

import cv2
import numpy as np
import torch

from pyslam.utilities.persistent_thread import PersistentThread

from .feature_base import BaseFeature2D

kVerbose = False


class DeDoDeFeature2D(BaseFeature2D):
    def __init__(self, num_features=2000, detector_weights="L-C4-v2", descriptor_weights="B-upright",
                 device=None, keypoint_size=1.0):
        import kornia.feature as KF

        self.lock = RLock()
        self.num_features = num_features
        self.keypoint_size = keypoint_size
        self.device = torch.device(device) if device is not None else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.model = KF.DeDoDe.from_pretrained(detector_weights=detector_weights,
                                               descriptor_weights=descriptor_weights).eval().to(self.device)
        # Inference always runs on this model's own long-lived thread (see persistent_thread.py):
        # calls from per-frame threads would otherwise rebuild CUDA/cuDNN state every time.
        self.worker = PersistentThread("dedode")
        self.frame = None
        self.kps = []
        self.des = None
        print(f"DeDoDeFeature2D: detector {detector_weights}, descriptor {descriptor_weights}, device {self.device}")

    def setMaxFeatures(self, num_features):
        self.num_features = num_features

    def _to_tensor(self, image):
        if image.ndim == 2:
            image = np.repeat(image[:, :, None], 3, axis=2)
        tensor = torch.from_numpy(np.ascontiguousarray(image)).to(self.device, non_blocking=True)
        return tensor.permute(2, 0, 1)[None].float().div_(255.0)

    def extract(self, image):
        return self.worker.run(self._extract, image)

    @torch.inference_mode()
    def _extract(self, image):
        keypoints, _, descriptors = self.model(self._to_tensor(image), n=self.num_features)
        keypoints = keypoints[0].float().cpu().numpy()
        descriptors = torch.nn.functional.normalize(descriptors[0].float(), p=2, dim=-1).cpu().numpy()
        return keypoints, descriptors

    def detectAndCompute(self, frame, mask=None):  # mask is not supported (as for ALIKED)
        with self.lock:
            self.frame = frame
            pts, self.des = self.extract(frame)
            self.kps = cv2.KeyPoint.convert(np.ascontiguousarray(pts, dtype=np.float32), size=self.keypoint_size)
            self.kps = list(self.kps)
            if kVerbose:
                print("detector: DEDODE, descriptor: DEDODE, #features:", len(self.kps), ", frame res:", frame.shape[0:2])
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
