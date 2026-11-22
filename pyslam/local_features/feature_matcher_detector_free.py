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

Detector-free pair matchers (no per-keypoint descriptors).

These matchers return matched keypoints for an image pair, but no descriptors that a map
point could keep. They fit frame-to-frame tracking, initialization, loop/relocalization
verification and main_feature_matching.py, like LoFTR/MASt3R, not descriptor-based
projection search.

DetectorFreeMatcher handles everything that is model independent:
- grayscale conversion and an aspect-preserving resize to a multiple of `divisible_by`
  with the long side <= `max_side` (no square squashing of e.g. 1241x376 KITTI frames);
- sub-pixel keypoints rescaled to the original image (no integer truncation);
- confidence threshold, image-bounds filtering, an optional cap on the number of matches
  and optional MAGSAC filtering;
- row filtering for rectified stereo (on original-image coordinates);
- the FeatureMatchingResult contract used by the trackers and main_feature_matching.py
  (cv2.KeyPoint arrays, built with the vectorized cv2.KeyPoint.convert).
A subclass implements only `_infer(t0, t1)`.
"""

import hashlib
import os
import threading
from pathlib import Path

import cv2
import numpy as np
import torch

from pyslam.utilities.logging import Printer

from .feature_matcher import FeatureMatcher, FeatureMatcherTypes, FeatureMatchingResult, MatcherUtils
from .feature_types import FeatureDescriptorTypes, FeatureDetectorTypes


def _default_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resized_shape(height, width, max_side, divisible_by):
    """Aspect-preserving target (H', W'): long side <= max_side, both multiples of divisible_by."""
    scale = 1.0 if not max_side else min(1.0, float(max_side) / max(height, width))
    new_h = max(divisible_by, int(round(height * scale)) // divisible_by * divisible_by)
    new_w = max(divisible_by, int(round(width * scale)) // divisible_by * divisible_by)
    return new_h, new_w


kDetectorFreeOptions = ("max_side", "conf_threshold", "max_num_matches", "use_magsac", "magsac_threshold",
                        "use_amp", "device")


def detector_free_options(kwargs):
    """Detector-free matcher options from a tracker preset (other preset keys are ignored)."""
    return {key: kwargs[key] for key in kDetectorFreeOptions if key in kwargs}


def _to_numpy(*values):
    return tuple(np.asarray(x.float().cpu().numpy() if torch.is_tensor(x) else x, dtype=np.float32) for x in values)


def keypoints_from_points(points, size=1.0):
    """Vectorized float (sub-pixel) cv2.KeyPoint array from an (N, 2) array."""
    points = np.ascontiguousarray(points, dtype=np.float32).reshape(-1, 2)
    if len(points) == 0:
        return np.array([], dtype=object)
    return np.array(cv2.KeyPoint.convert(points, size=size), dtype=object)


class DetectorFreeMatcher(FeatureMatcher):
    detector_free = True

    def __init__(
        self,
        matcher_type,
        max_side=1024,
        divisible_by=32,
        conf_threshold=0.2,
        max_num_matches=None,
        use_magsac=False,
        magsac_threshold=1.0,
        use_amp=False,
        device=None,
        ratio_test=1.0,
        **kwargs,
    ):
        super().__init__(
            norm_type=cv2.NORM_L2,
            cross_check=False,
            ratio_test=ratio_test,
            matcher_type=matcher_type,
            detector_type=FeatureDetectorTypes.NONE,
            descriptor_type=FeatureDescriptorTypes.NONE,
        )
        self.max_side = max_side
        self.divisible_by = int(divisible_by)
        self.conf_threshold = float(conf_threshold)
        self.max_num_matches = max_num_matches
        self.use_magsac = use_magsac
        self.magsac_threshold = magsac_threshold
        self.torch_device = torch.device(device) if device is not None else _default_device()
        # fp16 autocast is opt-in: it changes outputs and must pass the benchmark gates first.
        self.use_amp = bool(use_amp) and self.torch_device.type == "cuda"
        self._lock = threading.RLock()  # one model instance may be shared across threads

    # ---- model-specific ----------------------------------------------------------------
    def _infer(self, t0, t1):
        """t0, t1: float tensors (1, 1, H', W') in [0, 1] on self.torch_device, same size.
        Returns (kpts0, kpts1, conf): (N, 2) points in resized-pixel coordinates and (N,)."""
        raise NotImplementedError

    # ---- shared pipeline ---------------------------------------------------------------
    def _prepare(self, img, target_hw):
        gray = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        h, w = gray.shape[:2]
        th, tw = target_hw
        if (h, w) != (th, tw):
            gray = cv2.resize(gray, (tw, th), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(np.ascontiguousarray(gray)).to(self.torch_device, non_blocking=True)
        tensor = tensor.float().div_(255.0)[None, None]
        # Scale from resized-pixel to original-pixel coordinates (continuous coordinates).
        return tensor, (w / float(tw), h / float(th)), (h, w)

    def _infer_original(self, img1, img2):
        """Matches in original-image pixel coordinates: (k0, k1, conf) numpy arrays.
        Default: grayscale, aspect-preserving resize, _infer(), rescale. Models that handle
        their own preprocessing (e.g. RoMa) override this method instead of _infer."""
        h1, w1 = img1.shape[:2]
        target = resized_shape(h1, w1, self.max_side, self.divisible_by)
        t0, scale0, _ = self._prepare(img1, target)
        t1, scale1, _ = self._prepare(img2, target)
        with torch.inference_mode():
            if self.use_amp:
                with torch.autocast("cuda", dtype=torch.float16):
                    k0, k1, conf = self._infer(t0, t1)
            else:
                k0, k1, conf = self._infer(t0, t1)
        k0, k1, conf = _to_numpy(k0, k1, conf)
        return k0.reshape(-1, 2) * np.float32(scale0), k1.reshape(-1, 2) * np.float32(scale1), conf

    def match_points(self, img1, img2):
        """Matched (N, 2) float32 points in original image coordinates and (N,) confidences."""
        shape0, shape1 = img1.shape[:2], img2.shape[:2]
        with self._lock:
            k0, k1, conf = self._infer_original(img1, img2)
        k0, k1, conf = _to_numpy(k0, k1, conf)
        k0, k1, conf = k0.reshape(-1, 2), k1.reshape(-1, 2), conf.reshape(-1)
        keep = conf >= self.conf_threshold
        for points, (h, w) in ((k0, shape0), (k1, shape1)):
            keep &= (points[:, 0] >= 0) & (points[:, 0] < w) & (points[:, 1] >= 0) & (points[:, 1] < h)
        k0, k1, conf = k0[keep], k1[keep], conf[keep]
        if self.max_num_matches and len(conf) > self.max_num_matches:
            best = np.argsort(-conf, kind="stable")[: self.max_num_matches]
            best.sort()
            k0, k1, conf = k0[best], k1[best], conf[best]
        if self.use_magsac and len(k0) >= 8:
            _, mask = cv2.findFundamentalMat(k0, k1, cv2.USAC_MAGSAC, self.magsac_threshold, 0.999, 10000)
            if mask is not None:
                inliers = mask.ravel().astype(bool)
                k0, k1, conf = k0[inliers], k1[inliers], conf[inliers]
        return k0, k1, conf

    def match(
        self,
        img1,
        img2,
        des1=None,
        des2=None,
        kps1=None,
        kps2=None,
        ratio_test=None,
        row_matching=False,
        max_disparity=None,
        data=None,
    ):
        result = FeatureMatchingResult()
        if img2 is None:
            img2 = img1
        k0, k1, conf = self.match_points(img1, img2)
        idxs = np.arange(len(k0), dtype=np.int32)
        result.idxs1, result.idxs2 = idxs, idxs.copy()
        if row_matching:
            result.idxs1, result.idxs2 = MatcherUtils.filterNonRowMatches(
                k0, result.idxs1, k1, result.idxs2, max_disparity=max_disparity
            )
        result.kps1 = keypoints_from_points(k0)
        result.kps2 = keypoints_from_points(k1)
        result.des1 = result.des2 = None
        self.last_confidences = conf
        return result


# ==============================================================================
# RoMa / tiny-RoMa (Edstedt et al., "RoMa: Robust Dense Feature Matching", CVPR 2024).
# Code: Parskatt/RoMa (MIT), pinned commit, loaded from its source snapshot (no pip
# install). Weights: RoMa release artifacts (MIT) + DINOv2 ViT-L/14 (Apache-2.0); tiny-RoMa
# uses pyslam's vendored XFeat network (thirdparty/accelerated_features, Apache-2.0).
kRomaRepo = "Parskatt/RoMa:77f8d68803526dcddfd9b7a46bc76125bdc25f15"


def _import_romatch():
    """Import the pinned romatch package without installing it."""
    import sys
    import types

    from pyslam.loop_closing.global_feature_torchhub import github_snapshot

    root = github_snapshot(kRomaRepo)
    if root not in sys.path:
        sys.path.append(root)  # appended: "romatch" is a unique top-level name
    try:
        import loguru  # noqa: F401
    except ImportError:
        # romatch logs through loguru (not a pyslam dependency): route it to logging.
        import logging

        shim = types.ModuleType("loguru")
        shim.logger = logging.getLogger("romatch")
        sys.modules["loguru"] = shim
    import romatch

    return romatch


class RoMaMatcher(DetectorFreeMatcher):
    """Dense RoMa matches, sampled by certainty (balanced sampling, as upstream).
    Sampling is random (torch.multinomial): outputs vary between runs."""

    def __init__(self, matcher_type=FeatureMatcherTypes.ROMA, max_num_matches=2000, conf_threshold=0.0,
                 coarse_res=560, upsample_res=864, **kwargs):
        kwargs.pop("max_side", None)  # RoMa resizes internally (coarse/upsample resolutions)
        super().__init__(matcher_type=matcher_type, max_side=None, max_num_matches=max_num_matches,
                         conf_threshold=conf_threshold, **kwargs)
        self.num_samples = max_num_matches
        romatch = _import_romatch()
        self.matcher = self._build(romatch, coarse_res, upsample_res)
        self.matcher_name = type(self).__name__
        Printer.green(f"matcher: {self.matcher_name} ({self.torch_device})")

    def _build(self, romatch, coarse_res, upsample_res):
        # use_custom_corr=False: the optional fused-local-corr CUDA extension is not a pyslam
        # dependency; the pure PyTorch local correlation is used instead.
        return romatch.roma_outdoor(device=self.torch_device, coarse_res=coarse_res, upsample_res=upsample_res,
                                    use_custom_corr=False)

    @staticmethod
    def _pil(img):
        from PIL import Image

        return Image.fromarray(img if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2))

    def _match_warp(self, img1, img2):
        return self.matcher.match(self._pil(img1), self._pil(img2), device=self.torch_device)

    def _infer_original(self, img1, img2):
        (h1, w1), (h2, w2) = img1.shape[:2], img2.shape[:2]
        with torch.inference_mode():
            warp, certainty = self._match_warp(img1, img2)
            matches, cert = self.matcher.sample(warp, certainty, num=self.num_samples)
            k0, k1 = self.matcher.to_pixel_coordinates(matches, h1, w1, h2, w2)
        return k0, k1, cert


class TinyRoMaMatcher(RoMaMatcher):
    """tiny-RoMa v1 (XFeat backbone): much lighter than RoMa."""

    def __init__(self, matcher_type=FeatureMatcherTypes.TINY_ROMA, **kwargs):
        super().__init__(matcher_type=matcher_type, **kwargs)

    def _build(self, romatch, coarse_res, upsample_res):
        import pyslam.config as config
        from pyslam.utilities.system import import_from

        config.cfg.set_lib("xfeat")
        XFeat = import_from("modules.xfeat", "XFeat")
        xfeat = XFeat().net  # pyslam's vendored, locally stored XFeat weights (no torch.hub fetch)
        return romatch.tiny_roma_v1_outdoor(device=self.torch_device, xfeat=xfeat)

    def _match_warp(self, img1, img2):
        to_tensor = lambda img: torch.from_numpy(  # noqa: E731
            np.ascontiguousarray(img if img.ndim == 3 else np.repeat(img[:, :, None], 3, axis=2))
        ).to(self.torch_device).permute(2, 0, 1)[None].float().div_(255.0)
        return self.matcher.match(to_tensor(img1), to_tensor(img2), batched=False)
