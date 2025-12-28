"""
* This file is part of PYSLAM
* Adapted from https://github.com/cvlab-epfl/disk/blob/master/detect.py, see licence therein.
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

# adapted from https://github.com/cvlab-epfl/disk/blob/master/detect.py


import sys
import pyslam.config as config

config.cfg.set_lib("disk")
config.cfg.set_lib("torch-dimcheck")
config.cfg.set_lib("torch-localize")
config.cfg.set_lib("unets")

import cv2
from threading import RLock

from pyslam.utilities.logging import Printer
from pyslam.utilities.system import is_opencv_version_greater_equal
from pyslam.utilities.tensorflow import ensure_tensorflow_stub_for_tensorboard
from .feature_base import BaseFeature2D

import torch, h5py, imageio, os, argparse
import numpy as np
import torch.nn.functional as F
from functools import partial
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch_dimcheck import dimchecked

ensure_tensorflow_stub_for_tensorboard()
from disk import DISK, Features


kVerbose = True


class Image:
    def __init__(self, bitmap: ["C", "H", "W"], fname: str, orig_shape=None):
        self.bitmap = bitmap
        self.fname = fname
        if orig_shape is None:
            self.orig_shape = self.bitmap.shape[1:]
        else:
            self.orig_shape = orig_shape

    def resize_to(self, shape):
        return Image(
            self._pad(self._interpolate(self.bitmap, shape), shape),
            self.fname,
            orig_shape=self.bitmap.shape[1:],
        )

    @dimchecked
    def to_image_coord(self, xys: [2, "N"]) -> ([2, "N"], ["N"]):
        f, _size = self._compute_interpolation_size(self.bitmap.shape[1:])
        scaled = xys / f

        h, w = self.orig_shape
        x, y = scaled

        mask = (0 <= x) & (x < w) & (0 <= y) & (y < h)

        return scaled, mask

    def _compute_interpolation_size(self, shape):
        x_factor = self.orig_shape[0] / shape[0]
        y_factor = self.orig_shape[1] / shape[1]

        f = 1 / max(x_factor, y_factor)

        if x_factor > y_factor:
            new_size = (shape[0], int(f * self.orig_shape[1]))
        else:
            new_size = (int(f * self.orig_shape[0]), shape[1])

        return f, new_size

    @dimchecked
    def _interpolate(self, image: ["C", "H", "W"], shape) -> ["C", "h", "w"]:
        _f, size = self._compute_interpolation_size(shape)
        return F.interpolate(
            image.unsqueeze(0),
            size=size,
            mode="bilinear",
            align_corners=False,
        ).squeeze(0)

    @dimchecked
    def _pad(self, image: ["C", "H", "W"], shape) -> ["C", "h", "w"]:
        x_pad = shape[0] - image.shape[1]
        y_pad = shape[1] - image.shape[2]

        if x_pad < 0 or y_pad < 0:
            raise ValueError("Attempting to pad by negative value")

        return F.pad(image, (0, y_pad, 0, x_pad))


class ImageAdapter:
    def __init__(self, image, crop_size=(None, None)):
        self.image = image
        self.crop_size = crop_size

    def get(self):
        # name   = self.names[ix]
        # path   = os.path.join(self.image_path, name)
        # img    = np.ascontiguousarray(imageio.imread(path))
        # tensor = torch.from_numpy(img).to(torch.float32)

        img = np.ascontiguousarray(self.image)
        tensor = torch.from_numpy(img).to(torch.float32)

        if len(tensor.shape) == 2:  # some images may be grayscale
            tensor = tensor.unsqueeze(-1).expand(-1, -1, 3)

        bitmap = tensor.permute(2, 0, 1) / 255.0
        # extensionless_fname = os.path.splitext(name)[0]

        image = Image(bitmap, "")

        if self.crop_size != (None, None):
            image = image.resize_to(self.crop_size)
        return image

    def stack(self):
        images = [self.get()]
        bitmaps = torch.stack([im.bitmap for im in images], dim=0)
        return bitmaps, images


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints(pts, scores, size):
    assert len(pts) == len(scores)
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        if is_opencv_version_greater_equal(4, 5, 3):
            kps = [
                cv2.KeyPoint(p[0], p[1], size=size, response=s, octave=0)
                for p, s in zip(pts, scores)
            ]
        else:
            kps = [
                cv2.KeyPoint(p[0], p[1], _size=size, _response=s, _octave=0)
                for p, s in zip(pts, scores)
            ]
    return kps


# convert matrix of pts into list of keypoints
def convert_pts_to_keypoints_with_translation(pts, scores, size, deltax, deltay):
    assert len(pts) == len(scores)
    kps = []
    if pts is not None:
        # convert matrix [Nx2] of pts into list of keypoints
        if is_opencv_version_greater_equal(4, 5, 3):
            kps = [
                cv2.KeyPoint(p[0] + deltax, p[1] + deltay, size=size, response=s, octave=0)
                for p, s in zip(pts, scores)
            ]
        else:
            kps = [
                cv2.KeyPoint(p[0] + deltax, p[1] + deltay, _size=size, _response=s, _octave=0)
                for p, s in zip(pts, scores)
            ]
    return kps


# Interface for pySLAM
# NOTE: from Fig. 3 in the paper "DISK: Learning local features with policy gradient"
# "Our approach can match many more points and produce more accurate poses. It can deal with large changes in scale (4th and 5th columns) but not in rotation..."
class DiskFeature2D(BaseFeature2D):
    def __init__(
        self,
        num_features=2000,
        nms_window_size=5,  # NMS windows size
        desc_dim=128,  # descriptor dimension. Needs to match the checkpoint value
        mode="nms",  # choices=['nms', 'rng'], Whether to extract features using the non-maxima suppresion mode or through training-time grid sampling technique'
        do_cuda=True,
    ):
        print("Using DiskFeature2D")
        self.lock = RLock()

        self.num_features = num_features
        self.nms_window_size = nms_window_size
        self.desc_dim = desc_dim
        self.mode = mode
        self.model_base_path = config.cfg.root_folder + "/thirdparty/disk/depth-save.pth"

        self.do_cuda = do_cuda & torch.cuda.is_available()
        print("cuda:", self.do_cuda)

        self.DEV = torch.device("cuda" if self.do_cuda else "cpu")
        self.CPU = torch.device("cpu")
        self.state_dict = torch.load(self.model_base_path, map_location="cpu")

        # compatibility with older model saves which used the 'extractor' name
        if "extractor" in self.state_dict:
            weights = self.state_dict["extractor"]
        elif "disk" in self.state_dict:
            weights = self.state_dict["disk"]
        else:
            raise KeyError("Incompatible weight file!")
        self.disk = DISK(window=8, desc_dim=desc_dim)
        print("==> Loading pre-trained network.")
        self.disk.load_state_dict(weights)
        self.model = self.disk.to(self.DEV)
        print("==> Successfully loaded pre-trained network.")

        self.keypoint_size = 8  # just a representative size for visualization and in order to convert extracted points to cv2.KeyPoint

        self.pts = []
        self.kps = []
        self.des = []
        self.scales = []
        self.scores = []
        self.frame = None

        self.use_crop = False
        self.cropx = [0, 0]  # [startx, endx]
        self.cropy = [0, 0]  # [starty, endy]

    def setMaxFeatures(
        self, num_features
    ):  # use the cv2 method name for extractors (see https://docs.opencv.org/4.x/db/d95/classcv_1_1ORB.html#aca471cb82c03b14d3e824e4dcccf90b7)
        self.num_features = num_features

    def crop_center(self, img, cropx, cropy):
        y, x = img.shape
        startx = x // 2 - (cropx // 2)
        starty = y // 2 - (cropy // 2)
        return img[starty : starty + cropy, startx : startx + cropx]

    def extract(self, image):
        if self.mode == "nms":
            extract = partial(
                self.model.features,
                kind="nms",
                window_size=self.nms_window_size,
                cutoff=0.0,
                n=self.num_features,
            )
        else:
            extract = partial(model.features, kind="rng")

        self.use_crop = False
        print(f"image shape: {image.shape}, image.ndim: {image.ndim}")
        if image.ndim == 2:
            height, width = image.shape
        else:
            height, width, channels = image.shape
        cropx = width % 16
        cropy = height % 16
        if cropx != 0 or cropy != 0:
            self.use_crop = True
            half_cropx = cropx // 2
            rest_cropx = cropx % 2
            half_cropy = cropy // 2
            rest_cropy = cropy % 2
            self.cropx = [half_cropx, width - (half_cropx + rest_cropx)]
            self.cropy = [half_cropy, height - (half_cropy + rest_cropy)]
            if image.ndim == 3:
                cropped_image = image[
                    self.cropy[0] : self.cropy[1], self.cropx[0] : self.cropx[1], :
                ]
            elif image.ndim == 2:
                cropped_image = image[self.cropy[0] : self.cropy[1], self.cropx[0] : self.cropx[1]]
            image_adapter = ImageAdapter(cropped_image)
        else:
            image_adapter = ImageAdapter(image)
        bitmaps, images = image_adapter.stack()

        bitmaps = bitmaps.to(self.DEV, non_blocking=True)

        with torch.no_grad():
            try:
                batched_features = extract(bitmaps)
            except RuntimeError as e:
                if "U-Net failed" in str(e):
                    msg = (
                        "Please use input size which is multiple of 16 (or "
                        "adjust the --height and --width flags to let this "
                        "script rescale it automatically). This is because "
                        "we internally use a U-Net with 4 downsampling "
                        "steps, each by a factor of 2, therefore 2^4=16."
                    )

                    raise RuntimeError(msg) from e
                else:
                    raise

        for features, image in zip(batched_features.flat, images):
            features = features.to(self.CPU)

            kps_crop_space = features.kp.T
            kps_img_space, mask = image.to_image_coord(kps_crop_space)

            keypoints = kps_img_space.numpy().T[mask]
            descriptors = features.desc.numpy()[mask]
            scores = features.kp_logp.numpy()[mask]

            order = np.argsort(scores)[::-1]

            keypoints = keypoints[order]
            descriptors = descriptors[order]
            scores = scores[order]

            assert descriptors.shape[1] == self.desc_dim
            assert keypoints.shape[1] == 2
            return keypoints, descriptors, scores

    def compute_kps_des(self, im):
        with self.lock:
            keypoints, descriptors, scores = self.extract(im)
            # print('scales:',self.scales)
            if self.use_crop:
                self.kps = convert_pts_to_keypoints(keypoints, scores, self.keypoint_size)
            else:
                self.kps = convert_pts_to_keypoints_with_translation(
                    keypoints, scores, self.keypoint_size, self.cropx[0], self.cropy[0]
                )
            return self.kps, descriptors

    def detectAndCompute(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            self.frame = frame
            self.kps, self.des = self.compute_kps_des(frame)
            if kVerbose:
                print(
                    "detector: DISK, descriptor: DISK, #features: ",
                    len(self.kps),
                    ", frame res: ",
                    frame.shape[0:2],
                )
            return self.kps, self.des

    # return keypoints if available otherwise call detectAndCompute()
    def detect(self, frame, mask=None):  # mask is a fake input
        with self.lock:
            # if self.frame is not frame:
            self.detectAndCompute(frame)
            return self.kps

    # return descriptors if available otherwise call detectAndCompute()
    def compute(self, frame, kps=None, mask=None):  # kps is a fake input, mask is a fake input
        with self.lock:
            if self.frame is not frame:
                Printer.orange(
                    "WARNING: DISK is recomputing both kps and des on last input frame", frame.shape
                )
                self.detectAndCompute(frame)
            return self.kps, self.des

    # return descriptors if available otherwise call detectAndCompute()
    def compute(self, frame, kps=None, mask=None):  # kps is a fake input, mask is a fake input
        with self.lock:
            if self.frame is not frame:
                # Printer.orange('WARNING: DISK is recomputing both kps and des on last input frame', frame.shape)
                self.detectAndCompute(frame)
            return self.kps, self.des
