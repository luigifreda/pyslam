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


from pyslam.config import Config

config = Config()

from pyslam.utilities.file_management import gdrive_download_lambda
from pyslam.utilities.system import getchar
from pyslam.utilities.logging import Printer
from pyslam.utilities.img_management import (
    float_to_color,
    convert_float_to_colored_uint8_image,
    LoopCandidateImgs,
    ImgWriter,
)

import math
import cv2
import numpy as np

from pyslam.io.dataset_factory import dataset_factory

from pyslam.semantics.sky_mask_extractor import SkyMaskExtractor


if __name__ == "__main__":

    dataset = dataset_factory(config)

    sky_extractor = SkyMaskExtractor()
    img_writer = ImgWriter(font_scale=0.7)

    cv2.namedWindow("Semantic prediction")  # to get a resizable window

    img_id = 0  # 180, 340, 400   # you can start from a desired frame id if needed
    key = None
    while True:

        timestamp, img = None, None

        if dataset.is_ok:
            timestamp = dataset.getTimestamp()  # get current timestamp
            img = dataset.getImageColor(img_id)

        if img is not None:
            print("----------------------------------------")
            print(f"processing img {img_id}")

            sky_mask = sky_extractor.extract_mask(img, threshold=0.2)

            img_writer.write(img, f"id: {img_id}", (30, 30))
            cv2.imshow("img", img)

            # masked = cv2.bitwise_and(img, img, mask=(255 - sky_mask))  # sky = visible, non-sky = masked
            # cv2.imshow("masked sky", masked)

            # colorize sky as red (BGR: (0, 0, 255))
            colored = sky_extractor.colorize_sky_region(img, sky_mask, color=(0, 0, 255), alpha=0.9)
            cv2.imshow("sky red-colored", colored)

            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(100)

        if key == ord("q") or key == 27:
            break

        img_id += 1
