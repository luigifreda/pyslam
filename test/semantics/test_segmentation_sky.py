import os
import sys


from pyslam.config import Config

config = Config()

from pyslam.utilities.file_management import gdrive_download_lambda
from pyslam.utilities.system import getchar, Printer
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


# online loop closure detection by using DBoW3
if __name__ == "__main__":

    dataset = dataset_factory(config)

    sky_extractor = SkyMaskExtractor()
    img_writer = ImgWriter(font_scale=0.7)

    cv2.namedWindow("semantic prediction", cv2.WINDOW_NORMAL)  # to get a resizable window

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
            cv2.imshow("sky colored", colored)

            key = cv2.waitKey(1)
        else:
            key = cv2.waitKey(100)

        if key == ord("q") or key == 27:
            break

        img_id += 1
