"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
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
import cv2
import numpy as np
import onnxruntime
from typing import Optional

from pyslam.utilities.download import download_file_from_url

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."
kDataFolder = os.path.join(kRootFolder, "data")


class SkyMaskExtractor:
    def __init__(
        self,
        model_path: str = os.path.join(kDataFolder, "skyseg.onnx"),
        model_url: str = "https://huggingface.co/JianyuanWang/skyseg/resolve/main/skyseg.onnx",
    ):
        self.model_path = model_path
        self.model_url = model_url
        self.skyseg_session: Optional[onnxruntime.InferenceSession] = None
        self._ensure_model_downloaded()

    def _ensure_model_downloaded(self):
        if not os.path.exists(self.model_path):
            print(f"Downloading {self.model_path}...")
            download_file_from_url(self.model_url, self.model_path)
        else:
            print(f"{self.model_path} already exists, skipping download.")
        self.skyseg_session = onnxruntime.InferenceSession(self.model_path)

    def extract_mask(self, image: np.ndarray, threshold: float = 0.2) -> np.ndarray:
        """
        Run ONNX inference on a single input image to get the sky mask.

        Args:
            image (np.ndarray): Input BGR image (as loaded by OpenCV).

        Returns:
            np.ndarray: Binary sky mask (uint8) where 255 = non-sky, 0 = sky.
        """
        result_map = self._run_skyseg(self.skyseg_session, (320, 320), image)
        result_map_resized = cv2.resize(result_map, (image.shape[1], image.shape[0]))

        # Binary mask: 255 = non-sky, 0 = sky
        output_mask = np.zeros_like(result_map_resized, dtype=np.uint8)
        output_mask[result_map_resized < threshold] = 255

        return output_mask

    def _run_skyseg(self, session, input_size, image):
        """
        Internal helper for preprocessing + ONNX inference.

        Args:
            session: ONNX runtime session
            input_size: (width, height) tuple
            image: input BGR image

        Returns:
            np.ndarray: model output map (float32)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(image_rgb, input_size)
        normalized = resized.astype(np.float32) / 255.0
        input_tensor = normalized.transpose(2, 0, 1)[np.newaxis, ...]

        input_name = session.get_inputs()[0].name
        output_name = session.get_outputs()[0].name
        result = session.run([output_name], {input_name: input_tensor})[0]

        if result.ndim == 4:
            return result[0, 0]  # [1, 1, H, W] → [H, W]
        elif result.ndim == 3:
            return result[0]  # [1, H, W]
        else:
            raise RuntimeError(f"Unexpected model output shape: {result.shape}")

    @staticmethod
    def colorize_sky_region(
        image: np.ndarray, mask: np.ndarray, color: tuple = (0, 0, 255), alpha: float = 0.4
    ) -> np.ndarray:
        """
        Overlays a color on the sky region of an image with adjustable transparency.

        Args:
            image (np.ndarray): Original BGR image.
            mask (np.ndarray): Binary mask (255=non-sky, 0=sky), uint8.
            color (tuple): BGR color to apply to the sky.
            alpha (float): Opacity of the color overlay (0=no effect, 1=fully colored).

        Returns:
            np.ndarray: Image with colored sky region.
        """
        if image is None or mask is None:
            raise ValueError("Input image or mask is None.")
        if image.shape[:2] != mask.shape:
            raise ValueError("Image and mask size mismatch.")

        # Build solid color image
        color_layer = np.full_like(image, color, dtype=np.uint8)

        # Compute full blended image
        blended = cv2.addWeighted(image, 1 - alpha, color_layer, alpha, 0)

        # Apply only on sky pixels (mask == 0)
        sky_mask = mask == 0
        result = image.copy()
        result[sky_mask] = blended[sky_mask]

        return result
