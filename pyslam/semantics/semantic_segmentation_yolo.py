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
from enum import Enum
from pathlib import Path
import urllib.request

import numpy as np  # type: ignore[import-not-found]
import torch  # type: ignore[import-not-found]

from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_segmentation_output import SemanticSegmentationOutput
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_color_map_factory import semantic_color_map_factory
from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_instance_utils import ensure_unique_instance_ids

from pyslam.utilities.logging import Printer
from pyslam.config_parameters import Parameters


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = os.path.abspath(os.path.join(kScriptFolder, "..", ".."))
kModelsDir = os.path.join(kRootFolder, "data", "models")


class YoloSegModel(Enum):
    YOLO26N = ("v8.4.0", "yolo26n-seg.pt")
    YOLO11N = ("v8.4.0", "yolo11n-seg.pt")
    YOLOV8N = ("v8.4.0", "yolov8n-seg.pt")

    @property
    def release_tag(self) -> str:
        return self.value[0]

    @property
    def filename(self) -> str:
        return self.value[1]


class SemanticSegmentationYolo(SemanticSegmentationBase):
    """
    Semantic segmentation using Ultralytics YOLO segmentation models (instance masks).
    """

    supported_feature_types = [
        SemanticFeatureType.LABEL,
        SemanticFeatureType.PROBABILITY_VECTOR,
    ]

    def __init__(
        self,
        device=None,
        model_name=YoloSegModel.YOLO26N,
        weights_path="",
        model_url="",
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        image_size=(512, 512),
        semantic_feature_type=SemanticFeatureType.LABEL,
        confidence_threshold=0.3,
        download_if_missing=True,
        enforce_unique_instance_ids=Parameters.kSemanticSegmentationEnforceUniqueInstanceIds,
        unique_instance_min_pixels=Parameters.kSemanticSegmentationUniqueInstanceMinPixels,
        reserve_background_label=False,
        **kwargs,
    ):
        """
        Initialize YOLO semantic segmentation model.

        Args:
            device: torch device (cuda/cpu/mps) or None for auto-detection
            model_name: YOLO segmentation model name (e.g., YoloSegModel.YOLO26N)
            weights_path: Optional path to model weights (.pt). If empty, uses model_name.
            model_url: Optional URL for downloading weights (overrides default URL).
            semantic_dataset_type: Target dataset type for color mapping
            image_size: (height, width) - currently not used (model handles resizing)
            semantic_feature_type: LABEL or PROBABILITY_VECTOR
            confidence_threshold: Minimum confidence for predictions
            download_if_missing: Download weights if missing
            enforce_unique_instance_ids: If True, split disconnected components to unique IDs
            unique_instance_min_pixels: Minimum component size when splitting instance IDs
            reserve_background_label: If True, reserve label 0 for background and shift class IDs by +1
        """

        self.semantic_dataset_type = semantic_dataset_type
        self.confidence_threshold = confidence_threshold
        self.enforce_unique_instance_ids = enforce_unique_instance_ids
        self.unique_instance_min_pixels = unique_instance_min_pixels
        self.reserve_background_label = reserve_background_label

        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        device = self.init_device(device)
        model, transform = self.init_model(
            device,
            model_name=model_name,
            weights_path=weights_path,
            model_url=model_url,
            download_if_missing=download_if_missing,
        )

        super().__init__(model, transform, device, semantic_feature_type)

        num_classes = self.num_classes()
        self.class_id_offset = 1 if self.reserve_background_label else 0
        self.num_classes_with_background = int(num_classes + self.class_id_offset)
        self.semantic_color_map_obj = semantic_color_map_factory(
            semantic_dataset_type=semantic_dataset_type,
            semantic_feature_type=semantic_feature_type,
            num_classes=self.num_classes_with_background,
            semantic_segmentation_type=SemanticSegmentationType.YOLO,
        )
        self.semantic_color_map = self.semantic_color_map_obj.color_map
        self._last_result = None

    def _ensure_ultralytics(self):
        try:
            from ultralytics import YOLO  # type: ignore[import-not-found]  # noqa: F401
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Ultralytics is required. Install with: pip install ultralytics"
            ) from exc

    def init_model(
        self,
        device,
        model_name,
        weights_path,
        model_url,
        download_if_missing=True,
    ):
        self._ensure_ultralytics()
        from ultralytics import YOLO  # type: ignore[import-not-found]

        if isinstance(model_name, YoloSegModel):
            model_enum = model_name
            model_name = model_enum.filename
            if not model_url:
                model_url = (
                    "https://github.com/ultralytics/assets/releases/"
                    f"download/{model_enum.release_tag}/{model_enum.filename}"
                )

        if weights_path:
            weights_path = Path(weights_path)
            if not weights_path.is_absolute():
                weights_path = Path(kModelsDir) / weights_path
        else:
            weights_path = Path(kModelsDir) / model_name

        if download_if_missing and not weights_path.exists():
            weights_path.parent.mkdir(parents=True, exist_ok=True)
            if model_url:
                url = model_url
            else:
                url = f"https://github.com/ultralytics/assets/releases/download/v8.4.0/{model_name}"
            Printer.green(f"YOLO: Downloading weights to {weights_path}")
            urllib.request.urlretrieve(url, weights_path)

        if not weights_path.exists():
            raise FileNotFoundError(
                f"YOLO weights not found: {weights_path}. "
                f"Provide weights_path or enable download_if_missing."
            )

        model = YOLO(str(weights_path))
        return model, None

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if device.type == "cuda":
            Printer.green("SemanticSegmentationYolo: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise Exception("SemanticSegmentationYolo: MPS is not available")
            Printer.yellow("SemanticSegmentationYolo: Using MPS")
        else:
            Printer.yellow("SemanticSegmentationYolo: Using CPU")

        return device

    def num_classes(self):
        try:
            names = getattr(self.model, "names", None)
            if isinstance(names, dict) and names:
                keys = []
                for key in names.keys():
                    try:
                        keys.append(int(key))
                    except Exception:
                        pass
                if keys:
                    return int(max(keys) + 1)
                return int(len(names))
            if isinstance(names, (list, tuple)):
                return int(len(names))
        except Exception:
            pass

        return 91

    def _prepare_image(self, image):
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        if len(image.shape) == 2:
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:
            image = image[:, :, :3]
        elif image.shape[2] != 3:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return image

    @torch.no_grad()
    def infer(self, image) -> SemanticSegmentationOutput:
        import cv2  # type: ignore[import-not-found]

        image = self._prepare_image(image)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        results = self.model.predict(
            source=img_rgb,
            task="segment",
            conf=self.confidence_threshold,
            device=str(self.device),
            verbose=False,
        )
        first = results[0] if isinstance(results, list) and results else results
        self._last_result = first

        semantics, instances = self._results_to_semantics(first, image.shape[:2])
        self.semantics = semantics

        if instances is not None and self.enforce_unique_instance_ids:
            instances = ensure_unique_instance_ids(
                instances, background_id=0, min_pixels=self.unique_instance_min_pixels
            )

        return SemanticSegmentationOutput(semantics=self.semantics, instances=instances)

    def _results_to_semantics(self, result, image_shape):
        H, W = image_shape

        if result is None or result.masks is None or result.boxes is None:
            if self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
                num_classes = self.num_classes_with_background
                return np.zeros((H, W, num_classes), dtype=np.float32), None
            return np.zeros((H, W), dtype=np.int32), None

        masks = result.masks.data.cpu().numpy()
        if masks.shape[1:] != (H, W):
            import cv2  # type: ignore[import-not-found]

            resized = np.zeros((masks.shape[0], H, W), dtype=masks.dtype)
            for idx, mask in enumerate(masks):
                resized[idx] = cv2.resize(mask, (W, H), interpolation=cv2.INTER_LINEAR)
            masks = resized
        class_ids = result.boxes.cls.cpu().numpy().astype(np.int32)
        scores = None
        if hasattr(result.boxes, "conf") and result.boxes.conf is not None:
            scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        if scores is None:
            scores = np.ones(len(class_ids), dtype=np.float32)

        order = np.argsort(scores)[::-1]
        instances = np.zeros((H, W), dtype=np.int32)
        best_score = np.zeros((H, W), dtype=np.float32)

        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            semantic_labels = np.zeros((H, W), dtype=np.int32)
            for instance_id, det_idx in enumerate(order, start=1):
                mask = masks[det_idx] > 0.5
                class_id = int(class_ids[det_idx]) + self.class_id_offset
                score = float(scores[det_idx]) if scores is not None else 1.0
                update = mask & (score > best_score)
                if not np.any(update):
                    continue
                best_score[update] = score
                semantic_labels[update] = class_id
                instances[update] = instance_id
            return semantic_labels, instances

        num_classes = self.num_classes_with_background
        probs = np.zeros((H, W, num_classes), dtype=np.float32)
        for instance_id, det_idx in enumerate(order, start=1):
            mask = masks[det_idx] > 0.5
            class_id = int(class_ids[det_idx]) + self.class_id_offset
            score = float(scores[det_idx]) if scores is not None else 1.0
            if class_id >= num_classes:
                continue
            probs[mask, class_id] = np.maximum(probs[mask, class_id], score)
            update = mask & (score > best_score)
            if not np.any(update):
                continue
            best_score[update] = score
            instances[update] = instance_id

        sum_probs = probs.sum(axis=-1, keepdims=True)
        valid = sum_probs > 0
        probs[valid] = probs[valid] / sum_probs[valid]

        return probs, instances

    def sem_img_to_viz_rgb(self, semantics, bgr=False):
        if self._last_result is None:
            return self.semantic_color_map_obj.to_rgb(semantics, bgr=bgr)

        try:
            import cv2  # type: ignore[import-not-found]
        except Exception:
            return self.semantic_color_map_obj.to_rgb(semantics, bgr=bgr)

        try:
            annotated = self._last_result.plot()
        except Exception:
            return self.semantic_color_map_obj.to_rgb(semantics, bgr=bgr)

        if bgr:
            return annotated
        return cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return self.semantic_color_map_obj.sem_img_to_rgb(semantic_img, bgr=bgr)
