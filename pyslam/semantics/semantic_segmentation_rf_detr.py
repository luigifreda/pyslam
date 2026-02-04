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
import sys
from enum import Enum
from pathlib import Path

import numpy as np
import torch

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


class RFDETRSegmentationModel(Enum):
    SEG_PREVIEW = ("seg-preview", "rf-detr-seg-preview.pt")
    SEG_NANO = ("seg-nano", "rf-detr-seg-nano.pt")
    SEG_SMALL = ("seg-small", "rf-detr-seg-small.pt")
    SEG_MEDIUM = ("seg-medium", "rf-detr-seg-medium.pt")
    SEG_LARGE = ("seg-large", "rf-detr-seg-large.pt")
    SEG_XLARGE = ("seg-xlarge", "rf-detr-seg-xlarge.pt")
    SEG_2XLARGE = ("seg-2xlarge", "rf-detr-seg-xxlarge.pt")

    @property
    def variant_key(self) -> str:
        return self.value[0]

    @property
    def weights_name(self) -> str:
        return self.value[1]

    @classmethod
    def label_map(cls):
        return {
            "RFDETRSegPreview": cls.SEG_PREVIEW,
            "RFDETRSegNano": cls.SEG_NANO,
            "RFDETRSegSmall": cls.SEG_SMALL,
            "RFDETRSegMedium": cls.SEG_MEDIUM,
            "RFDETRSegLarge": cls.SEG_LARGE,
            "RFDETRSegXLarge": cls.SEG_XLARGE,
            "RFDETRSeg2XLarge": cls.SEG_2XLARGE,
        }


class RFDETRDetectionModel(Enum):
    NANO = ("nano", "rf-detr-nano.pth")
    SMALL = ("small", "rf-detr-small.pth")
    MEDIUM = ("medium", "rf-detr-medium.pth")
    LARGE = ("large", "rf-detr-large.pth")
    XLARGE = ("xlarge", "rf-detr-xlarge.pth")
    XXLARGE = ("2xlarge", "rf-detr-xxlarge.pth")

    @property
    def variant_key(self) -> str:
        return self.value[0]

    @property
    def weights_name(self) -> str:
        return self.value[1]

    @classmethod
    def label_map(cls):
        return {
            "RFDETRNano": cls.NANO,
            "RFDETRSmall": cls.SMALL,
            "RFDETRMedium": cls.MEDIUM,
            "RFDETRLarge": cls.LARGE,
            "RFDETRXLarge": cls.XLARGE,
            "RFDETR2XLarge": cls.XXLARGE,
        }


class SemanticSegmentationRfDetr(SemanticSegmentationBase):
    """
    Semantic segmentation using RF-DETR segmentation models (instance masks).

    This wrapper uses the RF-DETR predict() API and converts detections into
    semantic labels and optional instance IDs.
    """

    supported_feature_types = [SemanticFeatureType.LABEL, SemanticFeatureType.PROBABILITY_VECTOR]

    def __init__(
        self,
        device=None,
        model_variant=RFDETRSegmentationModel.SEG_SMALL,
        weights_path="",
        weights_name=None,
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        image_size=(512, 512),
        semantic_feature_type=SemanticFeatureType.LABEL,
        confidence_threshold=0.3,
        download_if_missing=True,
        enforce_unique_instance_ids=Parameters.kSemanticSegmentationEnforceUniqueInstanceIds,
        unique_instance_min_pixels=Parameters.kSemanticSegmentationUniqueInstanceMinPixels,
        **kwargs,
    ):
        """
        Initialize RF-DETR semantic segmentation model.

        Args:
            device: torch device (cuda/cpu/mps) or None for auto-detection
            model_variant: RF-DETR segmentation variant (e.g. "seg-medium", "RFDETRSegMedium")
            weights_path: Optional path to model weights (.pt). If empty, uses weights_name.
            weights_name: Optional hosted weights key (e.g. "rf-detr-seg-medium.pt")
            semantic_dataset_type: Target dataset type for color mapping
            image_size: (height, width) - currently not used (model handles resizing)
            semantic_feature_type: LABEL or PROBABILITY_VECTOR
            confidence_threshold: Minimum score for detections
            download_if_missing: Download weights if missing and hosted
            enforce_unique_instance_ids: If True, split disconnected components to unique IDs
            unique_instance_min_pixels: Minimum component size when splitting instance IDs
        """
        self.semantic_dataset_type = semantic_dataset_type
        self.confidence_threshold = confidence_threshold
        self.enforce_unique_instance_ids = enforce_unique_instance_ids
        self.unique_instance_min_pixels = unique_instance_min_pixels

        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        device = self.init_device(device)
        model, transform = self.init_model(
            device,
            model_variant=model_variant,
            weights_path=weights_path,
            weights_name=weights_name,
            download_if_missing=download_if_missing,
        )

        super().__init__(model, transform, device, semantic_feature_type)

        # Initialize semantic color map
        num_classes = self.num_classes()
        self.semantic_color_map_obj = semantic_color_map_factory(
            semantic_dataset_type=semantic_dataset_type,
            semantic_feature_type=semantic_feature_type,
            num_classes=num_classes,
            semantic_segmentation_type=SemanticSegmentationType.RFDETR,
        )
        self.semantic_color_map = self.semantic_color_map_obj.color_map
        self._viz_annotators = None
        self._last_detections = None
        self._last_image_rgb = None

    def _ensure_rfdetr_on_path(self):
        try:
            import rfdetr  # noqa: F401
        except ModuleNotFoundError:
            rf_detr_path = os.path.join(kRootFolder, "thirdparty", "rf_detr")
            if rf_detr_path not in sys.path:
                sys.path.insert(0, rf_detr_path)

    def init_model(
        self,
        device,
        model_variant,
        weights_path,
        weights_name,
        download_if_missing=True,
    ):
        self._ensure_rfdetr_on_path()

        from rfdetr import (
            RFDETRSegPreview,
            RFDETRSegNano,
            RFDETRSegSmall,
            RFDETRSegMedium,
            RFDETRSegLarge,
            RFDETRSegXLarge,
            RFDETRSeg2XLarge,
        )
        from rfdetr.main import HOSTED_MODELS
        from rfdetr.util.files import download_file

        if isinstance(model_variant, RFDETRSegmentationModel):
            variant_key = model_variant.variant_key
            if not weights_name:
                weights_name = model_variant.weights_name
        else:
            label_map = RFDETRSegmentationModel.label_map()
            if isinstance(model_variant, str) and model_variant in label_map:
                model_variant = label_map[model_variant]
                variant_key = model_variant.variant_key
                if not weights_name:
                    weights_name = model_variant.weights_name
            else:
                variant_key = str(model_variant).lower().strip()

        variant_map = {
            "seg-preview": (RFDETRSegPreview, "rf-detr-seg-preview.pt"),
            "seg-nano": (RFDETRSegNano, "rf-detr-seg-nano.pt"),
            "seg-small": (RFDETRSegSmall, "rf-detr-seg-small.pt"),
            "seg-medium": (RFDETRSegMedium, "rf-detr-seg-medium.pt"),
            "seg-large": (RFDETRSegLarge, "rf-detr-seg-large.pt"),
            "seg-xlarge": (RFDETRSegXLarge, "rf-detr-seg-xlarge.pt"),
            "seg-2xlarge": (RFDETRSeg2XLarge, "rf-detr-seg-xxlarge.pt"),
        }

        if variant_key not in variant_map:
            raise ValueError(
                f"Unsupported RF-DETR model_variant '{model_variant}'. "
                f"Supported: {sorted(variant_map.keys())}"
            )

        model_cls, default_weights_name = variant_map[variant_key]
        resolved_weights_name = weights_name if weights_name else default_weights_name

        if weights_path:
            weights_path = Path(weights_path)
            if not weights_path.is_absolute():
                weights_path = Path(kModelsDir) / weights_path
        else:
            if not resolved_weights_name:
                raise ValueError(
                    "RF-DETR: weights_name is empty and weights_path not provided. "
                    "Provide weights_path or a valid weights_name."
                )
            weights_path = Path(kModelsDir) / resolved_weights_name

        if download_if_missing and not weights_path.exists():
            if resolved_weights_name in HOSTED_MODELS:
                weights_path.parent.mkdir(parents=True, exist_ok=True)
                Printer.green(f"RF-DETR: Downloading weights to {weights_path}")
                download_file(HOSTED_MODELS[resolved_weights_name], str(weights_path))
            else:
                Printer.yellow(
                    f"RF-DETR: weights '{resolved_weights_name}' not in hosted list; "
                    f"expected file at {weights_path}"
                )

        if not weights_path.exists():
            raise FileNotFoundError(
                f"RF-DETR weights not found: {weights_path}. "
                f"Provide weights_path or enable download_if_missing."
            )

        model = model_cls(pretrain_weights=str(weights_path), device=device.type)
        return model, None

    def init_device(self, device):
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if device.type == "cuda":
            Printer.green("SemanticSegmentationRfDetr: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise Exception("SemanticSegmentationRfDetr: MPS is not available")
            Printer.yellow("SemanticSegmentationRfDetr: Using MPS")
        else:
            Printer.yellow("SemanticSegmentationRfDetr: Using CPU")

        return device

    def num_classes(self):
        try:
            class_names = getattr(self.model, "class_names", None)
            if isinstance(class_names, dict) and class_names:
                return int(max(class_names.keys()) + 1)
        except Exception:
            pass

        try:
            model_args = getattr(self.model, "model", None)
            if model_args is not None and hasattr(model_args, "args"):
                num_classes = int(model_args.args.num_classes) + 1
                return num_classes
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
        import cv2

        image = self._prepare_image(image)
        img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        detections = self.model.predict(img_rgb, threshold=self.confidence_threshold)
        if isinstance(detections, list):
            detections = detections[0] if detections else None
        self._last_detections = detections
        self._last_image_rgb = img_rgb

        semantics, instances = self._detections_to_semantics(detections, image.shape[:2])
        self.semantics = semantics

        if instances is not None and self.enforce_unique_instance_ids:
            instances = ensure_unique_instance_ids(
                instances, background_id=0, min_pixels=self.unique_instance_min_pixels
            )

        return SemanticSegmentationOutput(semantics=self.semantics, instances=instances)

    def _detections_to_semantics(self, detections, image_shape):
        H, W = image_shape

        if detections is None or len(detections) == 0:
            if self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
                num_classes = self.num_classes()
                return np.zeros((H, W, num_classes), dtype=np.float32), None
            return np.zeros((H, W), dtype=np.int32), None

        masks = getattr(detections, "mask", None)
        class_ids = getattr(detections, "class_id", None)
        scores = getattr(detections, "confidence", None)

        if masks is None or class_ids is None:
            if self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
                num_classes = self.num_classes()
                return np.zeros((H, W, num_classes), dtype=np.float32), None
            return np.zeros((H, W), dtype=np.int32), None

        masks = masks.astype(bool)
        class_ids = class_ids.astype(np.int32)
        if scores is None:
            scores = np.ones(len(class_ids), dtype=np.float32)

        order = np.argsort(scores)[::-1]
        instances = np.zeros((H, W), dtype=np.int32)
        best_score = np.zeros((H, W), dtype=np.float32)

        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            semantic_labels = np.zeros((H, W), dtype=np.int32)
            for instance_id, det_idx in enumerate(order, start=1):
                mask = masks[det_idx]
                class_id = int(class_ids[det_idx])
                score = float(scores[det_idx]) if scores is not None else 1.0
                update = mask & (score > best_score)
                if not np.any(update):
                    continue
                best_score[update] = score
                semantic_labels[update] = class_id
                instances[update] = instance_id
            return semantic_labels, instances

        num_classes = self.num_classes()
        probs = np.zeros((H, W, num_classes), dtype=np.float32)
        for instance_id, det_idx in enumerate(order, start=1):
            mask = masks[det_idx]
            class_id = int(class_ids[det_idx])
            score = float(scores[det_idx]) if scores is not None else 1.0
            if class_id >= num_classes:
                continue
            probs[mask, class_id] = np.maximum(probs[mask, class_id], score)
            update = mask & (score > best_score)
            if not np.any(update):
                continue
            best_score[update] = score
            instances[update] = instance_id

        # Normalize to a probability distribution per pixel when possible.
        sum_probs = probs.sum(axis=-1, keepdims=True)
        valid = sum_probs > 0
        probs[valid] = probs[valid] / sum_probs[valid]

        return probs, instances

    def sem_img_to_viz_rgb(self, semantics, bgr=False):
        if self._last_detections is None or self._last_image_rgb is None:
            return self.semantic_color_map_obj.to_rgb(semantics, bgr=bgr)

        try:
            import cv2
            import supervision as sv
        except Exception:
            return self.semantic_color_map_obj.to_rgb(semantics, bgr=bgr)

        if self._viz_annotators is None:
            self._viz_annotators = (
                sv.BoxAnnotator(),
                sv.MaskAnnotator(),
                sv.LabelAnnotator(),
            )

        box_annotator, mask_annotator, label_annotator = self._viz_annotators

        class_names = getattr(self.model, "class_names", {})
        labels = [
            str(class_names.get(class_id, class_id)) for class_id in self._last_detections.class_id
        ]

        annotated = mask_annotator.annotate(self._last_image_rgb, self._last_detections)
        annotated = box_annotator.annotate(annotated, self._last_detections)
        annotated = label_annotator.annotate(annotated, self._last_detections, labels)

        if bgr:
            return cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR)
        return annotated

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return self.semantic_color_map_obj.sem_img_to_rgb(semantic_img, bgr=bgr)
