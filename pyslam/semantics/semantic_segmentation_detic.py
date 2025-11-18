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

import numpy as np
import os
import sys
import platform

import torch
from PIL import Image
from torchvision import transforms

from .semantic_labels import get_ade20k_to_scannet40_map
from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_utils import (
    labels_color_map_factory,
    labels_to_image,
)
from .semantic_mapping_color_map import Detectron2ColorMapManager

from pyslam.utilities.system import Printer

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = os.path.abspath(os.path.join(kScriptFolder, "..", ".."))

kDeticPath = os.path.abspath(os.path.join(kRootFolder, "thirdparty", "detic"))


class SemanticSegmentationDetic(SemanticSegmentationBase):
    """
    Semantic segmentation using Detic (Detecting Twenty-thousand Classes using Image-level Supervision).

    Detic is an open-vocabulary object detection and segmentation model that can detect
    and segment objects from a large vocabulary (LVIS, COCO, OpenImages, Objects365, or custom).
    This wrapper provides a similar interface to SemanticSegmentationSegformer.
    """

    supported_feature_types = [SemanticFeatureType.LABEL, SemanticFeatureType.PROBABILITY_VECTOR]

    def __init__(
        self,
        device=None,
        config_file=None,
        model_weights="",
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        image_size=(512, 512),
        semantic_feature_type=SemanticFeatureType.LABEL,
        vocabulary="lvis",
        custom_vocabulary="",
        pred_all_class=False,
        confidence_threshold=0.5,
        **kwargs,
    ):
        """
        Initialize Detic semantic segmentation model.

        Args:
            device: torch device (cuda/cpu/mps) or None for auto-detection
            config_file: Path to Detic config file (YAML). If None, uses default.
            model_weights: Path to model weights file (.pth). If empty, uses default from config.
            semantic_dataset_type: Target dataset type for label mapping
            image_size: (height, width) - currently not used (model handles resizing)
            semantic_feature_type: LABEL or PROBABILITY_VECTOR
            vocabulary: Vocabulary to use ('lvis', 'coco', 'openimages', 'objects365', 'custom')
            custom_vocabulary: Custom vocabulary string (comma-separated) if vocabulary='custom'
            pred_all_class: Whether to predict all classes per proposal
            confidence_threshold: Minimum score for predictions
        """

        self.label_mapping = None

        device = self.init_device(device)

        # Initialize Detic model
        demo, transform = self.init_model(
            device,
            config_file,
            model_weights,
            vocabulary,
            custom_vocabulary,
            pred_all_class,
            confidence_threshold,
        )

        # Store demo as model (it contains the predictor)
        super().__init__(demo, transform, device, semantic_feature_type)

        # Extract color map from Detic's detectron2 metadata
        # IMPORTANT: Detic outputs category IDs that can be very large (up to 1203 for LVIS),
        # so we MUST use a large color map (3000 classes) to avoid clamping different objects to the same color
        from .semantic_segmentation_factory import SemanticSegmentationType

        # Initialize color map manager
        self.color_map_manager = Detectron2ColorMapManager(
            metadata=getattr(self.model, "metadata", None),
            semantic_feature_type=semantic_feature_type,
            semantic_dataset_type=semantic_dataset_type,
            semantic_segmentation_type=SemanticSegmentationType.DETIC,
        )

        try:
            if hasattr(self.model, "metadata") and hasattr(self.model.metadata, "thing_classes"):
                # Detic uses thing_classes (objects), not stuff_classes
                # Get the number of classes to determine color map size
                num_classes = self.num_classes() if hasattr(self, "num_classes") else None
                if num_classes is None:
                    num_classes = len(self.model.metadata.thing_classes)

                # Use unified color map extraction function
                if hasattr(self.model.metadata, "thing_colors"):
                    self.semantics_color_map = self.color_map_manager.extract_from_metadata(
                        color_attr="thing_colors",
                        num_classes=max(num_classes, 3000),
                        min_size=2000,
                        use_detectron2_padding=True,
                        verbose=True,
                    )
                else:
                    # thing_colors not available - use detectron2's colormap if possible
                    self.semantics_color_map = self.color_map_manager.create_fallback(
                        num_classes=3000
                    )
                    Printer.green(
                        f"Detic: Using detectron2 colormap (3000 classes) - thing_colors not available"
                    )
                Printer.green(
                    f"Detic: Using color map with {len(self.semantics_color_map)} classes"
                )
            else:
                # Fallback to large color map if metadata not available
                self.semantics_color_map = self.color_map_manager.create_fallback(num_classes=3000)
                Printer.yellow("Detic: Using fallback color map (metadata not available)")
        except Exception as e:
            # Fallback to large color map on error
            self.semantics_color_map = self.color_map_manager.create_fallback(num_classes=3000)
            Printer.yellow(f"Detic: Using fallback color map due to error: {e}")

        self.semantic_dataset_type = semantic_dataset_type

        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

    def init_model(
        self,
        device,
        config_file,
        model_weights,
        vocabulary,
        custom_vocabulary,
        pred_all_class,
        confidence_threshold,
    ):
        """
        Initialize Detic model using VisualizationDemo.

        Returns:
            demo: VisualizationDemo instance
            transform: None (transform is handled internally by detectron2)
        """
        import pyslam.config as config

        config.cfg.set_lib("detic", prepend=True)

        # # Add CenterNet2 to path (required for centernet imports)
        # centernet_path = os.path.join(kDeticPath, "third_party", "CenterNet2")
        # if centernet_path not in sys.path:
        #     sys.path.insert(0, centernet_path)

        from detectron2.config import get_cfg
        from centernet.config import add_centernet_config
        from detic.config import add_detic_config
        from detic.predictor import VisualizationDemo, BUILDIN_CLASSIFIER

        # Resolve BUILDIN_CLASSIFIER paths to absolute paths
        # These paths are used by reset_cls_test and need to be resolved
        resolved_buildin_classifier = {}
        for vocab, rel_path in BUILDIN_CLASSIFIER.items():
            if not os.path.isabs(rel_path):
                abs_path = os.path.abspath(os.path.join(kDeticPath, rel_path))
                if os.path.exists(abs_path):
                    resolved_buildin_classifier[vocab] = abs_path
                    # Only print if verbose (reduce noise)
                    # Printer.green(f"Detic: Resolved {vocab} classifier path to: {abs_path}")
                else:
                    resolved_buildin_classifier[vocab] = rel_path
                    Printer.yellow(
                        f"Detic: Classifier path not found: {abs_path}, using relative: {rel_path}"
                    )
            else:
                resolved_buildin_classifier[vocab] = rel_path

        # Patch BUILDIN_CLASSIFIER with resolved paths
        import detic.predictor as predictor_module

        predictor_module.BUILDIN_CLASSIFIER = resolved_buildin_classifier

        # Create a simple args-like object for VisualizationDemo
        class Args:
            def __init__(self):
                self.vocabulary = vocabulary
                self.custom_vocabulary = custom_vocabulary
                self.pred_all_class = pred_all_class
                self.confidence_threshold = confidence_threshold

        args = Args()

        # Setup config
        cfg = get_cfg()
        add_centernet_config(cfg)
        add_detic_config(cfg)

        # Use default config if not provided
        if config_file is None:
            config_file = os.path.join(
                kDeticPath, "configs", "Detic_LCOCOI21k_CLIP_SwinB_896b32_4x_ft4x_max-size.yaml"
            )

        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Config file not found: {config_file}\n"
                f"Please provide a valid config file path."
            )

        cfg.merge_from_file(config_file)

        # Defrost config to modify settings
        cfg.defrost()

        # Resolve relative paths in config to absolute paths relative to Detic root
        # Detic's code uses os.path.join(kDeticRootPath, CAT_FREQ_PATH), so we need to
        # ensure the path is relative to Detic root, not absolute
        def resolve_detic_path(rel_path):
            """Resolve a relative path relative to Detic root directory."""
            if not rel_path or os.path.isabs(rel_path):
                return rel_path
            # Keep it relative - Detic will join it with kDeticRootPath
            # But ensure it doesn't have leading slashes
            rel_path = rel_path.lstrip("/")
            # Normalize the path to resolve any ../ or ./
            # First, join with Detic root to get absolute path, then make it relative again
            abs_path = os.path.join(kDeticPath, rel_path)
            abs_path = os.path.normpath(abs_path)
            # Convert back to relative path from Detic root
            try:
                rel_path = os.path.relpath(abs_path, kDeticPath)
                # Ensure it doesn't go outside Detic root
                if rel_path.startswith(".."):
                    Printer.yellow(
                        f"Detic: Path goes outside Detic root: {rel_path}, using original"
                    )
                    return rel_path.lstrip("/").replace("../", "")
                return rel_path
            except ValueError:
                # Different drives on Windows, return normalized path
                return os.path.normpath(rel_path).lstrip("/").replace("../", "")

        # Resolve CAT_FREQ_PATH if it exists and is relative
        # Detic's custom_rcnn.py does: os.path.join(kDeticRootPath, CAT_FREQ_PATH)
        # So we need to ensure CAT_FREQ_PATH is relative to Detic root
        if hasattr(cfg.MODEL.ROI_BOX_HEAD, "CAT_FREQ_PATH"):
            cat_freq_path = cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH
            if cat_freq_path:
                # If it's already absolute, check if it's within Detic root
                if os.path.isabs(cat_freq_path):
                    # Check if it's within Detic root
                    try:
                        rel_path = os.path.relpath(cat_freq_path, kDeticPath)
                        if not rel_path.startswith(".."):
                            # It's within Detic root, use relative path
                            cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = rel_path
                            Printer.green(f"Detic: Converted CAT_FREQ_PATH to relative: {rel_path}")
                        else:
                            Printer.yellow(
                                f"Detic: CAT_FREQ_PATH is outside Detic root: {cat_freq_path}"
                            )
                    except ValueError:
                        # Different drives on Windows, keep absolute
                        pass
                else:
                    # It's relative, clean it up and normalize
                    cleaned_path = resolve_detic_path(cat_freq_path)
                    # Use os.path.normpath to resolve any remaining .. or .
                    cleaned_path = os.path.normpath(cleaned_path)
                    # Verify the resolved path exists
                    full_path = os.path.join(kDeticPath, cleaned_path)
                    if os.path.exists(full_path):
                        cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = cleaned_path
                        Printer.green(f"Detic: Using CAT_FREQ_PATH: {cleaned_path}")
                    else:
                        # Try the original path as-is
                        full_path_orig = os.path.join(kDeticPath, cat_freq_path)
                        if os.path.exists(full_path_orig):
                            cfg.MODEL.ROI_BOX_HEAD.CAT_FREQ_PATH = cat_freq_path
                            Printer.green(f"Detic: Using original CAT_FREQ_PATH: {cat_freq_path}")
                        else:
                            Printer.yellow(
                                f"Detic: CAT_FREQ_PATH not found: {full_path} or {full_path_orig}"
                            )

        # Resolve model weights path from config if it exists and is relative
        # This needs to happen before we override with model_weights parameter
        if hasattr(cfg.MODEL, "WEIGHTS") and cfg.MODEL.WEIGHTS:
            config_weights = cfg.MODEL.WEIGHTS
            if not os.path.isabs(config_weights):
                # Try relative to detic root first
                abs_weights = os.path.join(kDeticPath, config_weights)
                if os.path.exists(abs_weights):
                    cfg.MODEL.WEIGHTS = abs_weights
                    Printer.green(f"Detic: Resolved config model weights to: {abs_weights}")
                else:
                    # Try relative to config file directory
                    config_dir = os.path.dirname(config_file)
                    abs_weights = os.path.join(config_dir, config_weights)
                    if os.path.exists(abs_weights):
                        cfg.MODEL.WEIGHTS = abs_weights
                        Printer.green(f"Detic: Resolved config model weights to: {abs_weights}")
                    else:
                        # Try to find a matching model file in the models directory
                        models_dir = os.path.join(kDeticPath, "models")
                        if os.path.exists(models_dir):
                            # Get config filename without extension to match model name
                            config_basename = os.path.splitext(os.path.basename(config_file))[0]
                            # Try to find a model file that matches the config name
                            matching_model = os.path.join(models_dir, f"{config_basename}.pth")
                            if os.path.exists(matching_model):
                                cfg.MODEL.WEIGHTS = matching_model
                                Printer.green(
                                    f"Detic: Auto-matched model weights: {os.path.basename(matching_model)} "
                                    f"(config specified: {config_weights})"
                                )
                            else:
                                # List available models
                                available_models = [
                                    f for f in os.listdir(models_dir) if f.endswith(".pth")
                                ]
                                if available_models:
                                    Printer.yellow(
                                        f"Detic: Config model weights not found: {config_weights}, "
                                        f"available models: {available_models}"
                                    )
                                else:
                                    Printer.yellow(
                                        f"Detic: Config model weights not found: {config_weights}, "
                                        f"and no models found in {models_dir}"
                                    )
                        else:
                            Printer.yellow(
                                f"Detic: Config model weights not found: {config_weights}, "
                                f"tried {os.path.join(kDeticPath, config_weights)} and {abs_weights}"
                            )

        # Set score thresholds
        cfg.MODEL.RETINANET.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = confidence_threshold
        cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = confidence_threshold
        cfg.MODEL.ROI_BOX_HEAD.ZEROSHOT_WEIGHT_PATH = "rand"  # load later
        if not pred_all_class:
            cfg.MODEL.ROI_HEADS.ONE_CLASS_PER_PROPOSAL = True

        # Override model weights if provided
        if model_weights:
            # Resolve relative paths relative to the config file directory or detic root
            if not os.path.isabs(model_weights):
                # Try relative to detic root first
                abs_weights = os.path.join(kDeticPath, model_weights)
                if os.path.exists(abs_weights):
                    model_weights = abs_weights
                else:
                    # Try relative to config file directory
                    config_dir = os.path.dirname(config_file)
                    abs_weights = os.path.join(config_dir, model_weights)
                    if os.path.exists(abs_weights):
                        model_weights = abs_weights

            if not os.path.exists(model_weights):
                raise FileNotFoundError(
                    f"Model weights not found: {model_weights}\n"
                    f"Please provide a valid weights file path."
                )

            cfg.MODEL.WEIGHTS = model_weights
            Printer.green(f"Detic: Loading weights from: {model_weights}")

        # Set device in config
        if device.type == "cuda":
            cfg.MODEL.DEVICE = "cuda"
        elif device.type == "mps":
            # MPS not directly supported by detectron2, fallback to CPU
            Printer.yellow("Detic: MPS not supported by detectron2, using CPU")
            cfg.MODEL.DEVICE = "cpu"
        else:
            cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()

        # Initialize VisualizationDemo
        demo = VisualizationDemo(cfg, args)

        # Transform is handled internally by detectron2, so we return None
        return demo, None

    def init_device(self, device):
        """Initialize and validate device."""
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if device.type == "cuda":
            Printer.green("SemanticSegmentationDetic: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise Exception("SemanticSegmentationDetic: MPS is not available")
            Printer.yellow("SemanticSegmentationDetic: Using MPS (may fallback to CPU)")
        else:
            Printer.yellow("SemanticSegmentationDetic: Using CPU")

        return device

    def num_classes(self):
        """Get number of output classes."""
        # Detic uses open vocabulary, so the number of classes depends on the vocabulary
        try:
            metadata = self.model.metadata
            if hasattr(metadata, "thing_classes"):
                return len(metadata.thing_classes)
        except Exception:
            pass

        # Try to get from model config
        try:
            if hasattr(self.model, "predictor") and hasattr(self.model.predictor, "model"):
                cfg = self.model.predictor.cfg
                if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "ROI_HEADS"):
                    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
                    if num_classes:
                        return int(num_classes)
        except Exception:
            pass

        # Last resort: probe with a dummy image
        try:
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            with torch.no_grad():
                predictions, _ = self.model.run_on_image(dummy_image)
                if "panoptic_seg" in predictions:
                    panoptic_seg, segments_info = predictions["panoptic_seg"]
                    # Count unique category IDs
                    unique_cats = set()
                    for seg_info in segments_info:
                        unique_cats.add(seg_info.get("category_id", 0))
                    return len(unique_cats) if unique_cats else 1
                elif "instances" in predictions:
                    instances = predictions["instances"]
                    if hasattr(instances, "pred_classes"):
                        return (
                            int(instances.pred_classes.max().item() + 1)
                            if len(instances) > 0
                            else 1
                        )
        except Exception as e:
            Printer.red(f"SemanticSegmentationDetic: Failed to get number of classes: {e}")

        # Default fallback
        Printer.yellow(
            "SemanticSegmentationDetic: Could not determine number of classes, using default"
        )
        return 1000  # Common default for LVIS-like datasets

    @torch.no_grad()
    def infer(self, image):
        """
        Run semantic segmentation inference on an image.

        Args:
            image: numpy array of shape (H, W, 3) in BGR format (OpenCV format)

        Returns:
            semantics: numpy array of shape (H, W) for LABEL, or (H, W, num_classes) for PROBABILITY_VECTOR
        """
        # Ensure image is in correct format (BGR, uint8)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # Detic expects BGR format (OpenCV format)
        if len(image.shape) == 2:
            # Grayscale to BGR
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:
            # RGBA to BGR
            image = image[:, :, :3]
        elif image.shape[2] == 3:
            # Assume BGR (OpenCV format)
            pass
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        # Run inference
        predictions, visualized_output = self.model.run_on_image(image)

        # Store visualized_output for use in to_rgb
        self._last_visualized_output = visualized_output

        # Extract semantic segmentation
        if "panoptic_seg" in predictions:
            # Panoptic segmentation: convert to semantic labels
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            semantic_labels = self._panoptic_to_semantic_labels(panoptic_seg, segments_info)

            # Store segments_info and original image for color mapping
            import copy

            self._last_segments_info = copy.deepcopy(segments_info)
            self._last_panoptic_seg = panoptic_seg
            self._last_image = image.copy()

            # Get original image dimensions
            orig_h, orig_w = image.shape[:2]
            pred_h, pred_w = semantic_labels.shape

            # Resize if needed (simple nearest neighbor for labels)
            if orig_h != pred_h or orig_w != pred_w:
                import cv2

                semantic_labels = cv2.resize(
                    semantic_labels.astype(np.uint16),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)

            self.semantics = semantic_labels

        elif "instances" in predictions:
            # Instance segmentation: convert to semantic labels
            instances = predictions["instances"]
            semantic_labels = self._instances_to_semantic_labels(instances, image.shape[:2])

            # Get original image dimensions
            orig_h, orig_w = image.shape[:2]
            pred_h, pred_w = semantic_labels.shape

            # Resize if needed
            if orig_h != pred_h or orig_w != pred_w:
                import cv2

                semantic_labels = cv2.resize(
                    semantic_labels.astype(np.uint16),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)

            self.semantics = semantic_labels

        elif "sem_seg" in predictions:
            # Semantic segmentation output
            sem_seg = predictions["sem_seg"]  # Shape: [num_classes, H, W]
            sem_seg_np = sem_seg.cpu().numpy()

            # Get original image dimensions
            orig_h, orig_w = image.shape[:2]
            pred_h, pred_w = sem_seg_np.shape[1], sem_seg_np.shape[2]

            # Resize if needed
            if orig_h != pred_h or orig_w != pred_w:
                resize_transform = transforms.Resize(
                    (orig_h, orig_w), interpolation=transforms.InterpolationMode.BILINEAR
                )
                sem_seg_tensor = torch.from_numpy(sem_seg_np).unsqueeze(0)
                sem_seg_resized = resize_transform(sem_seg_tensor)[0]
                sem_seg_np = sem_seg_resized.numpy()

            # Apply softmax to get probabilities
            probs = self._softmax_2d(sem_seg_np, axis=0)  # Shape: [num_classes, H, W]
            probs = probs.transpose(1, 2, 0)  # Shape: [H, W, num_classes]

            # Apply label mapping if needed
            if self.label_mapping is not None:
                if self.semantic_feature_type == SemanticFeatureType.LABEL:
                    labels = np.argmax(probs, axis=-1)
                    self.semantics = self.label_mapping[labels]
                else:
                    self.semantics = self.aggregate_probabilities(probs, self.label_mapping)
            else:
                if self.semantic_feature_type == SemanticFeatureType.LABEL:
                    self.semantics = np.argmax(probs, axis=-1).astype(np.int32)
                else:
                    self.semantics = probs
        else:
            raise ValueError("No semantic segmentation found in predictions")

        return self.semantics

    def _panoptic_to_semantic_labels(self, panoptic_seg, segments_info):
        """
        Convert panoptic segmentation directly to semantic labels.

        Args:
            panoptic_seg: Tensor of shape [H, W] with segment IDs
            segments_info: List of dicts with segment information

        Returns:
            semantic_labels: numpy array of shape [H, W] with category IDs
        """
        # Create a mapping from segment ID to category ID
        segment_to_category = {}
        category_ids_seen = set()
        for seg_info in segments_info:
            seg_id = seg_info["id"]
            category_id = seg_info["category_id"]
            segment_to_category[seg_id] = category_id
            category_ids_seen.add(category_id)

        # Debug: print category ID range
        if category_ids_seen:
            min_cat_id = min(category_ids_seen)
            max_cat_id = max(category_ids_seen)
            num_unique = len(category_ids_seen)
            color_map_size = (
                len(self.semantics_color_map) if hasattr(self, "semantics_color_map") else 0
            )
            Printer.green(
                f"Detic: Category IDs range: {min_cat_id}-{max_cat_id} ({num_unique} unique), "
                f"color map size: {color_map_size}"
            )
            if max_cat_id >= color_map_size:
                Printer.red(
                    f"Detic: ERROR - max category ID {max_cat_id} >= color map size {color_map_size}! "
                    f"Colors will be clamped incorrectly."
                )

        # Convert panoptic to semantic labels directly
        panoptic_np = panoptic_seg.cpu().numpy()
        semantic_labels = np.zeros_like(panoptic_np, dtype=np.int32)

        # Map segment IDs to category IDs
        for seg_id, category_id in segment_to_category.items():
            mask = panoptic_np == seg_id
            semantic_labels[mask] = category_id

        return semantic_labels

    def _instances_to_semantic_labels(self, instances, image_shape):
        """
        Convert instance segmentation to semantic labels.

        Args:
            instances: detectron2 Instances object with pred_masks and pred_classes
            image_shape: (height, width) of the image

        Returns:
            semantic_labels: numpy array of shape [H, W] with category IDs
        """
        H, W = image_shape
        semantic_labels = np.zeros((H, W), dtype=np.int32)

        if not hasattr(instances, "pred_masks") or len(instances) == 0:
            return semantic_labels

        # Get masks and classes
        masks = instances.pred_masks.cpu().numpy()  # Shape: [N, H, W]
        classes = instances.pred_classes.cpu().numpy()  # Shape: [N]

        # Overlay masks with class IDs (later instances overwrite earlier ones)
        for i in range(len(instances)):
            mask = masks[i]
            class_id = int(classes[i])
            semantic_labels[mask] = class_id

        return semantic_labels

    def _softmax_2d(self, x, axis=0):
        """Apply softmax along specified axis."""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

    def aggregate_probabilities(
        self, semantics: np.ndarray, label_mapping: np.ndarray
    ) -> np.ndarray:
        """
        Aggregates original probabilities into output probabilities using a label mapping.

        Args:
            semantics: np.ndarray of shape [H, W, original num classes] - softmaxed class probabilities.
            label_mapping: np.ndarray of shape [original num classes] - maps original class indices to output classes.

        Returns:
            np.ndarray of shape [H, W, num output classes] - aggregated probabilities.
        """
        H, W, num_original_classes = semantics.shape
        num_output_classes = int(label_mapping.max() + 1)

        aggregated = np.zeros((H, W, num_output_classes), dtype=semantics.dtype)

        for in_idx, out_idx in enumerate(label_mapping):
            aggregated[..., int(out_idx)] += semantics[..., in_idx]

        return aggregated

    def to_rgb(self, semantics, bgr=False):
        """Convert semantics to RGB visualization using color map.

        Args:
            semantics: Can be either:
                - 1D array of shape (N,) with label IDs for N points
                - 2D array of shape (H, W) with label IDs for an image
            bgr: If True, return BGR format; otherwise RGB

        Returns:
            RGB/BGR array:
                - For 1D input: shape (N, 3) with RGB colors for each point
                - For 2D input: shape (H, W, 3) with RGB image
        """
        # Update color map manager with current color map
        self.color_map_manager.color_map = self.semantics_color_map

        # Prepare panoptic data if available
        panoptic_data = None
        if hasattr(self, "_last_segments_info") and hasattr(self, "_last_panoptic_seg"):
            from detectron2.utils.visualizer import ColorMode

            panoptic_data = {
                "panoptic_seg": self._last_panoptic_seg,
                "segments_info": self._last_segments_info,
                "image": getattr(self, "_last_image", None),
                "visualizer_class": None,  # Use default detectron2 Visualizer
                "instance_mode": getattr(self.model, "instance_mode", ColorMode.IMAGE),
            }

        # Use color map manager's to_rgb method
        return self.color_map_manager.to_rgb(
            semantics,
            panoptic_data=panoptic_data,
            visualized_output=getattr(self, "_last_visualized_output", None),
            bgr=bgr,
        )
