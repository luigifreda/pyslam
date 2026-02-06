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
import copy

import torch
from PIL import Image
from torchvision import transforms

from .semantic_labels import get_ade20k_to_scannet40_map
from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_segmentation_output import SemanticSegmentationOutput
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_color_utils import (
    labels_color_map_factory,
    labels_to_image,
)
from .semantic_color_map_factory import semantic_color_map_factory
from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_instance_utils import ensure_unique_instance_ids

from pyslam.utilities.logging import Printer
from pyslam.config_parameters import Parameters


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder + "/../.."

kEovSegPath = os.path.join(kRootFolder, "thirdparty", "eov_segmentation")
kCacheDir = os.path.join(kRootFolder, "results", "cache")


kVerboseInstanceSegmentation = False


def check_open_clip_torch_version():
    # Check open-clip-torch version
    open_clip_version = None
    try:
        # Try importlib.metadata first (Python 3.8+)
        try:
            import importlib.metadata

            open_clip_version = importlib.metadata.version("open-clip-torch")
        except (ImportError, importlib.metadata.PackageNotFoundError):
            # Fallback for older Python versions
            try:
                import pkg_resources

                open_clip_version = pkg_resources.get_distribution("open-clip-torch").version
            except Exception:
                pass
    except Exception:
        pass

    if open_clip_version != "2.24.0":
        if open_clip_version is None:
            Printer.red(
                "⚠️  WARNING: EOV-Seg requires open-clip-torch==2.24.0, but the package was not found. "
                "Please install it: pip install open-clip-torch==2.24.0"
            )
        else:
            Printer.red(
                f"⚠️  WARNING: EOV-Seg requires open-clip-torch==2.24.0, but found version {open_clip_version}. "
                f"This may cause compatibility issues. Please install the correct version: "
                f"pip install open-clip-torch==2.24.0"
            )


class SemanticSegmentationEovSeg(SemanticSegmentationBase):
    """
    Semantic segmentation using EOV-Seg (Open Vocabulary Segmentation).

    EOV-Seg is an open-vocabulary segmentation model that can segment
    objects and stuff classes based on text prompts. This wrapper provides
    a similar interface to SemanticSegmentationSegformer.
    """

    check_open_clip_torch_version()
    supported_feature_types = [SemanticFeatureType.LABEL, SemanticFeatureType.PROBABILITY_VECTOR]

    def __init__(
        self,
        device=None,
        config_file=None,
        model_weights="",
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        image_size=(512, 512),
        semantic_feature_type=SemanticFeatureType.LABEL,
        cache_dir=None,
        use_compile=False,
        use_fp16=False,
        enforce_unique_instance_ids=Parameters.kSemanticSegmentationEnforceUniqueInstanceIds,
        unique_instance_min_pixels=Parameters.kSemanticSegmentationUniqueInstanceMinPixels,
        **kwargs,
    ):
        """
        Initialize EOV-Seg semantic segmentation model.

        Args:
            device: torch device (cuda/cpu/mps) or None for auto-detection
            config_file: Path to EOV-Seg config file (YAML). If None, uses default (eov_seg_R50.yaml).
                        Available configs: eov_seg_R50.yaml, eov_seg_r50x4.yaml, eov_seg_convnext_l.yaml
                        Note: convnext_l requires convnext_large_d_320 model which may not be available in all open_clip versions.
            model_weights: Path to model weights file (.pth). If empty, uses default from config.
            semantic_dataset_type: Target dataset type for label mapping
            image_size: (height, width) - currently not used (model handles resizing)
            semantic_feature_type: LABEL or PROBABILITY_VECTOR
            cache_dir: Directory for vocabulary cache
            use_compile: Use torch.compile() for optimization (PyTorch 2.0+)
            use_fp16: Use mixed precision (FP16) for faster inference
            enforce_unique_instance_ids: If True, split disconnected components to unique IDs
            unique_instance_min_pixels: Minimum component size when splitting instance IDs
        """

        self.label_mapping = None

        device = self.init_device(device)

        if cache_dir is None:
            cache_dir = kCacheDir

        # Initialize EOV-Seg model
        demo, transform = self.init_model(
            device, config_file, model_weights, cache_dir, use_compile, use_fp16
        )

        # Store demo as model (it contains the predictor)
        super().__init__(demo, transform, device, semantic_feature_type)

        # Extract color map from EOV-Seg's detectron2 metadata using SemanticColorMap
        # This uses the same colors as detectron2's visualizer, extracted directly from metadata
        # Now self.model is available after super().__init__

        # Get the number of classes to determine color map size
        num_classes = None
        try:
            if hasattr(self.model, "metadata") and hasattr(self.model.metadata, "stuff_classes"):
                num_classes = self.num_classes() if hasattr(self, "num_classes") else None
                if num_classes is None:
                    num_classes = len(self.model.metadata.stuff_classes)
        except Exception:
            pass

        # Initialize semantic color map using factory - it will create SemanticColorMapDetectron2 if metadata is available
        try:
            metadata = getattr(self.model, "metadata", None)
            self.semantic_color_map_obj = semantic_color_map_factory(
                semantic_dataset_type=semantic_dataset_type,
                semantic_feature_type=semantic_feature_type,
                num_classes=num_classes,
                semantic_segmentation_type=SemanticSegmentationType.EOV_SEG,
                metadata=metadata,
            )
            # Extract the color map array for backward compatibility
            self.semantic_color_map = self.semantic_color_map_obj.color_map
        except Exception as e:
            # Fallback to factory-based color map on error
            Printer.yellow(f"EOV-Seg: Using fallback color map due to error: {e}")
            self.semantic_color_map_obj = semantic_color_map_factory(
                semantic_dataset_type=semantic_dataset_type,
                semantic_feature_type=semantic_feature_type,
                num_classes=num_classes or 150,
                semantic_segmentation_type=SemanticSegmentationType.EOV_SEG,
            )
            self.semantic_color_map = self.semantic_color_map_obj.color_map

        self.semantic_dataset_type = semantic_dataset_type
        self.enforce_unique_instance_ids = enforce_unique_instance_ids
        self.unique_instance_min_pixels = unique_instance_min_pixels

        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

    def init_model(self, device, config_file, model_weights, cache_dir, use_compile, use_fp16):
        """
        Initialize EOV-Seg model using VisualizationDemo.

        Returns:
            demo: VisualizationDemo instance
            transform: None (transform is handled internally by detectron2)
        """
        import pyslam.config as config

        config.cfg.set_lib("eov_segmentation", prepend=True)

        from .detectron2_utils import check_detectron2_import

        check_detectron2_import()

        from detectron2.config import get_cfg
        from detectron2.projects.deeplab import add_deeplab_config

        from eov_seg import add_eov_config
        from eov_segmentation.demo.predictor import VisualizationDemo

        # Setup config
        cfg = get_cfg()
        add_deeplab_config(cfg)
        add_eov_config(cfg)

        # Use default config if not provided
        # Default to convnext_l config to match the demo (run_demo.sh)
        if config_file is None:
            config_file = os.path.join(kEovSegPath, "configs", "eov_seg", "eov_seg_convnext_l.yaml")

        if not os.path.exists(config_file):
            raise FileNotFoundError(
                f"Config file not found: {config_file}\n"
                f"Please provide a valid config file path."
            )

        cfg.merge_from_file(config_file)
        cfg.freeze()

        # Set default model weights if not provided (matching demo)
        if not model_weights:
            # Default to convnext-l.pth to match the demo
            default_weights = os.path.join(kEovSegPath, "checkpoints", "convnext-l.pth")
            if os.path.exists(default_weights):
                model_weights = default_weights
                Printer.green(f"EOV-Seg: Using default weights: {model_weights}")
            else:
                Printer.yellow(
                    f"EOV-Seg: Default weights not found at {default_weights}, using config default"
                )

        # Override model weights if provided
        if model_weights:
            cfg.defrost()
            # Resolve relative paths relative to the config file directory or eov_segmentation root
            if not os.path.isabs(model_weights):
                # Try relative to eov_segmentation root first
                abs_weights = os.path.join(kEovSegPath, model_weights)
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
            Printer.green(f"EOV-Seg: Loading weights from: {model_weights}")
            cfg.freeze()

        # Set device in config
        cfg.defrost()
        if device.type == "cuda":
            cfg.MODEL.DEVICE = "cuda"
        elif device.type == "mps":
            # MPS not directly supported by detectron2, fallback to CPU
            Printer.yellow("EOV-Seg: MPS not supported by detectron2, using CPU")
            cfg.MODEL.DEVICE = "cpu"
        else:
            cfg.MODEL.DEVICE = "cpu"
        cfg.freeze()

        # Initialize VisualizationDemo
        demo = VisualizationDemo(
            cfg, cache_dir=cache_dir, use_compile=use_compile, use_fp16=use_fp16
        )

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
            Printer.green("SemanticSegmentationEovSeg: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise Exception("SemanticSegmentationEovSeg: MPS is not available")
            Printer.yellow("SemanticSegmentationEovSeg: Using MPS (may fallback to CPU)")
        else:
            Printer.yellow("SemanticSegmentationEovSeg: Using CPU")

        return device

    def num_classes(self):
        """Get number of output classes."""
        # EOV-Seg uses open vocabulary, so the number of classes depends on the metadata
        try:
            metadata = self.model.metadata
            if hasattr(metadata, "stuff_classes"):
                return len(metadata.stuff_classes)
        except Exception:
            pass

        # Try to get from model config
        try:
            if hasattr(self.model, "predictor") and hasattr(self.model.predictor, "model"):
                cfg = self.model.predictor.cfg
                if hasattr(cfg, "MODEL") and hasattr(cfg.MODEL, "SEM_SEG_HEAD"):
                    num_classes = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
                    if num_classes:
                        return int(num_classes)
        except Exception:
            pass

        # Last resort: probe with a dummy image
        try:
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            with torch.no_grad():
                predictions, _ = self.model.run_on_image(dummy_image)
                if "sem_seg" in predictions:
                    return int(predictions["sem_seg"].shape[0])
                elif "panoptic_seg" in predictions:
                    # For panoptic, we need to check metadata
                    metadata = self.model.metadata
                    if hasattr(metadata, "stuff_classes"):
                        return len(metadata.stuff_classes)
        except Exception as e:
            Printer.red(f"SemanticSegmentationEovSeg: Failed to get number of classes: {e}")

        # Default fallback
        Printer.yellow(
            "SemanticSegmentationEovSeg: Could not determine number of classes, using default"
        )
        return 150  # Common default for ADE20K-like datasets

    @torch.no_grad()
    def infer(self, image) -> SemanticSegmentationOutput:
        """
        Run semantic segmentation inference on an image.

        Args:
            image: numpy array of shape (H, W, 3) in BGR format (OpenCV format)

        Returns:
            SemanticSegmentationOutput: object containing semantics and optionally instances
        """
        # Ensure image is in correct format (BGR, uint8)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # EOV-Seg expects BGR format (OpenCV format)
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
        # EOV-Seg's VisualizationDemo.run_on_image() returns both predictions and visualized_output
        # This is the preferred method as it provides pre-computed visualization with overlays
        predictions, visualized_output = self.model.run_on_image(image)

        # Store visualized_output for use in to_rgb
        # The centralized visualization system in semantic_color_map_obj.to_rgb() will use
        # this pre-computed visualization if available, ensuring consistent output across models
        self._last_visualized_output = visualized_output

        instances = None
        semantics = None

        if (
            self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR
            and "sem_seg" not in predictions
        ):
            raise ValueError(
                "EOV-Seg probability vectors require a semantic segmentation head "
                "('sem_seg' output). Current EOV-Seg outputs are panoptic only."
            )

        # Extract semantic segmentation - optimized path for panoptic
        if "panoptic_seg" in predictions:
            # Direct extraction from panoptic (faster, no softmax needed)
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            semantic_labels = self._panoptic_to_semantic_labels(panoptic_seg, segments_info)
            instances = self._panoptic_to_instances(panoptic_seg, segments_info)

            # Debug: Log when instances are not available
            if instances is None and kVerboseInstanceSegmentation:
                SemanticSegmentationBase.print(
                    f"EOV-Seg: No instance IDs extracted from panoptic segmentation. "
                    f"Total segments: {len(segments_info)}"
                )
                # Dump segments_info for debugging - show first few segments
                SemanticSegmentationBase.print(
                    "EOV-Seg: Dumping segments_info structure (first 10 segments):"
                )
                for i, seg_info in enumerate(segments_info[:10]):
                    isthing_val = seg_info.get("isthing", "N/A")
                    isthing_type = type(isthing_val).__name__ if isthing_val != "N/A" else "N/A"
                    SemanticSegmentationBase.print(
                        f"  Segment {i}: id={seg_info.get('id', 'N/A')}, "
                        f"category_id={seg_info.get('category_id', 'N/A')}, "
                        f"isthing={isthing_val} (type: {isthing_type})"
                    )
                if len(segments_info) > 10:
                    SemanticSegmentationBase.print(
                        f"  ... and {len(segments_info) - 10} more segments"
                    )

            # Store segments_info and original image for color mapping
            # Make a deep copy to ensure segments_info is not modified

            self._last_segments_info = copy.deepcopy(segments_info)
            self._last_panoptic_seg = panoptic_seg
            # Store original image (BGR) for visualization - convert to RGB when needed
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

                if instances is not None:
                    instances = cv2.resize(
                        instances.astype(np.int32),
                        (orig_w, orig_h),
                        interpolation=cv2.INTER_NEAREST,
                    ).astype(np.int32)

            if instances is not None and self.enforce_unique_instance_ids:
                instances = ensure_unique_instance_ids(
                    instances, background_id=0, min_pixels=self.unique_instance_min_pixels
                )

            semantics = semantic_labels

        elif "sem_seg" in predictions:
            # Fallback to semantic segmentation output
            sem_seg = predictions["sem_seg"]  # Shape: [num_classes, H, W]
            sem_seg_np = sem_seg.cpu().numpy()

            # Get original image dimensions
            orig_h, orig_w = image.shape[:2]
            pred_h, pred_w = sem_seg_np.shape[1], sem_seg_np.shape[2]

            # Resize if needed
            if orig_h != pred_h or orig_w != pred_w:
                from torchvision import transforms

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
                    semantics = self.label_mapping[labels]
                else:
                    semantics = self.aggregate_probabilities(probs, self.label_mapping)
            else:
                if self.semantic_feature_type == SemanticFeatureType.LABEL:
                    semantics = np.argmax(probs, axis=-1).astype(np.int32)
                else:
                    semantics = probs
        else:
            raise ValueError("No semantic segmentation found in predictions")

        self.semantics = semantics
        return SemanticSegmentationOutput(semantics=self.semantics, instances=instances)

    def _panoptic_to_semantic_labels(self, panoptic_seg, segments_info):
        """
        Convert panoptic segmentation directly to semantic labels (faster, no softmax).

        Args:
            panoptic_seg: Tensor of shape [H, W] with segment IDs
            segments_info: List of dicts with segment information

        Returns:
            semantic_labels: numpy array of shape [H, W] with category IDs
        """
        # Get number of classes from metadata for validation
        num_classes = (
            len(self.model.metadata.stuff_classes)
            if hasattr(self.model.metadata, "stuff_classes")
            else 150
        )

        # Create a mapping from segment ID to category ID
        segment_to_category = {}
        for seg_info in segments_info:
            seg_id = seg_info["id"]
            category_id = seg_info["category_id"]
            # Store category_id as-is (don't clamp here, handle in visualization)
            segment_to_category[seg_id] = category_id

        # Convert panoptic to semantic labels directly
        panoptic_np = panoptic_seg.cpu().numpy()
        semantic_labels = np.zeros_like(panoptic_np, dtype=np.int32)

        # Map segment IDs to category IDs
        for seg_id, category_id in segment_to_category.items():
            mask = panoptic_np == seg_id
            semantic_labels[mask] = category_id

        return semantic_labels

    def _panoptic_to_instances(self, panoptic_seg, segments_info):
        """
        Extract instance IDs from panoptic segmentation.

        Args:
            panoptic_seg: Tensor of shape [H, W] with segment IDs
            segments_info: List of dicts with segment information

        Returns:
            instance_ids: numpy array of shape [H, W] with instance IDs (0 for background/stuff)
        """
        panoptic_np = panoptic_seg.cpu().numpy()

        # Verify segment IDs in panoptic_seg match segments_info
        unique_segment_ids_in_image = np.unique(panoptic_np)
        segment_ids_in_info = {seg_info["id"] for seg_info in segments_info}
        # Note: 0 is typically background/void, so we exclude it
        unique_segment_ids_in_image = set(unique_segment_ids_in_image) - {0}

        # Debug: Check for mismatches
        missing_in_info = unique_segment_ids_in_image - segment_ids_in_info
        missing_in_image = segment_ids_in_info - unique_segment_ids_in_image
        if (missing_in_info or missing_in_image) and kVerboseInstanceSegmentation:
            SemanticSegmentationBase.print(
                f"EOV-Seg: _panoptic_to_instances: Segment ID mismatch detected! "
                f"IDs in image but not in info: {missing_in_info}, "
                f"IDs in info but not in image: {missing_in_image}"
            )

        # Get metadata for fallback checking if isthing is not reliable
        metadata = getattr(self.model, "metadata", None)
        thing_category_ids = None
        num_thing_classes = None
        if metadata is not None:
            if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
                # Get the set of thing category IDs from metadata
                thing_category_ids = set(metadata.thing_dataset_id_to_contiguous_id.values())
            # Also get number of thing classes as additional fallback
            if hasattr(metadata, "thing_classes"):
                num_thing_classes = len(metadata.thing_classes)

        # Initialize instance_ids as None - will be created only if instances are found
        instance_ids = None

        # Debug counters
        num_thing_segments = 0
        num_stuff_segments = 0
        num_missing_isthing = 0

        # Map segment IDs to instance IDs (> 0 for actual instances, 0 for background/stuff)
        # We only need instance IDs > 0, not necessarily contiguous
        # Collect thing segment IDs that need remapping (if seg_id <= 0)
        segment_id_to_instance_id = {}
        next_instance_id = 1  # Start from 1 (0 is reserved for background/stuff)
        thing_seg_ids = []  # Track all thing segment IDs

        # First pass: identify all thing segments and create mapping
        for seg_info in segments_info:
            seg_id = seg_info["id"]
            category_id = seg_info.get("category_id", None)

            # Determine if this is a "thing" class (instance)
            # First try to get from segments_info
            is_thing = seg_info.get("isthing", None)

            # Handle both boolean and integer (0/1) values
            if is_thing is None:
                num_missing_isthing += 1
                # Fallback: check metadata using category_id
                if category_id is not None:
                    if thing_category_ids is not None:
                        is_thing = category_id in thing_category_ids
                    elif num_thing_classes is not None:
                        # Fallback: assume thing classes come first (common in detectron2)
                        is_thing = category_id < num_thing_classes
                    else:
                        is_thing = False
                else:
                    is_thing = False
            else:
                # Convert to boolean if it's an integer (0/1)
                if isinstance(is_thing, (int, np.integer)):
                    is_thing = bool(is_thing)
                # If it's already boolean, use as-is

            # Only assign instance IDs to "thing" classes (instances)
            # "stuff" classes (background regions) remain 0
            if is_thing:
                num_thing_segments += 1
                thing_seg_ids.append(seg_id)
                # Only remap if segment ID is <= 0 (shouldn't happen, but be safe)
                if seg_id <= 0:
                    segment_id_to_instance_id[seg_id] = next_instance_id
                    next_instance_id += 1
            else:
                num_stuff_segments += 1

        # Second pass: optimized remapping using segment IDs directly if possible
        if len(thing_seg_ids) > 0:
            # Use segment IDs directly if they're all > 0 (most common case)
            if len(segment_id_to_instance_id) == 0:
                # All segment IDs are > 0, use them directly - no remapping needed!
                instance_ids = np.zeros_like(panoptic_np, dtype=np.int32)
                # Use vectorized assignment for all thing segments at once
                thing_mask = np.isin(panoptic_np, thing_seg_ids)
                instance_ids[thing_mask] = panoptic_np[thing_mask]
            else:
                # Some segment IDs need remapping (seg_id <= 0)
                # Find max segment ID to determine lookup table size
                max_seg_id = int(panoptic_np.max()) if panoptic_np.size > 0 else 0
                max_seg_id = max(max_seg_id, max(thing_seg_ids))
                min_seg_id = int(panoptic_np.min()) if panoptic_np.size > 0 else 0
                min_seg_id = min(min_seg_id, min(thing_seg_ids))
                min_seg_id = min(min_seg_id, min(segment_id_to_instance_id.keys()))
                offset = -min_seg_id if min_seg_id < 0 else 0
                lookup_size = max_seg_id + offset + 1

                # Guard against huge sparse IDs that would blow up the lookup table.
                if lookup_size > 5_000_000:
                    instance_ids = np.zeros_like(panoptic_np, dtype=np.int32)
                    for seg_id in thing_seg_ids:
                        if seg_id > 0:
                            instance_ids[panoptic_np == seg_id] = np.int32(seg_id)
                    for seg_id, instance_id in segment_id_to_instance_id.items():
                        instance_ids[panoptic_np == seg_id] = np.int32(instance_id)
                else:
                    # Create lookup table: lookup[seg_id + offset] = instance_id, default 0 (background)
                    # Use int32 to match panoptic_np dtype
                    lookup = np.zeros(lookup_size, dtype=np.int32)
                    # First, copy valid segment IDs (> 0) directly
                    for seg_id in thing_seg_ids:
                        if seg_id > 0:
                            lookup[seg_id + offset] = np.int32(seg_id)
                    # Then, remap problematic ones (<= 0)
                    for seg_id, instance_id in segment_id_to_instance_id.items():
                        lookup[seg_id + offset] = np.int32(instance_id)

                    # Vectorized remapping: single pass through image using advanced indexing
                    instance_ids = lookup[panoptic_np + offset]

        # Debug logging when instances are not available
        if num_thing_segments == 0 and kVerboseInstanceSegmentation:
            SemanticSegmentationBase.print(
                f"EOV-Seg: _panoptic_to_instances: No instances found. "
                f"Total segments: {len(segments_info)}, "
                f"Thing segments: {num_thing_segments}, "
                f"Stuff segments: {num_stuff_segments}, "
                f"Missing 'isthing' field: {num_missing_isthing}"
            )
            SemanticSegmentationBase.print(
                f"EOV-Seg: NOTE: This is expected when all segments are 'stuff' classes "
                f"(background regions like walls, floors, sky, etc.). "
                f"Only 'thing' classes (objects like person, car, chair) have instance IDs."
            )
            if metadata is not None:
                SemanticSegmentationBase.print(
                    f"EOV-Seg: Metadata available - "
                    f"thing_dataset_id_to_contiguous_id: {hasattr(metadata, 'thing_dataset_id_to_contiguous_id')}, "
                    f"thing_classes: {hasattr(metadata, 'thing_classes')}"
                )
                if hasattr(metadata, "thing_dataset_id_to_contiguous_id"):
                    SemanticSegmentationBase.print(
                        f"EOV-Seg: thing_category_ids from metadata: {thing_category_ids}"
                    )
                if hasattr(metadata, "thing_classes"):
                    SemanticSegmentationBase.print(
                        f"EOV-Seg: num_thing_classes from metadata: {num_thing_classes}"
                    )
            else:
                SemanticSegmentationBase.print(
                    "EOV-Seg: No metadata available for fallback checking"
                )
        elif num_thing_segments > 0 and kVerboseInstanceSegmentation:
            SemanticSegmentationBase.print(
                f"EOV-Seg: _panoptic_to_instances: Successfully extracted {num_thing_segments} instances "
                f"from {len(segments_info)} total segments"
            )

        # Return None if no instances were found (all segments are stuff classes)
        # Return array with contiguous instance IDs if instances are found
        return instance_ids

    def _panoptic_to_semantic(self, panoptic_seg, segments_info):
        """
        Convert panoptic segmentation to semantic segmentation logits (for probability vectors).

        Args:
            panoptic_seg: Tensor of shape [H, W] with segment IDs
            segments_info: List of dicts with segment information

        Returns:
            sem_seg: Tensor of shape [num_classes, H, W] with semantic logits
        """
        # Get number of classes from metadata
        num_classes = (
            len(self.model.metadata.stuff_classes)
            if hasattr(self.model.metadata, "stuff_classes")
            else 150
        )

        # Get semantic labels first
        semantic_labels = self._panoptic_to_semantic_labels(panoptic_seg, segments_info)

        # Don't clamp - use category IDs as-is (matching demo behavior)
        # Note: Some category IDs may exceed num_classes, but that's handled by the model

        # Convert to one-hot encoding (hard assignment for panoptic)
        H, W = semantic_labels.shape
        sem_seg = np.zeros((num_classes, H, W), dtype=np.float32)

        # Use high logit value for assigned classes, low for others
        for c in range(num_classes):
            mask = semantic_labels == c
            sem_seg[c, mask] = 10.0  # High logit value
            sem_seg[c, ~mask] = -10.0  # Low logit value

        return torch.from_numpy(sem_seg)

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

    def sem_img_to_viz_rgb(self, semantics, bgr=False):
        """Convert semantics to RGB visualization using extracted detectron2 color map.

        This method uses the color map extracted from detectron2's metadata (same colors
        as detectron2's visualizer) for efficient direct color mapping. The visualizer
        is only used for panoptic segmentation where full visualization with overlays is needed.

        This method handles two cases:
        1. 1D array of point labels -> convert each label to RGB color directly
        2. 2D semantic image -> use extracted color map for direct color mapping

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
        # Prepare panoptic data if available
        panoptic_data = None
        if hasattr(self, "_last_segments_info") and hasattr(self, "_last_panoptic_seg"):
            from .detectron2_utils import check_detectron2_import

            check_detectron2_import()

            from detectron2.utils.visualizer import ColorMode

            # EOV-Seg uses OpenVocabVisualizer
            try:
                from eov_segmentation.demo.predictor import OpenVocabVisualizer

                visualizer_class = OpenVocabVisualizer
            except ImportError:
                visualizer_class = None

            panoptic_data = {
                "panoptic_seg": self._last_panoptic_seg,
                "segments_info": self._last_segments_info,
                "image": getattr(self, "_last_image", None),
                "visualizer_class": visualizer_class,
                "instance_mode": getattr(self.model, "instance_mode", ColorMode.IMAGE),
            }

        # Use semantic color map's centralized to_rgb method
        # This method handles visualization consistently across all detectron2-based models:
        # - Uses pre-computed visualized_output if available (DETIC, EOV_SEG)
        # - Falls back to panoptic_data visualization if needed (supports OpenVocabVisualizer)
        # - Uses direct color mapping as final fallback
        return self.semantic_color_map_obj.to_rgb(
            semantics,
            panoptic_data=panoptic_data,
            visualized_output=getattr(self, "_last_visualized_output", None),
            bgr=bgr,
        )

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return self.semantic_color_map_obj.sem_img_to_rgb(semantic_img, bgr=bgr)
