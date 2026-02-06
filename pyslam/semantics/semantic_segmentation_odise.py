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

import copy
import importlib.util
import os
import sys
import time
from typing import Optional, Tuple, Dict, Any

import numpy as np
import torch
import cv2
from torchvision import transforms

from .semantic_segmentation_base import SemanticSegmentationBase
from .semantic_segmentation_output import SemanticSegmentationOutput
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_color_map_factory import semantic_color_map_factory
from .semantic_instance_utils import ensure_unique_instance_ids

from pyslam.utilities.logging import Printer
from pyslam.config_parameters import Parameters


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = os.path.abspath(os.path.join(kScriptFolder, "..", ".."))
kOdisePath = os.path.abspath(os.path.join(kRootFolder, "thirdparty", "odise"))


class OdiseDemoWrapper:
    """
    Demo wrapper for ODISE inference that handles preprocessing and model execution.
    """

    def __init__(self, inference_model, metadata, aug):
        self.inference_model = inference_model
        self.metadata = metadata
        self.aug = aug
        self.cpu_device = torch.device("cpu")

    def predict(self, original_image: np.ndarray) -> Dict[str, Any]:
        """
        Run inference on an image.

        Args:
            original_image: RGB image as numpy array of shape (H, W, 3)

        Returns:
            predictions: Dictionary containing model predictions
        """
        # Ensure model is in eval mode (required for ODISE)
        self.inference_model.eval()

        height, width = original_image.shape[:2]
        Printer.green(f"ODISE: Preprocessing image {width}x{height}...")

        # Preprocessing/augmentation
        preprocess_start = time.time()
        from .detectron2_utils import check_detectron2_import

        check_detectron2_import()

        from detectron2.data import transforms as T

        aug_input = T.AugInput(original_image, sem_seg=None)
        self.aug(aug_input)
        image = aug_input.image
        image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
        preprocess_time = time.time() - preprocess_start

        # Get actual input size after augmentation
        if hasattr(image, "shape"):
            _, aug_h, aug_w = image.shape
            Printer.green(
                f"ODISE: Augmented image size: {aug_w}x{aug_h} (preprocessing: {preprocess_time:.3f}s)"
            )

        inputs = {"image": image, "height": height, "width": width}

        # Model forward pass
        Printer.green("ODISE: Running model forward pass...")
        forward_start = time.time()

        predictions = self._run_forward_pass(inputs)
        forward_time = time.time() - forward_start

        # Log prediction details
        self._log_prediction_details(predictions, forward_time)

        return predictions

    def _run_forward_pass(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute model forward pass with error handling for cuDNN issues.

        Args:
            inputs: Preprocessed input dictionary

        Returns:
            predictions: Model predictions
        """
        # Use torch.amp.autocast for PyTorch 2.0+ compatibility
        if hasattr(torch.amp, "autocast"):
            autocast_context = torch.amp.autocast(
                "cuda" if next(self.inference_model.parameters()).is_cuda else "cpu"
            )
        else:
            from torch.cuda.amp import autocast

            autocast_context = autocast()

        with autocast_context:
            try:
                return self.inference_model([inputs])[0]
            except RuntimeError as e:
                if "cuDNN" in str(e) or "cudnn" in str(e).lower():
                    Printer.warning(
                        f"ODISE: cuDNN error detected: {e}. Disabling cuDNN and retrying..."
                    )
                    torch.backends.cudnn.enabled = False
                    predictions = self.inference_model([inputs])[0]
                    Printer.green("ODISE: Inference continuing on GPU without cuDNN")
                    return predictions
                else:
                    raise

    def _log_prediction_details(self, predictions: Dict[str, Any], forward_time: float):
        """Log details about the predictions."""
        if "panoptic_seg" in predictions:
            panoptic_seg, segments_info = predictions["panoptic_seg"]
            num_segments = len(segments_info) if segments_info else 0
            Printer.green(
                f"ODISE: Forward pass completed in {forward_time:.3f}s "
                f"(panoptic segmentation: {num_segments} segments)"
            )
        elif "sem_seg" in predictions:
            sem_seg = predictions["sem_seg"]
            Printer.green(
                f"ODISE: Forward pass completed in {forward_time:.3f}s "
                f"(semantic segmentation: {sem_seg.shape[0]} classes)"
            )
        else:
            Printer.green(f"ODISE: Forward pass completed in {forward_time:.3f}s")


# ============================================================================


class SemanticSegmentationOdise(SemanticSegmentationBase):
    """
    Semantic segmentation using ODISE (Open-vocabulary DIffusion-based panoptic SEgmentation).

    ODISE is an open-vocabulary panoptic segmentation model that can segment
    objects and stuff classes from COCO and ADE20K datasets. This wrapper provides
    a similar interface to SemanticSegmentationEovSeg and SemanticSegmentationDetic.
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
        vocab="",
        label_list=None,
        input_size=None,
        max_size=None,
        use_fp16=True,
        enforce_unique_instance_ids=Parameters.kSemanticSegmentationEnforceUniqueInstanceIds,
        unique_instance_min_pixels=Parameters.kSemanticSegmentationUniqueInstanceMinPixels,
        **kwargs,
    ):
        """
        Initialize ODISE semantic segmentation model.

        Args:
            device: torch device (cuda/cpu/mps) or None for auto-detection
            config_file: Path to ODISE config file (Python). If None, uses default from model_zoo.
            model_weights: Path to model weights file (.pth). If empty, uses default from config.
            semantic_dataset_type: Target dataset type for label mapping
            image_size: (height, width) - used to derive input_size if input_size is None
            semantic_feature_type: LABEL or PROBABILITY_VECTOR
            vocab: Extra vocabulary for segmentation (format: 'a1,a2;b1,b2')
            label_list: List of categories to use (e.g., ['COCO (133 categories)', 'ADE (150 categories)'])
            input_size: Input image short edge size. If None, derived from image_size or defaults to 512.
                       Smaller values (384, 512) are faster but less accurate
            max_size: Maximum input image size. If None, defaults to 1.5 * input_size
            use_fp16: Use float16 for faster inference (default: True)
            enforce_unique_instance_ids: If True, split disconnected components to unique IDs
            unique_instance_min_pixels: Minimum component size when splitting instance IDs
        """
        self.label_mapping = None
        self.semantic_dataset_type = semantic_dataset_type
        self.enforce_unique_instance_ids = enforce_unique_instance_ids
        self.unique_instance_min_pixels = unique_instance_min_pixels

        # Validate feature type
        if semantic_feature_type not in self.supported_feature_types:
            raise ValueError(
                f"Semantic feature type {semantic_feature_type} is not supported for {self.__class__.__name__}"
            )

        # Derive input size parameters
        input_size, max_size = self._derive_input_sizes(image_size, input_size, max_size)
        Printer.green(f"ODISE: Input size: {input_size}, Max size: {max_size}")

        # Initialize device
        device = self.init_device(device)

        # Initialize ODISE model
        demo, transform = self._initialize_model(
            device,
            config_file,
            model_weights,
            vocab,
            label_list,
            input_size,
            max_size,
            use_fp16,
        )

        # Store demo as model (it contains the inference wrapper)
        super().__init__(demo, transform, device, semantic_feature_type)

        # Initialize color map using SemanticColorMap
        self._initialize_color_map(semantic_feature_type, semantic_dataset_type)

    def _derive_input_sizes(
        self, image_size: Tuple[int, int], input_size: Optional[int], max_size: Optional[int]
    ) -> Tuple[int, int]:
        """
        Derive input_size and max_size from image_size if not provided.

        Args:
            image_size: (height, width) tuple
            input_size: Optional input size override
            max_size: Optional max size override

        Returns:
            (input_size, max_size) tuple
        """
        if input_size is None:
            if image_size and isinstance(image_size, (tuple, list)) and len(image_size) >= 2:
                # Use the smaller dimension as input_size, but cap at reasonable values
                input_size = min(image_size[0], image_size[1])
                # Cap between 384 and 1024 for reasonable performance
                input_size = max(384, min(input_size, 1024))
            else:
                # Default to 512 for good balance between speed and accuracy
                input_size = 512

        if max_size is None:
            max_size = int(input_size * 1.5)  # 1.5x ratio is reasonable

        return input_size, max_size

    def _initialize_color_map(
        self,
        semantic_feature_type: SemanticFeatureType,
        semantic_dataset_type: SemanticDatasetType,
    ):
        """Initialize color map from ODISE's detectron2 metadata using semantic_color_map_factory."""
        from .semantic_segmentation_factory import SemanticSegmentationType

        # Get metadata
        demo_metadata = getattr(self, "_demo_metadata", None)
        if demo_metadata is None:
            demo_metadata = getattr(self.model, "metadata", None)

        # Get the number of classes to determine color map size
        num_classes = None
        try:
            num_classes = self.num_classes() if hasattr(self, "num_classes") else None
        except Exception:
            pass

        # Initialize semantic color map using factory - it will create SemanticColorMapDetectron2 if metadata is available
        try:
            self.semantic_color_map_obj = semantic_color_map_factory(
                semantic_dataset_type=semantic_dataset_type,
                semantic_feature_type=semantic_feature_type,
                num_classes=num_classes,
                semantic_segmentation_type=SemanticSegmentationType.ODISE,
                metadata=demo_metadata,
            )
            # Extract the color map array for backward compatibility
            self.semantic_color_map = self.semantic_color_map_obj.color_map
        except Exception as e:
            # Fallback to factory-based color map on error
            Printer.yellow(f"ODISE: Using fallback color map due to error: {e}")
            self.semantic_color_map_obj = semantic_color_map_factory(
                semantic_dataset_type=semantic_dataset_type,
                semantic_feature_type=semantic_feature_type,
                num_classes=num_classes or 300,
                semantic_segmentation_type=SemanticSegmentationType.ODISE,
            )
            self.semantic_color_map = self.semantic_color_map_obj.color_map

    # ========================================================================
    # Model Initialization
    # ========================================================================

    def _initialize_model(
        self,
        device: torch.device,
        config_file: Optional[str],
        model_weights: str,
        vocab: str,
        label_list: Optional[list],
        input_size: int,
        max_size: int,
        use_fp16: bool,
    ) -> Tuple[OdiseDemoWrapper, None]:
        """
        Initialize ODISE model using OpenPanopticInference wrapper.

        Returns:
            demo: OdiseDemoWrapper instance (wraps OpenPanopticInference)
            transform: None (transform is handled internally by detectron2)
        """
        # Load compatibility patches
        self._load_compatibility_patches()

        # Setup config
        import pyslam.config as config

        config.cfg.set_lib("odise", prepend=True)

        # Import ODISE dependencies
        from .detectron2_utils import check_detectron2_import

        check_detectron2_import()

        from detectron2.config import instantiate
        from detectron2.data import transforms as T
        from detectron2.utils.env import seed_all_rng
        from odise import model_zoo
        from odise.checkpoint import ODISECheckpointer
        from odise.config import instantiate_odise
        from odise.modeling.wrapper import OpenPanopticInference

        # Set random seed for reproducibility
        seed_all_rng(42)

        # Load and configure model config
        cfg = self._load_config(config_file)

        # Setup device configuration
        self._setup_device_config(cfg, device)

        # Resolve model weights path
        self._resolve_model_weights(cfg, model_weights)

        # Setup augmentations
        aug = self._setup_augmentations(cfg, input_size, max_size, instantiate, T)

        # Load and prepare model
        model = self._load_and_prepare_model(cfg, instantiate_odise, ODISECheckpointer, use_fp16)

        # Build classes and metadata
        if label_list is None:
            label_list = ["COCO (133 categories)", "ADE (150 categories)"]

        demo_classes, demo_metadata = self.build_demo_classes_and_metadata(vocab, label_list)

        # Create inference wrapper
        inference_model = self._create_inference_wrapper(
            model, demo_classes, demo_metadata, OpenPanopticInference
        )

        # Store references for metadata access
        self._inference_model = inference_model
        self._demo_metadata = demo_metadata

        # Create demo wrapper
        demo = OdiseDemoWrapper(inference_model, demo_metadata, aug)

        return demo, None

    def _load_compatibility_patches(self):
        """Load and apply ODISE compatibility patches."""
        compatibility_patches_path = os.path.join(kOdisePath, "compatibility_patches.py")
        if os.path.exists(compatibility_patches_path):
            try:
                # Add ODISE path to sys.path if not already there
                if kOdisePath not in sys.path:
                    sys.path.insert(0, kOdisePath)

                # Import compatibility patches
                spec = importlib.util.spec_from_file_location(
                    "compatibility_patches", compatibility_patches_path
                )
                if spec and spec.loader:
                    compatibility_patches = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(compatibility_patches)
                    # Apply torchmetrics patch
                    if hasattr(compatibility_patches, "_patch_torchmetrics"):
                        compatibility_patches._patch_torchmetrics()
            except Exception as e:
                Printer.yellow(f"ODISE: Could not load compatibility patches: {e}")

        # Ensure torchmetrics is patched before config loading
        try:
            from compatibility_patches import _patch_torchmetrics

            _patch_torchmetrics()
        except ImportError:
            pass

    def _load_config(self, config_file: Optional[str]):
        """
        Load ODISE configuration from model_zoo.

        Args:
            config_file: Optional path to config file

        Returns:
            cfg: Loaded configuration object
        """
        from odise import model_zoo

        if config_file is None:
            config_file = "Panoptic/odise_label_coco_50e.py"
            Printer.green(f"ODISE: Using default config: {config_file}")

        # Get config from model_zoo
        cfg = model_zoo.get_config(config_file, trained=True)
        cfg.model.overlap_threshold = 0

        # Disable checkpointing for faster inference
        if hasattr(cfg.model, "backbone") and hasattr(cfg.model.backbone, "use_checkpoint"):
            cfg.model.backbone.use_checkpoint = False
            Printer.green("ODISE: Disabled gradient checkpointing for faster inference")

        return cfg

    def _setup_device_config(self, cfg, device: torch.device):
        """
        Configure device settings in config.

        Args:
            cfg: Configuration object to modify
            device: Target device
        """
        if device.type == "cuda":
            cfg.train.device = "cuda"
            torch.cuda.empty_cache()
            Printer.green("ODISE: Using CUDA for inference")

            # Enable cuDNN benchmark mode for faster inference (like the demo)
            if torch.backends.cudnn.is_available():
                if not torch.backends.cudnn.benchmark:
                    torch.backends.cudnn.benchmark = True
                    Printer.green("ODISE: Enabled cuDNN benchmark mode for faster inference")
        elif device.type == "mps":
            # MPS not directly supported by detectron2, fallback to CPU
            Printer.yellow("ODISE: MPS not supported by detectron2, using CPU")
            cfg.train.device = "cpu"
        else:
            cfg.train.device = "cpu"
            Printer.yellow("ODISE: Using CPU for inference")

    def _resolve_model_weights(self, cfg, model_weights: str):
        """
        Resolve and set model weights path in config.

        Args:
            cfg: Configuration object to modify
            model_weights: Path to model weights (may be relative or absolute)
        """
        if not model_weights:
            return

        # Resolve relative paths relative to ODISE root
        if not os.path.isabs(model_weights):
            abs_weights = os.path.join(kOdisePath, model_weights)
            if os.path.exists(abs_weights):
                model_weights = abs_weights
            else:
                # Try relative to config directory
                config_dir = (
                    os.path.dirname(model_weights) if os.path.exists(model_weights) else None
                )
                if config_dir:
                    abs_weights = os.path.join(config_dir, model_weights)
                    if os.path.exists(abs_weights):
                        model_weights = abs_weights

        if not os.path.exists(model_weights):
            Printer.yellow(f"ODISE: Model weights not found: {model_weights}, using config default")
        else:
            cfg.train.init_checkpoint = model_weights
            Printer.green(f"ODISE: Loading weights from: {model_weights}")

    def _setup_augmentations(self, cfg, input_size: int, max_size: int, instantiate, T):
        """
        Setup and configure data augmentations.

        Args:
            cfg: Configuration object
            input_size: Input image short edge size
            max_size: Maximum input image size
            instantiate: detectron2 instantiate function
            T: detectron2 transforms module

        Returns:
            aug: Configured augmentation object
        """
        # Get dataset config and augmentation
        dataset_cfg = cfg.dataloader.test
        aug = instantiate(dataset_cfg.mapper).augmentations

        # Optimize input resolution for faster inference
        if hasattr(aug, "augs"):
            aug_list = aug.augs
        elif hasattr(aug, "__iter__"):
            aug_list = aug
        else:
            aug_list = [aug]

        for transform in aug_list:
            if isinstance(transform, T.ResizeShortestEdge):
                transform.short_edge_length = (input_size, input_size)
                transform.max_size = max_size
                Printer.green(
                    f"ODISE: Set input resolution: short_edge={input_size}, max_size={max_size}"
                )
                break

        return aug

    def _load_and_prepare_model(self, cfg, instantiate_odise, ODISECheckpointer, use_fp16: bool):
        """
        Instantiate, load weights, and prepare model for inference.

        Args:
            cfg: Configuration object
            instantiate_odise: ODISE model instantiation function
            ODISECheckpointer: ODISE checkpoint loader class
            use_fp16: Whether to use float16 precision

        Returns:
            model: Prepared model ready for inference
        """
        # Instantiate model on CPU first to avoid OOM during loading
        Printer.green("ODISE: Instantiating model on CPU...")
        model = instantiate_odise(cfg.model)

        # Load checkpoint on CPU first
        Printer.green("ODISE: Loading checkpoint...")
        ODISECheckpointer(model).load(cfg.train.init_checkpoint)

        # Set model to evaluation mode (required for ODISE inference)
        model.eval()
        Printer.green("ODISE: Model set to evaluation mode")

        # Convert to float16 and move to device
        if use_fp16:
            Printer.green(f"ODISE: Converting to float16 and moving to {cfg.train.device}...")
            model.to(torch.float16)
        else:
            Printer.green(f"ODISE: Moving to {cfg.train.device}...")

        # Move model to device with error handling
        # Note: _move_model_to_device will handle dtype conversion if falling back to CPU
        self._move_model_to_device(model, cfg)

        # Ensure model stays in eval mode after moving to device
        model.eval()

        return model

    def _move_model_to_device(self, model, cfg):
        """
        Move model to target device with error handling for OOM.

        Args:
            model: Model to move
            cfg: Configuration object with device settings
        """
        if cfg.train.device == "cuda":
            try:
                model.to(cfg.train.device)
                torch.cuda.empty_cache()
                device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A"
                Printer.green(f"ODISE: Model successfully moved to GPU: {device_name}")
            except RuntimeError as e:
                if "out of memory" in str(e):
                    Printer.warning(f"ODISE: GPU out of memory, falling back to CPU: {e}")
                    cfg.train.device = "cpu"
                    # Check if model is in float16 - CPU needs float32
                    current_dtype = next(model.parameters()).dtype
                    if current_dtype == torch.float16:
                        Printer.green(
                            "ODISE: Converting model from float16 to float32 for CPU inference..."
                        )
                        model.to(torch.float32)
                    model.to(cfg.train.device)
                    Printer.green("ODISE: Model moved to CPU (fallback)")
                else:
                    raise
        else:
            # If explicitly using CPU, ensure model is in float32
            current_dtype = next(model.parameters()).dtype
            if current_dtype == torch.float16:
                Printer.green(
                    "ODISE: Converting model from float16 to float32 for CPU inference..."
                )
                model.to(torch.float32)
            model.to(cfg.train.device)
            Printer.green("ODISE: Model moved to CPU")

    def _create_inference_wrapper(self, model, demo_classes, demo_metadata, OpenPanopticInference):
        """
        Create OpenPanopticInference wrapper for model.

        Args:
            model: Base ODISE model
            demo_classes: List of class labels
            demo_metadata: Metadata object
            OpenPanopticInference: Inference wrapper class

        Returns:
            inference_model: Configured inference wrapper
        """
        inference_model = OpenPanopticInference(
            model=model,
            labels=demo_classes,
            metadata=demo_metadata,
            semantic_on=False,
            instance_on=False,
            panoptic_on=True,
        )
        # Ensure inference wrapper is in eval mode
        inference_model.eval()
        return inference_model

    # ========================================================================
    # Metadata and Classes Building
    # ========================================================================

    def build_demo_classes_and_metadata(self, vocab: str, label_list: list):
        """
        Build classes and metadata for segmentation.

        Args:
            vocab: Extra vocabulary for segmentation (format: 'a1,a2;b1,b2')
            label_list: List of categories to use

        Returns:
            (demo_classes, demo_metadata) tuple
        """
        from .detectron2_utils import check_detectron2_import

        check_detectron2_import()

        from detectron2.data import MetadataCatalog
        from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES
        from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES
        from detectron2.utils.visualizer import random_color
        from odise.data import get_openseg_labels

        # Initialize with extra classes from vocab
        extra_classes = []
        if vocab:
            for words in vocab.split(";"):
                extra_classes.append([word.strip() for word in words.split(",")])
        extra_colors = [random_color(rgb=True, maximum=1) for _ in range(len(extra_classes))]

        demo_thing_classes = extra_classes
        demo_stuff_classes = []
        demo_thing_colors = extra_colors
        demo_stuff_colors = []

        # Add COCO classes if requested
        if any("COCO" in label for label in label_list):
            coco_classes, coco_colors = self._get_coco_classes_and_colors(get_openseg_labels)
            demo_thing_classes += coco_classes["thing"]
            demo_stuff_classes += coco_classes["stuff"]
            demo_thing_colors += coco_colors["thing"]
            demo_stuff_colors += coco_colors["stuff"]

        # Add ADE classes if requested
        if any("ADE" in label for label in label_list):
            ade_classes, ade_colors = self._get_ade_classes_and_colors(get_openseg_labels)
            demo_thing_classes += ade_classes["thing"]
            demo_stuff_classes += ade_classes["stuff"]
            demo_thing_colors += ade_colors["thing"]
            demo_stuff_colors += ade_colors["stuff"]

        # Build metadata
        demo_metadata = self._build_metadata(
            demo_thing_classes,
            demo_stuff_classes,
            demo_thing_colors,
            demo_stuff_colors,
            MetadataCatalog,
        )

        demo_classes = demo_thing_classes + demo_stuff_classes

        return demo_classes, demo_metadata

    def _get_coco_classes_and_colors(self, get_openseg_labels):
        """Extract COCO classes and colors."""
        from .detectron2_utils import check_detectron2_import

        check_detectron2_import()

        from detectron2.data.datasets.builtin_meta import COCO_CATEGORIES

        coco_labels = get_openseg_labels("coco_panoptic", True)
        coco_thing_classes = [
            label for idx, label in enumerate(coco_labels) if COCO_CATEGORIES[idx]["isthing"] == 1
        ]
        coco_stuff_classes = [
            label for idx, label in enumerate(coco_labels) if COCO_CATEGORIES[idx]["isthing"] == 0
        ]
        coco_thing_colors = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 1]
        coco_stuff_colors = [c["color"] for c in COCO_CATEGORIES if c["isthing"] == 0]

        return (
            {"thing": coco_thing_classes, "stuff": coco_stuff_classes},
            {"thing": coco_thing_colors, "stuff": coco_stuff_colors},
        )

    def _get_ade_classes_and_colors(self, get_openseg_labels):
        """Extract ADE20K classes and colors."""
        from mask2former.data.datasets.register_ade20k_panoptic import ADE20K_150_CATEGORIES

        ade_labels = get_openseg_labels("ade20k_150", True)
        ade_thing_classes = [
            label
            for idx, label in enumerate(ade_labels)
            if ADE20K_150_CATEGORIES[idx]["isthing"] == 1
        ]
        ade_stuff_classes = [
            label
            for idx, label in enumerate(ade_labels)
            if ADE20K_150_CATEGORIES[idx]["isthing"] == 0
        ]
        ade_thing_colors = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 1]
        ade_stuff_colors = [c["color"] for c in ADE20K_150_CATEGORIES if c["isthing"] == 0]

        return (
            {"thing": ade_thing_classes, "stuff": ade_stuff_classes},
            {"thing": ade_thing_colors, "stuff": ade_stuff_colors},
        )

    def _build_metadata(
        self,
        demo_thing_classes,
        demo_stuff_classes,
        demo_thing_colors,
        demo_stuff_colors,
        MetadataCatalog,
    ):
        """Build detectron2 metadata object."""
        MetadataCatalog.pop("odise_demo_metadata", None)
        demo_metadata = MetadataCatalog.get("odise_demo_metadata")
        demo_metadata.thing_classes = [c[0] for c in demo_thing_classes]
        demo_metadata.stuff_classes = [
            *demo_metadata.thing_classes,
            *[c[0] for c in demo_stuff_classes],
        ]
        demo_metadata.thing_colors = demo_thing_colors
        demo_metadata.stuff_colors = demo_thing_colors + demo_stuff_colors
        demo_metadata.stuff_dataset_id_to_contiguous_id = {
            idx: idx for idx in range(len(demo_metadata.stuff_classes))
        }
        demo_metadata.thing_dataset_id_to_contiguous_id = {
            idx: idx for idx in range(len(demo_metadata.thing_classes))
        }
        return demo_metadata

    # ========================================================================
    # Device Management
    # ========================================================================

    def init_device(self, device: Optional[torch.device]) -> torch.device:
        """
        Initialize and validate device.

        Args:
            device: Optional device specification

        Returns:
            device: Validated torch device
        """
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            if device.type != "cuda":
                device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
        elif isinstance(device, str):
            device = torch.device(device)

        if device.type == "cuda":
            Printer.green("SemanticSegmentationOdise: Using CUDA")
        elif device.type == "mps":
            if not torch.backends.mps.is_available():
                raise Exception("SemanticSegmentationOdise: MPS is not available")
            Printer.yellow("SemanticSegmentationOdise: Using MPS (may fallback to CPU)")
        else:
            Printer.yellow("SemanticSegmentationOdise: Using CPU")

        return device

    # ========================================================================
    # Model Information
    # ========================================================================

    def num_classes(self) -> int:
        """
        Get number of output classes.

        Returns:
            Number of classes
        """
        try:
            metadata = getattr(self, "_demo_metadata", None)
            if metadata is None:
                metadata = getattr(self.model, "metadata", None)
            if metadata is not None and hasattr(metadata, "stuff_classes"):
                return len(metadata.stuff_classes)
        except Exception:
            pass

        # Try to get from inference model
        try:
            if hasattr(self, "_inference_model"):
                return 300  # Default for COCO + ADE
        except Exception:
            pass

        # Last resort: probe with a dummy image
        try:
            dummy_image = np.zeros((100, 100, 3), dtype=np.uint8)
            with torch.no_grad():
                predictions = self.model.predict(dummy_image)
                if "panoptic_seg" in predictions:
                    metadata = getattr(self, "_demo_metadata", None)
                    if metadata is None:
                        metadata = getattr(self.model, "metadata", None)
                    if metadata is not None and hasattr(metadata, "stuff_classes"):
                        return len(metadata.stuff_classes)
        except Exception as e:
            Printer.red(f"SemanticSegmentationOdise: Failed to get number of classes: {e}")

        # Default fallback
        Printer.yellow(
            "SemanticSegmentationOdise: Could not determine number of classes, using default"
        )
        return 300  # Common default for COCO + ADE

    # ========================================================================
    # Inference
    # ========================================================================

    @torch.no_grad()
    def infer(self, image: np.ndarray) -> SemanticSegmentationOutput:
        """
        Run semantic segmentation inference on an image.

        Args:
            image: numpy array of shape (H, W, 3) in BGR format (OpenCV format)

        Returns:
            SemanticSegmentationOutput: object containing semantics and optionally instances
        """
        # Preprocess image
        rgb_image = self._preprocess_image(image)

        # Run inference
        Printer.green(f"ODISE: Starting inference on image shape {rgb_image.shape[:2]}...")
        predictions = self.model.predict(rgb_image)

        if (
            self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR
            and "sem_seg" not in predictions
        ):
            raise ValueError(
                "ODISE probability vectors require a semantic segmentation head "
                "('sem_seg' output). Current ODISE outputs are panoptic only."
            )

        # Generate visualized output using centralized visualization generation
        # ODISE's demo wrapper doesn't provide visualized_output like DETIC/EOV_SEG do,
        # so we generate it on-demand using the consolidated visualization method.
        # This ensures consistent visualization across all detectron2-based models.
        visualized_output = None
        if hasattr(self, "semantic_color_map_obj") and self.semantic_color_map_obj is not None:
            if hasattr(self.semantic_color_map_obj, "generate_visualization"):
                # Use SemanticColorMapDetectron2's centralized generate_visualization method
                # This method consolidates visualization logic from DETIC, EOV_SEG, and ODISE
                visualized_output = self.semantic_color_map_obj.generate_visualization(
                    predictions, rgb_image
                )
        self._last_visualized_output = visualized_output

        # Postprocess predictions
        semantics, instances = self._postprocess_predictions(predictions, rgb_image)

        self.semantics = semantics
        return SemanticSegmentationOutput(semantics=self.semantics, instances=instances)

    def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """
        Preprocess image for ODISE inference (BGR to RGB conversion, format validation).

        Args:
            image: Input image in BGR format

        Returns:
            rgb_image: Preprocessed image in RGB format
        """
        # Ensure image is in correct format (BGR, uint8)
        if image.dtype != np.uint8:
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)

        # ODISE expects RGB format (convert from BGR)
        if len(image.shape) == 2:
            # Grayscale to RGB
            image = np.stack([image, image, image], axis=2)
        elif image.shape[2] == 4:
            # RGBA to RGB
            image = image[:, :, :3]
        elif image.shape[2] == 3:
            # Convert BGR to RGB
            image = image[:, :, ::-1]
        else:
            raise ValueError(f"Unsupported image shape: {image.shape}")

        return image

    def _postprocess_predictions(
        self, predictions: Dict[str, Any], original_image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Postprocess model predictions into semantic labels or probabilities.

        Args:
            predictions: Model predictions dictionary
            original_image: Original RGB image for size reference

        Returns:
            (semantics, instances): Tuple of processed semantic output and optional instance IDs
        """
        if "panoptic_seg" in predictions:
            return self._process_panoptic_predictions(predictions, original_image)
        elif "sem_seg" in predictions:
            semantics = self._process_semantic_predictions(predictions, original_image)
            return semantics, None
        else:
            raise ValueError("No semantic segmentation found in predictions")

    def _process_panoptic_predictions(
        self, predictions: Dict[str, Any], original_image: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Process panoptic segmentation predictions.

        Args:
            predictions: Predictions dictionary containing panoptic_seg
            original_image: Original RGB image

        Returns:
            (semantic_labels, instance_ids): Tuple of semantic labels and instance IDs
        """
        panoptic_seg, segments_info = predictions["panoptic_seg"]
        semantic_labels = self._panoptic_to_semantic_labels(panoptic_seg, segments_info)
        instance_ids = self._panoptic_to_instances(panoptic_seg, segments_info)

        # Store segments_info and original image for color mapping
        self._last_segments_info = copy.deepcopy(segments_info)
        self._last_panoptic_seg = panoptic_seg
        self._last_image = original_image.copy()

        # Resize if needed (simple nearest neighbor for labels)
        orig_h, orig_w = original_image.shape[:2]
        pred_h, pred_w = semantic_labels.shape

        if orig_h != pred_h or orig_w != pred_w:
            semantic_labels = cv2.resize(
                semantic_labels.astype(np.uint16),
                (orig_w, orig_h),
                interpolation=cv2.INTER_NEAREST,
            ).astype(np.int32)

            if instance_ids is not None:
                instance_ids = cv2.resize(
                    instance_ids.astype(np.int32),
                    (orig_w, orig_h),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)

        if instance_ids is not None and self.enforce_unique_instance_ids:
            instance_ids = ensure_unique_instance_ids(
                instance_ids, background_id=0, min_pixels=self.unique_instance_min_pixels
            )

        return semantic_labels, instance_ids

    def _process_semantic_predictions(
        self, predictions: Dict[str, Any], original_image: np.ndarray
    ) -> np.ndarray:
        """
        Process semantic segmentation predictions.

        Args:
            predictions: Predictions dictionary containing sem_seg
            original_image: Original RGB image

        Returns:
            semantics: Processed semantic output (labels or probabilities)
        """
        sem_seg = predictions["sem_seg"]  # Shape: [num_classes, H, W]
        sem_seg_np = sem_seg.cpu().numpy()

        # Get original image dimensions
        orig_h, orig_w = original_image.shape[:2]
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
                return self.label_mapping[labels]
            else:
                return self.aggregate_probabilities(probs, self.label_mapping)
        else:
            if self.semantic_feature_type == SemanticFeatureType.LABEL:
                return np.argmax(probs, axis=-1).astype(np.int32)
            else:
                return probs

    # ========================================================================
    # Postprocessing Utilities
    # ========================================================================

    def _panoptic_to_semantic_labels(self, panoptic_seg: torch.Tensor, segments_info: list):
        """
        Convert panoptic segmentation directly to semantic labels (faster, no softmax).

        Args:
            panoptic_seg: Tensor of shape [H, W] with segment IDs
            segments_info: List of dicts with segment information

        Returns:
            semantic_labels: numpy array of shape [H, W] with category IDs
        """
        # Get number of classes from metadata for validation
        metadata = getattr(self, "_demo_metadata", None)
        if metadata is None:
            metadata = getattr(self.model, "metadata", None)

        # Create a mapping from segment ID to category ID
        segment_to_category = {}
        for seg_info in segments_info:
            seg_id = seg_info["id"]
            category_id = seg_info["category_id"]
            segment_to_category[seg_id] = category_id

        # Convert panoptic to semantic labels directly
        panoptic_np = panoptic_seg.cpu().numpy()
        semantic_labels = np.zeros_like(panoptic_np, dtype=np.int32)

        # Map segment IDs to category IDs
        for seg_id, category_id in segment_to_category.items():
            mask = panoptic_np == seg_id
            semantic_labels[mask] = category_id

        return semantic_labels

    def _panoptic_to_instances(
        self, panoptic_seg: torch.Tensor, segments_info: list
    ) -> Optional[np.ndarray]:
        """
        Extract instance IDs from panoptic segmentation.

        Args:
            panoptic_seg: Tensor of shape [H, W] with segment IDs
            segments_info: List of dicts with segment information

        Returns:
            instance_ids: numpy array of shape [H, W] with instance IDs (0 for background/stuff)
        """
        panoptic_np = panoptic_seg.cpu().numpy()

        # Get metadata for fallback checking if isthing is not reliable
        metadata = getattr(self, "_demo_metadata", None)
        if metadata is None:
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

        # Map segment IDs to instance IDs (> 0 for actual instances, 0 for background/stuff)
        # We only need instance IDs > 0, not necessarily contiguous
        # Collect thing segment IDs that need remapping (if seg_id <= 0)
        segment_id_to_instance_id = {}
        next_instance_id = 1  # Start from 1 (0 is reserved for background/stuff)
        thing_seg_ids = []  # Track all thing segment IDs

        # First pass: identify all thing segments
        for seg_info in segments_info:
            seg_id = seg_info["id"]
            category_id = seg_info.get("category_id", None)

            # Determine if this is a "thing" class (instance)
            # First try to get from segments_info
            is_thing = seg_info.get("isthing", None)

            # Handle both boolean and integer (0/1) values
            if is_thing is None:
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
                thing_seg_ids.append(seg_id)
                # Only remap if segment ID is <= 0 (shouldn't happen, but be safe)
                if seg_id <= 0:
                    segment_id_to_instance_id[seg_id] = next_instance_id
                    next_instance_id += 1

        # Return None if no instances were found (all segments are stuff classes)
        if len(thing_seg_ids) == 0:
            return None

        # Optimized remapping: use segment IDs directly if they're all > 0
        # Otherwise, remap only the problematic ones
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
                        instance_ids[panoptic_np == seg_id] = seg_id
                for seg_id, instance_id in segment_id_to_instance_id.items():
                    instance_ids[panoptic_np == seg_id] = instance_id
            else:
                # Create lookup table: lookup[seg_id + offset] = instance_id, default 0 (background)
                # Use int32 to match panoptic_np dtype
                lookup = np.zeros(lookup_size, dtype=np.int32)
                # First, copy valid segment IDs (> 0) directly
                for seg_id in thing_seg_ids:
                    if seg_id > 0:
                        lookup[seg_id + offset] = seg_id
                # Then, remap problematic ones (<= 0)
                for seg_id, instance_id in segment_id_to_instance_id.items():
                    lookup[seg_id + offset] = instance_id

                # Vectorized remapping: single pass through image using advanced indexing
                instance_ids = lookup[panoptic_np + offset]

        return instance_ids

    def _softmax_2d(self, x: np.ndarray, axis: int = 0) -> np.ndarray:
        """
        Apply softmax along specified axis.

        Args:
            x: Input array
            axis: Axis along which to apply softmax

        Returns:
            Softmax output
        """
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

    # ========================================================================
    # Visualization
    # ========================================================================

    def sem_img_to_viz_rgb(self, semantics, bgr=False):
        """
        Convert semantics to RGB visualization using extracted detectron2 color map.

        This method uses the color map extracted from detectron2's metadata (same colors
        as detectron2's visualizer) for efficient direct color mapping. The visualizer
        is only used for panoptic segmentation where full visualization with overlays is needed.

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

            panoptic_data = {
                "panoptic_seg": self._last_panoptic_seg,
                "segments_info": self._last_segments_info,
                "image": getattr(self, "_last_image", None),
                "visualizer_class": None,  # Use default detectron2 Visualizer
                "instance_mode": ColorMode.IMAGE,
            }

        # Use semantic color map's centralized to_rgb method
        # This method handles visualization consistently across all detectron2-based models:
        # - Uses pre-computed visualized_output if available (generated via generate_visualization())
        # - Falls back to panoptic_data visualization if needed
        # - Uses direct color mapping as final fallback
        # ODISE generates visualization on-demand since its demo wrapper doesn't provide it automatically
        return self.semantic_color_map_obj.to_rgb(
            semantics,
            panoptic_data=panoptic_data,
            visualized_output=getattr(self, "_last_visualized_output", None),
            bgr=bgr,
        )

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        return self.semantic_color_map_obj.sem_img_to_rgb(semantic_img, bgr=bgr)
