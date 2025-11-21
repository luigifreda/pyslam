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

import cv2
import numpy as np
import torch

from .semantic_types import SemanticFeatureType
from .semantic_feature_processing import SemanticFeatureProcessing
from .semantic_types import SemanticDatasetType
from .semantic_utils import (
    labels_color_map_factory,
    single_label_to_color,
    similarity_heatmap_point,
    similarity_heatmap_image,
    labels_to_image,
)

try:
    from pyslam.utilities.system import Printer
except ImportError:
    Printer = None

kVerbose = True
kTimerVerbose = False

kSemanticMappingSleepTime = 5e-3  # [s]


# ============================================================================
# SemanticMappingColorMap Class
# ============================================================================


# This class is used to manage the color map for semantic mapping. It is used by SemanticSegmentationProcess.
class SemanticMappingColorMap:
    def __init__(
        self,
        semantic_dataset_type: SemanticDatasetType,
        semantic_feature_type: SemanticFeatureType,
        num_classes=None,
        text_embs=None,
        device="cpu",
        sim_scale=1.0,
    ):
        self.semantic_dataset_type = semantic_dataset_type
        self.semantic_feature_type = semantic_feature_type
        self.num_classes = num_classes
        self.text_embs = text_embs
        self.device = device
        self.sim_scale = sim_scale
        if num_classes is None:
            self.color_map = labels_color_map_factory(semantic_dataset_type)
        else:
            self.color_map = labels_color_map_factory(
                semantic_dataset_type, num_classes=num_classes
            )
        if semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
            self.color_map = None
        # check consistency of text_embs and device
        if device == "cpu":
            if isinstance(text_embs, torch.Tensor):
                text_embs = text_embs.cpu().detach().numpy()
        elif device == "gpu":
            if isinstance(text_embs, np.ndarray):
                text_embs = torch.from_numpy(text_embs).to(device)
        else:
            raise ValueError("Invalid device: {device}")

    def to_rgb(self, semantics, bgr=False):
        # Check if input is an array (image) or scalar (single value)
        # Arrays have ndim >= 1 and size > 1 (or are 1D+ arrays)
        is_array = isinstance(semantics, np.ndarray) and (
            semantics.ndim >= 2 or (semantics.ndim == 1 and semantics.size > 1)
        )

        if is_array:
            # For arrays (images), use sem_img_to_rgb which handles images properly
            return self.sem_img_to_rgb(semantics, bgr=bgr)
        else:
            # For scalars (single values), use the original single_label_to_color logic
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                return similarity_heatmap_point(semantics, colormap=cv2.COLORMAP_JET, bgr=bgr)
            else:
                return single_label_to_color(semantics, self.color_map, bgr=bgr)

    def sem_des_to_rgb(self, semantic_des, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return single_label_to_color(semantic_des, self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return single_label_to_color(np.argmax(semantic_des, axis=-1), self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                # transform features to similarities
                sims = SemanticFeatureProcessing.features_to_sims(
                    semantic_des, self.text_embs, self.device
                )
                return similarity_heatmap_point(
                    sims,
                    colormap=cv2.COLORMAP_JET,
                    sim_scale=self.sim_scale,
                    bgr=bgr,
                )
            else:
                label = SemanticFeatureProcessing.features_to_labels(
                    semantic_des, self.text_embs, self.device
                )
                return single_label_to_color(label, self.color_map, bgr=bgr)

    def sem_img_to_rgb(self, semantic_img, bgr=False):
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            return labels_to_image(semantic_img, self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            return labels_to_image(np.argmax(semantic_img, axis=-1), self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.FEATURE_VECTOR:
            # Transform semantic to tensor
            # TODO: check if doing these operations (and functions below) in CPU is more efficient (it probably is)
            semantics = torch.from_numpy(semantic_img).to(self.device)
            # Convert text_embs to tensor if needed
            if isinstance(self.text_embs, np.ndarray):
                text_embs = torch.from_numpy(self.text_embs).to(self.device)
            else:
                text_embs = self.text_embs.to(self.device)
            # Compute similarity
            sims = semantics @ text_embs.T  # (H, W, D) @ (D, N) -> (H, W, N)
            if self.semantic_dataset_type == SemanticDatasetType.FEATURE_SIMILARITY:
                # For FEATURE_SIMILARITY, take maximum similarity across all text embeddings
                sim_map = sims.max(dim=-1)[0]  # (H, W)
                return similarity_heatmap_image(
                    sim_map.cpu().detach().numpy(),
                    colormap=cv2.COLORMAP_JET,
                    sim_scale=self.sim_scale,
                    bgr=bgr,
                )
            else:
                pred = sims.argmax(dim=-1)
                return labels_to_image(pred.cpu().detach().numpy(), self.color_map, bgr=bgr)


# ============================================================================
# Detectron2ColorMapManager Class
# ============================================================================


class Detectron2ColorMapManager:
    """
    Manager class for color map operations with detectron2-based models (Detic, EOV-Seg).

    This class provides a unified interface for:
    - Extracting color maps from detectron2 metadata
    - Creating fallback color maps
    - Converting semantics to RGB visualizations
    - Visualizing panoptic segmentations

    Example usage:
        # For Detic (uses thing_colors)
        manager = Detectron2ColorMapManager(metadata=model.metadata)
        color_map = manager.extract_from_metadata(
            color_attr="thing_colors",
            num_classes=3000,
            min_size=2000
        )
        rgb_image = manager.to_rgb(semantics, bgr=True)

        # For EOV-Seg (uses stuff_colors)
        manager = Detectron2ColorMapManager(
            metadata=model.metadata,
            semantic_feature_type=SemanticFeatureType.LABEL
        )
        color_map = manager.extract_from_metadata(
            color_attr="stuff_colors",
            min_size=150
        )
        manager.color_map = color_map  # Set the color map
        rgb_image = manager.to_rgb(semantics, bgr=True)

        # Fallback color map
        manager = Detectron2ColorMapManager(
            semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
            semantic_segmentation_type=SemanticSegmentationType.DETIC
        )
        color_map = manager.create_fallback(num_classes=3000)
    """

    @staticmethod
    def get_detectron2_colormap(num_colors=None):
        """
        Get detectron2's color palette.

        Args:
            num_colors: Number of colors needed. If None, returns the default palette (80 colors).
                       If larger, repeats the palette.

        Returns:
            np.ndarray: Color map array of shape (num_colors, 3) with RGB values (0-255)
        """
        try:
            from detectron2.utils.colormap import colormap

            detectron2_colors = colormap(rgb=True, maximum=255).astype(np.uint8)
            if num_colors is None:
                return detectron2_colors

            # Repeat the palette to fill the requested number
            num_repeats = (num_colors + len(detectron2_colors) - 1) // len(detectron2_colors)
            repeated_colors = np.tile(detectron2_colors, (num_repeats, 1))[:num_colors]
            return repeated_colors
        except ImportError:
            # Fallback to generic colors if detectron2 not available
            from .semantic_labels import get_generic_color_map

            if num_colors is None:
                num_colors = 80
            return get_generic_color_map(num_colors)

    def __init__(
        self,
        metadata=None,
        color_map=None,
        semantic_feature_type=None,
        semantic_dataset_type=None,
        semantic_segmentation_type=None,
    ):
        """
        Initialize Detectron2ColorMapManager.

        Args:
            metadata: Optional detectron2 MetadataCatalog object
            color_map: Optional pre-computed color map array
            semantic_feature_type: Optional SemanticFeatureType enum
            semantic_dataset_type: Optional SemanticDatasetType enum (for fallback)
            semantic_segmentation_type: Optional SemanticSegmentationType enum (for fallback)
        """
        self.metadata = metadata
        self.color_map = color_map
        self.semantic_feature_type = semantic_feature_type
        self.semantic_dataset_type = semantic_dataset_type
        self.semantic_segmentation_type = semantic_segmentation_type

    def extract_from_metadata(
        self,
        color_attr="stuff_colors",
        num_classes=None,
        min_size=2000,
        use_detectron2_padding=True,
        verbose=True,
    ):
        """
        Extract color map from detectron2 metadata with automatic padding if needed.

        This method handles both 'stuff_colors' (EOV-Seg) and 'thing_colors' (Detic).

        Args:
            color_attr: Attribute name to extract colors from ('stuff_colors' or 'thing_colors')
            num_classes: Optional target number of classes. If None, uses available colors.
                        If larger than available, pads with detectron2 colormap.
            min_size: Minimum size for color map (will pad if smaller)
            use_detectron2_padding: If True, use detectron2's colormap for padding; otherwise use generic colors
            verbose: If True, print status messages

        Returns:
            np.ndarray: Color map array of shape (num_classes, 3) with RGB values (0-255)
        """
        if self.metadata is None:
            raise ValueError("metadata must be provided to extract color map")

        if not hasattr(self.metadata, color_attr):
            raise ValueError(f"Metadata does not have {color_attr} attribute")

        # Extract colors from metadata
        colors = getattr(self.metadata, color_attr)

        # Convert to numpy array if needed
        if not isinstance(colors, np.ndarray):
            colors = np.array(colors, dtype=np.uint8)

        # Ensure it's 2D (num_colors, 3)
        if colors.ndim == 1:
            colors = colors.reshape(-1, 3)

        # Ensure colors are in the right format (uint8, 0-255)
        if colors.dtype != np.uint8:
            if colors.max() <= 1.0:
                colors = (colors * 255).astype(np.uint8)
            else:
                colors = colors.astype(np.uint8)

        num_available_colors = len(colors)

        # Determine target size
        if num_classes is None:
            num_classes = max(num_available_colors, min_size)
        else:
            num_classes = max(num_classes, min_size)

        # Pad if needed
        if num_classes > num_available_colors:
            padding_size = num_classes - num_available_colors
            if verbose and Printer:
                Printer.yellow(
                    f"Color map has only {num_available_colors} classes, "
                    f"padding to {num_classes} to handle large category IDs"
                )

            if use_detectron2_padding:
                try:
                    padding_colors = self.get_detectron2_colormap(padding_size)
                    colors = np.vstack([colors, padding_colors])
                    if verbose and Printer:
                        Printer.green(
                            f"Padded color map using detectron2's colormap (now {len(colors)} classes)"
                        )
                except ImportError:
                    # Fallback to generic colors
                    if verbose and Printer:
                        Printer.yellow("detectron2 colormap not available, using generic colors")
                    from .semantic_labels import get_generic_color_map

                    padding_colors = get_generic_color_map(padding_size)
                    colors = np.vstack([colors, padding_colors])
            else:
                from .semantic_labels import get_generic_color_map

                padding_colors = get_generic_color_map(padding_size)
                colors = np.vstack([colors, padding_colors])

        self.color_map = colors
        return self.color_map

    def create_fallback(self, num_classes=3000):
        """
        Create a fallback color map using detectron2's colormap or generic colors.

        Args:
            num_classes: Number of classes for the color map

        Returns:
            np.ndarray: Color map array of shape (num_classes, 3) with RGB values (0-255)
        """
        # Try to use detectron2's colormap first
        try:
            colors = self.get_detectron2_colormap(num_classes)
            if Printer:
                Printer.yellow(f"Using detectron2 colormap ({num_classes} classes)")
            self.color_map = colors
            return self.color_map
        except ImportError:
            # Fallback to dataset-specific color map
            if self.semantic_dataset_type is None or self.semantic_segmentation_type is None:
                raise ValueError(
                    "semantic_dataset_type and semantic_segmentation_type must be provided for fallback"
                )

            self.color_map = labels_color_map_factory(
                self.semantic_dataset_type,
                semantic_segmentation_type=self.semantic_segmentation_type,
                num_classes=num_classes,
            )
            return self.color_map

    def visualize_panoptic(
        self,
        panoptic_seg,
        segments_info,
        image=None,
        visualizer_class=None,
        instance_mode=None,
        target_shape=None,
        bgr=False,
    ):
        """
        Visualize panoptic segmentation using detectron2's visualizer.

        Args:
            panoptic_seg: Panoptic segmentation tensor/array of shape [H, W] with segment IDs
            segments_info: List of dicts with segment information
            image: Optional original image (BGR format). If None, creates a white image.
            visualizer_class: Optional custom visualizer class. If None, uses detectron2's Visualizer.
            instance_mode: ColorMode for visualization
            target_shape: Optional (H, W) to resize output to
            bgr: If True, return BGR format; otherwise RGB

        Returns:
            np.ndarray: RGB/BGR visualization image of shape (H, W, 3)
        """
        if self.metadata is None:
            raise ValueError("metadata must be provided for panoptic visualization")

        from detectron2.utils.visualizer import ColorMode

        # Determine output shape
        if target_shape is not None:
            H, W = target_shape
        else:
            if hasattr(panoptic_seg, "shape"):
                H, W = panoptic_seg.shape[:2]
            else:
                panoptic_np = (
                    panoptic_seg.cpu().numpy() if hasattr(panoptic_seg, "cpu") else panoptic_seg
                )
                H, W = panoptic_np.shape[:2]

        # Get the original image (BGR format from OpenCV)
        if image is not None:
            orig_img = image.copy()
            # Resize original image to match target shape if needed
            if target_shape is not None and orig_img.shape[:2] != (H, W):
                orig_img = cv2.resize(orig_img, (W, H), interpolation=cv2.INTER_LINEAR)
        else:
            # Fallback: create a white image if original not available
            orig_img = np.ones((H, W, 3), dtype=np.uint8) * 255

        # Convert image from OpenCV BGR format to Matplotlib RGB format
        vis_image = orig_img[:, :, ::-1]

        # Create visualizer with RGB image and metadata
        if instance_mode is None:
            instance_mode = ColorMode.IMAGE

        if visualizer_class is None:
            from detectron2.utils.visualizer import Visualizer

            visualizer_class = Visualizer

        visualizer = visualizer_class(
            vis_image, metadata=self.metadata, instance_mode=instance_mode
        )

        # Resize panoptic_seg to match target shape if needed
        if target_shape is not None:
            panoptic_np = (
                panoptic_seg.cpu().numpy() if hasattr(panoptic_seg, "cpu") else panoptic_seg
            )
            if panoptic_np.shape[:2] != (H, W):
                panoptic_np = cv2.resize(
                    panoptic_np.astype(np.uint16),
                    (W, H),
                    interpolation=cv2.INTER_NEAREST,
                ).astype(np.int32)
                panoptic_seg = torch.from_numpy(panoptic_np)

        # Ensure panoptic_seg is a tensor for the visualizer
        cpu_device = torch.device("cpu")
        if not isinstance(panoptic_seg, torch.Tensor):
            panoptic_seg = torch.from_numpy(np.asarray(panoptic_seg))
        panoptic_seg = panoptic_seg.to(cpu_device)

        # Choose the appropriate visualization method
        if hasattr(visualizer, "draw_panoptic_seg"):
            # EOV-Seg uses draw_panoptic_seg
            vis_output = visualizer.draw_panoptic_seg(panoptic_seg, segments_info)
        elif hasattr(visualizer, "draw_panoptic_seg_predictions"):
            # Detic uses draw_panoptic_seg_predictions
            vis_output = visualizer.draw_panoptic_seg_predictions(panoptic_seg, segments_info)
        else:
            raise ValueError("Visualizer does not have panoptic segmentation method")

        # Get RGB image from visualizer output
        rgb_output = vis_output.get_image()

        # Convert back to BGR if requested
        if bgr:
            rgb_output = rgb_output[:, :, ::-1]

        return rgb_output

    def to_rgb(
        self,
        semantics,
        panoptic_data=None,
        visualized_output=None,
        bgr=False,
        standard_image_sizes=None,
    ):
        """
        Convert semantics to RGB visualization.

        Args:
            semantics: Can be:
                - 1D array of shape (N,) with label IDs for N points
                - 2D array of shape (H, W) with label IDs for an image
                - 3D array of shape (H, W, num_classes) for probability vectors
            panoptic_data: Optional dict with keys:
                - 'panoptic_seg': Panoptic segmentation tensor/array
                - 'segments_info': List of segment info dicts
                - 'image': Original image (BGR format)
                - 'visualizer_class': Optional custom visualizer class
                - 'instance_mode': Optional ColorMode
            visualized_output: Optional pre-computed visualized output from model
            bgr: If True, return BGR format; otherwise RGB
            standard_image_sizes: Optional list of (H, W) tuples for detecting flattened images

        Returns:
            RGB/BGR array:
                - For 1D input: shape (N, 3) with RGB colors for each point
                - For 2D input: shape (H, W, 3) with RGB image
        """
        if self.color_map is None:
            raise ValueError("color_map must be set before calling to_rgb")

        if self.semantic_feature_type is None:
            raise ValueError("semantic_feature_type must be set before calling to_rgb")

        semantics = np.asarray(semantics)
        color_map_size = len(self.color_map)

        # Standard image sizes for detecting flattened images
        if standard_image_sizes is None:
            standard_image_sizes = [
                (480, 640),
                (720, 1280),
                (1080, 1920),
                (640, 480),
            ]

        # Case 1: 1D array - could be point cloud labels or flattened image
        if semantics.ndim == 1:
            size = semantics.size

            # Check if it matches a standard image size (flattened image)
            is_standard_image_size = False
            for H, W in standard_image_sizes:
                if size == H * W:
                    is_standard_image_size = True
                    semantics = semantics.reshape(H, W)
                    break

            if not is_standard_image_size:
                # Treat as point cloud labels (1D array of labels, one per point)
                if self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
                    labels = np.argmax(semantics.reshape(-1, semantics.shape[-1]), axis=-1)
                else:
                    labels = semantics.astype(np.int32)

                # Check if any labels exceed color map size (warn but don't fail)
                if labels.size > 0:
                    max_label = np.max(labels)
                    if max_label >= color_map_size and Printer:
                        Printer.yellow(
                            f"Warning - max label {max_label} exceeds color map size {color_map_size}, "
                            f"some objects may have incorrect colors. Consider increasing color map size."
                        )

                labels_clamped = np.clip(labels, 0, color_map_size - 1)
                colors = self.color_map[labels_clamped]

                if bgr:
                    colors = colors[:, ::-1]  # RGB to BGR

                return colors

        # Case 2: 2D image
        # Use pre-computed visualized output if available (matches demo.py exactly)
        if visualized_output is not None:
            rgb_output = visualized_output.get_image()

            # Resize if needed to match semantics shape
            H, W = semantics.shape[:2]
            if rgb_output.shape[:2] != (H, W):
                rgb_output = cv2.resize(rgb_output, (W, H), interpolation=cv2.INTER_LINEAR)

            # Convert to BGR if requested
            if bgr:
                rgb_output = rgb_output[:, :, ::-1]

            return rgb_output

        # For panoptic segmentation, use visualizer (needs overlays and instance info)
        if panoptic_data is not None:
            return self.visualize_panoptic(
                panoptic_data.get("panoptic_seg"),
                panoptic_data.get("segments_info"),
                image=panoptic_data.get("image"),
                visualizer_class=panoptic_data.get("visualizer_class"),
                instance_mode=panoptic_data.get("instance_mode"),
                target_shape=semantics.shape[:2] if semantics.ndim >= 2 else None,
                bgr=bgr,
            )

        # For semantic segmentation, use color map directly (much faster than visualizer)
        if self.semantic_feature_type == SemanticFeatureType.LABEL:
            # Check if any labels exceed color map size (warn but don't fail)
            max_label = np.max(semantics) if semantics.size > 0 else 0
            if max_label >= color_map_size and Printer:
                Printer.yellow(
                    f"Warning - max label {max_label} exceeds color map size {color_map_size}, "
                    f"some objects may have incorrect colors. Consider increasing color map size."
                )
            # Clamp labels to valid range and map to colors
            semantics_int = np.clip(semantics.astype(np.int32), 0, color_map_size - 1)
            return labels_to_image(semantics_int, self.color_map, bgr=bgr)
        elif self.semantic_feature_type == SemanticFeatureType.PROBABILITY_VECTOR:
            labels = np.argmax(semantics, axis=-1)
            # Check if any labels exceed color map size (warn but don't fail)
            max_label = np.max(labels) if labels.size > 0 else 0
            if max_label >= color_map_size and Printer:
                Printer.yellow(
                    f"Warning - max label {max_label} exceeds color map size {color_map_size}, "
                    f"some objects may have incorrect colors. Consider increasing color map size."
                )
            # Clamp labels to valid range and map to colors
            labels_int = np.clip(labels.astype(np.int32), 0, color_map_size - 1)
            return labels_to_image(labels_int, self.color_map, bgr=bgr)
        else:
            raise ValueError(f"Unsupported semantic feature type: {self.semantic_feature_type}")
