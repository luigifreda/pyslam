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

from .semantic_types import SemanticFeatureType, SemanticDatasetType
from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_color_map_base import SemanticColorMapBase
from .semantic_color_utils import labels_to_image

try:
    import cv2
except ImportError:
    cv2 = None

try:
    import torch
except ImportError:
    torch = None

from pyslam.utilities.logging import Printer


class SemanticColorMapDetectron2(SemanticColorMapBase):
    """
    Semantic color map implementation for detectron2-based models (Detic, EOV-Seg, ODISE).

    This class extracts color maps from detectron2 metadata and handles panoptic visualization.

    **Centralized Visualization Generation:**

    This class provides a unified interface for visualization generation across all detectron2-based
    segmentation models. Previously, each model (DETIC, EOV_SEG, ODISE) had its own visualization
    generation code, leading to code duplication. Now, all visualization logic is consolidated in
    the `generate_visualization()` method, which:

    1. Handles panoptic, semantic, and instance segmentation visualization
    2. Supports custom visualizer classes (e.g., OpenVocabVisualizer for EOV-Seg)
    3. Automatically selects the appropriate visualization method based on prediction type
    4. Ensures consistent visualization output across all models

    Models can either:
    - Use pre-computed visualized_output from their demo's run_on_image() method (DETIC, EOV_SEG)
    - Generate visualization on-demand using generate_visualization() (ODISE)

    Both approaches are handled transparently by the to_rgb() method.
    """

    def __init__(
        self,
        semantic_dataset_type: SemanticDatasetType,
        semantic_feature_type: SemanticFeatureType,
        metadata,
        num_classes=None,
        text_embs=None,
        device="cpu",
        sim_scale=1.0,
        semantic_segmentation_type=None,
    ):
        """
        Initialize detectron2-based semantic color map.

        Args:
            semantic_dataset_type: Type of semantic dataset
            semantic_feature_type: Type of semantic feature (LABEL, PROBABILITY_VECTOR, FEATURE_VECTOR)
            metadata: detectron2 MetadataCatalog object
            num_classes: Number of classes (optional)
            text_embs: Text embeddings for feature-based models (optional)
            device: Device to use ("cpu" or "gpu")
            sim_scale: Scale factor for similarity visualization
            semantic_segmentation_type: Type of semantic segmentation model (DETIC, EOV_SEG, ODISE)
        """
        # Initialize base class
        super().__init__(
            semantic_dataset_type=semantic_dataset_type,
            semantic_feature_type=semantic_feature_type,
            num_classes=num_classes,
            text_embs=text_embs,
            device=device,
            sim_scale=sim_scale,
            semantic_segmentation_type=semantic_segmentation_type,
        )

        self.metadata = metadata

        # Create detectron2 color map manager
        self._detectron2_manager = Detectron2ColorMapManager(
            metadata=metadata,
            semantic_feature_type=semantic_feature_type,
            semantic_dataset_type=semantic_dataset_type,
            semantic_segmentation_type=semantic_segmentation_type,
        )

        # Extract color map from metadata based on segmentation type
        try:
            if semantic_segmentation_type == SemanticSegmentationType.DETIC:
                # Detic uses thing_colors
                if hasattr(metadata, "thing_colors"):
                    self.color_map = self._detectron2_manager.extract_from_metadata(
                        color_attr="thing_colors",
                        num_classes=max(num_classes if num_classes is not None else 3000, 3000),
                        min_size=2000,
                        use_detectron2_padding=True,
                        verbose=True,
                    )
                else:
                    # thing_colors not available - use detectron2's colormap
                    self.color_map = self._detectron2_manager.create_fallback(num_classes=3000)
                    Printer.green(
                        f"Detic: Using detectron2 colormap (3000 classes) - thing_colors not available"
                    )
                    Printer.green(f"Detic: Using color map with {len(self.color_map)} classes")
            elif semantic_segmentation_type == SemanticSegmentationType.EOV_SEG:
                # EOV-Seg uses stuff_colors
                if hasattr(metadata, "stuff_colors"):
                    self.color_map = self._detectron2_manager.extract_from_metadata(
                        color_attr="stuff_colors",
                        num_classes=num_classes,
                        min_size=150,  # EOV-Seg typically uses ADE20K (150 classes)
                        use_detectron2_padding=True,
                        verbose=True,
                    )
                    Printer.green(
                        f"EOV-Seg: Extracted color map from metadata with {len(self.color_map)} classes"
                    )
                else:
                    self.color_map = self._detectron2_manager.create_fallback()
                    Printer.yellow("EOV-Seg: Using fallback color map (metadata not available)")
            elif semantic_segmentation_type == SemanticSegmentationType.ODISE:
                # ODISE uses stuff_colors
                if hasattr(metadata, "stuff_colors"):
                    self.color_map = self._detectron2_manager.extract_from_metadata(
                        color_attr="stuff_colors",
                        num_classes=num_classes,
                        min_size=300,  # ODISE typically uses COCO (133) + ADE (150) = 283+ classes
                        use_detectron2_padding=True,
                        verbose=True,
                    )
                    Printer.green(
                        f"ODISE: Extracted color map from metadata with {len(self.color_map)} classes"
                    )
                else:
                    self.color_map = self._detectron2_manager.create_fallback()
                    Printer.yellow("ODISE: Using fallback color map (metadata not available)")
            else:
                # Generic detectron2 model - try stuff_colors first, then thing_colors
                if hasattr(metadata, "stuff_colors"):
                    self.color_map = self._detectron2_manager.extract_from_metadata(
                        color_attr="stuff_colors",
                        num_classes=num_classes,
                        min_size=150,
                        use_detectron2_padding=True,
                        verbose=True,
                    )
                elif hasattr(metadata, "thing_colors"):
                    self.color_map = self._detectron2_manager.extract_from_metadata(
                        color_attr="thing_colors",
                        num_classes=num_classes,
                        min_size=2000,
                        use_detectron2_padding=True,
                        verbose=True,
                    )
                else:
                    self.color_map = self._detectron2_manager.create_fallback(
                        num_classes=num_classes or 3000
                    )
        except Exception as e:
            # Fallback to standard color map on error
            Printer.yellow(
                f"SemanticColorMapDetectron2: Failed to extract color map from metadata: {e}, "
                f"using fallback"
            )
            # Re-raise to let factory handle fallback
            raise

    def generate_visualization(self, predictions, image, visualizer_class=None, instance_mode=None):
        """
        Generate visualized output from predictions using detectron2's visualizer.

        **Centralized Visualization Generation:**

        This method consolidates visualization generation logic that was previously duplicated
        across different segmentation models (DETIC, EOV_SEG, ODISE). Instead of each model
        implementing its own visualization code, they all use this unified method.

        **Usage Pattern:**

        Models can use this method in two ways:

        1. **On-demand generation (ODISE)**: Call this method after inference to generate
           visualization when the model's demo doesn't provide it automatically.

        2. **Pre-computed visualization (DETIC, EOV_SEG)**: These models get visualized_output
           directly from their demo's run_on_image() method, so they don't need to call this.
           However, this method can be used as a fallback if needed.

        **Supported Prediction Types:**

        - Panoptic segmentation: Uses draw_panoptic_seg() or draw_panoptic_seg_predictions()
        - Semantic segmentation: Uses draw_sem_seg() with argmax of logits
        - Instance segmentation: Uses draw_instance_predictions()

        Args:
            predictions: Model predictions dictionary containing one of:
                - 'panoptic_seg': Tuple of (panoptic_seg tensor, segments_info list)
                - 'sem_seg': Tensor of shape [num_classes, H, W] with logits
                - 'instances': detectron2 Instances object
            image: RGB image array (numpy array) of shape (H, W, 3)
            visualizer_class: Optional custom visualizer class. If None, uses detectron2's
                default Visualizer. Can be set to OpenVocabVisualizer for EOV-Seg models.
            instance_mode: Optional ColorMode for visualization (default: ColorMode.IMAGE).
                Controls how instances are colored in the visualization.

        Returns:
            vis_output: VisImage object from detectron2's visualizer, or None if visualization
                cannot be generated (e.g., missing metadata or unsupported prediction type).
                The VisImage object can be passed to to_rgb() for final RGB/BGR conversion.

        Example:
            >>> # After model inference
            >>> predictions = model.predict(image)
            >>> vis_output = semantic_color_map_obj.generate_visualization(predictions, rgb_image)
            >>> rgb_vis = semantic_color_map_obj.to_rgb(semantics, visualized_output=vis_output)
        """
        # Delegate to detectron2 manager (which contains the actual implementation)
        return self._detectron2_manager.generate_visualization(
            predictions, image, visualizer_class=visualizer_class, instance_mode=instance_mode
        )

    def to_rgb(
        self,
        semantics,
        bgr=False,
        panoptic_data=None,
        visualized_output=None,
    ):
        """
        Convert semantics to RGB visualization.

        **Visualization Priority:**

        This method uses the following priority order for visualization:

        1. **Pre-computed visualized_output** (highest priority): If provided, uses the
           pre-computed visualization directly. This is the fastest path and is used by
           DETIC and EOV_SEG models that get visualization from their demo's run_on_image().

        2. **Panoptic data**: If panoptic_data is provided, generates visualization using
           detectron2's visualizer with panoptic segmentation overlays and labels.

        3. **Direct color mapping** (fallback): If neither is provided, uses direct color
           mapping from the color map (fastest but no overlays/labels).

        **Centralized Visualization:**

        All detectron2-based models (DETIC, EOV_SEG, ODISE) use this same method, ensuring
        consistent visualization output. The visualization generation is handled by the
        Detectron2ColorMapManager, which consolidates all visualization logic.

        Args:
            semantics: Can be:
                - Scalar (single label value)
                - 1D array of shape (N,) with label IDs for N points
                - 2D array of shape (H, W) with label IDs for an image
            bgr: If True, return BGR format; otherwise RGB
            panoptic_data: Optional dict with panoptic segmentation data (for detectron2 models).
                Should contain:
                - 'panoptic_seg': Panoptic segmentation tensor/array
                - 'segments_info': List of segment info dicts
                - 'image': Original image (BGR format)
                - 'visualizer_class': Optional custom visualizer class
                - 'instance_mode': Optional ColorMode
            visualized_output: Optional pre-computed VisImage object from detectron2's visualizer.
                If provided, this takes priority over panoptic_data. Models like DETIC and EOV_SEG
                get this from their demo's run_on_image() method.

        Returns:
            RGB/BGR array:
                - For 1D input: shape (N, 3) with RGB colors for each point
                - For 2D input: shape (H, W, 3) with RGB image
        """
        # If we have panoptic data or visualized_output, delegate to detectron2 manager
        # This ensures we get proper visualization with labels and bounding boxes for detectron2 models
        if panoptic_data is not None or visualized_output is not None:
            # Ensure the detectron2 manager's color_map is up to date
            self._detectron2_manager.color_map = self.color_map
            return self._detectron2_manager.to_rgb(
                semantics,
                panoptic_data=panoptic_data,
                visualized_output=visualized_output,
                bgr=bgr,
            )

        # Otherwise, use base class implementation (direct color mapping)
        return super().to_rgb(
            semantics,
            bgr=bgr,
            panoptic_data=panoptic_data,
            visualized_output=visualized_output,
        )

    def sem_img_to_rgb(self, semantic_img, bgr=False, panoptic_data=None, visualized_output=None):
        """
        Convert semantic image to RGB visualization.

        Args:
            semantic_img: 2D array of shape (H, W) with label IDs, or 3D array for probability vectors
            bgr: If True, return BGR format; otherwise RGB
            panoptic_data: Optional dict with panoptic segmentation data (for detectron2 models)
            visualized_output: Optional pre-computed visualized output from model

        Returns:
            RGB/BGR image array of shape (H, W, 3)
        """
        # If we have panoptic data or visualized output, delegate to detectron2 manager
        if panoptic_data is not None or visualized_output is not None:
            # Ensure the detectron2 manager's color_map is up to date
            self._detectron2_manager.color_map = self.color_map
            return self._detectron2_manager.to_rgb(
                semantic_img,
                panoptic_data=panoptic_data,
                visualized_output=visualized_output,
                bgr=bgr,
            )

        # Otherwise, use base class implementation
        return super().sem_img_to_rgb(
            semantic_img,
            bgr=bgr,
            panoptic_data=panoptic_data,
            visualized_output=visualized_output,
        )


# ============================================================================
# Detectron2ColorMapManager Class
# ============================================================================


class Detectron2ColorMapManager:
    """
    Manager class for color map operations with detectron2-based models (Detic, EOV-Seg, ODISE).

    This class provides a unified interface for:
    - Extracting color maps from detectron2 metadata
    - Creating fallback color maps
    - Converting semantics to RGB visualizations
    - Visualizing panoptic segmentations
    - **Centralized visualization generation** (see generate_visualization())

    **Centralized Visualization Generation:**

    The `generate_visualization()` method consolidates visualization logic that was previously
    duplicated across DETIC, EOV_SEG, and ODISE models. This ensures:
    - Consistent visualization output across all models
    - Single point of maintenance for visualization code
    - Support for custom visualizers (e.g., OpenVocabVisualizer)
    - Automatic method selection based on prediction type

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

        # Generate visualization from predictions (centralized method)
        predictions = model.predict(image)
        vis_output = manager.generate_visualization(predictions, rgb_image)
        rgb_vis = manager.to_rgb(semantics, visualized_output=vis_output)

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
            if verbose:
                Printer.yellow(
                    f"Color map has only {num_available_colors} classes, "
                    f"padding to {num_classes} to handle large category IDs"
                )

            if use_detectron2_padding:
                try:
                    padding_colors = self.get_detectron2_colormap(padding_size)
                    colors = np.vstack([colors, padding_colors])
                    if verbose:
                        Printer.green(
                            f"Padded color map using detectron2's colormap (now {len(colors)} classes)"
                        )
                except ImportError:
                    # Fallback to generic colors
                    if verbose:
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

    def generate_visualization(
        self,
        predictions,
        image,
        visualizer_class=None,
        instance_mode=None,
    ):
        """
        Generate visualized output from predictions using detectron2's visualizer.

        **Centralized Visualization Generation:**

        This method consolidates visualization generation logic that was previously duplicated
        across DETIC, EOV_SEG, and ODISE models. Instead of each model implementing its own
        visualization code, they all use this unified method.

        **How it works:**

        1. Creates a detectron2 Visualizer (or custom visualizer if provided)
        2. Detects the prediction type (panoptic_seg, sem_seg, or instances)
        3. Automatically selects the appropriate visualization method
        4. Handles tensor-to-CPU conversion and format conversions
        5. Returns a VisImage object that can be used by to_rgb()

        **Model-specific behavior:**

        - **DETIC**: Uses draw_panoptic_seg_predictions() for panoptic segmentation
        - **EOV_SEG**: Uses draw_panoptic_seg() for panoptic segmentation (can use OpenVocabVisualizer)
        - **ODISE**: Uses draw_panoptic_seg() for panoptic segmentation

        The method automatically detects which method is available and uses it.

        Args:
            predictions: Model predictions dictionary containing one of:
                - 'panoptic_seg': Tuple of (panoptic_seg tensor, segments_info list)
                - 'sem_seg': Tensor of shape [num_classes, H, W] with logits
                - 'instances': detectron2 Instances object
            image: RGB image array (numpy array) of shape (H, W, 3)
            visualizer_class: Optional custom visualizer class. If None, uses detectron2's
                default Visualizer. Can be set to OpenVocabVisualizer for EOV-Seg models.
            instance_mode: Optional ColorMode for visualization (default: ColorMode.IMAGE).
                Controls how instances are colored in the visualization.

        Returns:
            vis_output: VisImage object from detectron2's visualizer, or None if visualization
                cannot be generated (e.g., missing metadata or unsupported prediction type).
                The VisImage object can be passed to to_rgb() for final RGB/BGR conversion.

        Raises:
            None: This method catches all exceptions and returns None on error, ensuring
                that visualization failures don't break the main inference pipeline.
        """
        if self.metadata is None:
            return None

        try:
            from detectron2.utils.visualizer import Visualizer, ColorMode

            if instance_mode is None:
                instance_mode = ColorMode.IMAGE

            # Create visualizer (expects RGB image)
            if visualizer_class is None:
                visualizer_class = Visualizer

            visualizer = visualizer_class(
                image, metadata=self.metadata, instance_mode=instance_mode
            )

            # Generate visualization based on prediction type
            vis_output = None
            if "panoptic_seg" in predictions:
                panoptic_seg, segments_info = predictions["panoptic_seg"]
                # Convert to CPU tensor if needed
                if hasattr(panoptic_seg, "cpu"):
                    panoptic_seg = panoptic_seg.cpu()
                # Choose the appropriate visualization method
                if hasattr(visualizer, "draw_panoptic_seg"):
                    # EOV-Seg uses draw_panoptic_seg
                    vis_output = visualizer.draw_panoptic_seg(panoptic_seg, segments_info)
                elif hasattr(visualizer, "draw_panoptic_seg_predictions"):
                    # Detic uses draw_panoptic_seg_predictions
                    vis_output = visualizer.draw_panoptic_seg_predictions(
                        panoptic_seg, segments_info
                    )
                else:
                    # Fallback to draw_panoptic_seg if available
                    vis_output = visualizer.draw_panoptic_seg(panoptic_seg, segments_info)
            elif "sem_seg" in predictions:
                sem_seg = predictions["sem_seg"]
                # Convert to CPU tensor and get argmax
                if hasattr(sem_seg, "cpu"):
                    sem_seg = sem_seg.cpu()
                if hasattr(sem_seg, "argmax"):
                    sem_seg_labels = sem_seg.argmax(dim=0)
                else:
                    sem_seg_labels = np.argmax(sem_seg, axis=0)
                vis_output = visualizer.draw_sem_seg(sem_seg_labels)
            elif "instances" in predictions:
                instances = predictions["instances"]
                # Convert to CPU if needed
                if hasattr(instances, "to") and torch is not None:
                    instances = instances.to(torch.device("cpu"))
                vis_output = visualizer.draw_instance_predictions(predictions=instances)

            return vis_output
        except Exception as e:
            Printer.yellow(f"Detectron2ColorMapManager: Failed to generate visualization: {e}")
            return None

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
                    panoptic_np.astype(np.int32),
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
        # Visualization priority: visualized_output > panoptic_data > direct color mapping

        # Priority 1: Use pre-computed visualized output if available (fastest, highest quality)
        # This is used by DETIC and EOV_SEG models that get visualization from run_on_image()
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

        # Priority 2: For panoptic segmentation, use visualizer (needs overlays and instance info)
        # This generates visualization on-demand using panoptic data
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

        # Priority 3: For semantic segmentation, use color map directly (fastest but no overlays)
        # This is the fallback when no visualization data is available
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
