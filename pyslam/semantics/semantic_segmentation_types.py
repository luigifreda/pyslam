"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present Luigi Freda <luigi dot freda at gmail dot com>
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
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

from pyslam.utilities.serialization import SerializableEnum, register_class


# Just for science
"""
|----------------------------|---------------------------|---------------------------|---------------------------------|
|      **Aspect**            | **Semantic Segmentation** | **Instance Segmentation** | **Panoptic Segmentation**       |
|----------------------------|---------------------------|---------------------------|---------------------------------|
| Objects ("things")         |       ❌ no               |         ✅ yes            |         ✅ yes                  | 
| Background ("stuff")       |       ✅ yes              |         ❌ no             |         ✅ yes                  | 
| Every pixel labeled        |       ✅ yes              |         ❌ no             |         ✅ yes                  | 
| Instance IDs               |       ❌ no               |         ✅ yes            |         ✅ yes (things only)    | 
| Overlapping objects        |       ❌ no               |         ❌ limited        |         ❌ no                   | 
| Object parts distinguished |       ❌ no               |         ❌ no             |         ❌ no                   | 
| Scene understanding        |       Medium              |         Partial           |         Complete                | 
| Typical output             |       Per-pixel class map |         Mask per object   |  Unified map (class + instance) |
"""


#
# - Panoptic segmentation:
#   * DETIC: Object detection-based (CenterNet2 + CLIP), supports large vocabularies (LVIS/COCO/OpenImages/Objects365).
#            Architecture: Object detector (CenterNet2) detects individual objects first, then segments each detection.
#            Can output both "instances" (direct instance segmentation) and "panoptic_seg" formats.
#            Instance extraction: Direct from object detections - each detected object = one instance ID.
#            Result: Robust instance segmentation - each detected object gets a unique instance ID, even for multiple
#            objects of the same category (e.g., two pillows = two separate instances).
#   * ODISE: Diffusion-based panoptic segmentation, leverages diffusion models for segmentation.
#            Architecture: Panoptic segmentation model that segments image into regions first, then classifies regions.
#            Only outputs "panoptic_seg" format - instances extracted from panoptic segments via "isthing" flag.
#            Instance extraction: Derived from panoptic segments - one segment may contain multiple objects if model
#            groups them together (e.g., spatially connected objects of same category).
#            Result: Instance segmentation may merge multiple objects of the same category into a single instance
#            (e.g., two pillows may be detected as one "pillow" instance).
#   * EOV_SEG: Dual-backbone (CNN + ViT) with CLIP, text-prompt driven open vocabulary.
#            Architecture: Panoptic segmentation model (similar to ODISE) - segments image into regions first.
#            Only outputs "panoptic_seg" format - instances extracted from panoptic segments via "isthing" flag.
#            Instance extraction: Same as ODISE - derived from panoptic segments, may group multiple objects.
#            Result: Similar to ODISE, instance segmentation may group multiple objects of the same category together
#            (e.g., two pillows may be detected as one "pillow" instance).
#
# - Semantic segmentation:
#   * DEEPLABV3: Semantic segmentation model from torchvision DeepLab's v3.
#   * SEGFORMER: Semantic segmentation model from transformer's Segformer.
#   * CLIP: uses CLIP patch embeddings + text similarity to produce labels/probabilities (it is not a dedicated "segmentation head").
#
# - Instance segmentation:
#   * RFDETR: RF-DETR segmentation models (instance masks, COCO classes by default).
#             It comes with a segmentation model and a detection model. Different variants are available.
#             Nano, Small, Medium, Large, XLarge, 2XLarge. See pyslam/semantics/semantic_segmentation_rf_detr.py for more details.
#   * YOLO: Ultralytics YOLO segmentation models (instance masks).
#             It comes with a segmentation model and a detection model. Different variants are available.
#             Nano, Small, Medium, Large, XLarge. See pyslam/semantics/semantic_segmentation_yolo.py for more details.


@register_class
class SemanticSegmentationType(SerializableEnum):
    DEEPLABV3 = 0  # Semantics from torchvision DeepLab's v3 [Semantic only:]
    SEGFORMER = 1  # Semantics from transformer's Segformer [Semantic only]
    CLIP = 2  # Semantics from CLIP's segmentation head [Semantic only]
    EOV_SEG = (
        3  # Semantics from EOV-Seg (Efficient Open Vocabulary Segmentation) [Panoptic/instance]
    )
    DETIC = 4  # Semantics from Detic (Detecting Twenty-thousand Classes) [Panoptic/instance]
    ODISE = 5  # Semantics from ODISE (Open-vocabulary DIffusion-based panoptic SEgmentation) [Panoptic/instance]
    RFDETR = 6  # Semantics from RF-DETR segmentation models [Instance segmentation]
    #             Different variants are available. Nano, Small, Medium, Large, XLarge. See pyslam/semantics/semantic_segmentation_rf_detr.py for more details.
    YOLO = 7  # Semantics from Ultralytics YOLO segmentation models [Instance segmentation]
    #           Different variants are available. Nano, Small, Medium, Large, XLarge. See pyslam/semantics/semantic_segmentation_yolo.py for more details.

    @staticmethod
    def from_string(name: str):
        try:
            return SemanticSegmentationType[name]
        except KeyError:
            raise ValueError(f"Invalid SemanticSegmentationType: {name}")
