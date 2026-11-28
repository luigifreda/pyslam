"""
* This file is part of PYSLAM
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com>
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
from pyslam.utilities.serialization import SerializableEnum, register_class


"""
NOTES: 
In order to add a new DETECTOR:
- add a new enum in FeatureDetectorTypes
- manage its 'case' in the detector inialization in feature_manager.py 
- add its spec in pyslam/visual_models/model_registry.py (MASK_MANAGED_DETECTORS if it filters keypoints itself)

In order to add a new DESCRIPTOR:
- add a new enum in FeatureDescriptorTypes
- add its metric (norm, max_descriptor_distance) in _DESCRIPTOR_METRIC of pyslam/visual_models/model_registry.py
- manage its 'case' in the descriptor inialization in feature_manager.py 
"""


@register_class
class FeatureDetectorTypes(SerializableEnum):
    NONE = 0
    SHI_TOMASI = 1  # "Good Features To Track"
    FAST = 2  # "Faster and better: a machine learning approachto corner detection"
    SIFT = 3  # "Object Recognition from Local Scale-Invariant Features"
    ROOT_SIFT = 4  # "Three things everyone should know to improve object retrieval"
    SURF = 5  # "SURF: Speeded Up Robust Features"
    ORB = 6  # "ORB: An efficient alternative to SIFT or SURF"
    ORB2 = 7  # interface for ORB-SLAM2 features (ORB + spatial keypoint filtering)
    BRISK = 8  # "BRISK: Binary Robust Invariant Scalable Keypoints"
    KAZE = 9  # "KAZE Features"
    AKAZE = 10  # "Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces"
    SUPERPOINT = 11  # [end-to-end] joint detector-descriptor - "SuperPoint: Self-Supervised Interest Point Detection and Description"
    AGAST = 12  # "AGAST Corner Detector: faster than FAST and even FAST-ER"
    GFTT = 13  # "Good Features To Track" (it includes SHI-TOMASI and HARRIS methods)
    MSER = 14  # "Robust Wide Baseline Stereo from Maximally Stable Extremal Regions"
    MSD = 15  # "Interest points via maximal self-dissimilarities" - Unfortunately it does not work on my setup!
    STAR = 16  # StarDetector - "Censure: Center surround extremas for realtime feature detection and matching"
    HL = 17  # Harris-Laplace - "Scale & affine invariant interest point detectors"
    D2NET = 18  # [end-to-end] joint detector-descriptor - "D2-Net: A Trainable CNN for Joint Detection and Description of Local Features".
    DELF = 19  # [end-to-end] joint detector-descriptor - "Large-Scale Image Retrieval with Attentive Deep Local Features".
    CONTEXTDESC = 20  # [end-to-end] only with CONTEXTDESC descriptor - "ContextDesc: Local Descriptor Augmentation with Cross-Modality Context"
    LFNET = (
        21  # [end-to-end] joint detector-descriptor - "LF-Net: Learning Local Features from Images"
    )
    R2D2 = 22  # [end-to-end] joint detector-descriptor - "R2D2: Repeatable and Reliable Detector and Descriptor"
    KEYNET = 23  # "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters"
    DISK = 24  # [end-to-end] joint detector-descriptor - "DISK: Learning local features with policy gradient"
    ALIKED = 25  # [end-to-end] joint detector-descriptor - "ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation"
    LIGHTGLUESIFT = 26  # To be used with LightGlue
    XFEAT = 27  # [end-to-end] joint detector-descriptor - "XFeat: Accelerated Features for Lightweight Image Matching"
    KEYNETAFFNETHARDNET = 28  # [kornia-based] Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor. "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters"
    DEDODE = 29  # [end-to-end, kornia-based] DeDoDe v2 detector + DeDoDe descriptor - "DeDoDe: Detect, Don't Describe - Describe, Don't Detect"
    RDD = 30  # [end-to-end] joint detector-descriptor - "RDD: Robust Feature Detector and Descriptor using Deformable Transformer"


@register_class
class FeatureDescriptorTypes(SerializableEnum):
    NONE = 0  # Used for LK tracker (in main_vo.py)
    SIFT = 1  # "Object Recognition from Local Scale-Invariant Features"
    ROOT_SIFT = 2  # "Three things everyone should know to improve object retrieval"
    SURF = 3  # "SURF: Speeded Up Robust Features"
    ORB = 4  # [binary] "ORB: An efficient alternative to SIFT or SURF"
    ORB2 = 5  # [binary] interface for ORBSLAM2 features
    BRISK = 6  # [binary] "BRISK: Binary Robust Invariant Scalable Keypoints"
    KAZE = 7  # only with KAZE or AKAZE detectors - "KAZE Features"
    AKAZE = 8  # [binary] only with KAZE or AKAZE detectors - "Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces"
    FREAK = 9  # [binary] only descriptor - "FREAK: Fast retina keypoint"
    SUPERPOINT = 10  # [end-to-end] only with SUPERPOINT detector - "SuperPoint: Self-Supervised Interest Point Detection and Description"
    TFEAT = 11  # only descriptor - " Learning local feature descriptors with triplets and shallow convolutional neural networks"
    BOOST_DESC = 12  # [binary] only descriptor - "Learning Image Descriptors with Boosting"
    DAISY = 13  # only descriptor - "Daisy: An efficient dense descriptor applied to wide baseline stereo"
    LATCH = 14  # [binary] only descriptor - "LATCH: Learned Arrangements of Three Patch Codes"
    LUCID = 15  # [binary] only descriptor - "Locally uniform comparison image descriptor" - (it requires a color image) <-- !N.B.: not producing good results!
    VGG = 16  # only descriptor - "Learning local feature descriptors using convex optimisation"
    HARDNET = 17  # only descriptor - "Working hard to know your neighbor’s margins: Local descriptor learning loss"
    GEODESC = 18  # only descriptor - "GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"
    SOSNET = 19  # only descriptor - "SOSNet:Second Order Similarity Regularization for Local Descriptor Learning"
    L2NET = 20  # only descriptor - "L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space"
    LOGPOLAR = 21  # only descriptor - "Beyond Cartesian Representations for Local Descriptors"
    D2NET = 22  # [end-to-end] joint detector-descriptor (only with D2NET detector) - "D2-Net: A Trainable CNN for Joint Detection and Description of Local Features".
    DELF = 23  # [end-to-end] joint detector-descriptor (only with DELF detector) - "Large-Scale Image Retrieval with Attentive Deep Local Features".
    CONTEXTDESC = 24  # [end-to-end] only with CONTEXTDESC detector - "ContextDesc: Local Descriptor Augmentation with Cross-Modality Context"
    LFNET = 25  # [end-to-end] joint detector-descriptor (only with LFNET detector) - "LF-Net: Learning Local Features from Images"
    R2D2 = 26  # [end-to-end] joint detector-descriptor (only with R2D2 detector) - "R2D2: Repeatable and Reliable Detector and Descriptor"
    KEYNET = 27  # keynet descriptor is HARDNET (only with KEYNET detector) - "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters"
    BEBLID = (
        28  # [binary] only descriptor - " BEBLID: Boosted Efficient Binary Local Image Descriptor"
    )
    DISK = 29  # [end-to-end] joint detector-descriptor - "DISK: Learning local features with policy gradient"
    ALIKED = 30  # [end-to-end] joint detector-descriptor - "ALIKED: A Lighter Keypoint and Descriptor Extraction Network via Deformable Transformation"
    LIGHTGLUESIFT = 31  # To be used with LightGlue
    XFEAT = 32  # [end-to-end] joint detector-descriptor - "XFeat: Accelerated Features for Lightweight Image Matching"
    KEYNETAFFNETHARDNET = 33  # [kornia-based] Convenience module, which implements KeyNet detector + AffNet + HardNet descriptor. "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters"
    # MAST3R             = 34  # "Grounding Image Matching in 3D with MASt3R"
    DEDODE = 35  # [end-to-end, kornia-based] only with DEDODE detector - "DeDoDe: Detect, Don't Describe - Describe, Don't Detect" (34 is reserved above)
    RDD = 36  # [end-to-end] only with RDD detector - "RDD: Robust Feature Detector and Descriptor using Deformable Transformer"
    # NOTE: The extraction of independent descriptors from a single image does not make sense for the MASt3R/DUST3R model. The model ground the image matching in a 3D context defined by two images of the same place.


# FeatureInfo class
# This class contains the information about the features and descriptors.
# The values come from the visual-model registry (pyslam/visual_models/model_registry.py),
# the single place where a model's matching metric is declared.
def _feature_info_tables():
    from pyslam.visual_models.model_registry import MASK_MANAGED_DETECTORS, get_spec

    norms = {"l2": cv2.NORM_L2, "hamming": cv2.NORM_HAMMING}
    # The registry stores cv2 norm values without importing cv2: keep them in sync.
    assert get_spec("ORB").feature_info()["norm_type"] == cv2.NORM_HAMMING
    assert get_spec("SIFT").feature_info()["norm_type"] == cv2.NORM_L2
    custom_detector_types = tuple(FeatureDetectorTypes[name] for name in MASK_MANAGED_DETECTORS)
    need_mask_management = {detector_type: True for detector_type in custom_detector_types}
    need_mask_management[FeatureDescriptorTypes.NONE] = False
    norm_type, max_descriptor_distance = {}, {}
    for descriptor_type in FeatureDescriptorTypes:
        spec = get_spec(descriptor_type.name)
        norm_type[descriptor_type] = norms[spec.norm]
        max_descriptor_distance[descriptor_type] = spec.max_descriptor_distance
    return custom_detector_types, need_mask_management, norm_type, max_descriptor_distance


class FeatureInfo(object):
    # custom_detector_types: detectors that manage the keypoint mask themselves
    # need_mask_management: detector type -> need mask management
    # norm_type: descriptor type -> norm type
    # max_descriptor_distance: descriptor type -> reference max descriptor distance, used by SLAM
    #   for locally searching matches around frame keypoints
    custom_detector_types, need_mask_management, norm_type, max_descriptor_distance = (
        _feature_info_tables()
    )
