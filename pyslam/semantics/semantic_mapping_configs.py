"""
* This file is part of PYSLAM
*
* Copyright (C) 2025-present David Morilla-Cabello <davidmorillacabello at gmail dot com>
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

"""
A collection of ready-to-used semantic mapping configurations 
"""

from .semantic_mapping_base import SemanticMappingType
from .semantic_segmentation_factory import SemanticSegmentationType
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from pyslam.io.dataset_types import DatasetType
from pyslam.utilities.system import Printer


class SemanticMappingConfig:
    def __init__(
        self,
        semantic_mapping_type=SemanticMappingType.DENSE,
        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        semantic_feature_type=SemanticFeatureType.LABEL,
        encoder_name="",
        model_path="",
        custom_set_labels=[],
        sim_text_query="",
        image_size=(512, 512),
        device=None,
    ):
        self.semantic_mapping_type = semantic_mapping_type
        self.semantic_segmentation_type = semantic_segmentation_type
        self.semantic_dataset_type = semantic_dataset_type
        self.semantic_feature_type = semantic_feature_type
        self.encoder_name = encoder_name
        self.model_path = model_path
        self.custom_set_labels = custom_set_labels
        self.sim_text_query = sim_text_query
        self.image_size = image_size
        self.device = device

    def from_dict(self, config_dict: dict):
        self.semantic_mapping_type = config_dict.get("semantic_mapping_type", None)
        self.semantic_segmentation_type = config_dict.get("semantic_segmentation_type", None)
        self.semantic_feature_type = config_dict.get("semantic_feature_type", None)
        self.semantic_dataset_type = config_dict.get("semantic_dataset_type", None)
        self.encoder_name = config_dict.get("encoder_name", "")
        self.model_path = config_dict.get("model_path", "")
        self.custom_set_labels = config_dict.get("custom_set_labels", [])
        self.sim_text_query = config_dict.get("sim_text_query", "")
        self.image_size = config_dict.get("image_size", (512, 512))
        self.device = config_dict.get("device", None)

    def to_dict(self) -> dict:
        return {
            "semantic_mapping_type": self.semantic_mapping_type,
            "semantic_segmentation_type": self.semantic_segmentation_type,
            "semantic_feature_type": self.semantic_feature_type,
            "semantic_dataset_type": self.semantic_dataset_type,
            "encoder_name": self.encoder_name,
            "model_path": self.model_path,
            "custom_set_labels": self.custom_set_labels,
            "sim_text_query": self.sim_text_query,
            "image_size": self.image_size,
            "device": self.device,
        }

    def __str__(self) -> str:
        return f"SemanticMappingConfig: \
            semantic_mapping_type = {self.semantic_mapping_type.name}, \
            semantic_segmentation_type = {self.semantic_segmentation_type.name}, \
            semantic_feature_type = {self.semantic_feature_type.name}, \
            semantic_dataset_type = {self.semantic_dataset_type.name}, \
            image_size = {self.image_size}, \
            device = {self.device}"


class SemanticMappingConfigs:

    @staticmethod
    def get_config_from_name(config_name):
        config_dict = getattr(SemanticMappingConfigs, config_name, None)
        if config_dict is not None:
            Printer.cyan("SemanticMappingConfigs: Configuration loaded:", config_dict)
        else:
            Printer.red(f"SemanticMappingConfigs: No configuration found for '{config_name}'")
        return config_dict

    # For convenience, we offer already prepared configurations for some SLAM datasets
    @staticmethod
    def get_config_from_slam_dataset(slam_dataset_name):
        if slam_dataset_name == DatasetType.KITTI:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.TUM:
            Printer.red("Semantics in TUM dataset will be bad with current model!")
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                # semantic_segmentation_type=SemanticSegmentationType.DEEPLABV3,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                # semantic_dataset_type=SemanticDatasetType.ADE20K,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.ICL_NUIM:
            Printer.red("Semantics in ICL_NUIM dataset will be bad with current model!")
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                # semantic_segmentation_type=SemanticSegmentationType.DEEPLABV3,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                # semantic_dataset_type=SemanticDatasetType.ADE20K,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.EUROC:
            Printer.red("Semantics in TUM dataset will be bad with current model!")
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.REPLICA:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.TARTANAIR:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.VIDEO:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.FOLDER:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.ROS1BAG:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.ROS2BAG:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.LIVE:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.SCANNET:
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        else:
            raise ValueError(
                f"SemanticMappingConfigs: No configuration found for SLAM dataset '{slam_dataset_name}'"
            )

    # =====================================
    # Dense-based semantic mapping

    SEGFORMER = dict(
        semantic_mapping_type=SemanticMappingType.DENSE,
        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        semantic_feature_type=SemanticFeatureType.LABEL,
    )

    DEEPLABV3 = dict(
        semantic_mapping_type=SemanticMappingType.DENSE,
        semantic_segmentation_type=SemanticSegmentationType.DEEPLABV3,
        semantic_dataset_type=SemanticDatasetType.VOC,
        semantic_feature_type=SemanticFeatureType.LABEL,
    )
