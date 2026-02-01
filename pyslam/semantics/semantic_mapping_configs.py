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
from .semantic_segmentation_types import SemanticSegmentationType
from .semantic_types import SemanticFeatureType, SemanticDatasetType
from pyslam.io.dataset_types import DatasetType
from pyslam.utilities.logging import Printer


class SemanticMappingConfig:
    def __init__(
        self,
        semantic_mapping_type=SemanticMappingType.DENSE,
        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
        semantic_feature_type=SemanticFeatureType.LABEL,
        encoder_name="",
        model_path="",
        custom_set_labels=None,
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
        self.custom_set_labels = custom_set_labels if custom_set_labels is not None else []
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


def get_semantic_segmentation_type_from_str(semantic_segmentation_type_str):
    if (
        semantic_segmentation_type_str is None
        or semantic_segmentation_type_str == "DEFAULT"
        or semantic_segmentation_type_str == "default"
        or semantic_segmentation_type_str == ""
        or semantic_segmentation_type_str == "None"
        or semantic_segmentation_type_str == "none"
    ):
        return None
    semantic_segmentation_type_str = semantic_segmentation_type_str.upper()
    # iterate over SemanticSegmentationType and return the type if it matches
    for semantic_segmentation_type in SemanticSegmentationType:
        if semantic_segmentation_type.name == semantic_segmentation_type_str:
            return semantic_segmentation_type
    raise ValueError(
        f"SemanticSegmentationType: Unknown semantic segmentation type '{semantic_segmentation_type_str}'"
    )
    return None


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
    def get_config_from_slam_dataset(slam_dataset_name, semantic_segmentation_type_str=None):
        """
        Get a semantic mapping configuration for a given SLAM dataset.
        If semantic_segmentation_type is not provided, a default model will be selected based on the dataset.
        """
        semantic_segmentation_type = get_semantic_segmentation_type_from_str(
            semantic_segmentation_type_str
        )

        default_semantic_segmentation_type = SemanticSegmentationType.DETIC

        if slam_dataset_name == DatasetType.KITTI:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=default_semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.TUM:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
                if semantic_segmentation_type == SemanticSegmentationType.SEGFORMER:
                    Printer.red(
                        f"Semantics in TUM dataset will be bad with current model {semantic_segmentation_type.name}!"
                    )
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                # semantic_dataset_type=SemanticDatasetType.ADE20K,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.ICL_NUIM:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                # semantic_dataset_type=SemanticDatasetType.ADE20K,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.EUROC:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
                if semantic_segmentation_type == SemanticSegmentationType.SEGFORMER:
                    Printer.red(
                        f"Semantics in EUROC dataset will be bad with current model {semantic_segmentation_type.name}!"
                    )
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.REPLICA:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.TARTANAIR:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.VIDEO:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.ADE20K,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.FOLDER:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.ROS1BAG:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.ROS2BAG:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.MCAP:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.LIVE:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        elif slam_dataset_name == DatasetType.SCANNET:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                semantic_feature_type=SemanticFeatureType.LABEL,
            )
        else:
            if semantic_segmentation_type is None:
                semantic_segmentation_type = default_semantic_segmentation_type
                Printer.yellow(
                    f"SemanticMappingConfigs: No configuration found for SLAM dataset '{slam_dataset_name}', using default model DETIC"
                )
            return dict(
                semantic_mapping_type=SemanticMappingType.DENSE,
                semantic_segmentation_type=semantic_segmentation_type,
                semantic_dataset_type=SemanticDatasetType.NYU40,
                # semantic_dataset_type=SemanticDatasetType.ADE20K,
                semantic_feature_type=SemanticFeatureType.LABEL,
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
