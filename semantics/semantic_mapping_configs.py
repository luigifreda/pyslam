"""
* This file is part of PYSLAM 
*
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


"""
A collection of ready-to-used semantic mapping configurations 
"""
from dataset_types import DatasetType
from semantic_utils import SemanticDatasetType
from semantic_mapping import SemanticMappingType
from semantic_segmentation_factory import SemanticSegmentationType
from semantic_types import SemanticFeatureType
from utils_sys import Printer

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
        if(slam_dataset_name == DatasetType.KITTI):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.TUM):
            Printer.red("Semantics in TUM dataset will be bad with current model!")
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.NYU40,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.EUROC):
            Printer.red("Semantics in TUM dataset will be bad with current model!")
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.REPLICA):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.NYU40,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.TARTANAIR):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.VIDEO):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.FOLDER):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.ROS1BAG):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.NYU40,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.ROS2BAG):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.NYU40,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.LIVE):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.NYU40,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        elif(slam_dataset_name == DatasetType.SCANNET):
            return dict(semantic_mapping_type=SemanticMappingType.DENSE,
                        semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                        semantic_dataset_type=SemanticDatasetType.NYU40,
                        semantic_feature_type=SemanticFeatureType.LABEL)
        else:
            raise ValueError(f"SemanticMappingConfigs: No configuration found for SLAM dataset '{slam_dataset_name}'")

    # =====================================
    # Dense-based semantic mapping

    SEGFORMER = dict(semantic_mapping_type=SemanticMappingType.DENSE,
                     semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                     semantic_dataset_type=SemanticDatasetType.CITYSCAPES,
                     semantic_feature_type=SemanticFeatureType.LABEL)
    
    DEEPLABV3 = dict(semantic_mapping_type=SemanticMappingType.DENSE,
                     semantic_segmentation_type=SemanticSegmentationType.DEEPLABV3,
                     semantic_dataset_type=SemanticDatasetType.VOC,
                     semantic_feature_type=SemanticFeatureType.LABEL)