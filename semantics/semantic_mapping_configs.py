"""
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present David Morilla-Cabello <davidmorillacabello at gmail dot com> 
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
from semantic_mapping import SemanticMappingType
from semantic_segmentation_factory import SemanticSegmentationType
from utils_sys import Printer


class SemanticMappingConfigs:

    @staticmethod
    def get_config_from_name(config_name):
        config_dict = getattr(SemanticMappingConfigs, config_name, None)
        if config_dict is not None:
            Printer.cyan("FeatureTrackerConfigs: Configuration loaded:", config_dict)
        else:
            Printer.red(f"FeatureTrackerConfigs: No configuration found for '{config_name}'")
        return config_dict
    
    # =====================================
    # Dense-based semantic mapping

    SEGFORMER = dict(semantic_mapping_type=SemanticMappingType.DENSE,
                     semantic_segmentation_type=SemanticSegmentationType.SEGFORMER,
                     dataset_name='cityscapes',
                     semantic_feature_type='label')
    
    DEEPLABV3 = dict(semantic_mapping_type=SemanticMappingType.DENSE,
                     semantic_segmentation_type=SemanticSegmentationType.DEEPLABV3,
                     dataset_name='cityscapes',
                     semantic_feature_type='label')