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


import os
import time
import math 
import numpy as np
import cv2
from enum import Enum

from utils_sys import getchar, Printer 

from parameters import Parameters

from feature_manager import feature_manager_factory
from feature_manager_configs import FeatureManagerConfigs
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes

from loop_detector_base import LoopDetectorTaskType, LoopDetectKeyframeData, LoopDetectorTask, LoopDetectorOutput, LoopDetectorBase
from loop_detector_dbow3 import LoopDetectorDBoW3
from loop_detector_dbow2 import LoopDetectorDBoW2
from loop_detector_obindex2 import LoopDetectorOBIndex2
from loop_detector_ibow import LoopDetectorIBow
from loop_detector_vpr import LoopDetectorHdcDelf, LoopDetectorEigenPlaces, LoopDetectorNetVLAD, LoopDetectorSad, LoopDetectorAlexNet, LoopDetectorCosPlace
from loop_detector_vlad import LoopDetectorVlad

from loop_detector_vocabulary import OrbVocabularyData, VocabularyData, VladVocabularyData


kVerbose = True
kPrintTrackebackDetails = True 

kTimerVerbose = False

kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = kScriptFolder
kDataFolder = kRootFolder + '/data'

   
class GlobalDescriptorType(Enum):
    DBOW2       = 0     # Bags of Words (BoW). This implementation only works with ORB local features. It needs an ORB vocabulary (available).
    DBOW3       = 1     # Bags of Words (BoW). It needs a vocabulary (available with ORB).
    OBINDEX2    = 2     # Hierarchical indexing scheme. Incremental Bags of binary Words. Incrementally builds a vocabulary. If needed transform the non-binary local descriptors into binary descriptors for better performance.
    IBOW        = 3     # Incremental Bags of binary Words (iBoW). It incrementally builds a vocabulary. Built on the top of OBINDEX2. If needed transform the non-binary local descriptors into binary descriptors for better performance.
    HDC_DELF    = 4     # Local DELF descriptor + Hyperdimensional Computing (HDC).
    SAD         = 5     # Sum of Absolute Differences (SAD) ["SeqSLAM: Visual Route-Based Navigation for Sunny Summer Days and Stormy Winter Nights"]). 
    ALEXNET     = 6     # AlexNetConv3Extractor. 
    NETVLAD     = 7     # PatchNetVLADFeatureExtractor. 
    COSPLACE    = 8     # CosPlaceFeatureExtractor model. 
    EIGENPLACES = 9     # EigenPlacesFeatureExtractor model. 
    VLAD        = 10    # VLAD, Vector of Locally Aggregated Descriptors.
    
    
# Additional information about used local descriptors aggregation method
class LocalDescriptorAggregationType(Enum):
    NONE        = 0
    DBOW2       = 1   # Bags of Words (BoW). Works only with ORB local features
    DBOW3       = 2   # Bags of Words (BoW)
    OBINDEX2    = 3   # Hierarchical indexing scheme. Incremental Bags of binary Words.
    IBOW        = 4   # Incremental Bags of binary Words (iBoW). Built on the top of OBINDEX2.
    HDC         = 5   # Hyperdimensional Computing (HDC).
    NETVLAD     = 6   # Patch-level features from NetVLAD residuals
    VLAD        = 7   # VLAD, Vector of Locally Aggregated Descriptors. 
    

"""
A collection of ready-to-used loop detection configurations. These configurations are used by LoopDetectingProcess.
"""
class LoopDetectorConfigs(object):   
    
    DBOW2 = dict(
        global_descriptor_type = GlobalDescriptorType.DBOW2,
        local_feature_manager_config = None,                                      # If None the frontend local descriptors will be re-used (depending on the used descriptor aggregator and vocabulary)
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.DBOW2,
        vocabulary_data = OrbVocabularyData())                                     # Must be a vocabulary built with the frontend local descriptor type

    DBOW2_INDEPENDENT = dict(
        global_descriptor_type = GlobalDescriptorType.DBOW2,
        local_feature_manager_config = FeatureManagerConfigs.ORB2,                # Use an independent ORB2 local feature manager for loop detection (must be compatible with the used descriptor aggregator and vocabulary)
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.DBOW2,
        vocabulary_data = OrbVocabularyData())                                     # Must be a vocabulary built with the frontend local descriptor type  
        
    DBOW3 = dict(
        global_descriptor_type = GlobalDescriptorType.DBOW3,
        local_feature_manager_config = None,                                      # If None the frontend local descriptors will be re-used (depending on the used descriptor aggregator and vocabulary)
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.DBOW3,
        vocabulary_data = OrbVocabularyData())                                     # Must be a vocabulary built with the frontend local descriptor type

    DBOW3_INDEPENDENT = dict(
        global_descriptor_type = GlobalDescriptorType.DBOW3,
        local_feature_manager_config = FeatureManagerConfigs.ORB2,                # Use an independent ORB2 local feature manager for loop detection (must be compatible with the used descriptor aggregator and vocabulary)
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.VLAD,
        vocabulary_data = OrbVocabularyData())                                     # Must be a vocabulary built with the adopted local descriptor type                       

    VLAD = dict(
        global_descriptor_type = GlobalDescriptorType.VLAD,
        local_feature_manager_config = None,                                      # If None the frontend local descriptors will be re-used (depending on the used descriptor aggregator and vocabulary)
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.VLAD,
        vocabulary_data = VladVocabularyData())                                    # Must be a vocabulary built with the adopted local descriptor type

    VLAD_INDEPENDENT = dict(
        global_descriptor_type = GlobalDescriptorType.VLAD,
        local_feature_manager_config = FeatureManagerConfigs.ORB2,                # Use an independent ORB2 local feature manager for loop detection (must be compatible with the used descriptor aggregator and vocabulary)
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.VLAD,
        vocabulary_data = VladVocabularyData())                                    # Must be a vocabulary built with the adopted local descriptor type

    OBINDEX2 = dict(
        global_descriptor_type = GlobalDescriptorType.OBINDEX2,
        local_feature_manager_config = None,                                        # If None the frontend local descriptors will be re-used 
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.OBINDEX2,
        vocabulary_data = None)                                                     # OBIndex2 does not need a vocabulary. It incrementally builds it.

    IBOW = dict(
        global_descriptor_type = GlobalDescriptorType.IBOW,
        local_feature_manager_config = None,                                     # If None the frontend local descriptors will be re-used 
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.IBOW,
        vocabulary_data = None)                                             # IBow does not need a vocabulary. It incrementally builds it.
    
    IBOW_INDEPENDENT = dict(
        global_descriptor_type = GlobalDescriptorType.IBOW,
        local_feature_manager_config = FeatureManagerConfigs.ORB2,               # Use an independent ORB2 local feature manager for loop detection (must be compatible with the used descriptor aggregator and vocabulary)
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.IBOW,
        vocabulary_data = None)                                                  # IBow does not need a vocabulary. It incrementally builds it.
        
    # NOTE: HDC_DELF seems very slow when combined with SLAM. The online computation of global descriptors on KFs 
    #       does not seem able to keep the pace with SLAM. 
    HDC_DELF = dict(
        global_descriptor_type = GlobalDescriptorType.HDC_DELF,
        local_feature_manager_config = None,                                     # It does use its own local feature manager: Delf.
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.HDC,
        vocabulary_data = None)                                                  # It does not need a vocabulary

    SAD = dict(
        global_descriptor_type = GlobalDescriptorType.SAD,
        local_feature_manager_config = None,                                     # Not needed.
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.NONE,
        vocabulary_data = None)                                                  # It does not need a vocabulary

    ALEXNET = dict(
        global_descriptor_type = GlobalDescriptorType.ALEXNET,
        local_feature_manager_config = None,                                     # Not needed.
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.NONE,
        vocabulary_data = None)                                                  # It does not need a vocabulary
    
    NETVLAD = dict(
        global_descriptor_type = GlobalDescriptorType.NETVLAD,
        local_feature_manager_config = None,                                        # Not needed.
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.NETVLAD,
        vocabulary_data = None)                                                     # It does not need a vocabulary

    COSPLACE = dict(
        global_descriptor_type = GlobalDescriptorType.COSPLACE,
        local_feature_manager_config = None,                                       # Not needed.
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.NONE,
        vocabulary_data = None)                                                    # It does not need a vocabulary

    EIGENPLACES = dict(
        global_descriptor_type = GlobalDescriptorType.EIGENPLACES,
        local_feature_manager_config = None,                                       # Not needed.
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.NONE,
        vocabulary_data = None)                                                    # It does not need a vocabulary
    


def loop_detector_factory(
        global_descriptor_type = GlobalDescriptorType.DBOW3,
        local_feature_manager_config = None,                                      # If None the frontend local descriptors will be re-used (depending on the used descriptor aggregator and vocabulary)
        local_descriptor_aggregation_type = LocalDescriptorAggregationType.DBOW3,
        vocabulary_data = OrbVocabularyData()):
    
    if vocabulary_data is not None:
        vocabulary_data.check_download() # check if the vocabulary exists or we need to download it    
        
    # create an independent local feature manager if requested for the loop detection 
    if local_feature_manager_config is None:
        local_feature_manager = None # re-use the frontend local descriptors
    else:
        local_feature_manager = feature_manager_factory(**local_feature_manager_config)  # use an independent local feature manager in loop closing 
    
    loop_detector = None       
    if global_descriptor_type == GlobalDescriptorType.DBOW2:
        if local_feature_manager is not None:
            if local_feature_manager.descriptor_type != FeatureDescriptorTypes.ORB2 and local_feature_manager.descriptor_type != FeatureDescriptorTypes.ORB:
                raise ValueError('loop_detector_factory: local_feature_manager.descriptor_type must be ORB2 or ORB')
        loop_detector = LoopDetectorDBoW2(vocabulary_data=vocabulary_data, local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.DBOW3:
        loop_detector = LoopDetectorDBoW3(vocabulary_data=vocabulary_data, local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.VLAD:
        loop_detector = LoopDetectorVlad(vocabulary_data=vocabulary_data, local_feature_manager=local_feature_manager)        
    elif global_descriptor_type == GlobalDescriptorType.OBINDEX2:
        loop_detector = LoopDetectorOBIndex2(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.IBOW:
        loop_detector = LoopDetectorIBow(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.HDC_DELF:
        loop_detector = LoopDetectorHdcDelf(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.SAD:
        loop_detector = LoopDetectorSad(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.ALEXNET:
        loop_detector = LoopDetectorAlexNet(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.NETVLAD:
        loop_detector = LoopDetectorNetVLAD(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.COSPLACE:
        loop_detector = LoopDetectorCosPlace(local_feature_manager=local_feature_manager)
    elif global_descriptor_type == GlobalDescriptorType.EIGENPLACES:
        loop_detector = LoopDetectorEigenPlaces(local_feature_manager=local_feature_manager)
    else: 
        raise ValueError('loop_detector_factory: unknown global_descriptor_type')
    
    if loop_detector is not None:
        loop_detector.global_descriptor_type = global_descriptor_type
        loop_detector.local_descriptor_aggregation_type = local_descriptor_aggregation_type
        loop_detector.local_feature_manager = local_feature_manager
        loop_detector.vocabulary_data = vocabulary_data
    
    return loop_detector 



def loop_detector_check(loop_detector: LoopDetectorBase, slam_feature_descriptor_type: FeatureDescriptorTypes):
    descriptor_type_to_check = None
    if loop_detector.local_feature_manager is None:
        descriptor_type_to_check = slam_feature_descriptor_type
    else:
        descriptor_type_to_check = loop_detector.local_feature_manager.descriptor_type
        
    if loop_detector.vocabulary_data is not None:
        if descriptor_type_to_check != loop_detector.vocabulary_data.descriptor_type:
            message = f'loop_detector_check: ERROR: incompatible vocabulary type! vocabulary_data.descriptor_type must be {loop_detector.vocabulary_data.descriptor_type.name}.'
            message += f'\n\t If you dont have such vocabulary type you must create it. Otherwise, you can set the loop detector to work with an independent {loop_detector.vocabulary_data.descriptor_type.name} local_feature_manager.'
            message += f'\n\t See the file loop_detector_configs.py for more details.'
            Printer.red(message)
            raise ValueError(message)     
        
    if loop_detector.global_descriptor_type == GlobalDescriptorType.DBOW2:
        if descriptor_type_to_check != FeatureDescriptorTypes.ORB2 and descriptor_type_to_check != FeatureDescriptorTypes.ORB:
            message = f'loop_detector_check: ERROR: incompatible descriptor_type! With DBOW2, the local_feature_manager must be ORB2 or ORB.'
            Printer.red(message)
            raise ValueError(message)