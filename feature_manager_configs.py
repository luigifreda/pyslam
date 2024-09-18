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

from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo

from parameters import Parameters  


kNumFeatures=Parameters.kNumFeatures    

        
"""
Interface for feature manager configurations 
"""
class FeatureManagerConfigs(object):   
        
    # Template configuration (required for feature_manager_factory(), see feature_manager.py)
    TEMPLATE = dict(num_features=kNumFeatures,                     
                num_levels=8,     
                scale_factor = 1.2,
                sigma_level0 = Parameters.kSigmaLevel0,                                     
                detector_type = FeatureDetectorTypes.NONE,
                descriptor_type = FeatureDescriptorTypes.NONE)      
            
                 
    # get the useful sub-part of an input configuration dictionary (used for extracting sub-configurations from FeatureTrackerConfigs) 
    # (see for instance the file test_feature_manager.py)      
    @staticmethod        
    def extract_from(dict_in):
        dict_out = { key:dict_in[key] for key in FeatureManagerConfigs.TEMPLATE.keys() if key in dict_in }      
        return dict_out       
        
         
