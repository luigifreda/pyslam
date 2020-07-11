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

from feature_tracker import feature_tracker_factory, FeatureTrackerTypes 
from feature_manager import feature_manager_factory
from feature_types import FeatureDetectorTypes, FeatureDescriptorTypes, FeatureInfo
from feature_matcher import feature_matcher_factory, FeatureMatcherTypes

from parameters import Parameters  


# some default parameters 

kNumFeatures=Parameters.kNumFeatures    

kRatioTest=Parameters.kFeatureMatchRatioTest

kTrackerType = FeatureTrackerTypes.DES_BF      # default descriptor-based, brute force matching with knn 
#kTrackerType = FeatureTrackerTypes.DES_FLANN  # default descriptor-based, FLANN-based matching 
        
        
"""
A collection of ready-to-used feature tracker configurations 
"""
class FeatureTrackerConfigs(object):   
    
    # Test/Template configuration: you can use this to quickly test 
    # - your custom parameters and 
    # - favourite descriptor and detector (check the file feature_types.py)
    TEST = dict(num_features=kNumFeatures,                   
                num_levels = 8,                                  # N.B: some detectors/descriptors do not allow to set num_levels or they set it on their own
                scale_factor = 1.2,                              # N.B: some detectors/descriptors do not allow to set scale_factor or they set it on their own
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.ORB2, 
                match_ratio_test = kRatioTest,
                tracker_type = kTrackerType)
    
    # =====================================
    # LK trackers (these can only be used with VisualOdometry() ... at the present time)
    
    LK_SHI_TOMASI = dict(num_features=kNumFeatures,
                         num_levels = 3,
                         detector_type = FeatureDetectorTypes.SHI_TOMASI,
                         descriptor_type = FeatureDescriptorTypes.NONE, 
                         tracker_type = FeatureTrackerTypes.LK)

    LK_FAST = dict(num_features=kNumFeatures,
                   num_levels = 3,
                   detector_type = FeatureDetectorTypes.FAST, 
                   descriptor_type = FeatureDescriptorTypes.NONE, 
                   tracker_type = FeatureTrackerTypes.LK)


    # =====================================
    # Descriptor-based 'trackers' 
    
    SHI_TOMASI_ORB = dict(num_features=kNumFeatures,                   # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                          num_levels = 8, 
                          scale_factor = 1.2,
                          detector_type = FeatureDetectorTypes.SHI_TOMASI, 
                          descriptor_type = FeatureDescriptorTypes.ORB, 
                          match_ratio_test = kRatioTest,
                          tracker_type = kTrackerType)
    
    SHI_TOMASI_FREAK = dict(num_features=kNumFeatures,                     
                            num_levels=8,                      
                            scale_factor = 1.2,
                            detector_type = FeatureDetectorTypes.SHI_TOMASI, 
                            descriptor_type = FeatureDescriptorTypes.FREAK, 
                            match_ratio_test = kRatioTest,
                            tracker_type = kTrackerType)      

    FAST_ORB = dict(num_features=kNumFeatures,                         # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                    num_levels = 8, 
                    scale_factor = 1.2,
                    detector_type = FeatureDetectorTypes.FAST, 
                    descriptor_type = FeatureDescriptorTypes.ORB, 
                    match_ratio_test = kRatioTest,                         
                    tracker_type = kTrackerType) 
    
    FAST_FREAK = dict(num_features=kNumFeatures,                       
                      num_levels = 8,
                      scale_factor = 1.2,                    
                      detector_type = FeatureDetectorTypes.FAST, 
                      descriptor_type = FeatureDescriptorTypes.FREAK,      
                      match_ratio_test = kRatioTest,                          
                      tracker_type = kTrackerType)       
    
    BRISK_TFEAT = dict(num_features=kNumFeatures,                     
                       num_levels = 4, 
                       scale_factor = 1.2,
                       detector_type = FeatureDetectorTypes.BRISK, 
                       descriptor_type = FeatureDescriptorTypes.TFEAT, 
                       match_ratio_test = kRatioTest,                               
                       tracker_type = kTrackerType)        

    ORB = dict(num_features=kNumFeatures, 
               num_levels = 8, 
               scale_factor = 1.2, 
               detector_type = FeatureDetectorTypes.ORB, 
               descriptor_type = FeatureDescriptorTypes.ORB, 
               match_ratio_test = kRatioTest,                        
               tracker_type = kTrackerType)
    
    ORB2 = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.ORB2, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType)    
    
    ORB2_FREAK = dict(num_features=kNumFeatures, 
                      num_levels = 8, 
                      scale_factor = 1.2,                     
                      detector_type = FeatureDetectorTypes.ORB2, 
                      descriptor_type = FeatureDescriptorTypes.FREAK, 
                      match_ratio_test = kRatioTest,                        
                      tracker_type = kTrackerType)    
    

    BRISK = dict(num_features=kNumFeatures,
                 num_levels = 8,
                 detector_type = FeatureDetectorTypes.BRISK, 
                 descriptor_type = FeatureDescriptorTypes.BRISK,
                 match_ratio_test = kRatioTest,                           
                 tracker_type = kTrackerType)   

    KAZE = dict(num_features=kNumFeatures,
                num_levels = 8,
                detector_type = FeatureDetectorTypes.KAZE, 
                descriptor_type = FeatureDescriptorTypes.KAZE, 
                match_ratio_test = kRatioTest,                          
                tracker_type = kTrackerType)  
    
    AKAZE = dict(num_features=kNumFeatures,
                 num_levels = 8,
                 detector_type = FeatureDetectorTypes.AKAZE, 
                 descriptor_type = FeatureDescriptorTypes.AKAZE, 
                 match_ratio_test = kRatioTest,                          
                 tracker_type = kTrackerType)  
                
    SIFT = dict(num_features=kNumFeatures,
                detector_type = FeatureDetectorTypes.SIFT, 
                descriptor_type = FeatureDescriptorTypes.SIFT, 
                match_ratio_test = kRatioTest,                         
                tracker_type = kTrackerType)
    
    ROOT_SIFT = dict(num_features=kNumFeatures,
                     detector_type = FeatureDetectorTypes.ROOT_SIFT, 
                     descriptor_type = FeatureDescriptorTypes.ROOT_SIFT, 
                     match_ratio_test = kRatioTest,                              
                     tracker_type = kTrackerType)    
    
    SURF = dict(num_features=kNumFeatures,
                num_levels = 8,
                detector_type = FeatureDetectorTypes.SURF, 
                descriptor_type = FeatureDescriptorTypes.SURF, 
                match_ratio_test = kRatioTest,                         
                tracker_type = kTrackerType)
        
    SUPERPOINT = dict(num_features=kNumFeatures,                            # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.SUPERPOINT, 
                      descriptor_type = FeatureDescriptorTypes.SUPERPOINT, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = kTrackerType)

    CONTEXTDESC = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.CONTEXTDESC, 
                       descriptor_type = FeatureDescriptorTypes.CONTEXTDESC, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    
    
