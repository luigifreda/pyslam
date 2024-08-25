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
from feature_matcher import FeatureMatcherTypes

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

    BRISK = dict(num_features=kNumFeatures,                     
                num_levels = 4, 
                scale_factor = 1.2,
                detector_type = FeatureDetectorTypes.BRISK, 
                descriptor_type = FeatureDescriptorTypes.BRISK, 
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
    
    # NOTE: SURF is a patented algorithm and not included in the new opencv versions 
    #       If you want to test it, you can install and old version of opencv that supports it: run 
    #       $ pip3 uninstall opencv-contrib-python
    #       $ pip3 install opencv-contrib-python==3.4.2.16
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
    
    XFEAT = dict(num_features=kNumFeatures,                            # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.XFEAT, 
                      descriptor_type = FeatureDescriptorTypes.XFEAT, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = kTrackerType)      

    XFEAT_XFEAT = dict(num_features=kNumFeatures,                            # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.XFEAT, 
                      descriptor_type = FeatureDescriptorTypes.XFEAT, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = FeatureTrackerTypes.XFEAT)  # <= Using XFEAT matcher here!
        
        
    LIGHTGLUE = dict(num_features=kNumFeatures,                            # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.SUPERPOINT, 
                      descriptor_type = FeatureDescriptorTypes.SUPERPOINT, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = FeatureTrackerTypes.LIGHTGLUE)
    
    LIGHTGLUE_DISK = dict(num_features=kNumFeatures,                            # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.DISK, 
                      descriptor_type = FeatureDescriptorTypes.DISK, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = FeatureTrackerTypes.LIGHTGLUE)    
    
    LIGHTGLUE_ALIKED = dict(num_features=kNumFeatures,                            # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.ALIKED, 
                      descriptor_type = FeatureDescriptorTypes.ALIKED, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = FeatureTrackerTypes.LIGHTGLUE)      
    
    LIGHTGLUESIFT = dict(num_features=kNumFeatures,                            
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.LIGHTGLUESIFT, 
                      descriptor_type = FeatureDescriptorTypes.LIGHTGLUESIFT, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = FeatureTrackerTypes.LIGHTGLUE)         
    
    DELF = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.DELF, 
                       descriptor_type = FeatureDescriptorTypes.DELF, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    D2NET = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.D2NET, 
                       descriptor_type = FeatureDescriptorTypes.D2NET, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    
    R2D2 = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.R2D2, 
                       descriptor_type = FeatureDescriptorTypes.R2D2, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    
    LFNET = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.LFNET, 
                       descriptor_type = FeatureDescriptorTypes.LFNET, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    
    CONTEXTDESC = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.CONTEXTDESC, 
                       descriptor_type = FeatureDescriptorTypes.CONTEXTDESC, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    
    KEYNET = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.KEYNET, 
                       descriptor_type = FeatureDescriptorTypes.KEYNET, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
        
    DISK = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.DISK, 
                       descriptor_type = FeatureDescriptorTypes.DISK, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)
    ALIKED = dict(num_features=kNumFeatures,                   
                       num_levels = 1,                                  
                       scale_factor = 1.2,                              
                       detector_type = FeatureDetectorTypes.ALIKED, 
                       descriptor_type = FeatureDescriptorTypes.ALIKED, 
                       match_ratio_test = kRatioTest,
                       tracker_type = kTrackerType)    
    
    KEYNETAFFNETHARDNET = dict(num_features=kNumFeatures,                            # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                      num_levels = 1, 
                      scale_factor = 1.2,
                      detector_type = FeatureDetectorTypes.KEYNETAFFNETHARDNET, 
                      descriptor_type = FeatureDescriptorTypes.KEYNETAFFNETHARDNET, 
                      match_ratio_test = kRatioTest,                               
                      tracker_type = kTrackerType) 
    
    # =====================================
    # Descriptor-based 'trackers' with ORB2
    
    ORB2_FREAK = dict(num_features=kNumFeatures, 
                      num_levels = 8, 
                      scale_factor = 1.2,                     
                      detector_type = FeatureDetectorTypes.ORB2, 
                      descriptor_type = FeatureDescriptorTypes.FREAK, 
                      match_ratio_test = kRatioTest,                        
                      tracker_type = kTrackerType)    
    
    ORB2_BEBLID = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.BEBLID, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType)    
    
    ORB2_HARDNET = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.HARDNET, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType)    
    
    ORB2_SOSNET = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.SOSNET, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType)   
    
    ORB2_L2NET = dict(num_features=kNumFeatures, 
                num_levels = 8, 
                scale_factor = 1.2, 
                detector_type = FeatureDetectorTypes.ORB2, 
                descriptor_type = FeatureDescriptorTypes.L2NET, 
                match_ratio_test = kRatioTest,                        
                tracker_type = kTrackerType) 

    # =====================================
    # Matcher-based 'trackers'
    # Note: The following matchers are NOT able to extract keypoints and descriptors on a single provided image. They work directly on a pair of images (img1, img2) and produce
    #       as a result a pair of corresponding keypoint vectors (kps1, kps2). 
    #       By design, if we feed these matchers with video images then the extracted keypoints are different on each image. That is, given: 
    #       - matcher(img1, img2) -> (kps1, kps2a)
    #       - matcher(img2, img3) -> (kps2b, kps3)
    #       we have that the keypoint kps2a[i], extrated on img2 the first time, does not necessarily correspond to kps2b[i] or to any other kps2b[j] extracted the second time on img2. 
    # WARNING: For the reasons explained above, at present, we cannot use these "pure" matchers with classic SLAM. In fact, mapping and localization processes need more than two observations 
    #          for each triangulated 3D point along different frames to obtain persistent map points and properly constrain camera pose optimizations in the Sim(3) manifold. WIP.
    
    LOFTR = dict(num_features=kNumFeatures,                            # N.B.: here, keypoints are not oriented! (i.e. keypoint.angle=0 always)
                num_levels = 1, 
                scale_factor = 1.2,
                detector_type = FeatureDetectorTypes.NONE, 
                descriptor_type = FeatureDescriptorTypes.NONE, 
                match_ratio_test = kRatioTest,                               
                tracker_type = FeatureTrackerTypes.LOFTR)       