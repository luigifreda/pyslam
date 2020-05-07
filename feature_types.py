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

from enum import Enum
import cv2 


'''
NOTES: 
In order to add a new DETECTOR:
- add a new enum in FeatureDetectorTypes
- manage its 'case' in the detector inialization in feature_manager.py 

In order to add a new DESCRIPTOR:
- add a new enum in FeatureDescriptorTypes
- add the related information in the class FeatureInfo below
- manage its 'case' in the descriptor inialization in feature_manager.py 
'''

class FeatureDetectorTypes(Enum):   
    NONE        = 0 
    SHI_TOMASI  = 1   # "Good Features To Track"
    FAST        = 2   # "Faster and better: a machine learning approachto corner detection" 
    SIFT        = 3   # "Object Recognition from Local Scale-Invariant Features"
    ROOT_SIFT   = 4   # "Three things everyone should know to improve object retrieval"
    SURF        = 5   # "SURF: Speeded Up Robust Features"
    ORB         = 6   # "ORB: An efficient alternative to SIFT or SURF"
    ORB2        = 7   # interface for ORB-SLAM2 features (ORB + spatial keypoint filtering)
    BRISK       = 8   # "BRISK: Binary Robust Invariant Scalable Keypoints"
    KAZE        = 9   # "KAZE Features"
    AKAZE       = 10  # "Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces"
    SUPERPOINT  = 11  # [end-to-end] joint detector-descriptor - "SuperPoint: Self-Supervised Interest Point Detection and Description"
    AGAST       = 12  # "AGAST Corner Detector: faster than FAST and even FAST-ER"
    GFTT        = 13  # "Good Features To Track" (it includes SHI-TOMASI and HARRIS methods)
    MSER        = 14  # "Robust Wide Baseline Stereo from Maximally Stable Extremal Regions"
    MSD         = 15  # "Interest points via maximal self-dissimilarities" - Unfortunately it does not work on my setup! 
    STAR        = 16  # StarDetector - "Censure: Center surround extremas for realtime feature detection and matching"
    HL          = 17  # Harris-Laplace - "Scale & affine invariant interest point detectors"
    D2NET       = 18  # [end-to-end] joint detector-descriptor - "D2-Net: A Trainable CNN for Joint Detection and Description of Local Features".  
    DELF        = 19  # [end-to-end] joint detector-descriptor - "Large-Scale Image Retrieval with Attentive Deep Local Features".  
    CONTEXTDESC = 20  # [end-to-end] only with CONTEXTDESC descriptor - "ContextDesc: Local Descriptor Augmentation with Cross-Modality Context"  
    LFNET       = 21  # [end-to-end] joint detector-descriptor - "LF-Net: Learning Local Features from Images"
    R2D2        = 22  # [end-to-end] joint detector-descriptor - "R2D2: Repeatable and Reliable Detector and Descriptor"   
    KEYNET      = 23  # "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters"       


class FeatureDescriptorTypes(Enum):
    NONE        = 0   # used for LK tracker (in main_vo.py)
    SIFT        = 1   # "Object Recognition from Local Scale-Invariant Features"
    ROOT_SIFT   = 2   # "Three things everyone should know to improve object retrieval"
    SURF        = 3   # "SURF: Speeded Up Robust Features"
    ORB         = 4   # [binary] "ORB: An efficient alternative to SIFT or SURF"
    ORB2        = 5   # [binary] interface for ORBSLAM2 features     
    BRISK       = 6   # [binary] "BRISK: Binary Robust Invariant Scalable Keypoints"    
    KAZE        = 7   # only with KAZE or AKAZE detectors - "KAZE Features"
    AKAZE       = 8   # [binary] only with KAZE or AKAZE detectors - "Fast Explicit Diffusion for Accelerated Features in Nonlinear Scale Spaces"
    FREAK       = 9   # [binary] only descriptor - "FREAK: Fast retina keypoint"
    SUPERPOINT  = 10  # [end-to-end] only with SUPERPOINT detector - "SuperPoint: Self-Supervised Interest Point Detection and Description"
    TFEAT       = 11  # only descriptor - " Learning local feature descriptors with triplets and shallow convolutional neural networks"
    BOOST_DESC  = 12  # [binary] only descriptor - "Learning Image Descriptors with Boosting" 
    DAISY       = 13  # only descriptor - "Daisy: An efficient dense descriptor applied to wide baseline stereo" 
    LATCH       = 14  # [binary] only descriptor - "LATCH: Learned Arrangements of Three Patch Codes" 
    LUCID       = 15  # [binary] only descriptor - "Locally uniform comparison image descriptor" - (it requires a color image) <-- !N.B.: not producing good results!
    VGG         = 16  # only descriptor - "Learning local feature descriptors using convex optimisation" 
    HARDNET     = 17  # only descriptor - "Working hard to know your neighbor’s margins: Local descriptor learning loss"
    GEODESC     = 18  # only descriptor - "GeoDesc: Learning Local Descriptors by Integrating Geometry Constraints"
    SOSNET      = 19  # [end-to-end] joint detector-descriptor (only with SUPERPOINT detector) - "SOSNet:Second Order Similarity Regularization for Local Descriptor Learning"
    L2NET       = 20  # only descriptor - "L2-Net: Deep Learning of Discriminative Patch Descriptor in Euclidean Space"
    LOGPOLAR    = 21  # only descriptor - "Beyond Cartesian Representations for Local Descriptors"
    D2NET       = 22  # [end-to-end] joint detector-descriptor (only with D2NET detector) - "D2-Net: A Trainable CNN for Joint Detection and Description of Local Features".  
    DELF        = 23  # [end-to-end] joint detector-descriptor (only with DELF detector) - "Large-Scale Image Retrieval with Attentive Deep Local Features".
    CONTEXTDESC = 24  # [end-to-end] only with CONTEXTDESC detector - "ContextDesc: Local Descriptor Augmentation with Cross-Modality Context"      
    LFNET       = 25  # [end-to-end] joint detector-descriptor (only with LFNET detector) - "LF-Net: Learning Local Features from Images"
    R2D2        = 26  # [end-to-end] joint detector-descriptor (only with R2D2 detector) - "R2D2: Repeatable and Reliable Detector and Descriptor" 
    KEYNET      = 27  # keynet descriptor is HARDNET (only with KEYNET detector) - "Key.Net: Keypoint Detection by Handcrafted and Learned CNN Filters"                  
    
    
class FeatureInfo(object): 
    norm_type = dict() 
    max_descriptor_distance = dict()   # initial reference max descriptor distances used by SLAM for locally searching matches around frame keypoints; 
                                       # these are initialized and then updated by using standard deviation robust estimation (MAD) and exponential smoothing 
                                       # N.B.: these intial reference distances can be easily estimated by using test_feature_matching.py 
                                       #       where (3 x sigma_mad) is computed 
    # 
    norm_type[FeatureDescriptorTypes.NONE] = cv2.NORM_L2
    max_descriptor_distance[FeatureDescriptorTypes.NONE] = float('inf')    
    # 
    norm_type[FeatureDescriptorTypes.SIFT] = cv2.NORM_L2
    max_descriptor_distance[FeatureDescriptorTypes.SIFT] = 450          # SIFT     
    #
    norm_type[FeatureDescriptorTypes.ROOT_SIFT] = cv2.NORM_L2      
    max_descriptor_distance[FeatureDescriptorTypes.ROOT_SIFT] = 0.9     # ROOT_SIFT                 
    #    
    norm_type[FeatureDescriptorTypes.SURF] = cv2.NORM_L2  
    max_descriptor_distance[FeatureDescriptorTypes.SURF] = 0.45         # SURF                    
    #
    norm_type[FeatureDescriptorTypes.ORB] = cv2.NORM_HAMMING
    max_descriptor_distance[FeatureDescriptorTypes.ORB] = 100           # ORB    
    #
    norm_type[FeatureDescriptorTypes.ORB2] = cv2.NORM_HAMMING  
    max_descriptor_distance[FeatureDescriptorTypes.ORB2] = 100          # ORB                   
    #
    norm_type[FeatureDescriptorTypes.BRISK] = cv2.NORM_HAMMING   
    max_descriptor_distance[FeatureDescriptorTypes.BRISK] = 200         # BRISK             
    #
    norm_type[FeatureDescriptorTypes.KAZE] = cv2.NORM_L2    
    max_descriptor_distance[FeatureDescriptorTypes.KAZE] = 1.2          # KAZE                    
    #
    norm_type[FeatureDescriptorTypes.AKAZE] = cv2.NORM_HAMMING       
    max_descriptor_distance[FeatureDescriptorTypes.AKAZE] = 190         # AKAZE             
    #
    norm_type[FeatureDescriptorTypes.FREAK] = cv2.NORM_HAMMING     
    max_descriptor_distance[FeatureDescriptorTypes.FREAK] = 180         # FREAK                                                           
    #
    norm_type[FeatureDescriptorTypes.SUPERPOINT] = cv2.NORM_L2  
    max_descriptor_distance[FeatureDescriptorTypes.SUPERPOINT] = 1.30   # SUPERPOINT       
    #
    norm_type[FeatureDescriptorTypes.TFEAT] = cv2.NORM_L2    
    max_descriptor_distance[FeatureDescriptorTypes.TFEAT] = 11          # TFEAT          
    #
    norm_type[FeatureDescriptorTypes.BOOST_DESC] = cv2.NORM_HAMMING            
    max_descriptor_distance[FeatureDescriptorTypes.BOOST_DESC] = 100    # BOOST_DESC          
    #
    norm_type[FeatureDescriptorTypes.DAISY] = cv2.NORM_L2 
    max_descriptor_distance[FeatureDescriptorTypes.DAISY] = 0.4         # DAISY            
    #
    norm_type[FeatureDescriptorTypes.LATCH] = cv2.NORM_HAMMING    
    max_descriptor_distance[FeatureDescriptorTypes.LATCH] = 120         # LATCH             
    #
    norm_type[FeatureDescriptorTypes.LUCID] = cv2.NORM_HAMMING 
    max_descriptor_distance[FeatureDescriptorTypes.LUCID] = 100         # LUCID               
    #
    norm_type[FeatureDescriptorTypes.VGG] = cv2.NORM_L2    
    max_descriptor_distance[FeatureDescriptorTypes.VGG] = 5             # VGG             
    #
    norm_type[FeatureDescriptorTypes.HARDNET] = cv2.NORM_L2    
    max_descriptor_distance[FeatureDescriptorTypes.HARDNET] = 1.8       # HARDNET          
    #
    norm_type[FeatureDescriptorTypes.GEODESC] = cv2.NORM_L2             # unless GeodescFeature2D.quantize == True 
    max_descriptor_distance[FeatureDescriptorTypes.GEODESC] = 0.4       # GEODESC         
    #
    norm_type[FeatureDescriptorTypes.SOSNET] = cv2.NORM_L2   
    max_descriptor_distance[FeatureDescriptorTypes.SOSNET] = 2          # SOSNET             
    #
    norm_type[FeatureDescriptorTypes.L2NET] = cv2.NORM_L2   
    max_descriptor_distance[FeatureDescriptorTypes.L2NET] = 2.9         # L2NET            
    #
    norm_type[FeatureDescriptorTypes.LOGPOLAR] = cv2.NORM_L2   
    max_descriptor_distance[FeatureDescriptorTypes.LOGPOLAR] = 3.2      # LOGPOLAR                    
    #
    norm_type[FeatureDescriptorTypes.D2NET] = cv2.NORM_L2   
    max_descriptor_distance[FeatureDescriptorTypes.D2NET] = 2.8         # D2NET         
    #
    norm_type[FeatureDescriptorTypes.DELF] = cv2.NORM_L2   
    max_descriptor_distance[FeatureDescriptorTypes.DELF] = 2.1          # DELF                
    #
    norm_type[FeatureDescriptorTypes.CONTEXTDESC] = cv2.NORM_L2         # unless ContextDescFeature2D.quantize == True 
    max_descriptor_distance[FeatureDescriptorTypes.CONTEXTDESC] = 1.6   # CONTEXTDESC    
    #
    norm_type[FeatureDescriptorTypes.LFNET] = cv2.NORM_L2   
    max_descriptor_distance[FeatureDescriptorTypes.LFNET] = 2.2         # LFNET               
    #
    norm_type[FeatureDescriptorTypes.R2D2] = cv2.NORM_L2   
    max_descriptor_distance[FeatureDescriptorTypes.R2D2] = 1.4          # R2D2       
    #
    norm_type[FeatureDescriptorTypes.KEYNET] = cv2.NORM_L2   
    max_descriptor_distance[FeatureDescriptorTypes.KEYNET] = 1.6        # KEYNET      
