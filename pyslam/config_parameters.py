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


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = os.path.join(kScriptFolder, '..')


# List of shared static parameters for configuring SLAM modules 
class Parameters:   

    kLogsFolder = kRootFolder + '/logs'              # Folder where logs are stored. This can be changed by pyslam/config.py to redirect the logs in a different folder.
    
    # SLAM tracking-mapping threads 
    kLocalMappingOnSeparateThread = True             # True: move local mapping on a separate thread, False: tracking and then local mapping in a single thread 
    kTrackingWaitForLocalMappingToGetIdle = False    # True: wait for local mapping to get idle before starting tracking, False: tracking and then local mapping in a single thread  
    kWaitForLocalMappingTimeout = 0.5                # [s]  # Timeout for waiting local mapping to be idle (if kTrackingWaitForLocalMappingToGetIdle is True)   (was previously 1.5)
    kParallelLBAWaitIdleTimeout = 0.3                # [s]  # Timeout for parallel LBA process to finish
    
    # Number of desired keypoints per frame 
    kNumFeatures=2000                               # Default number of keypoints (can be overridden by settings file)
    
    # Adaptive stats 
    kUseDynamicDesDistanceTh = True                 # Use dynamic descriptor distance threshold
    kUseDescriptorSigmaMadv2 = False                # [Experimental, WIP] In theory, if we use this we should accordingly update all the current descriptor thresholds

    # Point triangulation 
    kCosMaxParallaxInitializer=0.99998  # 0.99998   # Max cos angle for triangulation (min parallax angle) in the Initializer
    kCosMaxParallax=0.9998 # 0.9999, 0.9998         # Max cos angle for triangulation (min parallax angle)   
    kMinRatioBaselineDepth = 0.01      
    
    
    # Point visibility 
    kViewingCosLimitForPoint=0.5    # Must be viewing cos < kViewingCosLimitForPoint (viewing angle must be less than 60 deg)
    kScaleConsistencyFactor=1.5
    kMaxDistanceToleranceFactor=1.2 
    kMinDistanceToleranceFactor=0.8   


    # Feature management
    kSigmaLevel0 = 1.0                        # Sigma of the keypoint localization at level 0 (default value is 1 for ORB detector); can be changed by selected feature    
    kFeatureMatchDefaultRatioTest = 0.7       # This is the default ratio test used by all feature matchers. It can be configured per descriptor in feature_tracker_configs.py
    #kInitializerFeatureMatchRatioTest        # Ratio test used by Initializer 
    #
    kKdtNmsRadius = 3 # pixels  #3            # Radius for kd-tree based Non-Maxima Suppression
    #
    kCheckFeaturesOrientation = True 


    # Initializer 
    kInitializerDesiredMedianDepth = 1         # When initializing, the initial median depth is computed and forced to this value (for better visualization is > 1) 
    kInitializerMinRatioDepthBaseline = 100    # Compare to 1/kMinRatioBaselineDepth
    kInitializerNumMinFeatures = 100
    kInitializerNumMinFeaturesStereo = 500
    kInitializerNumMinTriangulatedPoints = 150
    kInitializerNumMinTriangulatedPointsStereo = 100
    kInitializerFeatureMatchRatioTest = 0.9    # Ratio test used by Initializer   # TODO: put it in an table and make it configurable per descriptor
    kInitializerNumMinNumPointsForPnPWithDepth = 15 
    kInitializerUseCellCoverageCheck = True
    kInitializerUseMinFrameDistanceCheck = True


    # Tracking 
    kUseMotionModel = True                            # Use or not the motion model for computing a first guess pose (that will be optimized by pose optimization)  
    kUseSearchFrameByProjection = True                # Match frames by using frame map points projection and epipolar lines; here, the current available interframe pose estimate is used for computing the fundamental mat F
    kMinNumMatchedFeaturesSearchFrameByProjection=20  # If the number of tracked features is below this, then the search fails 
    kUseEssentialMatrixFitting = False                # Fit an essential matrix; orientation and keypoint match inliers are estimated by fitting an essential mat (5 points algorithm), 
                                                      # WARNING: essential matrix fitting comes with some limitations (please, read the comments of the method slam.estimate_pose_ess_mat())
    kMinNumMatchedFeaturesSearchReferenceFrame = 15
    kMaxNumOfKeyframesInLocalMap = 80
    kNumBestCovisibilityKeyFrames = 10
    kUseVisualOdometryPoints = True
    kUseInterruptLocalMapping = False                 # Use interrupt to stop local mapping when a new keyframe is created so that the new keyframe can be processed ASAP.
                                                      # WARNING: This may degrade the performance of the local mapping under some circumstances. This makes more sense when real-time performance is attainable.
    
    kMaxOutliersRatioInPoseOptimization = 0.9         # Maximum ratio of outliers in pose optimization (if the ratio is greater, the pose optimization is not performed)
    
    kUseMotionBlurDection = True                      # Use motion blur detection to find inliers and estimate the inter-frame transformation (assuming frames are very close in space). Only between consecutive frames.
    
    # Keyframe generation 
    kNumMinPointsForNewKf = 15                               # Minimum number of matched map points for spawning a new KeyFrame 
    kNumMinTrackedClosePointsForNewKfNonMonocular = 100      # Minimum number of tracked close map points that for not spawning a new KeyFrame in case of a non-monocular system
    kNumMaxNonTrackedClosePointsForNewKfNonMonocular = 70    # Maximum number of non-tracked close map points for not spawning a new KeyFrame in case of a non-monocular system    
    kThNewKfRefRatioMonocular = 0.9                          # For determining if a new KF must be spawned, condition 3
    kThNewKfRefRatioStereo = 0.75                            # For determining if a new KF must be spawned, condition 3, in the case non-monocular
    kThNewKfRefRatioNonMonocular = 0.25                      # For determining if a new KF must be spawned in case the system is not monocular, condition 2b
    kUseFeatureCoverageControlForNewKf = False               # [Experimental] check if all the matched map points in the current frame well cover the image (by using an image grid check)
    
    
    # Keyframe culling
    kKeyframeCullingRedundantObsRatio = 0.9  
    kKeyframeMaxTimeDistanceInSecForCulling = 0.5 # [s]  # Use float('inf') for disabling it   
    kKeyframeCullingMinNumPoints = 0 


    # Stereo matching 
    kStereoMatchingMaxRowDistance = 1.1       # [pixels] 
    kStereoMatchingShowMatchedPoints = False  # Show the frame stereo matches (debug stereo matching)


    # Search matches by projection 
    kMaxReprojectionDistanceFrame = 7         # [pixels]    o:7
    kMaxReprojectionDistanceFrameRgbd = 15    # [pixels]    o:15
    kMaxReprojectionDistanceMap = 3           # [pixels]    o:1  
    kMaxReprojectionDistanceMapRgbd = 3       # [pixels]    o:3
    kMaxReprojectionDistanceMapReloc = 5      # [pixels]    o:5
    kMaxReprojectionDistanceFuse = 3          # [pixels]    o:3
    kMaxReprojectionDistanceSim3 = 7.5        # [pixels]    o:7.5
    #
    kMatchRatioTestFrameByProjection = 0.9
    kMatchRatioTestMap = 0.8
    kMatchRatioTestEpipolarLine = 0.8      # Used just for test function find_matches_along_line()
    #
    # Reference max descriptor distance (used for initial checks and then updated and adapted)                   
    kMaxDescriptorDistance = 0 # It is initialized by the first created instance of feature_manager.py at runtime 
    

    # Search matches for triangulation by using epipolar lines 
    kMinDistanceFromEpipole = 10            # [pixels] Used with search by epipolar lines 


    # Local Mapping 
    kLocalMappingParallelKpsMatching = True           # True: use parallel keypoint matching in local mapping, False: use serial keypoint matching
    kLocalMappingParallelKpsMatchingNumWorkers = 2
    kLocalMappingParallelFusePointsNumWorkers = 2
    kLocalMappingDebugAndPrintToFile = True    
    kLocalMappingNumNeighborKeyFramesStereo = 10      #  [# frames]   for generating new points and fusing them under stereo or RGBD              
    kLocalMappingNumNeighborKeyFramesMonocular = 20   #  [# frames]   for generating new points and fusing them under monocular              
    kLocalMappingTimeoutPopKeyframe = 0.5             # [s]


    # Covisibility graph 
    kMinNumOfCovisiblePointsForCreatingConnection = 15 
    
    
    # Sparse map visualization 
    kSparseImageColorPatchDelta = 1  # center +- delta


    # Optimization engine 
    kOptimizationFrontEndUseGtsam = False        # Use GTSAM in pose optimization in the frontend
    kOptimizationBundleAdjustUseGtsam = False    # Use GTSAM for LBA and GBA
    kOptimizationLoopClosingUseGtsam = False     # [Experimental,WIP] Use GTSAM for loop closing (relocalization and PGO)
    
    
    # Bundle Adjustment (BA)
    kLocalBAWindow=20                 #  [# frames]   
    kUseLargeWindowBA=False           # True: perform BA over a large window; False: do not perform large window BA       
    kEveryNumFramesLargeWindowBA=10   # Number of frames between two large window BA  
    kLargeBAWindow=20                 #  [# frames] 
    kUseParallelProcessLBA = False    # [Experimental] Not super stable yet! 

        
    # Global Bundle Adjustment (GBA)
    kUseGBA = True                      # Activated by loop closing
    kGBADebugAndPrintToFile = True
    kGBAUseRobustKernel = True

    
    # Loop closing
    kUseLoopClosing = True                                  # To enable/disable loop closing.
    kMinDeltaFrameForMeaningfulLoopClosure = 10
    kMaxResultsForLoopClosure = 5
    kLoopDetectingTimeoutPopKeyframe = 0.5 # [s]
    kLoopClosingDebugWithLoopDetectionImages = False
    kLoopClosingDebugWithSimmetryMatrix = True
    kLoopClosingDebugAndPrintToFile = True
    kLoopClosingDebugWithLoopConsistencyCheckImages = True
    kLoopClosingDebugShowLoopMatchedPoints = False
    kLoopClosingParallelKpsMatching=True    
    kLoopClosingParallelKpsMatchingNumWorkers = 2
    kLoopClosingGeometryCheckerMinKpsMatches = 20            # o:20
    kLoopClosingTh2 = 10
    kLoopClosingMaxReprojectionDistanceMapSearch = 10        # [pixels]    o:10
    kLoopClosingMinNumMatchedMapPoints = 40
    kLoopClosingMaxReprojectionDistanceFuse = 4              # [pixels]    o:4
    kLoopClosingFeatureMatchRatioTest = 0.75                 # TODO: put it in an table and make it configurable per descriptor


    # Relocatization 
    kRelocalizationDebugAndPrintToFile = False
    kRelocalizationMinKpsMatches = 15                       # o:15
    kRelocalizationParallelKpsMatching=True    
    kRelocalizationParallelKpsMatchingNumWorkers = 2    
    kRelocalizationFeatureMatchRatioTest = 0.75                   # TODO: put it in an table and make it configurable per descriptor
    kRelocalizationFeatureMatchRatioTestLarge = 0.9               # o:0.9
    kRelocalizationPoseOpt1MinMatches = 10                        # o:10
    kRelocalizationDoPoseOpt2NumInliers = 50                      # o:50
    kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 10    # [pixels]    o:10 
    kRelocalizationMaxReprojectionDistanceMapSearchFine = 3       # [pixels]    o:3       
    
        
    # Volumetric Integration
    kUseVolumetricIntegration = False                  # To enable/disable volumetric integration (dense mapping)  
    kVolumetricIntegrationType = "TSDF"                # "TSDF", "GAUSSIAN_SPLATTING" (see volumetric_integrator_factory.py)
    kVolumetricIntegrationDebugAndPrintToFile = True
    kVolumetricIntegrationExtractMesh = False          # Extract mesh or point cloud as output
    kVolumetricIntegrationVoxelLength = 0.015          # [m]
    kVolumetricIntegrationSdfTrunc = 0.04              # [m]
    kVolumetricIntegrationDepthTruncIndoor = 4.0       # [m] 
    kVolumetricIntegrationDepthTruncOutdoor = 10.0     # [m]
    kVolumetricIntegrationMinNumLBATimes = 1           # We integrate only the keyframes that have been processed by LBA at least kVolumetricIntegrationMinNumLBATimes times.
    kVolumetricIntegrationOutputTimeInterval = 1.0     # [s]
    kVolumetricIntegrationUseDepthEstimator = False    # Use depth estimator for volumetric integration in the back-end. 
                                                       # Since the depth inference time may be above 1 second, the volumetric integrator may be very slow.
                                                       # NOTE: The depth estimator estimates a metric depth (with an absolute scale). You can't combine it with a MONOCULAR SLAM since the SLAM sparse map scale will not be consistent.
    kVolumetricIntegrationDepthEstimatorType = "DEPTH_RAFT_STEREO"  # "DEPTH_PRO","DEPTH_ANYTHING_V2, "DEPTH_SGBM", "DEPTH_RAFT_STEREO", "DEPTH_CRESTEREO_PYTORCH"  (see depth_estimator_factory.py)
    kVolumetricIntegrationDepthEstimationFilterShadowPoints = True


    # Depth estimator (experimental usage in the front-end, WIP)
    kUseDepthEstimatorInFrontEnd = False                # To enable/disable depth estimation with monocular front-end.
                                                        # You can directly set your desired depth estimator in main_slam.py.
    kDepthEstimatorRemoveShadowPointsInFrontEnd = True
    
    # Semantic mapping. NOTE: By activating the semantic mapping, semantics will be used for the whole SLAM pipeline
    kDoSemanticMapping = False                          # To enable/disable semantic mapping  (disabled by default since it is still problematic under mac, TODO: fix it)
    kSemanticMappingOnSeparateThread = True
    kSemanticMappingDebugAndPrintToFile = True
    kUseSemanticsInOptimization = False
    kSemanticMappingTimeoutPopKeyframe = 0.5 # [s]
    kSemanticMappingNumNeighborKeyFrames = 10             #  [# frames]   only used in object-based to decide the best semantic descriptor       

    # Other parameters 
    kChi2Mono = 5.991   # chi-square 2 DOFs, used for reprojection error  (Hartley Zisserman pg 119)
    kChi2Stereo = 7.815 # chi-square 3 DOFs, used for reprojection error  (Hartley Zisserman pg 119)



def set_from_dict(cls, config):
    for key, value in config.items():
        if hasattr(cls, key):  # Ensures it is a defined class attribute
            setattr(cls, key, value)
        else:
            print(f"Unknown config key: {key}")