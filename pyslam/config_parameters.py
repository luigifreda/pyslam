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

# NOTE: The actual path of this file is pyslam/config_parameters.py.
#       For convenience, it is also exposed as a symlink in the root folder of the project.


kScriptPath = os.path.realpath(__file__)
kScriptFolder = os.path.dirname(kScriptPath)
kRootFolder = os.path.join(kScriptFolder, "..")


# List of shared static parameters for configuring SLAM modules
class Parameters:

    # ================================================================
    # C++ core
    # ================================================================
    USE_CPP_CORE = True  # True: use the C++ core; False: use the Python core

    # ================================================================
    # Logs
    # ================================================================
    # Folder where logs are stored. This can be changed by pyslam/config.py to redirect the logs in a different folder.
    kLogsFolder = kRootFolder + "/logs"

    # ================================================================
    # Sparse SLAM
    # ================================================================

    # SLAM tracking-mapping threads
    kLocalMappingOnSeparateThread = True  # True: move local mapping on a separate thread, False: tracking and then local mapping in a single thread
    kTrackingWaitForLocalMappingToGetIdle = False  # True: wait for local mapping to get idle before starting tracking, False: tracking and then local mapping in a single thread
    kWaitForLocalMappingTimeout = (
        0.5 if not USE_CPP_CORE else 0.05
    )  # [s]  # Timeout for waiting local mapping to be idle (if kTrackingWaitForLocalMappingToGetIdle is True)   (was previously 1.5)
    kParallelLBAWaitIdleTimeout = (
        0.3 if not USE_CPP_CORE else 0.03
    )  # [s]  # Timeout for parallel LBA process to finish

    # Number of desired keypoints per frame
    kNumFeatures = 2000  # Default number of keypoints (can be overridden by the "settings" file)

    # Adaptive stats
    kUseDynamicDesDistanceTh = True  # Use dynamic descriptor distance threshold
    kUseDescriptorSigmaMadv2 = False  # [Experimental, WIP] In theory, if we use this we should accordingly update all the current descriptor thresholds

    # Point triangulation
    kCosMaxParallaxInitializer = 0.99998  # 0.99998   # Max cos angle for triangulation (min parallax angle) in the Initializer
    kCosMaxParallax = (
        0.9998  # 0.9999, 0.9998         # Max cos angle for triangulation (min parallax angle)
    )
    kMinRatioBaselineDepth = 0.01

    # Point visibility
    kViewingCosLimitForPoint = 0.5  # Must be viewing cos < kViewingCosLimitForPoint (viewing angle must be less than 60 deg)
    kScaleConsistencyFactor = 1.5
    kMaxDistanceToleranceFactor = 1.2
    kMinDistanceToleranceFactor = 0.8

    # Feature management
    kSigmaLevel0 = 1.0  # Sigma of the keypoint localization at level 0 (default value is 1 for ORB detector); can be changed by selected feature
    kFeatureMatchDefaultRatioTest = 0.7  # This is the default ratio test used by all feature matchers. It can be configured per descriptor in feature_tracker_configs.py
    # kInitializerFeatureMatchRatioTest        # Ratio test used by Initializer
    #
    kKdtNmsRadius = 3  # pixels  #3            # Radius for kd-tree based Non-Maxima Suppression
    #
    kCheckFeaturesOrientation = (
        True  # Check the consistency of matched features orientations with RotationHistogram
    )

    # Initializer
    kInitializerDesiredMedianDepth = 1  # When initializing, the initial median depth is computed and forced to this value (for better visualization is > 1)
    kInitializerMinRatioDepthBaseline = 100  # Compare to 1/kMinRatioBaselineDepth
    kInitializerNumMinFeatures = 100
    kInitializerNumMinFeaturesStereo = 500
    kInitializerNumMinTriangulatedPoints = 150
    kInitializerNumMinTriangulatedPointsStereo = 100
    kInitializerFeatureMatchRatioTest = 0.9  # Ratio test used by Initializer   # TODO: put it in an table and make it configurable per descriptor
    kInitializerNumMinNumPointsForPnPWithDepth = 15
    kInitializerUseCellCoverageCheck = True
    kInitializerUseMinFrameDistanceCheck = True

    # Tracking
    kUseMotionModel = True  # Use or not the motion model for computing a first guess pose (that will be optimized by pose optimization)
    kUseSearchFrameByProjection = True  # Match frames by using frame map points projection and epipolar lines; here, the current available interframe pose estimate is used for computing the fundamental mat F
    kMinNumMatchedFeaturesSearchFrameByProjection = (
        20  # If the number of tracked features is below this, then the search fails
    )
    kUseEssentialMatrixFitting = False  # Fit an essential matrix; orientation and keypoint match inliers are estimated by fitting an essential mat (5 points algorithm),
    # WARNING: essential matrix fitting comes with some limitations (please, read the comments of the method slam.estimate_pose_ess_mat())
    kMinNumMatchedFeaturesSearchReferenceFrame = 15
    kMaxNumOfKeyframesInLocalMap = 80
    kNumBestCovisibilityKeyFrames = 10
    kUseVisualOdometryPoints = True
    kMaxNumVisualOdometryPoints = 100
    kMaxNumStereoPointsOnNewKeyframe = 100
    kUseInterruptLocalMapping = False  # Use interrupt to stop local mapping when a new keyframe is created so that the new keyframe can be processed ASAP.
    # WARNING: This may degrade the performance of the local mapping under some circumstances. This makes more sense when real-time performance is attainable.

    kMaxOutliersRatioInPoseOptimization = 0.9  # Maximum ratio of outliers in pose optimization (if the ratio is greater, the pose optimization is not performed)

    kUseMotionBlurDection = True  # Use motion blur detection to find inliers and estimate the inter-frame transformation (assuming frames are very close in space). Only between consecutive frames.
    kMotionBlurDetectionLalacianVarianceThreshold = 100.0  # Threshold for motion blur detection via Laplacian variance (used in detect_blur_laplacian())
    kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography = 30  # Maximum number of matched keypoints for enabling RANSAC with homography (used in track())

    # Keyframe generation
    kNumMinPointsForNewKf = 15  # Minimum number of matched map points for spawning a new KeyFrame
    kNumMinTrackedClosePointsForNewKfNonMonocular = 100  # Minimum number of tracked close map points that for not spawning a new KeyFrame in case of a non-monocular system
    kNumMaxNonTrackedClosePointsForNewKfNonMonocular = 70  # Maximum number of non-tracked close map points for not spawning a new KeyFrame in case of a non-monocular system
    kThNewKfRefRatioMonocular = 0.9  # For determining if a new KF must be spawned, condition 3
    kThNewKfRefRatioStereo = (
        0.75  # For determining if a new KF must be spawned, condition 3, in the case non-monocular
    )
    kThNewKfRefRatioNonMonocular = 0.25  # For determining if a new KF must be spawned in case the system is not monocular, condition 2b
    kUseFeatureCoverageControlForNewKf = False  # [Experimental] check if all the matched map points in the current frame well cover the image (by using an image grid check)
    kUseFovCentersBasedKfGeneration = False  # Use FOV centers based keyframe generation; not considered if KeyFrame.useFovCentersBasedGeneration is set in yaml
    kMaxFovCentersDistanceForKfGeneration = 0.2  # [m] Maximum distance between FOV centers for keyframe generation; not considered if KeyFrame.maxFovCentersDistance is set in yaml

    # Keyframe culling
    kKeyframeCullingRedundantObsRatio = 0.9
    kKeyframeMaxTimeDistanceInSecForCulling = 0.5  # [s]  # Use float('inf') for disabling it
    kKeyframeCullingMinNumPoints = 0

    # Stereo matching
    kStereoMatchingMaxRowDistance = 1.1  # [pixels]
    kStereoMatchingShowMatchedPoints = (
        False  # Show the frame stereo matches (debug stereo matching)
    )

    # Search matches by projection
    kMaxReprojectionDistanceFrame = 7  # [pixels]    o:7
    kMaxReprojectionDistanceFrameNonStereo = 15  # [pixels]    o:15
    kMaxReprojectionDistanceMap = 3  # [pixels]    o:1
    kMaxReprojectionDistanceMapRgbd = 3  # [pixels]    o:3
    kMaxReprojectionDistanceMapReloc = 5  # [pixels]    o:5
    kMaxReprojectionDistanceFuse = 3  # [pixels]    o:3
    kMaxReprojectionDistanceSim3 = 7.5  # [pixels]    o:7.5
    #
    kMatchRatioTestFrameByProjection = 0.9
    kMatchRatioTestMap = 0.8
    kMatchRatioTestEpipolarLine = 0.8  # Used just for test function find_matches_along_line()
    #
    # Reference max descriptor distance (used for initial checks and then updated and adapted)
    kMaxDescriptorDistance = (
        0  # It is initialized by the first created instance of feature_manager.py at runtime
    )

    # Search matches for triangulation by using epipolar lines
    kMinDistanceFromEpipole = 10  # [pixels] Used with search by epipolar lines

    # Local Mapping
    kLocalMappingParallelKpsMatching = True  # True: use parallel keypoint matching in local mapping, False: use serial keypoint matching
    kLocalMappingParallelKpsMatchingNumWorkers = 2
    kLocalMappingParallelFusePointsNumWorkers = 2
    kLocalMappingDebugAndPrintToFile = True
    kLocalMappingNumNeighborKeyFramesStereo = (
        10  #  [# frames]   for generating new points and fusing them under stereo or RGBD
    )
    kLocalMappingNumNeighborKeyFramesMonocular = (
        20  #  [# frames]   for generating new points and fusing them under monocular
    )
    kLocalMappingTimeoutPopKeyframe = 0.5  # [s]

    # Covisibility graph
    kMinNumOfCovisiblePointsForCreatingConnection = 15

    # ================================================================
    # Optimization engine
    # ================================================================
    kOptimizationAllUseGtsam = False  # Use GTSAM for all optimization tasks (it seems to have problems under mac, TODO: fix it)
    kOptimizationFrontEndUseGtsam = (
        False or kOptimizationAllUseGtsam
    )  # Use GTSAM in pose optimization in the frontend
    kOptimizationBundleAdjustUseGtsam = (
        False or kOptimizationAllUseGtsam
    )  # Use GTSAM for LBA and GBA
    kOptimizationLoopClosingUseGtsam = (
        False
        or kOptimizationAllUseGtsam
        # [Experimental,WIP] Use GTSAM for loop closing (relocalization and PGO)
    )

    # ================================================================
    # Bundle Adjustment (BA and LBA)
    # ================================================================
    kLocalBAWindowSize = 20  #  [# frames]
    kUseLargeWindowBA = (
        False  # True: perform BA over a large window; False: do not perform large window BA
    )
    kEveryNumFramesLargeWindowBA = 10  # Number of frames between two large window BA
    kLargeBAWindowSize = 20  #  [# frames]
    kUseParallelProcessLBA = False  # [Experimental] Running LBA in a parallel process (not thread). Not super stable yet!

    # ================================================================
    # Global Bundle Adjustment (GBA)
    # ================================================================
    kUseGBA = (
        True  # To enable/disable global bundle adjustment (GBA). It is activated by loop closing.
    )
    #
    kGBADebugAndPrintToFile = True
    kGBAUseRobustKernel = True

    # ================================================================
    # Loop closing
    # ================================================================
    kUseLoopClosing = True  # To enable/disable loop closing.
    #
    kMinDeltaFrameForMeaningfulLoopClosure = 10
    kMaxResultsForLoopClosure = 5
    kLoopDetectingTimeoutPopKeyframe = 0.5  # [s]
    kLoopClosingDebugWithLoopDetectionImages = False
    kLoopClosingDebugWithSimmetryMatrix = True
    kLoopClosingDebugAndPrintToFile = True
    kLoopClosingDebugWithLoopConsistencyCheckImages = True
    kLoopClosingDebugShowLoopMatchedPoints = False
    kLoopClosingParallelKpsMatching = True
    kLoopClosingParallelKpsMatchingNumWorkers = 2
    kLoopClosingGeometryCheckerMinKpsMatches = 20  # o:20
    kLoopClosingTh2 = 10
    kLoopClosingMaxReprojectionDistanceMapSearch = 10  # [pixels]    o:10
    kLoopClosingMinNumMatchedMapPoints = 40
    kLoopClosingMaxReprojectionDistanceFuse = 4  # [pixels]    o:4
    kLoopClosingFeatureMatchRatioTest = (
        0.75  # TODO: put it in an table and make it configurable per descriptor
    )

    # ================================================================
    # Relocatization
    # ================================================================
    kRelocalizationDebugAndPrintToFile = True
    kRelocalizationMinKpsMatches = 15  # o:15
    kRelocalizationParallelKpsMatching = True
    kRelocalizationParallelKpsMatchingNumWorkers = 2
    kRelocalizationFeatureMatchRatioTest = (
        0.75  # TODO: put it in an table and make it configurable per descriptor
    )
    kRelocalizationFeatureMatchRatioTestLarge = 0.9  # o:0.9
    kRelocalizationPoseOpt1MinMatches = 10  # o:10
    kRelocalizationDoPoseOpt2NumInliers = 50  # o:50
    kRelocalizationMaxReprojectionDistanceMapSearchCoarse = 10  # [pixels]    o:10
    kRelocalizationMaxReprojectionDistanceMapSearchFine = 3  # [pixels]    o:3

    # ================================================================
    # Depth image undistortion
    # ================================================================
    kDepthImageUndistortionUseOptimalNewCameraMatrixWithAlphaScale = True  # True: use optimal new camera matrix for undistorting the depth image, False: use current K as is
    kDepthImageUndistortionOptimalNewCameraMatrixWithAlphaScaleValue = (
        0.7  # Alpha value for the optimal new camera matrix (0.0: no scaling, 1.0: full scaling)
    )

    # ================================================================
    # Dtype for the dense mapping
    # ================================================================
    kDenseMappingDtypeVertices = "float32"  # "float32" or "float64"
    kDenseMappingDtypeColors = "float32"
    kDenseMappingDtypeDepth = "float32"
    kDenseMappingDtypeSemantics = "int32"  # negative values are used for unassigned voxels
    kDenseMappingDtypeObjectIds = "int32"  # negative values are used for unassigned voxels
    kDenseMappingDtypeTriangles = "uint32"

    # ================================================================
    # Volumetric Integration
    # ================================================================
    kDoVolumetricIntegration = False  # To enable/disable volumetric integration (dense mapping)
    #
    # kVolumetricIntegrationType: "VOXEL_GRID",
    #                             "VOXEL_SEMANTIC_GRID",  (to be used with semantic mapping activated)
    #                             "VOXEL_SEMANTIC_PROBABILISTIC_GRID", best for semantic mapping (to be used with semantic mapping activated)
    #                             "TSDF", allows mesh extraction as output
    #                             "GAUSSIAN_SPLATTING", requires CUDA
    #                              see pyslam/dense/volumetric_integrator_types.py and cpp/volumetric/README.md
    kVolumetricIntegrationType = "VOXEL_GRID"
    kVolumetricIntegrationDebugAndPrintToFile = True
    #
    kVolumetricIntegrationVoxelLength = 0.015  # [m]
    kVolumetricIntegrationUseVoxelBlocks = True  # Use (or not) the more efficient voxel blocks (indirect hashing to blocks instead of direct voxel hashing) for volumetric integration
    kVolumetricIntegrationBlockSize = 8  # [voxels] Numer of voxels per side of the block
    kVolumetricIntegrationTBBThreads = (
        2  # Number of threads for TBB parallel operations (global setting)
    )
    kVolumetricIntegrationFpsThrottleEnabled = (
        True  # Enable FPS-based throttle for volumetric integration (control max FPS allowed)
    )
    kVolumetricIntegrationFpsThrottleMinQueueSize = (
        10  # Minimum queue size for enabling FPS-based throttle
    )
    kVolumetricIntegrationFpsMaxThreshold = 10.0  # max FPS allowed
    kVolumetricIntegrationFpsThrottleBaseDelay = 0.01  # base delay [s]
    kVolumetricIntegrationFpsThrottleScale = 0.1  # delay scale per FPS over threshold
    #
    kVolumetricIntegrationVoxelGridMinCount = (
        3  # Minimum number of point observations per voxel for grid integration
    )
    kVolumetricIntegrationVoxelGridMinConfidence = (
        0.6  # Minimum confidence for grid semantic integration (0.5-0.7 is a good range)
    )
    #
    kVolumetricIntegrationVoxelGridUseCarving = False  # Use carving to remove voxels that are inconsistent with the depth image (in front or behind by more than the threshold).
    kVolumetricIntegrationVoxelGridCarvingDepthMin = (
        1e-2  # Minimum depth [m] for carving. Voxels with depth less than this will not be carved.
    )
    kVolumetricIntegrationVoxelGridCarvingDepthMaxIndoor = 8.0  # Maximum depth [m] for carving. Voxels with depth greater than this will not be carved.
    kVolumetricIntegrationVoxelGridCarvingDepthMaxOutdoor = 15.0  # Maximum depth [m] for carving. Voxels with depth greater than this will not be carved.
    kVolumetricIntegrationVoxelGridCarvingDepthThreshold = 3e-2  # Depth threshold [m] for carving. Voxels that differ from the depth image by more than this threshold will be removed.
    #
    kVolumetricIntegrationVoxelGridShadowPointsFilter = (
        True  # Filter shadow points in the grid semantic integration
    )
    #
    kVolumetricIntegrationTsdfExtractMesh = (
        True  # Extract mesh or point cloud as output; only for TSDF
    )
    kVolumetricIntegrationTSdfTrunc = 0.04  # [m]
    kVolumetricIntegrationTsdfDepthTruncIndoor = 4.0  # [m]
    kVolumetricIntegrationTsdfDepthTruncOutdoor = 10.0  # [m]
    #
    kVolumetricIntegrationMinNumLBATimes = 1  # We integrate only the keyframes that have been processed by LBA at least kVolumetricIntegrationMinNumLBATimes times.
    kVolumetricIntegrationOutputTimeInterval = 1.0  # [s]
    #
    kVolumetricIntegrationUseDepthEstimator = (
        False  # Use depth estimator for volumetric integration in the back-end.
    )
    # Since the depth inference time may be above 1 second, the volumetric integrator may be very slow.
    # NOTE: The depth estimator estimates a metric depth (with an absolute scale). You can't combine it with a MONOCULAR SLAM since the SLAM sparse map scale will not be consistent.
    kVolumetricIntegrationDepthEstimatorType = "DEPTH_RAFT_STEREO"  # "DEPTH_PRO","DEPTH_ANYTHING_V2, "DEPTH_ANYTHING_V3", "DEPTH_SGBM", "DEPTH_RAFT_STEREO", "DEPTH_CRESTEREO_PYTORCH"  (see depth_estimator_factory.py)
    kVolumetricIntegrationDepthEstimationFilterShadowPoints = True
    #
    # Semantic integration
    # probabilistic integration: if depth < depth_threshold, the confidence weight is 1.0, otherwise it exponentially decays with the depth decay rate
    # non-probabilistic integration: the confidence counter is only updated when the depth is below the depth threshold
    kVolumetricSemanticProbabilisticIntegrationUseDepth = (
        True  # enable/disable the use of depths for computing semantics confidence
    )
    kVolumetricSemanticProbabilisticIntegrationDepthThresholdIndoor = 5.0  # [m] depth threshold for updating semantics with depth, after this threshold the confidence decays exponentially with the depth decay rate
    kVolumetricSemanticProbabilisticIntegrationDepthThresholdOutdoor = 10.0  # [m] depth threshold for updating semantics with depth, after this threshold the confidence decays exponentially with the depth decay rate
    kVolumetricSemanticProbabilisticIntegrationDepthDecayRateIndoor = (
        0.1  # [1/m] depth decay rate for updating semantics with depth
    )
    kVolumetricSemanticProbabilisticIntegrationDepthDecayRateOutdoor = 0.05  # [1/m] depth decay rate for updating semantics with depth (less aggressive decay rate than indoor)
    kVolumetricSemanticIntegrationUseInstanceIds = (
        True  # Use image instance IDs for 3D semantic integration.
    )
    kVolumetricSemanticIntegrationMinVoteRatio = 0.5  # Minimum vote ratio for semantic integration, while assingning 3D object IDs to 2D semantic instance IDs with assign_object_ids_to_instance_ids()
    kVolumetricSemanticIntegrationMinVotes = 3  # Minimum votes for semantic integration, while assingning 3D object IDs to 2D semantic instance IDs with assign_object_ids_to_instance_ids()

    # ================================================================
    # Sparse semantic mapping and image segmentation
    # ================================================================
    kDoSparseSemanticMappingAndSegmentation = False  # To enable/disable _sparse_ semantic mapping and image segmentation (TODO: may be problematic under mac, fix it)
    #
    # NOTE: If you want _volumetric_ semantic mapping, set:
    # kDoSparseSemanticMappingAndSegmentation = True and
    # kDoVolumetricIntegration = True and
    # kVolumetricIntegrationType = "VOXEL_SEMANTIC_PROBABILISTIC_GRID" or "VOXEL_SEMANTIC_GRID"
    #
    # kSemanticSegmentationType: None/"", "DEEPLABV3", "SEGFORMER", "CLIP", "EOV_SEG", "DETIC", "ODISE", "RFDETR", "YOLO"
    # (see pyslam/semantics/semantic_segmentation_types.py)
    kSemanticSegmentationType = ""  # Override the semantic model selected in semantic_mapping_configs.py. If kSemanticSegmentationType = "" or None, a default model will be selected based on the dataset (see semantic_mapping_configs.py)
    kSemanticMappingOnSeparateThread = (
        True  # To enable/disable the use of a separate thread for semantic mapping
    )
    kSemanticMappingMoveSemanticSegmentationToSeparateProcess = (
        True  # To enable/disable the use of a separate process for semantic segmentation
    )
    kSemanticMappingDebugAndPrintToFile = True
    kUseSemanticsInOptimization = False  # To enable/disable the use of semantics in optimization (disabled by default, it's still very experimental)
    kSemanticMappingTimeoutPopKeyframe = 0.5  # [s]
    #
    kSemanticSegmentationEnforceUniqueInstanceIds = (
        True  # Enforce unique instance IDs for semantic mapping
    )
    kSemanticSegmentationUniqueInstanceMinPixels = (
        20  # Minimum number of pixels for a unique instance ID
    )

    # ================================================================
    # Depth estimator in the front-end (EXPERIMENTAL, WIP)
    # ================================================================
    kUseDepthEstimatorInFrontEnd = (
        False  # To enable/disable depth estimation with monocular front-end.
    )
    #
    # You can directly set your desired depth estimator in main_slam.py.
    kDepthEstimatorRemoveShadowPointsInFrontEnd = True

    # ===============================================================
    # Visualization
    # ===============================================================
    kMaxFeatureTrailLength = 16  # Maximum length of a feature trail

    # Sparse map visualization
    kSparseImageColorPatchDelta = 1  # center +- delta
    kMinWeightForDrawingCovisibilityEdge = 100
    kMaxSparseMapPointsToVisualize = 1e6  # Sparse pointcloud downsampling for very large clouds to reduce queue bandwidth and GL load

    kViewerDrawSlamMapOnSeparateThread = True  # True: draw the slam map on a separate thread, False: draw the slam map on the main thread

    # ================================================================
    # Other parameters
    # ================================================================
    kChi2Mono = 5.991  # chi-square 2 DOFs, used for reprojection error  (Hartley Zisserman pg 119)
    kChi2Stereo = (
        7.815  # chi-square 3 DOFs, used for reprojection error  (Hartley Zisserman pg 119)
    )
    kMinDepth = 1e-2

    # ================================================================
    # Multiprocessing/Multithreading
    # ================================================================
    kMultiprocessingProcessJoinDefaultTimeout = (
        5  # [s] Timeout for multiprocessing process join operations
    )
    kMultithreadingThreadJoinDefaultTimeout = (
        5  # [s] Timeout for multithreading thread join operations
    )


# ================================================================
# Set and get parameters from/to a dictionary
# ================================================================
# NOTE: We put the following functions outside the class as a workaround to avoid pickling problems with multiprocessing.


def set_from_dict(cls, config):
    for key, value in config.items():
        if hasattr(cls, key):  # Ensures it is a defined class attribute
            setattr(cls, key, value)
        else:
            print(f"Unknown config key: {key}")


def to_dict(cls):
    return {key: getattr(cls, key) for key in dir(cls) if not key.startswith("__")}


# ================================================================
# Get numpy dtype from string
# ================================================================


def get_np_dtype(dtype_str):
    allowed = {
        "float32",
        "float64",
        "int32",
        "int64",
        "uint32",
        "uint64",
    }
    if dtype_str not in allowed:
        raise ValueError(f"Unsupported dense mapping dtype: {dtype_str}")
    try:
        import importlib

        np = importlib.import_module("numpy")
    except Exception:
        return dtype_str
    try:
        np_dtype = np.dtype(dtype_str)
    except Exception as exc:
        raise ValueError(f"Unsupported dense mapping dtype: {dtype_str}") from exc
    if np_dtype.name != dtype_str:
        raise ValueError(f"Unsupported dense mapping dtype: {dtype_str}")
    return np_dtype
