/*
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
 */

#pragma once

#include <array>
#include <cstddef>
#include <string_view>

namespace pyslam {

class Parameters {
  public:
    // ----------------------------------------------------------------------------
    // Constants
    // ----------------------------------------------------------------------------

    static constexpr float kChi2Mono =
        5.991; // chi-square 2 DOFs, used for reprojection error  (Hartley Zisserman pg 119)
    static constexpr float kChi2Stereo =
        7.815; // chi-square 3 DOFs, used for reprojection error  (Hartley Zisserman pg 119)

    static constexpr double kThHuberMono = 2.447;   // sqrt(kChi2Mono)
    static constexpr double kThHuberStereo = 2.796; // sqrt(kChi2Stereo)

    static constexpr int kSparseImageColorPatchDelta = 1; // centre +- delta
    static constexpr std::size_t kMinWeightForDrawingCovisibilityEdge = 100;
    static constexpr std::size_t kMaxSparseMapPointsToVisualize =
        1000000; // Sparse pointcloud downsampling for very large clouds to reduce queue bandwidth
                 // and GL load

    static constexpr double kMinDepth = 1e-2; // [meters] Minimum depth for a map point

    // Threshold constant (equivalent to Parameters.kMinNumOfCovisiblePointsForCreatingConnection =
    // 15)
    static constexpr int kMinNumOfCovisiblePointsForCreatingConnection = 15;

    static constexpr int kLargeBAWindowSize = 20;
    static constexpr std::size_t kMaxLenFrameDeque = 20;
    static constexpr int kMaxNumOfKeyframesInLocalMap = 80;
    static constexpr int kNumBestCovisibilityKeyFrames = 10;

    static constexpr double kMaxOutliersRatioInPoseOptimization = 0.9;

    // Tracking: Point visibility
    static constexpr float kViewingCosLimitForPoint =
        0.5; // Must be viewing cos < kViewingCosLimitForPoint (viewing angle must be less than 60
    // deg)
    static constexpr double kScaleConsistencyFactor = 1.5;
    static constexpr double kMaxDistanceToleranceFactor = 1.2;
    static constexpr double kMinDistanceToleranceFactor = 0.8;

    // Tracking: Constants
    static constexpr double kRansacThresholdNormalized =
        0.0004; // 0.0003 // metric threshold used for normalized image coordinates
    static constexpr double kRansacProb = 0.999;
    static constexpr int kNumMinInliersEssentialMat = 8;
    // static constexpr int kNumMinInliersPoseOptimizationTrackFrame = 10;
    // static constexpr int kNumMinInliersPoseOptimizationTrackLocalMap = 20;
    // static constexpr int kNumMinInliersTrackLocalMapForNotWaitingLocalMappingIdle = 50;
    // static constexpr int kNumMinObsForKeyFrameDefault = 3;
    static constexpr int kNumMinObsForKeyFrameTrackedPoints = 3;

    static constexpr double kRansacReprojThreshold = 5;
    static constexpr int kRansacMinNumInliers = 15;

    // Stereo Matching: Show matched points
    static constexpr bool kStereoMatchingShowMatchedPoints = false;

    // Tracking: Visual Odometry
    static constexpr int kMaxNumVisualOdometryPoints = 100;
    static constexpr int kMaxNumStereoPointsOnNewKeyframe = 100;

    // Search matches by projection
    static constexpr float kMaxReprojectionDistanceFrame = 7.0f;           // [pixels]    o:7
    static constexpr float kMaxReprojectionDistanceFrameNonStereo = 15.0f; // [pixels]    o:15
    static constexpr float kMaxReprojectionDistanceMap = 3.0f;             // [pixels]    o:1
    static constexpr float kMaxReprojectionDistanceMapRgbd = 3.0f;         // [pixels]    o:3
    static constexpr float kMaxReprojectionDistanceMapReloc = 5.0f;        // [pixels]    o:5
    static constexpr float kMaxReprojectionDistanceFuse = 3.0f;            // [pixels]    o:3
    static constexpr float kMaxReprojectionDistanceSim3 = 7.5f;            // [pixels]    o:7.5
    //
    static constexpr float kMatchRatioTestFrameByProjection = 0.9f;
    static constexpr float kMatchRatioTestMap = 0.8f;

    // Search matches for triangulation by using epipolar lines
    static constexpr float kMinDistanceFromEpipole =
        10.0f; // [pixels] Used with search by epipolar lines

    static constexpr bool kCheckFeaturesOrientation = true;

    static constexpr int kLocalBAWindowSize = 10;

    // Point triangulation
    static constexpr double kCosMaxParallax = 0.9998;
    static constexpr double kMinRatioBaselineDepth = 0.01;

    // Loop closing
    static constexpr float kLoopClosingMaxReprojectionDistanceFuse = 4.0f; // [pixels]    o:4

    // Local mapping
    static constexpr float kKeyframeCullingRedundantObsRatio = 0.9f;
    static constexpr double kKeyframeMaxTimeDistanceInSecForCulling = 0.5;
    static constexpr int kKeyframeCullingMinNumPoints = 0;

    static constexpr int kLocalMappingParallelFusePointsNumWorkers = 2;

    static constexpr int kLocalMappingNumNeighborKeyFramesStereo =
        10; //  [# frames]   for generating new points and fusing them under stereo or RGBD

    static constexpr int kLocalMappingNumNeighborKeyFramesMonocular =
        20; //  [# frames]   for generating new points and fusing them under monocular

    // ----------------------------------------------------------------------------
    // Modifiable parameters
    // ----------------------------------------------------------------------------

    static float kFeatureMatchDefaultRatioTest; // This is the default ratio test used by all
                                                // feature matchers. It can be configured per
                                                // descriptor in feature_tracker_configs.py

    static float kMaxDescriptorDistance; // It is initialized by the first created instance of
                                         // feature_manager.py at runtime
};

} // namespace pyslam