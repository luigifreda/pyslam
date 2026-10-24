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
#include "config_parameters.h"

#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

namespace pyslam {

std::atomic<float> Parameters::kMaxDescriptorDistance{
    0.0f}; // It is initialized by the first created instance of feature_manager.py at runtime

float Parameters::kFeatureMatchDefaultRatioTest =
    0.7; // This is the default ratio test used by all feature matchers. It can be configured
         // per descriptor in feature_tracker_configs.py

bool Parameters::kLocalMappingOnSeparateThread = true;
bool Parameters::kTrackingWaitForLocalMappingToGetIdle = false;
double Parameters::kWaitForLocalMappingTimeout = 0.05;
double Parameters::kParallelLBAWaitIdleTimeout = 0.03;
double Parameters::kLocalMappingTimeoutPopKeyframe = 0.5;
bool Parameters::kUseDynamicDesDistanceTh = true;
bool Parameters::kUseDescriptorSigmaMadv2 = false;
double Parameters::kCosMaxParallaxInitializer = 0.99998;
double Parameters::kInitializerDesiredMedianDepth = 1.0;
double Parameters::kInitializerMinRatioDepthBaseline = 100.0;
int Parameters::kInitializerNumMinFeatures = 100;
int Parameters::kInitializerNumMinFeaturesStereo = 500;
int Parameters::kInitializerNumMinTriangulatedPoints = 150;
int Parameters::kInitializerNumMinTriangulatedPointsStereo = 100;
float Parameters::kInitializerFeatureMatchRatioTest = 0.9f;
int Parameters::kInitializerNumMinNumPointsForPnPWithDepth = 15;
bool Parameters::kInitializerUseCellCoverageCheck = true;
bool Parameters::kInitializerUseMinFrameDistanceCheck = true;
bool Parameters::kUseMotionModel = true;
bool Parameters::kUseSearchFrameByProjection = true;
int Parameters::kMinNumMatchedFeaturesSearchFrameByProjection = 20;
bool Parameters::kUseEssentialMatrixFitting = false;
int Parameters::kMinNumMatchedFeaturesSearchReferenceFrame = 15;
bool Parameters::kUseVisualOdometryPoints = true;
bool Parameters::kUseInterruptLocalMapping = false;
bool Parameters::kUseMotionBlurDection = true;
double Parameters::kMotionBlurDetectionLalacianVarianceThreshold = 100.0;
int Parameters::kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography = 30;
int Parameters::kNumMinPointsForNewKf = 15;
int Parameters::kNumMinTrackedClosePointsForNewKfNonMonocular = 100;
int Parameters::kNumMaxNonTrackedClosePointsForNewKfNonMonocular = 70;
double Parameters::kThNewKfRefRatioMonocular = 0.9;
double Parameters::kThNewKfRefRatioStereo = 0.75;
double Parameters::kThNewKfRefRatioNonMonocular = 0.25;
bool Parameters::kUseFeatureCoverageControlForNewKf = false;
bool Parameters::kLocalMappingParallelKpsMatching = true;
int Parameters::kLocalMappingParallelKpsMatchingNumWorkers = 2;
bool Parameters::kUseLargeWindowBA = false;
int Parameters::kEveryNumFramesLargeWindowBA = 10;
bool Parameters::kUseParallelProcessLBA = false;
bool Parameters::kOptimizationFrontEndUseGtsam = false;
bool Parameters::kUseLoopClosing = true;
bool Parameters::kDoVolumetricIntegration = false;
bool Parameters::kDoSparseSemanticMappingAndSegmentation = false;
bool Parameters::kSemanticMappingOnSeparateThread = true;

#ifdef USE_PYTHON
namespace {

template <typename T> void read_attr(const py::object &params, const char *name, T &dst) {
    if (py::hasattr(params, name)) {
        dst = py::cast<T>(params.attr(name));
    }
}

} // namespace

void Parameters::sync_runtime_from_python() {
    py::object params = py::module_::import("pyslam.config_parameters").attr("Parameters");
    read_attr(params, "kLocalMappingOnSeparateThread", kLocalMappingOnSeparateThread);
    read_attr(params, "kTrackingWaitForLocalMappingToGetIdle",
              kTrackingWaitForLocalMappingToGetIdle);
    read_attr(params, "kWaitForLocalMappingTimeout", kWaitForLocalMappingTimeout);
    read_attr(params, "kParallelLBAWaitIdleTimeout", kParallelLBAWaitIdleTimeout);
    read_attr(params, "kLocalMappingTimeoutPopKeyframe", kLocalMappingTimeoutPopKeyframe);
    read_attr(params, "kUseDynamicDesDistanceTh", kUseDynamicDesDistanceTh);
    read_attr(params, "kUseDescriptorSigmaMadv2", kUseDescriptorSigmaMadv2);
    read_attr(params, "kCosMaxParallaxInitializer", kCosMaxParallaxInitializer);
    read_attr(params, "kInitializerDesiredMedianDepth", kInitializerDesiredMedianDepth);
    read_attr(params, "kInitializerMinRatioDepthBaseline", kInitializerMinRatioDepthBaseline);
    read_attr(params, "kInitializerNumMinFeatures", kInitializerNumMinFeatures);
    read_attr(params, "kInitializerNumMinFeaturesStereo", kInitializerNumMinFeaturesStereo);
    read_attr(params, "kInitializerNumMinTriangulatedPoints", kInitializerNumMinTriangulatedPoints);
    read_attr(params, "kInitializerNumMinTriangulatedPointsStereo",
              kInitializerNumMinTriangulatedPointsStereo);
    read_attr(params, "kInitializerFeatureMatchRatioTest", kInitializerFeatureMatchRatioTest);
    read_attr(params, "kInitializerNumMinNumPointsForPnPWithDepth",
              kInitializerNumMinNumPointsForPnPWithDepth);
    read_attr(params, "kInitializerUseCellCoverageCheck", kInitializerUseCellCoverageCheck);
    read_attr(params, "kInitializerUseMinFrameDistanceCheck", kInitializerUseMinFrameDistanceCheck);
    read_attr(params, "kUseMotionModel", kUseMotionModel);
    read_attr(params, "kUseSearchFrameByProjection", kUseSearchFrameByProjection);
    read_attr(params, "kMinNumMatchedFeaturesSearchFrameByProjection",
              kMinNumMatchedFeaturesSearchFrameByProjection);
    read_attr(params, "kUseEssentialMatrixFitting", kUseEssentialMatrixFitting);
    read_attr(params, "kMinNumMatchedFeaturesSearchReferenceFrame",
              kMinNumMatchedFeaturesSearchReferenceFrame);
    read_attr(params, "kUseVisualOdometryPoints", kUseVisualOdometryPoints);
    read_attr(params, "kUseInterruptLocalMapping", kUseInterruptLocalMapping);
    read_attr(params, "kUseMotionBlurDection", kUseMotionBlurDection);
    read_attr(params, "kMotionBlurDetectionLalacianVarianceThreshold",
              kMotionBlurDetectionLalacianVarianceThreshold);
    read_attr(params, "kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography",
              kMotionBlurDetectionMaxNumMatchedKpsToEnablRansacHomography);
    read_attr(params, "kNumMinPointsForNewKf", kNumMinPointsForNewKf);
    read_attr(params, "kNumMinTrackedClosePointsForNewKfNonMonocular",
              kNumMinTrackedClosePointsForNewKfNonMonocular);
    read_attr(params, "kNumMaxNonTrackedClosePointsForNewKfNonMonocular",
              kNumMaxNonTrackedClosePointsForNewKfNonMonocular);
    read_attr(params, "kThNewKfRefRatioMonocular", kThNewKfRefRatioMonocular);
    read_attr(params, "kThNewKfRefRatioStereo", kThNewKfRefRatioStereo);
    read_attr(params, "kThNewKfRefRatioNonMonocular", kThNewKfRefRatioNonMonocular);
    read_attr(params, "kUseFeatureCoverageControlForNewKf", kUseFeatureCoverageControlForNewKf);
    read_attr(params, "kLocalMappingParallelKpsMatching", kLocalMappingParallelKpsMatching);
    read_attr(params, "kLocalMappingParallelKpsMatchingNumWorkers",
              kLocalMappingParallelKpsMatchingNumWorkers);
    read_attr(params, "kUseLargeWindowBA", kUseLargeWindowBA);
    read_attr(params, "kEveryNumFramesLargeWindowBA", kEveryNumFramesLargeWindowBA);
    read_attr(params, "kUseParallelProcessLBA", kUseParallelProcessLBA);
    read_attr(params, "kOptimizationFrontEndUseGtsam", kOptimizationFrontEndUseGtsam);
    read_attr(params, "kUseLoopClosing", kUseLoopClosing);
    read_attr(params, "kDoVolumetricIntegration", kDoVolumetricIntegration);
    read_attr(params, "kDoSparseSemanticMappingAndSegmentation",
              kDoSparseSemanticMappingAndSegmentation);
    read_attr(params, "kSemanticMappingOnSeparateThread", kSemanticMappingOnSeparateThread);
    if (py::hasattr(params, "kMaxDescriptorDistance")) {
        kMaxDescriptorDistance = py::cast<float>(params.attr("kMaxDescriptorDistance"));
    }
}
#endif

} // namespace pyslam
