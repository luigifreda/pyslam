#include "feature_shared_resources.h"
#include "config_parameters.h"
#include <limits>

namespace pyslam {

float FeatureSharedResources::scale_factor = std::numeric_limits<float>::infinity();
float FeatureSharedResources::inv_scale_factor = std::numeric_limits<float>::infinity();
float FeatureSharedResources::log_scale_factor = std::numeric_limits<float>::infinity();

std::vector<float> FeatureSharedResources::scale_factors;     // level -> scale factor
std::vector<float> FeatureSharedResources::inv_scale_factors; // level -> inverse scale factor
std::vector<float> FeatureSharedResources::level_sigmas;
std::vector<float> FeatureSharedResources::level_sigmas2;     // level -> sigma^2
std::vector<float> FeatureSharedResources::inv_level_sigmas2; // level -> inverse sigma^2

int FeatureSharedResources::num_levels = -1;
int FeatureSharedResources::num_features = -1; // maximum number of features

int FeatureSharedResources::detector_type = -1;
int FeatureSharedResources::descriptor_type = -1;

NormType FeatureSharedResources::norm_type = NormType::None;
SemanticFeatureType FeatureSharedResources::semantic_feature_type = SemanticFeatureType::NONE;
bool FeatureSharedResources::oriented_features = false;

float FeatureSharedResources::feature_match_ratio_test = Parameters::kFeatureMatchDefaultRatioTest;

FeatureDetectAndComputeCallback FeatureSharedResources::feature_detect_and_compute_callback =
    nullptr;
FeatureDetectAndComputeCallback FeatureSharedResources::feature_detect_and_compute_right_callback =
    nullptr;
FeatureMatchingCallback FeatureSharedResources::stereo_matching_callback = nullptr;
FeatureMatchingCallback FeatureSharedResources::feature_matching_callback = nullptr;

} // namespace pyslam