#include "feature_shared_info.h"
#include <limits>

namespace pyslam {

float FeatureSharedInfo::scale_factor = std::numeric_limits<float>::infinity();
float FeatureSharedInfo::inv_scale_factor = std::numeric_limits<float>::infinity();
float FeatureSharedInfo::log_scale_factor = std::numeric_limits<float>::infinity();

std::vector<float> FeatureSharedInfo::scale_factors;     // level -> scale factor
std::vector<float> FeatureSharedInfo::inv_scale_factors; // level -> inverse scale factor
std::vector<float> FeatureSharedInfo::level_sigmas;
std::vector<float> FeatureSharedInfo::level_sigmas2;     // level -> sigma^2
std::vector<float> FeatureSharedInfo::inv_level_sigmas2; // level -> inverse sigma^2

int FeatureSharedInfo::num_levels = -1;
int FeatureSharedInfo::num_features = -1; // maximum number of features

int FeatureSharedInfo::detector_type = -1;
int FeatureSharedInfo::descriptor_type = -1;

NormType FeatureSharedInfo::norm_type = NormType::None;
SemanticFeatureType FeatureSharedInfo::semantic_feature_type = SemanticFeatureType::NONE;

FeatureDetectAndComputeCallback FeatureSharedInfo::feature_detect_and_compute_callback = nullptr;
FeatureDetectAndComputeCallback FeatureSharedInfo::feature_detect_and_compute_right_callback =
    nullptr;
StereoMatchingCallback FeatureSharedInfo::stereo_matching_callback = nullptr;

} // namespace pyslam