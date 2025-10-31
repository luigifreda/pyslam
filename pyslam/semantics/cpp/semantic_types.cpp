#include "semantic_types.h"

#include <opencv2/core/core.hpp>

namespace pyslam {

// Normalize kps_sem type to avoid mixed-type issues downstream (e.g., push_back)
// Policy: LABEL -> CV_32S;
// PROBABILITY_VECTOR/FEATURE_VECTOR -> CV_32F
int get_cv_depth_for_semantic_feature_type(const SemanticFeatureType &type) {
    switch (type) {
    case SemanticFeatureType::LABEL:
        return CV_32S;
    case SemanticFeatureType::PROBABILITY_VECTOR:
    case SemanticFeatureType::FEATURE_VECTOR:
        return CV_32F;
    default:
        throw std::invalid_argument("Invalid semantic feature type");
    }
}

} // namespace pyslam