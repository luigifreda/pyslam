#include "semantic_serialization.h"

namespace pyslam {

// Helper function to serialize semantic descriptors to JSON
nlohmann::json serialize_semantic_des(const cv::Mat &semantic_des,
                                      SemanticFeatureType semantic_type) {
    if (semantic_des.empty() || semantic_type == SemanticFeatureType::NONE) {
        return nlohmann::json{{"type", nullptr}, {"value", nullptr}};
    }

    std::string type_str = semantic_feature_type_to_string(semantic_type);

    if (semantic_type == SemanticFeatureType::LABEL) {
        // For labels, extract the integer value
        if (semantic_des.type() == CV_32S) {
            int label = semantic_des.at<int>(0, 0);
            return nlohmann::json{{"type", type_str}, {"value", label}};
        }
    } else {
        // For vectors, convert cv::Mat to array
        std::vector<float> values;
        if (semantic_des.type() == CV_32F) {
            for (int i = 0; i < semantic_des.cols; ++i) {
                values.push_back(semantic_des.at<float>(0, i));
            }
            return nlohmann::json{{"type", type_str}, {"value", values}};
        }
    }

    // Fallback for unsupported types
    return nlohmann::json{{"type", nullptr}, {"value", nullptr}};
}

// Helper function to deserialize semantic descriptors from JSON
std::pair<cv::Mat, SemanticFeatureType> deserialize_semantic_des(const nlohmann::json &json_data) {
    if (json_data.is_null() || !json_data.contains("type") || !json_data.contains("value")) {
        return std::make_pair(cv::Mat(), SemanticFeatureType::NONE);
    }

    // Check if type is null before trying to get it as string
    if (json_data["type"].is_null()) {
        return std::make_pair(cv::Mat(), SemanticFeatureType::NONE);
    }

    std::string type_str = json_data["type"].get<std::string>();
    SemanticFeatureType semantic_type;

    try {
        semantic_type = string_to_semantic_feature_type(type_str);
    } catch (const std::invalid_argument &) {
        return std::make_pair(cv::Mat(), SemanticFeatureType::NONE);
    }

    if (json_data["value"].is_null()) {
        return std::make_pair(cv::Mat(), semantic_type);
    }

    if (semantic_type == SemanticFeatureType::LABEL) {
        // For labels, create a 1x1 matrix with the integer value
        int label = json_data["value"].get<int>();
        cv::Mat mat(1, 1, CV_32S);
        mat.at<int>(0, 0) = label;
        return std::make_pair(mat, semantic_type);
    } else {
        // For vectors, convert array to cv::Mat
        auto value_array = json_data["value"].get<std::vector<float>>();
        if (value_array.empty()) {
            return std::make_pair(cv::Mat(), semantic_type);
        }

        cv::Mat mat(1, value_array.size(), CV_32F);
        for (size_t i = 0; i < value_array.size(); ++i) {
            mat.at<float>(0, i) = value_array[i];
        }
        return std::make_pair(mat, semantic_type);
    }
}

} // namespace pyslam