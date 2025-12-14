#include "semantic_utils.h"
#include "semantic_labels.h"
#include "semantic_types.h"
#include <algorithm>
#include <cmath>
#include <limits>
#include <opencv2/opencv.hpp>
#include <stdexcept>

namespace pyslam {

cv::Mat similarity_heatmap_image(const cv::Mat &sim_map, int colormap, double sim_scale, bool bgr) {
    cv::Mat sim_map_normalized;
    sim_map.convertTo(sim_map_normalized, CV_32F);

    // Normalize to 0-1 for colormap
    sim_map_normalized = sim_map_normalized * sim_scale;
    cv::threshold(sim_map_normalized, sim_map_normalized, 1.0, 1.0, cv::THRESH_TRUNC);
    cv::threshold(sim_map_normalized, sim_map_normalized, 0.0, 0.0, cv::THRESH_TOZERO);

    // Convert to 0-255 range and invert
    sim_map_normalized = (1.0 - sim_map_normalized) * 255.0;
    sim_map_normalized.convertTo(sim_map_normalized, CV_8U);

    // Apply colormap
    cv::Mat sim_color;
    cv::applyColorMap(sim_map_normalized, sim_color, colormap);

    if (!bgr) {
        cv::cvtColor(sim_color, sim_color, cv::COLOR_BGR2RGB);
    }

    return sim_color;
}

cv::Vec3b similarity_heatmap_point(double sim_point, int colormap, double sim_scale, bool bgr) {
    // Create a 1x1 image for the single point
    cv::Mat point_mat(1, 1, CV_32F, cv::Scalar(sim_point));
    cv::Mat color_mat = similarity_heatmap_image(point_mat, colormap, sim_scale, bgr);
    return color_mat.at<cv::Vec3b>(0, 0);
}

cv::Mat labels_to_image(const cv::Mat &label_img, const std::vector<cv::Vec3b> &semantics_color_map,
                        bool bgr, const std::vector<int> &ignore_labels, const cv::Mat &rgb_image) {
    cv::Mat rgb_output(label_img.size(), CV_8UC3);

    // Convert color map to BGR if needed
    std::vector<cv::Vec3b> color_map = semantics_color_map;
    if (bgr) {
        for (auto &color : color_map) {
            std::swap(color[0], color[2]); // Swap R and B
        }
    }

    // Apply color mapping - handle both CV_8UC1 and CV_32SC1
    for (int y = 0; y < label_img.rows; ++y) {
        for (int x = 0; x < label_img.cols; ++x) {
            int label;
            if (label_img.type() == CV_8UC1) {
                label = static_cast<int>(label_img.at<uchar>(y, x));
            } else if (label_img.type() == CV_32SC1) {
                label = label_img.at<int>(y, x);
            } else {
                throw std::invalid_argument(
                    "Unsupported label image type. Expected CV_8UC1 or CV_32SC1");
            }

            if (label >= 0 && label < static_cast<int>(color_map.size())) {
                rgb_output.at<cv::Vec3b>(y, x) = color_map[label];
            }
        }
    }

    // Handle ignore labels
    if (!ignore_labels.empty()) {
        if (rgb_image.empty()) {
            throw std::invalid_argument("rgb_image must be provided if ignore_labels is not empty");
        }

        cv::Mat mask = cv::Mat::zeros(label_img.size(), CV_8UC1);
        for (int ignore_label : ignore_labels) {
            cv::Mat label_mask;
            if (label_img.type() == CV_8UC1) {
                cv::compare(label_img, static_cast<uchar>(ignore_label), label_mask, cv::CMP_EQ);
            } else {
                cv::compare(label_img, ignore_label, label_mask, cv::CMP_EQ);
            }
            cv::bitwise_or(mask, label_mask, mask);
        }
        rgb_image.copyTo(rgb_output, mask);
    }

    return rgb_output;
}

cv::Mat rgb_to_class(const cv::Mat &rgb_labels, const std::vector<cv::Vec3b> &label_map) {
    cv::Mat class_image(rgb_labels.size(), CV_8U);

    for (int y = 0; y < rgb_labels.rows; ++y) {
        for (int x = 0; x < rgb_labels.cols; ++x) {
            cv::Vec3b pixel_color = rgb_labels.at<cv::Vec3b>(y, x);

            // Find matching color in label map
            int best_match = -1;
            for (size_t i = 0; i < label_map.size(); ++i) {
                if (pixel_color == label_map[i]) {
                    best_match = static_cast<int>(i);
                    break;
                }
            }

            class_image.at<uchar>(y, x) = (best_match >= 0) ? best_match : 0;
        }
    }

    return class_image;
}

cv::Vec3b single_label_to_color(int label, const std::vector<cv::Vec3b> &semantics_color_map,
                                bool bgr) {
    if (label < 0 || label >= static_cast<int>(semantics_color_map.size())) {
        return cv::Vec3b(0, 0, 0); // Return black for invalid labels
    }

    cv::Vec3b color = semantics_color_map[label];
    if (bgr) {
        std::swap(color[0], color[2]); // Swap R and B
    }

    return color;
}

// ------------------------------------------------------------

std::vector<cv::Vec3b> labels_color_map_factory(SemanticDatasetType semantic_dataset_type,
                                                int num_classes) {
    switch (semantic_dataset_type) {
    case SemanticDatasetType::VOC:
        return get_voc_color_map();
    case SemanticDatasetType::CITYSCAPES:
        return get_cityscapes_color_map();
    case SemanticDatasetType::ADE20K:
        return get_ade20k_color_map();
    case SemanticDatasetType::NYU40:
        return get_nyu40_color_map();
    case SemanticDatasetType::CUSTOM_SET:
        if (num_classes <= 0) {
            throw std::invalid_argument(
                "num_classes must be provided and > 0 if semantic_dataset_type is CUSTOM_SET");
        }
        return get_generic_color_map(num_classes);
    default:
        throw std::invalid_argument("Unknown dataset type: " +
                                    std::to_string(static_cast<int>(semantic_dataset_type)));
    }
}

std::vector<std::string> labels_name_factory(SemanticDatasetType semantic_dataset_type) {
    switch (semantic_dataset_type) {
    case SemanticDatasetType::VOC:
        return get_voc_labels();
    case SemanticDatasetType::CITYSCAPES:
        return get_cityscapes_labels();
    case SemanticDatasetType::ADE20K:
        return get_ade20k_labels();
    case SemanticDatasetType::NYU40:
        return get_nyu40_labels();
    case SemanticDatasetType::CUSTOM_SET:
        throw std::invalid_argument("CUSTOM_SET does not have predefined labels");
    default:
        throw std::invalid_argument("Unknown dataset type: " +
                                    std::to_string(static_cast<int>(semantic_dataset_type)));
    }
}

std::vector<double> information_weights_factory(SemanticDatasetType semantic_dataset_type,
                                                int num_classes) {
    switch (semantic_dataset_type) {
    case SemanticDatasetType::VOC:
        return get_voc_information_weights();
    case SemanticDatasetType::CITYSCAPES:
        return get_cityscapes_information_weights();
    case SemanticDatasetType::ADE20K:
        return get_ade20k_information_weights();
    case SemanticDatasetType::NYU40:
        return get_nyu40_information_weights();
    case SemanticDatasetType::CUSTOM_SET:
        if (num_classes <= 0) {
            throw std::invalid_argument(
                "num_classes must be provided and > 0 if semantic_dataset_type is CUSTOM_SET");
        }
        return get_trivial_information(num_classes);
    default:
        throw std::invalid_argument("Unknown dataset type: " +
                                    std::to_string(static_cast<int>(semantic_dataset_type)));
    }
}

// ------------------------------------------------------------

std::vector<double> get_voc_information_weights() {
    return std::vector<double>(21, 1.0); // All weights are 1.0
}

std::vector<double> get_cityscapes_information_weights() {
    return {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.001, 1.0,
            1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
}

std::vector<double> get_ade20k_information_weights() { return get_trivial_information(150); }

std::vector<double> get_nyu40_information_weights() { return get_trivial_information(41); }

std::vector<double> get_trivial_information(int num_classes) {
    return std::vector<double>(num_classes, 1.0);
}

} // namespace pyslam
