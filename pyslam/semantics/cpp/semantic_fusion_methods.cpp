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

#include "semantic_fusion_methods.h"
#include "utils/messages.h"

#include <algorithm>
#include <map>

namespace pyslam {

int count_labels_fusion(const std::vector<int> &labels) {
    if (labels.empty()) {
        return -1; // Invalid label for empty input
    }

    // Count occurrences of each label
    std::map<int, int> label_counts;
    for (const int &label : labels) {
        label_counts[label]++;
    }

    // Find the label with maximum count
    auto max_element =
        std::max_element(label_counts.begin(), label_counts.end(),
                         [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                             return a.second < b.second;
                         });

    return max_element->first;
}

int count_labels_fusion(VecNdRef labels) {
    if (labels.size() == 0) {
        return -1; // Invalid label for empty input
    }

    // Count occurrences of each label
    std::map<int, int> label_counts;
    for (int i = 0; i < labels.size(); ++i) {
        const int label = static_cast<int>(std::round(labels(i)));
        label_counts[label]++;
    }

    // Find the label with maximum count
    auto max_element =
        std::max_element(label_counts.begin(), label_counts.end(),
                         [](const std::pair<int, int> &a, const std::pair<int, int> &b) {
                             return a.second < b.second;
                         });

    return max_element->first;
}

cv::Mat count_labels_fusion(const cv::Mat &labels) {
    if (labels.rows == 0) {
        return cv::Mat(); // Return empty matrix for empty input
    }

    MSG_ASSERT(labels.channels() == 1, "Labels must be a single channel");
    MSG_ASSERT(labels.type() == CV_32S, "Labels must be of type CV_32S");
    MSG_ASSERT(labels.cols == 1 || labels.rows == 1, "Labels must be a single column or row");

    // Count occurrences of each label
    std::map<int, int> label_counts;
    if (labels.cols == 1) {
        for (int i = 0; i < labels.rows; ++i) {
            const int label = static_cast<int>(std::round(labels.at<int>(i, 0)));
            label_counts[label]++;
        }
    } else { // 1 x N
        for (int j = 0; j < labels.cols; ++j) {
            const int label = static_cast<int>(std::round(labels.at<int>(0, j)));
            label_counts[label]++;
        }
    }

    // Find the label with maximum count
    auto it = std::max_element(label_counts.begin(), label_counts.end(),
                               [](const auto &a, const auto &b) { return a.second < b.second; });

    // TODO: can we do this more efficiently?
    // return a matrix with the most frequent label (1x1)
    cv::Mat result(1, 1, CV_32S);
    result.at<int>(0, 0) = it->first;
    return result;
}

// ================================
// Bayesian fusion
// ================================

VecNd bayesian_fusion(const std::vector<VecNd> &probs) {
    /**
     * Bayesian fusion of probability vectors.
     * https://en.wikipedia.org/wiki/Bayesian_inference#Bayesian_inference_for_parameter_estimation
     * Uses the following formula:
     * P(θ|D) = P(D|θ) * P(θ) / P(D)
     * where:
    - P(θ|D) is the posterior probability of the parameter θ given the data D
    - P(D|θ) is the likelihood of the data D given the parameter θ
    - P(θ) is the prior probability of the parameter θ
    - P(D) is the marginal likelihood of the data D
    */
    if (probs.empty()) {
        return VecNd(); // Return empty vector for empty input
    }

    const int num_classes = probs[0].size();

    // Initialize with uniform prior
    const VecNd prior = VecNd::Ones(num_classes) / static_cast<double>(num_classes);
    VecNd posterior = std::move(prior);

    // Iteratively update posterior with each observation
    for (const auto &obs : probs) {
        if (obs.size() != num_classes) {
            throw std::invalid_argument("All probability vectors must have the same size");
        }

        // Element-wise multiplication
        //
        posterior = posterior.cwiseProduct(obs);

        // Normalize to ensure probabilities sum to 1
        double sum = posterior.sum();
        if (sum > 0.0) {
            posterior /= sum;
        } else {
            MSG_ERROR("bayesian_fusion: Invalid posterior sum");
        }
    }

    return posterior;
}

VecNd bayesian_fusion(MatNxMdRef probs) {
    /**
     * Bayesian fusion of probability vectors.
     * https://en.wikipedia.org/wiki/Bayesian_inference#Bayesian_inference_for_parameter_estimation
     * Uses the following formula:
     * P(θ|D) = P(D|θ) * P(θ) / P(D)
     * where:
    - P(θ|D) is the posterior probability of the parameter θ given the data D
    - P(D|θ) is the likelihood of the data D given the parameter θ
    - P(θ) is the prior probability of the parameter θ
    - P(D) is the marginal likelihood of the data D
    */
    if (probs.rows() == 0) {
        return VecNd(); // Return empty vector for empty input
    }

    const int C = probs.cols();
    // Initialize with uniform prior
    VecNd posterior = VecNd::Ones(C) / static_cast<double>(C);

    for (int i = 0; i < probs.rows(); ++i) {
        VecNd obs = probs.row(i).transpose(); // transpose!
        posterior = posterior.cwiseProduct(obs);
        double s = posterior.sum();
        if (s > 0.0) {
            posterior /= s;
        } else {
            MSG_ERROR("bayesian_fusion: Invalid posterior sum");
        }
    }
    return posterior;
}

cv::Mat bayesian_fusion(const cv::Mat &probs) {
    /**
    * Bayesian fusion of probability vectors.
    * https://en.wikipedia.org/wiki/Bayesian_inference#Bayesian_inference_for_parameter_estimation
    * Uses the following formula:
    * P(θ|D) = P(D|θ) * P(θ) / P(D)
    * where:
    - P(θ|D) is the posterior probability of the parameter θ given the data D
    - P(D|θ) is the likelihood of the data D given the parameter θ
    - P(θ) is the prior probability of the parameter θ
    - P(D) is the marginal likelihood of the data D
    */
    if (probs.empty())
        return cv::Mat();

    MSG_ASSERT(probs.channels() == 1, "Probabilities must be a single channel");
    MSG_ASSERT(probs.type() == CV_32F, "Probabilities must be of type CV_32F");

    const int rows = probs.rows, C = probs.cols;

    // 1 x C prior/posterior
    cv::Mat posterior = cv::Mat::ones(1, C, CV_32F) / static_cast<float>(C);

    for (int i = 0; i < rows; ++i) {
        cv::Mat obs = probs.row(i); // 1 x C
        MSG_ASSERT(obs.size() == posterior.size(), "Invalid observation size");

        posterior = posterior.mul(obs); // 1 x C

        double s = cv::sum(posterior)[0];
        if (s > 0.0) {
            posterior /= s;
        } else {
            MSG_ERROR("bayesian_fusion: Invalid posterior sum");
        }
    }
    return posterior;
}

// ================================
// Average fusion
// ================================

VecNd average_fusion(const std::vector<VecNd> &features) {
    if (features.empty()) {
        return VecNd(); // Return empty vector for empty input
    }

    const int feature_dim = features[0].size();
    VecNd result = VecNd::Zero(feature_dim);

    // Sum all features
    for (const auto &feature : features) {
        if (feature.size() != feature_dim) {
            MSG_ERROR("average_fusion: All feature vectors must have the same size");
            return VecNd();
        }
        result += feature;
    }

    // Divide by number of features to get average
    result /= static_cast<double>(features.size());

    return result;
}

VecNd average_fusion(MatNxMdRef features) {
    if (features.rows() == 0)
        return VecNd();

    const int feature_dim = features.cols();
    VecNd result = VecNd::Zero(feature_dim);
    for (int i = 0; i < features.rows(); ++i) {
        result += features.row(i).transpose(); // transpose!
    }
    result /= static_cast<double>(features.rows());
    return result;
}

cv::Mat average_fusion(const cv::Mat &features) {
    if (features.rows == 0) {
        return cv::Mat(); // Return empty matrix for empty input
    }

    const int feature_dim = features.cols;
    cv::Mat result = cv::Mat::zeros(1, feature_dim, CV_32F);

    // Sum all features (rows)
    for (int i = 0; i < features.rows; ++i) {
        result += features.row(i);
    }

    // Divide by number of features to get average
    result /= static_cast<double>(features.rows);

    return result;
}

} // namespace pyslam