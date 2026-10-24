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

#include <Eigen/Core>

#include "eigen_aliases.h"

#include <algorithm>
#include <limits>
#include <vector>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <random>
#include <stdexcept>

namespace pyslam {

// Function to combine two images horizontally
cv::Mat combine_images_horizontally(const cv::Mat &img1, const cv::Mat &img2) {
    cv::Mat img1_rgb, img2_rgb;

    // Convert to RGB if needed
    if (img1.channels() == 1) {
        cv::cvtColor(img1, img1_rgb, cv::COLOR_GRAY2RGB);
    } else {
        img1_rgb = img1.clone();
    }

    if (img2.channels() == 1) {
        cv::cvtColor(img2, img2_rgb, cv::COLOR_GRAY2RGB);
    } else {
        img2_rgb = img2.clone();
    }

    int h1 = img1_rgb.rows;
    int w1 = img1_rgb.cols;
    int h2 = img2_rgb.rows;
    int w2 = img2_rgb.cols;

    // Create combined image
    cv::Mat combined(h1 > h2 ? h1 : h2, w1 + w2, CV_8UC3, cv::Scalar(0, 0, 0));

    // Copy images side by side
    img1_rgb.copyTo(combined(cv::Rect(0, 0, w1, h1)));
    img2_rgb.copyTo(combined(cv::Rect(w1, 0, w2, h2)));

    return combined;
}

// Function to combine two images vertically
cv::Mat combine_images_vertically(const cv::Mat &img1, const cv::Mat &img2) {
    cv::Mat img1_rgb, img2_rgb;

    // Convert to RGB if needed
    if (img1.channels() == 1) {
        cv::cvtColor(img1, img1_rgb, cv::COLOR_GRAY2RGB);
    } else {
        img1_rgb = img1.clone();
    }

    if (img2.channels() == 1) {
        cv::cvtColor(img2, img2_rgb, cv::COLOR_GRAY2RGB);
    } else {
        img2_rgb = img2.clone();
    }

    int h1 = img1_rgb.rows;
    int w1 = img1_rgb.cols;
    int h2 = img2_rgb.rows;
    int w2 = img2_rgb.cols;

    // Create combined image
    cv::Mat combined(h1 + h2, w1 > w2 ? w1 : w2, CV_8UC3, cv::Scalar(0, 0, 0));

    // Copy images vertically
    img1_rgb.copyTo(combined(cv::Rect(0, 0, w1, h1)));
    img2_rgb.copyTo(combined(cv::Rect(0, h1, w2, h2)));

    return combined;
}

// Draw a list of points with different random colors on an input image
cv::Mat draw_points(const cv::Mat &img, const MatNx2f &pts, int radius = 5) {
    cv::Mat result;
    if (img.channels() < 3) {
        cv::cvtColor(img, result, cv::COLOR_GRAY2BGR);
    } else {
        result = img.clone();
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    int num_points = pts.rows();
    for (int i = 0; i < num_points; ++i) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen));
        cv::Point2i pt(static_cast<int>(std::round(pts(i, 0))),
                       static_cast<int>(std::round(pts(i, 1))));
        cv::circle(result, pt, radius, color, -1);
    }
    return result;
}

// Draw corresponding points with the same random color on two separate images
std::pair<cv::Mat, cv::Mat> draw_points2(const cv::Mat &img1, const cv::Mat &img2,
                                         const MatNx2f &pts1, const MatNx2f &pts2, int radius = 5) {
    cv::Mat result1, result2;

    if (img1.channels() < 3) {
        cv::cvtColor(img1, result1, cv::COLOR_GRAY2BGR);
    } else {
        result1 = img1.clone();
    }

    if (img2.channels() < 3) {
        cv::cvtColor(img2, result2, cv::COLOR_GRAY2BGR);
    } else {
        result2 = img2.clone();
    }

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    int num_points = pts1.rows();
    for (int i = 0; i < num_points; ++i) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen));

        cv::Point2i pt1(static_cast<int>(std::round(pts1(i, 0))),
                        static_cast<int>(std::round(pts1(i, 1))));
        cv::Point2i pt2(static_cast<int>(std::round(pts2(i, 0))),
                        static_cast<int>(std::round(pts2(i, 1))));

        cv::circle(result1, pt1, radius, color, -1);
        cv::circle(result2, pt2, radius, color, -1);
    }

    return std::make_pair(result1, result2);
}

// Draw lines on an image; line_edges is assumed to be a list of 2D image points
cv::Mat draw_lines(const cv::Mat &img,
                   const std::vector<std::pair<cv::Point2f, cv::Point2f>> &line_edges,
                   const MatNx2f &pts = MatNx2f(), int radius = 5) {
    cv::Mat result = img.clone();

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    int num_lines = line_edges.size();
    for (int i = 0; i < num_lines; ++i) {
        cv::Scalar color(dis(gen), dis(gen), dis(gen));

        cv::Point2i pt1(static_cast<int>(std::round(line_edges[i].first.x)),
                        static_cast<int>(std::round(line_edges[i].first.y)));
        cv::Point2i pt2(static_cast<int>(std::round(line_edges[i].second.x)),
                        static_cast<int>(std::round(line_edges[i].second.y)));

        cv::line(result, pt1, pt2, color, 1);

        // Draw point if provided
        if (pts.rows() > 0 && i < pts.rows()) {
            cv::Point2i pt(static_cast<int>(std::round(pts(i, 0))),
                           static_cast<int>(std::round(pts(i, 1))));
            cv::circle(result, pt, radius, color, -1);
        }
    }
    return result;
}

// Draw features matches (images are combined horizontally)
cv::Mat draw_feature_matches_horizontally(const cv::Mat &img1, const cv::Mat &img2,
                                          const MatNx2f &kps1, const MatNx2f &kps2,
                                          const std::vector<float> &kps1_sizes = {},
                                          const std::vector<float> &kps2_sizes = {},
                                          int lineType = cv::LINE_AA, bool show_kp_sizes = true) {
    cv::Mat img3 = combine_images_horizontally(img1, img2);
    int h1 = img1.rows;
    int w1 = img1.cols;
    int N = kps1.rows();

    float default_size = 3.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (int i = 0; i < N; ++i) {
        cv::Point2i p1(static_cast<int>(std::round(kps1(i, 0))),
                       static_cast<int>(std::round(kps1(i, 1))));
        cv::Point2i p2(static_cast<int>(std::round(kps2(i, 0))),
                       static_cast<int>(std::round(kps2(i, 1))));

        float size1 = (!kps1_sizes.empty() && i < kps1_sizes.size()) ? kps1_sizes[i] : default_size;
        float size2 = (!kps2_sizes.empty() && i < kps2_sizes.size()) ? kps2_sizes[i] : default_size;

        cv::Scalar color(dis(gen), dis(gen), dis(gen));

        // Draw line connecting corresponding points
        cv::line(img3, p1, cv::Point2i(p2.x + w1, p2.y), color, 1);

        // Draw circles at keypoint locations
        cv::circle(img3, p1, 2, color, -1, lineType);
        cv::circle(img3, cv::Point2i(p2.x + w1, p2.y), 2, color, -1, lineType);

        // Draw keypoint size circles if requested
        if (show_kp_sizes) {
            cv::circle(img3, p1, static_cast<int>(size1), cv::Scalar(0, 255, 0), 1, lineType);
            cv::circle(img3, cv::Point2i(p2.x + w1, p2.y), static_cast<int>(size2),
                       cv::Scalar(0, 255, 0), 1, lineType);
        }
    }
    return img3;
}

// Draw features matches (images are combined vertically)
cv::Mat draw_feature_matches_vertically(const cv::Mat &img1, const cv::Mat &img2,
                                        const MatNx2f &kps1, const MatNx2f &kps2,
                                        const std::vector<float> &kps1_sizes = {},
                                        const std::vector<float> &kps2_sizes = {},
                                        int lineType = cv::LINE_AA, bool show_kp_sizes = true) {
    cv::Mat img3 = combine_images_vertically(img1, img2);
    int h1 = img1.rows;
    int w1 = img1.cols;
    int N = kps1.rows();

    float default_size = 2.0f;

    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<> dis(0, 255);

    for (int i = 0; i < N; ++i) {
        cv::Point2i p1(static_cast<int>(std::round(kps1(i, 0))),
                       static_cast<int>(std::round(kps1(i, 1))));
        cv::Point2i p2(static_cast<int>(std::round(kps2(i, 0))),
                       static_cast<int>(std::round(kps2(i, 1))));

        float size1 = (!kps1_sizes.empty() && i < kps1_sizes.size()) ? kps1_sizes[i] : default_size;
        float size2 = (!kps2_sizes.empty() && i < kps2_sizes.size()) ? kps2_sizes[i] : default_size;

        cv::Scalar color(dis(gen), dis(gen), dis(gen));

        // Draw line connecting corresponding points
        cv::line(img3, p1, cv::Point2i(p2.x, p2.y + h1), color, 1);

        // Draw circles at keypoint locations
        cv::circle(img3, p1, 2, color, -1, lineType);
        cv::circle(img3, cv::Point2i(p2.x, p2.y + h1), 2, color, -1, lineType);

        // Draw keypoint size circles if requested
        if (show_kp_sizes) {
            cv::circle(img3, p1, static_cast<int>(size1), cv::Scalar(0, 255, 0), 1, lineType);
            cv::circle(img3, cv::Point2i(p2.x, p2.y + h1), static_cast<int>(size2),
                       cv::Scalar(0, 255, 0), 1, lineType);
        }
    }
    return img3;
}

// Main function to draw features matches (images are combined horizontally or vertically)
cv::Mat draw_feature_matches(const cv::Mat &img1, const cv::Mat &img2, const MatNx2f &kps1,
                             const MatNx2f &kps2, const std::vector<float> &kps1_sizes = {},
                             const std::vector<float> &kps2_sizes = {}, bool horizontal = true,
                             int lineType = cv::LINE_AA, bool show_kp_sizes = true) {
    if (horizontal) {
        return draw_feature_matches_horizontally(img1, img2, kps1, kps2, kps1_sizes, kps2_sizes,
                                                 lineType, show_kp_sizes);
    } else {
        return draw_feature_matches_vertically(img1, img2, kps1, kps2, kps1_sizes, kps2_sizes,
                                               lineType, show_kp_sizes);
    }
}

// Overloaded version for backward compatibility with existing code
cv::Mat draw_feature_matches(const cv::Mat &img_left, const cv::Mat &img_right,
                             const MatNx2f &kps_left, const MatNx2f &kps_right, bool horizontal,
                             bool show_keypoint_sizes) {
    return draw_feature_matches(img_left, img_right, kps_left, kps_right, {}, {}, horizontal,
                                cv::LINE_AA, show_keypoint_sizes);
}

} // namespace pyslam