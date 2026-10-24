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

#include "eigen_aliases.h"
#include "messages.h"

#include <Eigen/Dense>
#include <array>
#include <opencv2/opencv.hpp>
#include <type_traits>
#include <vector>

namespace pyslam {
namespace utils {

// generic extractor for 2D point types
template <typename Elem, typename ReturnScalar>
inline std::pair<ReturnScalar, ReturnScalar> get_xy(const Elem &p) {
    if constexpr (std::is_same_v<Elem, cv::Point2f> || std::is_same_v<Elem, cv::Point2d>) {
        return {static_cast<ReturnScalar>(p.x), static_cast<ReturnScalar>(p.y)};
    } else if constexpr (std::is_same_v<Elem, Eigen::Vector2d>) {
        return {static_cast<ReturnScalar>(p.x()), static_cast<ReturnScalar>(p.y())};
    } else if constexpr (std::is_same_v<Elem, Eigen::Vector2f>) {
        return {static_cast<ReturnScalar>(p.x()), static_cast<ReturnScalar>(p.y())};
    } else if constexpr (std::is_same_v<Elem, Eigen::Matrix<double, 2, 1>>) {
        return {static_cast<ReturnScalar>(p(0)), static_cast<ReturnScalar>(p(1))};
    } else if constexpr (std::is_same_v<Elem, Eigen::Matrix<float, 2, 1>>) {
        return {static_cast<ReturnScalar>(p(0)), static_cast<ReturnScalar>(p(1))};
    } else if constexpr (std::is_same_v<Elem, std::array<double, 2>> ||
                         std::is_same_v<Elem, std::array<float, 2>>) {
        return {static_cast<ReturnScalar>(p[0]), static_cast<ReturnScalar>(p[1])};
    } else if constexpr (std::is_base_of_v<Eigen::MatrixBase<Elem>, Elem> &&
                         Elem::ColsAtCompileTime == 2) {
        // Handle Eigen::Block and other Eigen matrix expressions with 2 columns
        return {static_cast<ReturnScalar>(p(0)), static_cast<ReturnScalar>(p(1))};
    } else if constexpr (std::is_base_of_v<Eigen::Block<Elem>, Elem> &&
                         Elem::ColsAtCompileTime == 2) {
        // Handle Eigen::Block specifically
        return {static_cast<ReturnScalar>(p(0)), static_cast<ReturnScalar>(p(1))};
    } else {
        static_assert(sizeof(Elem) == 0, "get_xy: unsupported 2D point element type");
    }
}

// generic extractor for 2D point types at an index
template <typename Container, typename ReturnScalar>
inline std::pair<ReturnScalar, ReturnScalar> get_xy_at(const Container &container, size_t index) {
    // First, let's try to handle Eigen matrices generically
    if constexpr (std::is_base_of_v<Eigen::MatrixBase<Container>, Container>) {
        if (index >= static_cast<size_t>(container.rows())) {
            return {static_cast<ReturnScalar>(0), static_cast<ReturnScalar>(0)};
        }
        const auto &p = container.row(index);
        return {static_cast<ReturnScalar>(p(0)), static_cast<ReturnScalar>(p(1))};
    }
    // Then handle std::vector types
    else if constexpr (std::is_same_v<Container, std::vector<cv::Point2i>>) {
        if (index >= container.size()) {
            return {static_cast<ReturnScalar>(0), static_cast<ReturnScalar>(0)};
        }
        const auto &p = container[index];
        return {static_cast<ReturnScalar>(p.x), static_cast<ReturnScalar>(p.y)};
    } else if constexpr (std::is_same_v<Container, std::vector<cv::Point2f>>) {
        if (index >= container.size()) {
            return {static_cast<ReturnScalar>(0), static_cast<ReturnScalar>(0)};
        }
        const auto &p = container[index];
        return {static_cast<ReturnScalar>(p.x), static_cast<ReturnScalar>(p.y)};
    } else if constexpr (std::is_same_v<Container, std::vector<cv::Point2d>>) {
        if (index >= container.size()) {
            return {static_cast<ReturnScalar>(0), static_cast<ReturnScalar>(0)};
        }
        const auto &p = container[index];
        return {static_cast<ReturnScalar>(p.x), static_cast<ReturnScalar>(p.y)};
    } else if constexpr (std::is_same_v<Container, std::vector<Eigen::Vector2f>>) {
        if (index >= container.size()) {
            return {static_cast<ReturnScalar>(0), static_cast<ReturnScalar>(0)};
        }
        const auto &p = container[index];
        return {static_cast<ReturnScalar>(p.x()), static_cast<ReturnScalar>(p.y())};
    } else if constexpr (std::is_same_v<Container, std::vector<Eigen::Vector2d>>) {
        if (index >= container.size()) {
            return {static_cast<ReturnScalar>(0), static_cast<ReturnScalar>(0)};
        }
        const auto &p = container[index];
        return {static_cast<ReturnScalar>(p.x()), static_cast<ReturnScalar>(p.y())};
    } else if constexpr (std::is_same_v<Container, std::vector<std::array<float, 2>>>) {
        if (index >= container.size()) {
            return {static_cast<ReturnScalar>(0), static_cast<ReturnScalar>(0)};
        }
        const auto &p = container[index];
        return {static_cast<ReturnScalar>(p[0]), static_cast<ReturnScalar>(p[1])};
    } else if constexpr (std::is_same_v<Container, std::vector<std::array<double, 2>>>) {
        if (index >= container.size()) {
            return {static_cast<ReturnScalar>(0), static_cast<ReturnScalar>(0)};
        }
        const auto &p = container[index];
        return {static_cast<ReturnScalar>(p[0]), static_cast<ReturnScalar>(p[1])};
    } else {
        static_assert(sizeof(Container) == 0,
                      "get_xy_at: unsupported aggregated 2D coordinate container type");
    }
}

} // namespace utils
} // namespace pyslam