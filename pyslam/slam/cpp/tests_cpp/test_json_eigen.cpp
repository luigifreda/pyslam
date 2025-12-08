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
#include <iostream>
#include <nlohmann/json.hpp>

#include <Eigen/Dense>

#include "utils/serialization_json.h"
#include "utils/test_utils.h"

using pyslam::eigen_matrix_to_json_array;
using pyslam::flexible_json_to_eigen_matrix;
using namespace pyslam::test_utils;

static void test_eigen_matrix_to_json_and_back_numeric() {
    Eigen::Matrix4d M;
    M << 1.0, 0.0, 0.0, 0.1, 0.0, 1.0, 0.0, 0.2, 0.0, 0.0, 1.0, 0.3, 0.0, 0.0, 0.0, 1.0;

    nlohmann::json j = eigen_matrix_to_json_array(M);

    Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
    bool ok = flexible_json_to_eigen_matrix<double, 4, 4>(j, R);
    expect_true(ok, "numeric array parse failed");
    expect_true((R - M).norm() < 1e-12, "numeric roundtrip mismatch");
}

static void test_eigen_matrix_to_json_and_back_string() {
    Eigen::Matrix4d M;
    M << 0.999, 0.001, 0.0, 10.0, -0.001, 0.999, 0.0, 20.0, 0.0, 0.0, 1.0, 30.0, 0.0, 0.0, 0.0, 1.0;

    // Simulate Python double-encoded case: store json of the nested arrays into a string
    nlohmann::json arr = eigen_matrix_to_json_array(M);
    nlohmann::json wrapper;
    wrapper["pose"] = arr.dump();

    Eigen::Matrix4d R = Eigen::Matrix4d::Zero();
    bool ok = pyslam::flexible_json_to_eigen_matrix<double, 4, 4>(wrapper["pose"], R);
    expect_true(ok, "string array parse failed");
    expect_true((R - M).norm() < 1e-12, "string roundtrip mismatch");
}

static void test_bad_shapes_and_types() {
    // Wrong shape: 3x3
    nlohmann::json bad =
        nlohmann::json::array({nlohmann::json::array({1, 0, 0}), nlohmann::json::array({0, 1, 0}),
                               nlohmann::json::array({0, 0, 1})});
    Eigen::Matrix4d R;
    bool ok = flexible_json_to_eigen_matrix<double, 4, 4>(bad, R);
    expect_true(!ok, "expected failure on 3x3 input");

    // Corrupted string
    nlohmann::json s = "not-a-json";
    ok = flexible_json_to_eigen_matrix<double, 4, 4>(s, R);
    expect_true(!ok, "expected failure on invalid JSON string");
}

int main() {
    try {
        test_eigen_matrix_to_json_and_back_numeric();
        test_eigen_matrix_to_json_and_back_string();
        test_bad_shapes_and_types();
        std::cout << "All Eigen JSON serialization tests passed." << std::endl;
        return 0;
    } catch (const std::exception &e) {
        std::cerr << "Test failed: " << e.what() << std::endl;
        return 1;
    }
}
