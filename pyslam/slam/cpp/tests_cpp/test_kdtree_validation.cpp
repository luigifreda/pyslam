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

#include "ckdtree_eigen.h"
#include "eigen_aliases.h"

#include <Eigen/Core>
#include <algorithm>
#include <cassert>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>

using namespace pyslam;

// Brute force k-NN for ground truth
template <typename Scalar, int D>
std::pair<std::vector<Scalar>, std::vector<size_t>>
brute_force_knn(const Eigen::Matrix<Scalar, Eigen::Dynamic, D, Eigen::RowMajor> &points,
                const Eigen::Matrix<Scalar, D, 1> &query, size_t k) {

    std::vector<std::pair<Scalar, size_t>> distances;
    distances.reserve(points.rows());

    for (int i = 0; i < points.rows(); ++i) {
        Scalar dist_sq = 0;
        for (int j = 0; j < D; ++j) {
            Scalar diff = points(i, j) - query(j);
            dist_sq += diff * diff;
        }
        distances.emplace_back(std::sqrt(dist_sq), i);
    }

    std::sort(distances.begin(), distances.end());

    std::vector<Scalar> dists(k);
    std::vector<size_t> indices(k);

    for (size_t i = 0; i < k && i < distances.size(); ++i) {
        dists[i] = distances[i].first;
        indices[i] = distances[i].second;
    }

    return {dists, indices};
}

// Brute force radius search for ground truth
template <typename Scalar, int D>
std::vector<size_t>
brute_force_radius(const Eigen::Matrix<Scalar, Eigen::Dynamic, D, Eigen::RowMajor> &points,
                   const Eigen::Matrix<Scalar, D, 1> &query, Scalar radius) {

    std::vector<size_t> result;
    Scalar radius_sq = radius * radius;

    for (int i = 0; i < points.rows(); ++i) {
        Scalar dist_sq = 0;
        for (int j = 0; j < D; ++j) {
            Scalar diff = points(i, j) - query(j);
            dist_sq += diff * diff;
        }
        if (dist_sq <= radius_sq) {
            result.push_back(i);
        }
    }

    std::sort(result.begin(), result.end());
    return result;
}

// Test function for fixed dimension KDTree
template <typename Scalar, int D>
bool test_fixed_dimension(const std::string &test_name, size_t n_points, size_t n_queries) {
    std::cout << "\n=== Testing " << test_name << " ===" << std::endl;

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-10.0, 10.0);

    Eigen::Matrix<Scalar, Eigen::Dynamic, D, Eigen::RowMajor> points(n_points, D);
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < D; ++j) {
            points(i, j) = dis(gen);
        }
    }

    // Build KDTree
    CKDTreeEigen<Scalar, D> kdtree(points);

    bool all_tests_passed = true;

    // Test k-NN queries
    for (size_t q = 0; q < n_queries; ++q) {
        Eigen::Matrix<Scalar, D, 1> query;
        for (int j = 0; j < D; ++j) {
            query(j) = dis(gen);
        }

        // Test different k values
        for (size_t k : {size_t(1), size_t(3), size_t(5), std::min(n_points, size_t(10))}) {
            if (k > n_points)
                continue;

            // KDTree result
            auto [kd_dists, kd_indices] = kdtree.query(query, k);

            // Ground truth
            auto [gt_dists, gt_indices] = brute_force_knn(points, query, k);

            // Validate results
            bool test_passed = true;
            if (kd_dists.size() != gt_dists.size()) {
                std::cout << "ERROR: k-NN size mismatch for k=" << k << std::endl;
                test_passed = false;
            }

            for (size_t i = 0; i < std::min(kd_dists.size(), gt_dists.size()); ++i) {
                Scalar dist_diff = std::abs(kd_dists[i] - gt_dists[i]);
                if (dist_diff > 1e-6) {
                    std::cout << "ERROR: k-NN distance mismatch at i=" << i
                              << ", kd=" << kd_dists[i] << ", gt=" << gt_dists[i] << std::endl;
                    test_passed = false;
                }

                if (kd_indices[i] != gt_indices[i]) {
                    std::cout << "ERROR: k-NN index mismatch at i=" << i << ", kd=" << kd_indices[i]
                              << ", gt=" << gt_indices[i] << std::endl;
                    test_passed = false;
                }
            }

            if (!test_passed) {
                all_tests_passed = false;
            }
        }

        // Test radius search
        Scalar radius = Scalar(2.0);
        auto kd_radius_indices = kdtree.query_ball_point(query, radius);
        auto gt_radius_indices = brute_force_radius(points, query, radius);

        if (kd_radius_indices.size() != gt_radius_indices.size()) {
            std::cout << "ERROR: Radius search size mismatch" << std::endl;
            all_tests_passed = false;
        } else {
            for (size_t i = 0; i < kd_radius_indices.size(); ++i) {
                if (kd_radius_indices[i] != gt_radius_indices[i]) {
                    std::cout << "ERROR: Radius search index mismatch at i=" << i << std::endl;
                    all_tests_passed = false;
                }
            }
        }
    }

    if (all_tests_passed) {
        std::cout << "✓ All tests passed for " << test_name << std::endl;
    } else {
        std::cout << "✗ Some tests failed for " << test_name << std::endl;
    }

    return all_tests_passed;
}

// Test function for dynamic dimension KDTree
bool test_dynamic_dimension(const std::string &test_name, size_t n_points, int dim,
                            size_t n_queries) {
    std::cout << "\n=== Testing " << test_name << " ===" << std::endl;

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<double> dis(-10.0, 10.0);

    MatNxM<double> points(n_points, dim);
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < dim; ++j) {
            points(i, j) = dis(gen);
        }
    }

    // Build KDTree
    CKDTreeEigenDyn<double, size_t> kdtree(points);

    bool all_tests_passed = true;

    // Test k-NN queries
    for (size_t q = 0; q < n_queries; ++q) {
        Eigen::VectorXd query(dim);
        for (int j = 0; j < dim; ++j) {
            query(j) = dis(gen);
        }

        // Test different k values
        for (size_t k : {size_t(1), size_t(3), size_t(5), std::min(n_points, size_t(10))}) {
            if (k > n_points)
                continue;

            // KDTree result
            auto [kd_dists, kd_indices] = kdtree.query(query, k);

            // Ground truth (brute force)
            std::vector<std::pair<double, size_t>> distances;
            distances.reserve(n_points);

            for (int i = 0; i < n_points; ++i) {
                double dist_sq = 0;
                for (int j = 0; j < dim; ++j) {
                    double diff = points(i, j) - query(j);
                    dist_sq += diff * diff;
                }
                distances.emplace_back(std::sqrt(dist_sq), i);
            }

            std::sort(distances.begin(), distances.end());

            std::vector<double> gt_dists(k);
            std::vector<size_t> gt_indices(k);

            for (size_t i = 0; i < k && i < distances.size(); ++i) {
                gt_dists[i] = distances[i].first;
                gt_indices[i] = distances[i].second;
            }

            // Validate results
            bool test_passed = true;
            if (kd_dists.size() != gt_dists.size()) {
                std::cout << "ERROR: k-NN size mismatch for k=" << k << std::endl;
                test_passed = false;
            }

            for (size_t i = 0; i < std::min(kd_dists.size(), gt_dists.size()); ++i) {
                double dist_diff = std::abs(kd_dists[i] - gt_dists[i]);
                if (dist_diff > 1e-6) {
                    std::cout << "ERROR: k-NN distance mismatch at i=" << i
                              << ", kd=" << kd_dists[i] << ", gt=" << gt_dists[i] << std::endl;
                    test_passed = false;
                }

                if (kd_indices[i] != gt_indices[i]) {
                    std::cout << "ERROR: k-NN index mismatch at i=" << i << ", kd=" << kd_indices[i]
                              << ", gt=" << gt_indices[i] << std::endl;
                    test_passed = false;
                }
            }

            if (!test_passed) {
                all_tests_passed = false;
            }
        }
    }

    if (all_tests_passed) {
        std::cout << "✓ All tests passed for " << test_name << std::endl;
    } else {
        std::cout << "✗ Some tests failed for " << test_name << std::endl;
    }

    return all_tests_passed;
}

// Performance test
template <typename Scalar, int D>
void performance_test(const std::string &test_name, size_t n_points, size_t n_queries) {
    std::cout << "\n=== Performance Test: " << test_name << " ===" << std::endl;

    // Generate random data
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<Scalar> dis(-10.0, 10.0);

    Eigen::Matrix<Scalar, Eigen::Dynamic, D, Eigen::RowMajor> points(n_points, D);
    for (int i = 0; i < n_points; ++i) {
        for (int j = 0; j < D; ++j) {
            points(i, j) = dis(gen);
        }
    }

    // Build KDTree
    auto start = std::chrono::high_resolution_clock::now();
    CKDTreeEigen<Scalar, D> kdtree(points);
    auto end = std::chrono::high_resolution_clock::now();
    auto build_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "Build time: " << build_time.count() << " μs" << std::endl;

    // Generate queries
    std::vector<Eigen::Matrix<Scalar, D, 1>> queries(n_queries);
    for (size_t q = 0; q < n_queries; ++q) {
        for (int j = 0; j < D; ++j) {
            queries[q](j) = dis(gen);
        }
    }

    // KDTree queries
    start = std::chrono::high_resolution_clock::now();
    for (const auto &query : queries) {
        auto [dists, indices] = kdtree.query(query, 5);
    }
    end = std::chrono::high_resolution_clock::now();
    auto kdtree_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    // Brute force queries
    start = std::chrono::high_resolution_clock::now();
    for (const auto &query : queries) {
        auto [dists, indices] = brute_force_knn(points, query, 5);
    }
    end = std::chrono::high_resolution_clock::now();
    auto brute_time = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    std::cout << "KDTree query time: " << kdtree_time.count() << " μs" << std::endl;
    std::cout << "Brute force time: " << brute_time.count() << " μs" << std::endl;
    std::cout << "Speedup: " << double(brute_time.count()) / kdtree_time.count() << "x"
              << std::endl;
}

int main(int argc, char **argv) {
    std::cout << "=== KDTree Comprehensive Test Suite ===" << std::endl;

    bool all_passed = true;

    // Test fixed dimensions
    all_passed &= test_fixed_dimension<double, 2>("2D Double", 100, 20);
    all_passed &= test_fixed_dimension<float, 3>("3D Float", 200, 30);
    all_passed &= test_fixed_dimension<double, 4>("4D Double", 150, 25);

    // Test dynamic dimensions
    all_passed &= test_dynamic_dimension("Dynamic 5D", 100, 5, 20);
    all_passed &= test_dynamic_dimension("Dynamic 10D", 200, 10, 30);

    // Performance tests
    performance_test<double, 2>("2D Performance", 1000, 100);
    performance_test<float, 3>("3D Performance", 2000, 200);

    // Edge cases
    std::cout << "\n=== Edge Cases ===" << std::endl;

    // Single point
    MatNx2<double> single_point(1, 2);
    single_point << 0, 0;
    CKDTreeEigen<double, 2> single_kd(single_point);
    auto [d, i] = single_kd.query(Eigen::Vector2d(1, 1), 1);
    std::cout << "Single point test: distance=" << d[0] << ", index=" << i[0] << std::endl;

    // Empty query (should handle gracefully)
    MatNx2<double> empty_points(0, 2);
    // Note: This might cause issues depending on implementation

    std::cout << "\n=== Test Summary ===" << std::endl;
    if (all_passed) {
        std::cout << "✓ ALL TESTS PASSED!" << std::endl;
    } else {
        std::cout << "✗ SOME TESTS FAILED!" << std::endl;
    }

    return all_passed ? 0 : 1;
}