

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

#include "optimizer_g2o.h"

#include "camera.h"
#include "camera_pose.h"
#include "config_parameters.h"
#include "feature_shared_resources.h"
#include "frame.h"
#include "keyframe.h"
#include "map.h"
#include "map_point.h"
#include "utils/eigen_helpers.h"
#include "utils/messages.h"

#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/core/sparse_optimizer.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>
#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include <algorithm>
#include <cmath>
#include <iostream>
#include <mutex>

#ifdef USE_PYTHON
#include "py_module/py_wrappers.h"
#endif

namespace g2o {
using BlockSolverSE2 = BlockSolver_3_2;
using BlockSolverSE3 = BlockSolver_6_3;
using BlockSolverSim3 = BlockSolver_7_3;
} // namespace g2o

namespace pyslam {

// TODO: Add support for semantics in optimization

BundleAdjustmentResult OptimizerG2o::bundle_adjustment(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    std::optional<int> local_window_size, bool fixed_points, int rounds, int loop_kf_id,
    bool use_robust_kernel, bool *abort_flag, bool fill_result_dict, bool verbose) {

    BundleAdjustmentResult result;

    // Determine local frames
    std::vector<KeyFramePtr> local_frames;
    if (local_window_size.has_value()) {
        const int window_size = local_window_size.value();
        const int start_idx = std::max(0, static_cast<int>(keyframes.size()) - window_size);
        local_frames.assign(keyframes.begin() + start_idx, keyframes.end());
    } else {
        local_frames = keyframes;
    }

    int robust_rounds = use_robust_kernel ? rounds / 2 : 0;
    int final_rounds = rounds - robust_rounds;

    if (verbose) {
        std::cout << "bundle_adjustment: rounds: " << rounds << ", robust_rounds: " << robust_rounds
                  << ", final_rounds: " << final_rounds << std::endl;
    }

    // Create g2o optimizer
    g2o::SparseOptimizer optimizer;
    auto linearSolver =
        std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverSE3>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer.setAlgorithm(algorithm);

    if (abort_flag) {
        optimizer.setForceStopFlag(abort_flag);
    }

    // Maps to store graph elements
    std::unordered_map<KeyFramePtr, g2o::VertexSE3Expmap *> graph_keyframes;
    std::unordered_map<MapPointPtr, g2o::VertexSBAPointXYZ *> graph_points;
    std::unordered_map<g2o::EdgeSE3ProjectXYZ *, bool> graph_edges_mono;
    std::unordered_map<g2o::EdgeStereoSE3ProjectXYZ *, bool> graph_edges_stereo;

    int num_edges = 0;
    int num_bad_edges = 0;

    // Add keyframe vertices to graph
    // If points are fixed then consider just the local frames, otherwise we need all frames or at
    // least two frames for each point
    std::vector<KeyFramePtr> frames_to_optimize = fixed_points ? local_frames : keyframes;
    for (auto &kf : frames_to_optimize) {
        if (!kf || kf->is_bad()) {
            continue;
        }

        // Create SE3 vertex
        auto *vertex_se3 = new g2o::VertexSE3Expmap();
        g2o::SE3Quat se3(kf->Rcw(), kf->tcw());

        vertex_se3->setEstimate(se3);
        vertex_se3->setId(kf->kid * 2); // even ids (use f.kid here!)
        const bool is_kf_in_local_frames =
            std::find(local_frames.begin(), local_frames.end(), kf) != local_frames.end();
        vertex_se3->setFixed(kf->kid == 0 || !is_kf_in_local_frames); // (use f.kid here!)

        optimizer.addVertex(vertex_se3);
        graph_keyframes[kf] = vertex_se3;
    }

    const auto &inv_level_sigmas2 = FeatureSharedResources::inv_level_sigmas2;

    // Add point vertices to graph
    for (auto &p : points) {
        if (!p || p->is_bad()) {
            continue;
        }

        auto *vertex_point = new g2o::VertexSBAPointXYZ();
        vertex_point->setId(p->id * 2 + 1); // odd ids
        vertex_point->setEstimate(p->pt());
        vertex_point->setMarginalized(true);
        vertex_point->setFixed(fixed_points);

        optimizer.addVertex(vertex_point);
        graph_points[p] = vertex_point;

        // Add edges
        auto observations = p->observations();
        for (const auto &obs : observations) {
            KeyFramePtr kf = obs.first;
            const int idx = obs.second;

            const auto &it = graph_keyframes.find(kf);
            if (it == graph_keyframes.end()) {
                continue;
            }
            const auto v_se3_kf = it->second;

            // Get keypoint information
            const Eigen::Vector2d kpu = Eigen::Vector2d(kf->kpsu(idx, 0), kf->kpsu(idx, 1));

            const auto &kps_ur = kf->kps_ur;
            const bool is_stereo_obs = (kps_ur.size() > idx && kps_ur[idx] >= 0);

            // Get inverse sigma squared
            const auto &octaves = kf->octaves;
            double invSigma2 = inv_level_sigmas2[octaves[idx]];

            const auto &camera = kf->camera;

            if (is_stereo_obs) {
                auto *edge = new g2o::EdgeStereoSE3ProjectXYZ();
                edge->setVertex(0, vertex_point);
                edge->setVertex(1, v_se3_kf);

                Eigen::Vector3d obs_vec(kpu.x(), kpu.y(), kps_ur[idx]);
                edge->setMeasurement(obs_vec);
                edge->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                if (use_robust_kernel) {
                    auto *robust_kernel = new g2o::RobustKernelHuber();
                    robust_kernel->setDelta(Parameters::kThHuberStereo);
                    edge->setRobustKernel(robust_kernel);
                }

                edge->fx = camera->fx;
                edge->fy = camera->fy;
                edge->cx = camera->cx;
                edge->cy = camera->cy;
                edge->bf = camera->bf;

                optimizer.addEdge(edge);
                graph_edges_stereo[edge] = true;
            } else {
                auto *edge = new g2o::EdgeSE3ProjectXYZ();
                edge->setVertex(0, vertex_point);
                edge->setVertex(1, v_se3_kf);
                edge->setMeasurement(kpu);
                edge->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                if (use_robust_kernel) {
                    auto *robust_kernel = new g2o::RobustKernelHuber();
                    robust_kernel->setDelta(Parameters::kThHuberMono);
                    edge->setRobustKernel(robust_kernel);
                }

                edge->fx = camera->fx;
                edge->fy = camera->fy;
                edge->cx = camera->cx;
                edge->cy = camera->cy;

                optimizer.addEdge(edge);
                graph_edges_mono[edge] = true;
            }
            num_edges++;
        }
    }

    if (abort_flag && *abort_flag) {
        result.mean_squared_error = -1.0;
        if (fill_result_dict) {
            result.keyframe_updates = {};
            result.point_updates = {};
        }
        if (verbose) {
            MSG_WARN("bundle_adjustment: aborting due to abort_flag");
        }
        return result;
    }

    if (verbose) {
        optimizer.setVerbose(true);
    }

    // Initialize optimization
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    double initial_mean_squared_error = optimizer.activeChi2() / std::max(num_edges, 1);

    // Robust optimization rounds
    if (robust_rounds > 0) {
        optimizer.optimize(robust_rounds);

        // Check inliers and remove outliers
        for (auto &edge_pair : graph_edges_mono) {
            auto *edge = edge_pair.first;
            double edge_chi2 = edge->chi2();
            if (edge_chi2 > Parameters::kChi2Mono || !edge->isDepthPositive()) {
                edge->setLevel(1);
                num_bad_edges++;
            }
            edge->setRobustKernel(nullptr);
        }

        for (auto &edge_pair : graph_edges_stereo) {
            auto *edge = edge_pair.first;
            double edge_chi2 = edge->chi2();
            if (edge_chi2 > Parameters::kChi2Stereo || !edge->isDepthPositive()) {
                edge->setLevel(1);
                num_bad_edges++;
            }
            edge->setRobustKernel(nullptr);
        }
    }

    if (abort_flag && *abort_flag) {
        result.mean_squared_error = -1.0;
        if (fill_result_dict) {
            result.keyframe_updates = {};
            result.point_updates = {};
        }
        if (verbose) {
            MSG_WARN("bundle_adjustment: aborting due to abort_flag");
        }
        return result;
    }

    // Final optimization
    optimizer.initializeOptimization();
    optimizer.optimize(final_rounds);

    // Store updates
    if (fill_result_dict) {
        result.keyframe_updates = {};
        result.point_updates = {};
    }
    for (auto &kf_pair : graph_keyframes) {
        auto &kf = kf_pair.first;
        auto *vertex = kf_pair.second;
        g2o::SE3Quat est = vertex->estimate();
        Eigen::Matrix3d R = est.rotation().matrix();
        Eigen::Vector3d t = est.translation();
        Eigen::Matrix4d T = poseRt(R, t);
        if (fill_result_dict) {
            result.keyframe_updates[kf->id] = T;
        } else {
            if (loop_kf_id == 0) {
                // direct update on map
                kf->update_pose(T);
            } else {
                // update for loop closure
                kf->Tcw_GBA = T;
                kf->GBA_kf_id = loop_kf_id;
                kf->is_Tcw_GBA_valid = true;
            }
        }
    }

    // Update points if not fixed
    if (!fixed_points) {
        for (auto &p_pair : graph_points) {
            auto &p = p_pair.first;
            auto *vertex = p_pair.second;
            const Eigen::Vector3d new_position = vertex->estimate();
            if (fill_result_dict) {
                result.point_updates[p->id] = new_position;
            } else {
                if (loop_kf_id == 0) {
                    // direct update on map
                    p->update_position(new_position);
                    p->update_normal_and_depth(/*force=*/true);
                } else {
                    // update for loop closure
                    p->pt_GBA = new_position;
                    p->GBA_kf_id = loop_kf_id;
                    p->is_pt_GBA_valid = true;
                }
            }
        }
    }

    const int num_active_edges = num_edges - num_bad_edges;
    result.mean_squared_error = optimizer.activeChi2() / std::max(num_active_edges, 1);

    if (verbose) {
        std::cout << "bundle_adjustment: mean_squared_error: " << result.mean_squared_error
                  << ", initial_mean_squared_error: " << initial_mean_squared_error
                  << ", num_edges: " << num_edges << ", num_bad_edges: " << num_bad_edges
                  << " (perc: " << (num_bad_edges * 100.0 / std::max(num_edges, 1)) << "%)"
                  << std::endl;
    }

    return result;
}

BundleAdjustmentResult OptimizerG2o::global_bundle_adjustment(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points, int rounds,
    int loop_kf_id, bool use_robust_kernel, bool *abort_flag, bool fill_result_dict, bool verbose) {

    const bool fixed_points = false;
    return bundle_adjustment(keyframes, points, std::nullopt, fixed_points, rounds, loop_kf_id,
                             use_robust_kernel, abort_flag, fill_result_dict, verbose);
}

BundleAdjustmentResult
OptimizerG2o::global_bundle_adjustment_map(MapPtr &map, int rounds, int loop_kf_id,
                                           bool use_robust_kernel, bool *abort_flag,
                                           bool fill_result_dict, bool verbose) {

    auto keyframes = map->get_keyframes_vector();
    auto points = map->get_points_vector();

    return global_bundle_adjustment(keyframes, points, rounds, loop_kf_id, use_robust_kernel,
                                    abort_flag, fill_result_dict, verbose);
}

PoseOptimizationResult OptimizerG2o::pose_optimization(FramePtr &frame, bool verbose, int rounds) {

    PoseOptimizationResult result;
    result.is_ok = true;

    // Create g2o optimizer
    g2o::SparseOptimizer optimizer;
    auto linearSolver =
        std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverSE3>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer.setAlgorithm(algorithm);

    // Create SE3 vertex for frame pose
    auto *vertex_se3 = new g2o::VertexSE3Expmap();
    Eigen::Matrix3d Rcw = frame->Rcw();
    Eigen::Vector3d tcw = frame->tcw();
    g2o::SE3Quat se3(Rcw, tcw);

    vertex_se3->setEstimate(se3);
    vertex_se3->setId(0);
    vertex_se3->setFixed(false);
    optimizer.addVertex(vertex_se3);

    // Get frame points and create edges
    auto frame_points = frame->get_points();
    std::vector<std::pair<g2o::EdgeSE3ProjectXYZOnlyPose *, int>> point_edge_pairs_mono;
    point_edge_pairs_mono.reserve(frame_points.size());
    std::vector<std::pair<g2o::EdgeStereoSE3ProjectXYZOnlyPose *, int>> point_edge_pairs_stereo;
    point_edge_pairs_stereo.reserve(frame_points.size());

    int num_point_edges = 0;
    const auto &camera = frame->camera;

    const auto &inv_level_sigmas2 = FeatureSharedResources::inv_level_sigmas2;
    const auto &kps_ur = frame->kps_ur;
    const auto &octaves = frame->octaves;

    {
        std::lock_guard<std::mutex> lock(MapPoint::global_lock);

        for (size_t idx = 0; idx < frame_points.size(); ++idx) {
            const auto &p = frame_points[idx];
            if (!p) {
                continue;
            }

            const Eigen::Vector2d kpu = Eigen::Vector2d(frame->kpsu(idx, 0), frame->kpsu(idx, 1));
            const bool is_stereo_obs = (kps_ur.size() > idx && kps_ur[idx] >= 0);

            // Reset outlier flag
            frame->outliers[idx] = false;

            // Get inverse sigma squared
            const double invSigma2 = inv_level_sigmas2[octaves[idx]];

            if (is_stereo_obs) {
                auto *edge = new g2o::EdgeStereoSE3ProjectXYZOnlyPose();
                Eigen::Vector3d obs_vec(kpu.x(), kpu.y(), kps_ur[idx]);
                edge->setVertex(0, vertex_se3);
                edge->setMeasurement(obs_vec);
                edge->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                auto *robust_kernel = new g2o::RobustKernelHuber();
                robust_kernel->setDelta(Parameters::kThHuberStereo);
                edge->setRobustKernel(robust_kernel);

                edge->fx = camera->fx;
                edge->fy = camera->fy;
                edge->cx = camera->cx;
                edge->cy = camera->cy;
                edge->bf = camera->bf;
                edge->Xw = p->pt();

                optimizer.addEdge(edge);
                point_edge_pairs_stereo.push_back({edge, static_cast<int>(idx)});
            } else {
                auto *edge = new g2o::EdgeSE3ProjectXYZOnlyPose();
                edge->setVertex(0, vertex_se3);
                edge->setMeasurement(kpu);
                edge->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                auto *robust_kernel = new g2o::RobustKernelHuber();
                robust_kernel->setDelta(Parameters::kThHuberMono);
                edge->setRobustKernel(robust_kernel);

                edge->fx = camera->fx;
                edge->fy = camera->fy;
                edge->cx = camera->cx;
                edge->cy = camera->cy;
                edge->Xw = p->pt();

                optimizer.addEdge(edge);
                point_edge_pairs_mono.push_back({edge, static_cast<int>(idx)});
            }
            num_point_edges++;
        }
    }

    if (num_point_edges < 3) {
        MSG_WARN("pose_optimization: not enough correspondences!");
        result.is_ok = false;
        return result;
    }

    if (verbose) {
        optimizer.setVerbose(true);
    }

    int num_bad_point_edges = 0;

    // Perform 4 optimizations with outlier detection
    for (int it = 0; it < 4; ++it) {
        vertex_se3->setEstimate(g2o::SE3Quat(Rcw, tcw));
        optimizer.initializeOptimization();
        optimizer.optimize(rounds);

        num_bad_point_edges = 0;

        for (auto &pair : point_edge_pairs_mono) {
            auto *edge = pair.first;
            int idx = pair.second;

            if (frame->outliers[idx]) {
                edge->computeError();
            }

            double chi2 = edge->chi2();
            if (chi2 > Parameters::kChi2Mono) {
                frame->outliers[idx] = true;
                edge->setLevel(1);
                num_bad_point_edges++;
            } else {
                frame->outliers[idx] = false;
                edge->setLevel(0);
            }

            if (it == 2) {
                edge->setRobustKernel(nullptr);
            }
        }

        // Check stereo edges
        for (auto &pair : point_edge_pairs_stereo) {
            auto *edge = pair.first;
            int idx = pair.second;

            if (frame->outliers[idx]) {
                edge->computeError();
            }

            double chi2 = edge->chi2();
            if (chi2 > Parameters::kChi2Stereo) {
                frame->outliers[idx] = true;
                edge->setLevel(1);
                num_bad_point_edges++;
            } else {
                frame->outliers[idx] = false;
                edge->setLevel(0);
            }

            if (it == 2) {
                edge->setRobustKernel(nullptr);
            }
        }

        if (optimizer.edges().size() < 10) {
            std::cerr << "pose_optimization: stopped - not enough edges!" << std::endl;
            break;
        }
    }

    std::cout << "pose_optimization: available " << num_point_edges << " points, found "
              << num_bad_point_edges << " bad points" << std::endl;

    result.num_valid_points = num_point_edges - num_bad_point_edges;
    if (result.num_valid_points < 10) {
        MSG_RED_WARN("pose_optimization: not enough edges!");
        result.is_ok = false;
    }

    const double ratio_bad_points =
        static_cast<double>(num_bad_point_edges) / std::max(1, num_point_edges);
    if (result.num_valid_points > 15 &&
        ratio_bad_points > Parameters::kMaxOutliersRatioInPoseOptimization) {
        MSG_RED_WARN_STREAM("pose_optimization: percentage of bad points is too high: "
                            << ratio_bad_points * 100.0 << "%");
        result.is_ok = false;
    }

    // Update pose estimation
    if (result.is_ok) {
        g2o::SE3Quat est = vertex_se3->estimate();
        const Eigen::Matrix3d R = est.rotation().matrix();
        const Eigen::Vector3d t = est.translation();
        const Eigen::Matrix4d T = poseRt(R, t);
        frame->update_pose(T);
    }

    result.mean_squared_error = optimizer.activeChi2() / std::max(result.num_valid_points, 1);

    return result;
}

template <typename LockType>
std::pair<double, double> OptimizerG2o::local_bundle_adjustment(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    const std::vector<KeyFramePtr> &keyframes_ref, bool fixed_points, bool verbose, int rounds,
    bool *abort_flag, LockType *map_lock) {

    // Create g2o optimizer
    g2o::SparseOptimizer optimizer;
    auto linearSolver =
        std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverSE3>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer.setAlgorithm(algorithm);

    if (abort_flag) {
        optimizer.setForceStopFlag(abort_flag);
    }

    // Filter good keyframes and points
    std::vector<KeyFramePtr> good_keyframes;
    for (const auto &kf : keyframes) {
        if (!kf->is_bad()) {
            good_keyframes.push_back(kf);
        }
    }
    for (const auto &kf : keyframes_ref) {
        if (!kf->is_bad()) {
            good_keyframes.push_back(kf);
        }
    }

    // Maps to store graph elements
    std::unordered_map<KeyFramePtr, g2o::VertexSE3Expmap *> graph_keyframes;
    std::unordered_map<MapPointPtr, g2o::VertexSBAPointXYZ *> graph_points;
    std::unordered_map<g2o::EdgeSE3ProjectXYZ *, std::tuple<MapPointPtr, KeyFramePtr, int, bool>>
        graph_edges_mono;
    std::unordered_map<g2o::EdgeStereoSE3ProjectXYZ *,
                       std::tuple<MapPointPtr, KeyFramePtr, int, bool>>
        graph_edges_stereo;

    // Add keyframe vertices
    for (const auto &kf : good_keyframes) {
        auto *vertex_se3 = new g2o::VertexSE3Expmap();
        Eigen::Matrix3d Rcw = kf->Rcw();
        Eigen::Vector3d tcw = kf->tcw();
        g2o::SE3Quat se3(Rcw, tcw);

        vertex_se3->setEstimate(se3);
        vertex_se3->setId(kf->kid * 2);
        const bool is_kf_in_keyframes_ref =
            std::find(keyframes_ref.begin(), keyframes_ref.end(), kf) != keyframes_ref.end();
        vertex_se3->setFixed(kf->kid == 0 || is_kf_in_keyframes_ref);

        optimizer.addVertex(vertex_se3);
        graph_keyframes[kf] = vertex_se3;
    }

    int num_edges = 0;
    int num_bad_edges = 0;

    const auto &inv_level_sigmas2 = FeatureSharedResources::inv_level_sigmas2;

    // Add point vertices and edges
    for (const auto &p : points) {
        if (!p || p->is_bad()) {
            continue;
        }
        auto *vertex_point = new g2o::VertexSBAPointXYZ();
        vertex_point->setId(p->id * 2 + 1);
        vertex_point->setEstimate(p->pt());
        vertex_point->setMarginalized(true);
        vertex_point->setFixed(fixed_points);

        optimizer.addVertex(vertex_point);
        graph_points[p] = vertex_point;

        // Add edges for good observations
        auto observations = p->observations();
        for (const auto &obs : observations) {
            const auto &kf = obs.first;
            int p_idx = obs.second;

            if (kf->is_bad()) {
                continue;
            }
            const auto it = graph_keyframes.find(kf);
            if (it == graph_keyframes.end()) {
                continue;
            }
            const auto v_se3_kf = it->second;

            // Verify the observation is correct
            const auto &p_f = kf->get_point_match(p_idx);
            if (p_f != p) {
                continue; // Skip invalid observations
            }

            const Eigen::Vector2d kpu = Eigen::Vector2d(kf->kpsu(p_idx, 0), kf->kpsu(p_idx, 1));
            const auto &kps_ur = kf->kps_ur;
            bool is_stereo_obs = (kps_ur.size() > p_idx && kps_ur[p_idx] >= 0);

            const auto &octaves = kf->octaves;
            const double invSigma2 = inv_level_sigmas2[octaves[p_idx]];

            const auto &camera = kf->camera;

            if (is_stereo_obs) {
                auto *edge = new g2o::EdgeStereoSE3ProjectXYZ();
                Eigen::Vector3d obs_vec(kpu.x(), kpu.y(), kps_ur[p_idx]);
                edge->setMeasurement(obs_vec);
                edge->setInformation(Eigen::Matrix3d::Identity() * invSigma2);

                auto *robust_kernel = new g2o::RobustKernelHuber();
                robust_kernel->setDelta(Parameters::kThHuberStereo);
                edge->setRobustKernel(robust_kernel);

                edge->fx = camera->fx;
                edge->fy = camera->fy;
                edge->cx = camera->cx;
                edge->cy = camera->cy;
                edge->bf = camera->bf;

                edge->setVertex(0, vertex_point);
                edge->setVertex(1, v_se3_kf);
                optimizer.addEdge(edge);

                graph_edges_stereo[edge] = {p, kf, p_idx, is_stereo_obs};
            } else {
                auto *edge = new g2o::EdgeSE3ProjectXYZ();
                edge->setMeasurement(kpu);
                edge->setInformation(Eigen::Matrix2d::Identity() * invSigma2);

                auto *robust_kernel = new g2o::RobustKernelHuber();
                robust_kernel->setDelta(Parameters::kThHuberMono);
                edge->setRobustKernel(robust_kernel);

                edge->fx = camera->fx;
                edge->fy = camera->fy;
                edge->cx = camera->cx;
                edge->cy = camera->cy;

                edge->setVertex(0, vertex_point);
                edge->setVertex(1, v_se3_kf);
                optimizer.addEdge(edge);

                graph_edges_mono[edge] = {p, kf, p_idx, is_stereo_obs};
            }
            num_edges++;
        }
    }

    if (verbose) {
        optimizer.setVerbose(true);
    }

    if (abort_flag && *abort_flag) {
        return {-1.0, 0.0};
    }

    std::cout << "local_bundle_adjustment: starting optimization with " << graph_keyframes.size()
              << " keyframes, " << graph_points.size() << " points, " << num_edges << " edges"
              << std::endl;

    // Initial optimization
    optimizer.initializeOptimization();
    optimizer.optimize(5);

    // Check inliers and remove outliers
    if (!abort_flag || !*abort_flag) {
        for (auto &edge_pair : graph_edges_mono) {
            auto *edge = edge_pair.first;
            auto &edge_data = edge_pair.second;
            auto [p, kf, p_idx, is_stereo] = edge_data;

            double edge_chi2 = edge->chi2();
            bool chi2_check_failure = (edge_chi2 > Parameters::kChi2Mono);
            if (chi2_check_failure || !edge->isDepthPositive()) {
                edge->setLevel(1);
                num_bad_edges++;
            }
            edge->setRobustKernel(nullptr);
        }

        for (auto &edge_pair : graph_edges_stereo) {
            auto *edge = edge_pair.first;
            auto &edge_data = edge_pair.second;
            auto [p, kf, p_idx, is_stereo] = edge_data;

            double edge_chi2 = edge->chi2();
            bool chi2_check_failure = (edge_chi2 > Parameters::kChi2Stereo);
            if (chi2_check_failure || !edge->isDepthPositive()) {
                edge->setLevel(1);
                num_bad_edges++;
            }
            edge->setRobustKernel(nullptr);
        }

        // Optimize again without outliers
        optimizer.initializeOptimization();
        optimizer.optimize(rounds);
    }

    // Search for final outlier observations
    int num_bad_observations = 0;
    std::vector<std::tuple<MapPointPtr, KeyFramePtr, int, bool>> outliers_edge_data;

    for (auto &edge_pair : graph_edges_mono) {
        auto *edge = edge_pair.first;
        auto &edge_data = edge_pair.second;
        auto [p, kf, p_idx, is_stereo] = edge_data;

        if (edge->chi2() > Parameters::kChi2Mono || !edge->isDepthPositive()) {
            num_bad_observations++;
            outliers_edge_data.push_back(edge_data);
        }
    }

    for (auto &edge_pair : graph_edges_stereo) {
        auto *edge = edge_pair.first;
        auto &edge_data = edge_pair.second;
        auto [p, kf, p_idx, is_stereo] = edge_data;

        if (edge->chi2() > Parameters::kChi2Stereo || !edge->isDepthPositive()) {
            num_bad_observations++;
            outliers_edge_data.push_back(edge_data);
        }
    }

    // Apply updates with map lock
    // std::unique_lock<std::mutex> lock;
    std::unique_ptr<pyslam::PyLockGuard<LockType>> lock;
    if (map_lock) {
        lock = std::make_unique<pyslam::PyLockGuard<LockType>>(map_lock);
    }
    // Only bypass the map's internal locking when we already hold the map lock.
    const bool map_no_lock = (map_lock != nullptr);

    // Remove outlier observations
    for (const auto &outlier_data : outliers_edge_data) {
        auto [p, kf, p_idx, is_stereo] = outlier_data;
        const auto &p_f = kf->get_point_match(p_idx);
        if (p_f && p_f == p) {
            p->remove_observation(kf, p_idx, map_no_lock);
        }
    }

    // Update keyframe poses
    for (auto &kf_pair : graph_keyframes) {
        const auto &kf = kf_pair.first;
        auto *vertex = kf_pair.second;
        g2o::SE3Quat est = vertex->estimate();
        Eigen::Matrix3d R = est.rotation().matrix();
        Eigen::Vector3d t = est.translation();
        Eigen::Matrix4d T = poseRt(R, t);
        kf->update_pose(T);
        kf->lba_count++;
    }

    // Update point positions
    if (!fixed_points) {
        for (auto &p_pair : graph_points) {
            const auto &p = p_pair.first;
            auto *vertex = p_pair.second;
            Eigen::Vector3d new_position = vertex->estimate();
            p->update_position(new_position);
            p->update_normal_and_depth(true);
        }
    }

    int num_active_edges = num_edges - num_bad_edges;
    double mean_squared_error = optimizer.activeChi2() / std::max(num_active_edges, 1);
    double outlier_ratio = static_cast<double>(num_bad_observations) / std::max(1, num_edges);

    return {mean_squared_error, outlier_ratio};
}

template std::pair<double, double> OptimizerG2o::local_bundle_adjustment<pyslam::MapMutex>(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    const std::vector<KeyFramePtr> &keyframes_ref, bool fixed_points, bool verbose, int rounds,
    bool *abort_flag, pyslam::MapMutex *map_lock);

#ifdef USE_PYTHON

template std::pair<double, double> OptimizerG2o::local_bundle_adjustment<pyslam::PyLock>(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    const std::vector<KeyFramePtr> &keyframes_ref, bool fixed_points, bool verbose, int rounds,
    bool *abort_flag, pyslam::PyLock *map_lock);
#endif

Sim3OptimizationResult OptimizerG2o::optimize_sim3(
    KeyFramePtr &kf1, KeyFramePtr &kf2, const std::vector<MapPointPtr> &map_points1,
    const std::vector<MapPointPtr> &map_point_matches12, const Eigen::Matrix3d &R12,
    const Eigen::Vector3d &t12, double s12, double th2, bool fix_scale, bool verbose) {

    Sim3OptimizationResult result;
    result.num_inliers = 0;
    result.R = R12;
    result.t = t12;
    result.scale = s12;
    result.delta_error = 0.0;

    // Camera calibration
    const auto &cam1 = kf1->camera;
    const auto &cam2 = kf2->camera;

    Eigen::Matrix3d R1w = kf1->Rcw();
    Eigen::Vector3d t1w = kf1->tcw();
    Eigen::Matrix3d R2w = kf2->Rcw();
    Eigen::Vector3d t2w = kf2->tcw();

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    auto linearSolver =
        std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolverX::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverX>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    optimizer.setAlgorithm(algorithm);

    g2o::Sim3 sim3(R12, t12, s12);

    // Sim3 vertex
    auto *sim3_vertex = new g2o::VertexSim3Expmap();
    sim3_vertex->setEstimate(sim3);
    sim3_vertex->setId(0);
    sim3_vertex->setFixed(false);
    sim3_vertex->_fix_scale = fix_scale;
    sim3_vertex->_principle_point1 = Eigen::Vector2d(cam1->cx, cam1->cy);
    sim3_vertex->_focal_length1 = Eigen::Vector2d(cam1->fx, cam1->fy);
    sim3_vertex->_principle_point2 = Eigen::Vector2d(cam2->cx, cam2->cy);
    sim3_vertex->_focal_length2 = Eigen::Vector2d(cam2->fx, cam2->fy);
    optimizer.addVertex(sim3_vertex);

    // Process map points
    std::vector<MapPointPtr> actual_map_points1 = map_points1;
    if (actual_map_points1.empty()) {
        actual_map_points1 = kf1->get_points();
    }

    int num_matches = map_point_matches12.size();
    if (num_matches != static_cast<int>(actual_map_points1.size())) {
        return result; // Invalid input
    }

    if (verbose) {
        std::cout << "optimize_sim3: num_matches = " << num_matches << std::endl;
    }

    std::vector<g2o::EdgeSim3ProjectXYZ *> edges_12;
    std::vector<g2o::EdgeInverseSim3ProjectXYZ *> edges_21;
    std::vector<int> vertex_indices;

    const auto &inv_level_sigmas2 = FeatureSharedResources::inv_level_sigmas2;

    double delta_huber = std::sqrt(th2);
    int num_correspondences = 0;

    for (int i = 0; i < num_matches; ++i) {
        const auto &mp1 = actual_map_points1[i];
        if (!mp1 || mp1->is_bad()) {
            continue;
        }
        const auto &mp2 = map_point_matches12[i];
        if (!mp2 || mp2->is_bad()) {
            continue;
        }

        int vertex_id1 = 2 * i + 1;
        int vertex_id2 = 2 * (i + 1);
        int index2 = mp2->get_observation_idx(kf2);
        if (index2 >= 0) {
            // Create vertex for map point 1 (fixed)
            auto *v_mp1 = new g2o::VertexSBAPointXYZ();
            v_mp1->setEstimate(R1w * mp1->pt() + t1w);
            v_mp1->setId(vertex_id1);
            v_mp1->setFixed(true);
            optimizer.addVertex(v_mp1);

            // Create vertex for map point 2 (fixed)
            auto *v_mp2 = new g2o::VertexSBAPointXYZ();
            v_mp2->setEstimate(R2w * mp2->pt() + t2w);
            v_mp2->setId(vertex_id2);
            v_mp2->setFixed(true);
            optimizer.addVertex(v_mp2);

            // Create edge 12 (project mp2_2 on camera 1)
            auto *edge_12 = new g2o::EdgeSim3ProjectXYZ();
            edge_12->setVertex(0, optimizer.vertex(vertex_id2));
            edge_12->setVertex(1, optimizer.vertex(0));
            const Eigen::Vector2d kpu1 = Eigen::Vector2d(kf1->kpsu(i, 0), kf1->kpsu(i, 1));
            edge_12->setMeasurement(kpu1);

            const auto &octaves1 = kf1->octaves;
            const double invSigma2_12 = inv_level_sigmas2[octaves1[i]];

            edge_12->setInformation(Eigen::Matrix2d::Identity() * invSigma2_12);
            auto *robust_kernel_12 = new g2o::RobustKernelHuber();
            robust_kernel_12->setDelta(delta_huber);
            edge_12->setRobustKernel(robust_kernel_12);
            optimizer.addEdge(edge_12);

            // Create edge 21 (project mp1_1 on camera 2)
            auto *edge_21 = new g2o::EdgeInverseSim3ProjectXYZ();
            edge_21->setVertex(0, optimizer.vertex(vertex_id1));
            edge_21->setVertex(1, optimizer.vertex(0));
            const Eigen::Vector2d kpu2 =
                Eigen::Vector2d(kf2->kpsu(index2, 0), kf2->kpsu(index2, 1));
            edge_21->setMeasurement(kpu2);

            const auto &octaves2 = kf2->octaves;
            const double invSigma2_21 = inv_level_sigmas2[octaves2[index2]];

            edge_21->setInformation(Eigen::Matrix2d::Identity() * invSigma2_21);
            auto *robust_kernel_21 = new g2o::RobustKernelHuber();
            robust_kernel_21->setDelta(delta_huber);
            edge_21->setRobustKernel(robust_kernel_21);
            optimizer.addEdge(edge_21);

            edges_12.push_back(edge_12);
            edges_21.push_back(edge_21);
            vertex_indices.push_back(i);
            num_correspondences++;
        }
    }

    if (verbose) {
        std::cout << "optimize_sim3: num_correspondences = " << num_correspondences << std::endl;
    }

    // Optimize
    optimizer.initializeOptimization();
    if (verbose) {
        optimizer.setVerbose(true);
    }
    optimizer.optimize(5);
    double err = optimizer.activeChi2();

    // Check inliers
    int num_bad = 0;
    for (size_t i = 0; i < edges_12.size(); ++i) {
        auto *edge_12 = edges_12[i];
        auto *edge_21 = edges_21[i];

        if (edge_12->chi2() > th2 || !edge_12->isDepthPositive() || edge_21->chi2() > th2 ||
            !edge_21->isDepthPositive()) {
            int index = vertex_indices[i];
            actual_map_points1[index] = nullptr;
            optimizer.removeEdge(edge_12);
            optimizer.removeEdge(edge_21);
            edges_12[i] = nullptr;
            edges_21[i] = nullptr;
            num_bad++;
        }
    }

    int num_more_iterations = (num_bad > 0) ? 10 : 5;

    if (num_correspondences - num_bad < 10) {
        std::cout << "optimize_sim3: Too few inliers, num_correspondences = " << num_correspondences
                  << ", num_bad = " << num_bad << std::endl;
        return Sim3OptimizationResult();
    }

    // Optimize again with inliers
    optimizer.initializeOptimization();
    optimizer.optimize(num_more_iterations);

    result.delta_error = optimizer.activeChi2() - err;

    // Count final inliers
    for (size_t i = 0; i < edges_12.size(); ++i) {
        auto *edge_12 = edges_12[i];
        auto *edge_21 = edges_21[i];

        if (edge_12 && edge_21) {
            if (edge_12->chi2() > th2 || !edge_12->isDepthPositive() || edge_21->chi2() > th2 ||
                !edge_21->isDepthPositive()) {
                int index = vertex_indices[i];
                actual_map_points1[index] = nullptr;
            } else {
                result.num_inliers++;
            }
        }
    }

    // Recover optimized Sim3
    auto *sim3_vertex_recov = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(0));
    g2o::Sim3 sim3_opt = sim3_vertex_recov->estimate();
    result.R = sim3_opt.rotation().matrix();
    result.t = sim3_opt.translation();
    result.scale = sim3_opt.scale();

    return result;
}

double OptimizerG2o::optimize_essential_graph(
    MapPtr map_object, KeyFramePtr loop_keyframe, KeyFramePtr current_keyframe,
    const std::unordered_map<KeyFramePtr, Sim3Pose> &non_corrected_sim3_map,
    const std::unordered_map<KeyFramePtr, Sim3Pose> &corrected_sim3_map,
    const std::unordered_map<KeyFramePtr, std::vector<KeyFramePtr>> &loop_connections,
    bool fix_scale, bool verbose) {

    // Setup optimizer
    g2o::SparseOptimizer optimizer;
    optimizer.setVerbose(false);

    auto linearSolver =
        std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_7_3::PoseMatrixType>>();
    auto blockSolver = std::make_unique<g2o::BlockSolverSim3>(std::move(linearSolver));
    auto algorithm = new g2o::OptimizationAlgorithmLevenberg(std::move(blockSolver));
    algorithm->setUserLambdaInit(1e-16);
    optimizer.setAlgorithm(algorithm);

    const auto all_keyframes = map_object->get_keyframes_vector();
    const auto all_map_points = map_object->get_points_vector();

    const int max_keyframe_id = map_object->max_keyframe_id;

    std::vector<bool> vec_Scw_is_valid(max_keyframe_id + 1, false);
    std::vector<g2o::Sim3> vec_Scw(max_keyframe_id + 1);
    std::vector<g2o::Sim3> vec_corrected_Swc(max_keyframe_id + 1);

    int min_number_features = 100;

    // Set KeyFrame vertices
    for (const auto &keyframe : all_keyframes) {
        if (keyframe->is_bad()) {
            continue;
        }

        auto *vertex_sim3 = new g2o::VertexSim3Expmap();
        const int keyframe_id = keyframe->kid;

        g2o::Sim3 Siw;
        const auto corrected_it = corrected_sim3_map.find(keyframe);
        if (corrected_it != corrected_sim3_map.end()) {
            const auto &Siw_ = corrected_it->second;
            Siw = g2o::Sim3(Siw_.R(), Siw_.t(), Siw_.s());
        } else {
            Siw = g2o::Sim3(keyframe->Rcw(), keyframe->tcw(), 1.0);
        }
        vec_Scw[keyframe_id] = Siw;
        vec_Scw_is_valid[keyframe_id] = true;

        vertex_sim3->setEstimate(
            g2o::Sim3(Siw.rotation().matrix(), Siw.translation(), Siw.scale()));
        vertex_sim3->setFixed(keyframe == loop_keyframe);
        vertex_sim3->setId(keyframe_id);
        vertex_sim3->setMarginalized(false);
        vertex_sim3->_fix_scale = fix_scale;

        optimizer.addVertex(vertex_sim3);
    }

    int num_graph_edges = 0;
    std::set<std::pair<int, int>> inserted_loop_edges;

    // Set loop edges
    for (const auto &connection_pair : loop_connections) {
        const auto &keyframe = connection_pair.first;
        const auto &connections = connection_pair.second;

        const int keyframe_id = keyframe->kid;
        if (keyframe_id >= vec_Scw.size()) {
            std::cerr << "[optimize_essential_graph] keyframe_id " << keyframe_id
                      << " is out of bounds" << std::endl;
            continue;
        }
        const auto &Siw = vec_Scw[keyframe_id];
        if (!vec_Scw_is_valid[keyframe_id]) {
            MSG_RED_WARN_STREAM("optimize_essential_graph: SiW for keyframe " << keyframe_id
                                                                              << " is not valid");
            continue;
        }
        const auto Swi = Siw.inverse();

        for (const auto &connected_keyframe : connections) {
            const int connected_id = connected_keyframe->kid;

            // Accept (current_keyframe, loop_keyframe) and other loop edges with sufficient weight
            if ((keyframe_id != current_keyframe->kid || connected_id != loop_keyframe->kid) &&
                keyframe->get_weight(connected_keyframe) < min_number_features) {
                continue;
            }

            auto &Sjw = vec_Scw[connected_id];
            const auto Sji = Sjw * Swi;

            auto *edge = new g2o::EdgeSim3();
            edge->setVertex(1, optimizer.vertex(connected_id));
            edge->setVertex(0, optimizer.vertex(keyframe_id));
            edge->setMeasurement(
                g2o::Sim3(Sji.rotation().matrix(), Sji.translation(), Sji.scale()));
            edge->setInformation(Eigen::Matrix<double, 7, 7>::Identity());

            optimizer.addEdge(edge);
            num_graph_edges++;
            inserted_loop_edges.insert(
                {std::min(keyframe_id, connected_id), std::max(keyframe_id, connected_id)});
        }
    }

    // Set normal edges
    for (const auto &keyframe : all_keyframes) {
        const int keyframe_id = keyframe->kid;
        const auto &parent_keyframe = keyframe->get_parent();

        g2o::Sim3 Swi;
        const auto non_corrected_it = non_corrected_sim3_map.find(keyframe);
        if (non_corrected_it != non_corrected_sim3_map.end()) {
            const auto &Swi_ = non_corrected_it->second.inverse();
            Swi = g2o::Sim3(Swi_.R(), Swi_.t(), Swi_.s());
        } else {
            Swi = vec_Scw[keyframe_id].inverse();
        }

        // Spanning tree edge
        if (parent_keyframe) {
            const int parent_id = parent_keyframe->kid;

            g2o::Sim3 Sjw;
            const auto parent_non_corrected_it = non_corrected_sim3_map.find(parent_keyframe);
            if (parent_non_corrected_it != non_corrected_sim3_map.end()) {
                const auto &Sjw_ = parent_non_corrected_it->second;
                Sjw = g2o::Sim3(Sjw_.R(), Sjw_.t(), Sjw_.s());
            } else {
                Sjw = vec_Scw[parent_id];
            }

            const auto Sji = Sjw * Swi;

            auto *edge = new g2o::EdgeSim3();
            edge->setVertex(1, optimizer.vertex(parent_id));
            edge->setVertex(0, optimizer.vertex(keyframe_id));
            edge->setMeasurement(
                g2o::Sim3(Sji.rotation().matrix(), Sji.translation(), Sji.scale()));
            edge->setInformation(Eigen::Matrix<double, 7, 7>::Identity());
            optimizer.addEdge(edge);
            num_graph_edges++;
        }

        // Loop edges
        auto loop_edges = keyframe->get_loop_edges();
        for (const auto &loop_edge : loop_edges) {
            if (loop_edge->kid < keyframe_id) {
                g2o::Sim3 Slw;
                const auto loop_non_corrected_it = non_corrected_sim3_map.find(loop_edge);
                if (loop_non_corrected_it != non_corrected_sim3_map.end()) {
                    const auto &Slw_ = loop_non_corrected_it->second;
                    Slw = g2o::Sim3(Slw_.R(), Slw_.t(), Slw_.s());
                } else {
                    Slw = vec_Scw[loop_edge->kid];
                }

                const auto Sli = Slw * Swi;

                auto *edge = new g2o::EdgeSim3();
                edge->setVertex(1, optimizer.vertex(loop_edge->kid));
                edge->setVertex(0, optimizer.vertex(keyframe_id));
                edge->setMeasurement(
                    g2o::Sim3(Sli.rotation().matrix(), Sli.translation(), Sli.scale()));
                edge->setInformation(Eigen::Matrix<double, 7, 7>::Identity());
                optimizer.addEdge(edge);
                num_graph_edges++;
            }
        }

        // Covisibility graph edges
        auto covisible_keyframes = keyframe->get_covisible_by_weight(min_number_features);
        for (const auto &connected_keyframe : covisible_keyframes) {
            if (connected_keyframe != parent_keyframe && !keyframe->has_child(connected_keyframe) &&
                connected_keyframe->kid < keyframe_id && !connected_keyframe->is_bad() &&
                inserted_loop_edges.find({std::min(keyframe_id, connected_keyframe->kid),
                                          std::max(keyframe_id, connected_keyframe->kid)}) ==
                    inserted_loop_edges.end()) {

                g2o::Sim3 Snw;
                auto connected_non_corrected_it = non_corrected_sim3_map.find(connected_keyframe);
                if (connected_non_corrected_it != non_corrected_sim3_map.end()) {
                    const auto &Snw_ = connected_non_corrected_it->second;
                    Snw = g2o::Sim3(Snw_.R(), Snw_.t(), Snw_.s());
                } else {
                    Snw = vec_Scw[connected_keyframe->kid];
                }

                const auto Sni = Snw * Swi;

                auto *edge = new g2o::EdgeSim3();
                edge->setVertex(1, optimizer.vertex(connected_keyframe->kid));
                edge->setVertex(0, optimizer.vertex(keyframe_id));
                edge->setMeasurement(
                    g2o::Sim3(Sni.rotation().matrix(), Sni.translation(), Sni.scale()));
                edge->setInformation(Eigen::Matrix<double, 7, 7>::Identity());
                optimizer.addEdge(edge);
                num_graph_edges++;
            }
        }
    }

    if (verbose) {
        std::cout << "[optimize_essential_graph]: Total number of graph edges: " << num_graph_edges
                  << std::endl;
    }

    // Optimize
    optimizer.initializeOptimization();
    optimizer.optimize(20);

    // SE3 Pose Recovering. Sim3:[sR t;0 1] -> SE3:[R t/s;0 1]
    for (const auto &keyframe : all_keyframes) {
        const int keyframe_id = keyframe->kid;
        auto *vertex_sim3 = static_cast<g2o::VertexSim3Expmap *>(optimizer.vertex(keyframe_id));
        g2o::Sim3 corrected_Siw = vertex_sim3->estimate();

        const Eigen::Matrix3d R = corrected_Siw.rotation().matrix();
        const Eigen::Vector3d t = corrected_Siw.translation();
        const double s = corrected_Siw.scale();

        const g2o::Sim3 corrected_Siw_pose(R, t, s);
        vec_corrected_Swc[keyframe_id] = corrected_Siw_pose.inverse();

        // Convert Sim3 to SE3: [R t/s; 0 1]
        Eigen::Matrix4d Tiw = poseRt(R, t / s);
        keyframe->update_pose(Tiw);
    }

    // Correct points: Transform to "non-optimized" reference keyframe pose and transform back with
    // optimized pose
    for (const auto &map_point : all_map_points) {
        if (map_point->is_bad()) {
            continue;
        }

        int reference_id;
        if (map_point->corrected_by_kf == current_keyframe->kid) {
            reference_id = map_point->corrected_reference;
        } else {
            const auto &reference_keyframe = map_point->get_reference_keyframe();
            reference_id = reference_keyframe->kid;
        }

        const g2o::Sim3 &Srw = vec_Scw[reference_id];
        const g2o::Sim3 &corrected_Swr = vec_corrected_Swc[reference_id];

        const Eigen::Vector3d P3Dw = map_point->pt();
        const Eigen::Vector3d corrected_P3Dw = corrected_Swr.map(Srw.map(P3Dw));
        map_point->update_position(corrected_P3Dw);
        map_point->update_normal_and_depth();
    }

    const double mean_squared_error = optimizer.activeChi2() / std::max(num_graph_edges, 1);
    return mean_squared_error;
}

} // namespace pyslam
