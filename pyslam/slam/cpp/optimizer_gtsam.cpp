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

#include "optimizer_gtsam.h"

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

// Include GTSAM headers - Point3 must come first as it's a typedef used by other headers
#include <gtsam/geometry/Cal3_S2.h>
#include <gtsam/geometry/Cal3_S2Stereo.h>
#include <gtsam/geometry/Point3.h> // Must include full definition (it's a typedef to Vector3)
#include <gtsam/geometry/Pose3.h>
#include <gtsam/geometry/Similarity3.h>
#include <gtsam/geometry/StereoPoint2.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <gtsam/nonlinear/NonlinearFactorGraph.h>
#include <gtsam/slam/BetweenFactor.h>
#include <gtsam/slam/PriorFactor.h>

#include <gtsam_factors/numerical_derivative.h>
#include <gtsam_factors/optimizers.h>
#include <gtsam_factors/resectioning.h>
#include <gtsam_factors/similarity.h>
#include <gtsam_factors/switchable_robust_noise_model.h>
#include <gtsam_factors/weighted_projection_factors.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <iostream>
#include <mutex>
#include <sstream>
#include <thread>
#include <tuple>

#ifdef USE_PYTHON
#include "py_module/py_wrappers.h"
#endif

// Use the functions from similarity.h
using gtsam_factors::getSimilarity3;
using gtsam_factors::insertSimilarity3;

// Typedefs for PriorFactor types
using PriorFactorPose3 = gtsam::PriorFactor<gtsam::Pose3>;
using PriorFactorPoint3 = gtsam::PriorFactor<gtsam::Point3>;

namespace pyslam {

using namespace gtsam;
using symbol_shorthand::L;
using symbol_shorthand::X;

// Constants
constexpr double kSigmaForFixed = OptimizerGTSAM::kSigmaForFixed;
constexpr double kWeightForDisabledFactor = OptimizerGTSAM::kWeightForDisabledFactor;

// Helper function implementations
Eigen::Matrix4d OptimizerGTSAM::pose3_to_matrix(const gtsam::Pose3 &pose) {
    Eigen::Matrix4d T = Eigen::Matrix4d::Identity();
    T.block<3, 3>(0, 0) = pose.rotation().matrix();
    // Point3 is a typedef to Vector3, so explicit conversion needed
    T.block<3, 1>(0, 3) = Eigen::Vector3d(pose.translation());
    return T;
}

gtsam::Pose3 OptimizerGTSAM::matrix_to_pose3(const Eigen::Matrix4d &T) {
    Eigen::Matrix3d R = T.block<3, 3>(0, 0);
    Eigen::Vector3d t = T.block<3, 1>(0, 3);
    return gtsam::Pose3(gtsam::Rot3(R), gtsam::Point3(t));
}

// Bundle adjustment implementation
BundleAdjustmentResult OptimizerGTSAM::bundle_adjustment(
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

    std::cout << "bundle_adjustment: rounds: " << rounds << ", robust_rounds: " << robust_rounds
              << ", final_rounds: " << final_rounds << std::endl;

    // Create GTSAM factor graph
    NonlinearFactorGraph graph;
    Values initial_estimates;

    const auto &level_sigmas = FeatureSharedResources::level_sigmas;

    // Huber loss parameters
    const double th_huber_mono = std::sqrt(Parameters::kChi2Mono);
    const double th_huber_stereo = std::sqrt(Parameters::kChi2Stereo);

    // Maps to store graph elements
    std::unordered_map<KeyFramePtr, Key> keyframe_keys;
    std::unordered_map<MapPointPtr, Key> point_keys;
    std::vector<std::tuple<boost::shared_ptr<gtsam_factors::WeightedGenericProjectionFactorCal3_S2>,
                           boost::shared_ptr<gtsam_factors::SwitchableRobustNoiseModel>,
                           MapPointPtr, KeyFramePtr, int>>
        graph_factors_mono;
    std::vector<
        std::tuple<boost::shared_ptr<gtsam_factors::WeightedGenericStereoProjectionFactor3D>,
                   boost::shared_ptr<gtsam_factors::SwitchableRobustNoiseModel>, MapPointPtr,
                   KeyFramePtr, int>>
        graph_factors_stereo;

    int num_edges = 0;
    int num_bad_edges = 0;

    // Add keyframes as pose variables
    std::vector<KeyFramePtr> frames_to_optimize = fixed_points ? local_frames : keyframes;
    for (auto &kf : frames_to_optimize) {
        if (!kf || kf->is_bad()) {
            continue;
        }

        Key pose_key = X(kf->kid);
        keyframe_keys[kf] = pose_key;

        Eigen::Matrix4d kf_Twc = kf->Twc();
        Pose3 pose(Rot3(kf_Twc.block<3, 3>(0, 0)), Point3(kf_Twc.block<3, 1>(0, 3)));
        initial_estimates.insert(pose_key, pose);

        if (kf->kid == 0 ||
            std::find(local_frames.begin(), local_frames.end(), kf) == local_frames.end()) {
            auto prior_noise = noiseModel::Isotropic::Sigma(6, kSigmaForFixed);
            graph.add(PriorFactorPose3(pose_key, pose, prior_noise));
        }
    }

    // Add points as graph vertices
    for (auto &p : points) {
        if (!p || p->is_bad()) {
            continue;
        }

        Key point_key = L(p->id);
        point_keys[p] = point_key;
        Eigen::Vector3d pt = p->pt().head<3>();
        initial_estimates.insert(point_key, Point3(pt));

        if (fixed_points) {
            auto prior_noise = noiseModel::Isotropic::Sigma(3, kSigmaForFixed);
            graph.add(PriorFactorPoint3(point_key, Point3(pt), prior_noise));
        }

        // Add measurement factors
        auto observations = p->observations();
        for (const auto &obs : observations) {
            KeyFramePtr kf = obs.first;
            const int idx = obs.second;

            const auto &it = keyframe_keys.find(kf);
            if (it == keyframe_keys.end()) {
                continue;
            }
            Key pose_key = it->second;

            const Eigen::Vector2d kpu = Eigen::Vector2d(kf->kpsu(idx, 0), kf->kpsu(idx, 1));
            const auto &kps_ur = kf->kps_ur;
            const bool is_stereo_obs = (kps_ur.size() > idx && kps_ur[idx] >= 0);

            double sigma = level_sigmas[kf->octaves[idx]];

            // Create noise model
            auto noise_model = boost::make_shared<gtsam_factors::SwitchableRobustNoiseModel>(
                is_stereo_obs ? 3 : 2, sigma, is_stereo_obs ? th_huber_stereo : th_huber_mono);
            noise_model->setRobustModelActive(use_robust_kernel);

            if (is_stereo_obs) {
                auto calib = boost::make_shared<Cal3_S2Stereo>(kf->camera->fx, kf->camera->fy, 0.0,
                                                               kf->camera->cx, kf->camera->cy,
                                                               kf->camera->b);
                StereoPoint2 measurement(kpu.x(), kps_ur[idx], kpu.y());
                auto factor =
                    boost::make_shared<gtsam_factors::WeightedGenericStereoProjectionFactor3D>(
                        measurement, noise_model, pose_key, point_key, calib);

                graph_factors_stereo.push_back(std::make_tuple(factor, noise_model, p, kf, idx));
                graph.add(factor);
            } else {
                auto calib = boost::make_shared<Cal3_S2>(kf->camera->fx, kf->camera->fy, 0.0,
                                                         kf->camera->cx, kf->camera->cy);
                Point2 measurement(kpu.x(), kpu.y());
                auto factor =
                    boost::make_shared<gtsam_factors::WeightedGenericProjectionFactorCal3_S2>(
                        measurement, noise_model, pose_key, point_key, calib);

                graph_factors_mono.push_back(std::make_tuple(factor, noise_model, p, kf, idx));
                graph.add(factor);
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
        return result;
    }

    const double chi2Mono = Parameters::kChi2Mono;
    const double chi2Stereo = Parameters::kChi2Stereo;

    double initial_mean_squared_error = graph.error(initial_estimates) / std::max(num_edges, 1);

    Values current_values = initial_estimates;

    // Robust optimization rounds
    if (robust_rounds > 0) {
        // LevenbergMarquardtParams params;
        auto params = gtsam::LevenbergMarquardtParams().CeresDefaults();
        params.setlambdaInitial(1e-5);         // Matches g2o’s _tau
        params.setlambdaLowerBound(1e-7);      // Prevent over-reduction
        params.setlambdaUpperBound(1e3);       // Prevent excessive increase
        params.setlambdaFactor(2.0);           // Mimics g2o’s adaptive _ni
        params.setDiagonalDamping(true);       // Mimics g2o’s Hessian updates
        params.setUseFixedLambdaFactor(false); // Mimics g2o’s ni

        params.setMaxIterations(robust_rounds);

        if (verbose) {
            params.setVerbosityLM("SUMMARY");
        }

        LevenbergMarquardtOptimizer optimizer(graph, initial_estimates, params);
        current_values = optimizer.optimize();

        // Check mono inliers observation
        for (auto &factor_tuple : graph_factors_mono) {
            auto factor = std::get<0>(factor_tuple);
            auto noise_model = std::get<1>(factor_tuple);
            auto p = std::get<2>(factor_tuple);
            auto kf = std::get<3>(factor_tuple);
            auto idx = std::get<4>(factor_tuple);

            // reset weight to enable back the factor and its correct error computation
            factor->setWeight(1.0);
            // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
            // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
            double chi2 = 2.0 * factor->error(current_values);

            bool chi2_check_failure = (chi2 > chi2Mono);

            if (!chi2_check_failure) {
                Key point_key = point_keys[p];
                Key pose_key = keyframe_keys[kf];
                Point3 point_position = current_values.at<Point3>(point_key);
                Pose3 pose_wc = current_values.at<Pose3>(pose_key);

                // Compute depth check
                Eigen::Matrix4d Twc = pose3_to_matrix(pose_wc);
                Eigen::Matrix4d Tcw = inv_T(Twc);
                Eigen::Vector3d Pc =
                    (Tcw.block<3, 3>(0, 0) * point_position + Tcw.block<3, 1>(0, 3));
                double depth = Pc.z();
                chi2_check_failure = depth <= Parameters::kMinDepth;
            }

            if (chi2_check_failure) {
                num_bad_edges++;
                // Disable the weighted factor
                factor->setWeight(kWeightForDisabledFactor);
            }

            noise_model->setRobustModelActive(false);
        }

        // Check stereo inliers observation
        for (auto &factor_tuple : graph_factors_stereo) {
            auto factor = std::get<0>(factor_tuple);
            auto noise_model = std::get<1>(factor_tuple);
            auto p = std::get<2>(factor_tuple);
            auto kf = std::get<3>(factor_tuple);
            auto idx = std::get<4>(factor_tuple);

            // reset the factor weight to 1.0 to compute a meaningful chi2
            factor->setWeight(1.0);
            // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
            // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
            double chi2 = 2.0 * factor->error(current_values);

            bool chi2_check_failure = (chi2 > chi2Stereo);

            if (!chi2_check_failure) {
                Key point_key = point_keys[p];
                Key pose_key = keyframe_keys[kf];
                Point3 point_position = current_values.at<Point3>(point_key);
                Pose3 pose_wc = current_values.at<Pose3>(pose_key);

                // Compute depth check
                Eigen::Matrix4d Twc = pose3_to_matrix(pose_wc);
                Eigen::Matrix4d Tcw = inv_T(Twc);
                Eigen::Vector3d Pc =
                    (Tcw.block<3, 3>(0, 0) * point_position + Tcw.block<3, 1>(0, 3));
                double depth = Pc.z();
                chi2_check_failure = depth <= Parameters::kMinDepth;
            }

            if (chi2_check_failure) {
                num_bad_edges++;
                // Disable the weighted factor
                factor->setWeight(kWeightForDisabledFactor);
            }

            noise_model->setRobustModelActive(false);
        }
    }

    if (abort_flag && *abort_flag) {
        result.mean_squared_error = -1.0;
        if (fill_result_dict) {
            result.keyframe_updates = {};
            result.point_updates = {};
        }
        return result;
    }

    // Final optimization
    // LevenbergMarquardtParams params;
    auto params = gtsam::LevenbergMarquardtParams().CeresDefaults();
    params.setlambdaInitial(1e-5);         // Matches g2o’s _tau
    params.setlambdaLowerBound(1e-7);      // Prevent over-reduction
    params.setlambdaUpperBound(1e3);       // Prevent excessive increase
    params.setlambdaFactor(2.0);           // Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(true);       // Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(false); // Mimics g2o’s ni

    params.setMaxIterations(final_rounds);

    if (verbose) {
        params.setVerbosityLM("SUMMARY");
    }

    LevenbergMarquardtOptimizer optimizer(graph, current_values, params);
    Values final_values = optimizer.optimize();

    // Store updates
    if (fill_result_dict) {
        result.keyframe_updates = {};
        result.point_updates = {};
    }

    // Put frames back
    for (auto &kf_pair : keyframe_keys) {
        auto &kf = kf_pair.first;
        Key pose_key = kf_pair.second;
        Pose3 pose_estimated = final_values.at<Pose3>(pose_key);
        Eigen::Matrix4d Twc = pose3_to_matrix(pose_estimated);
        Eigen::Matrix4d Tcw = inv_T(Twc);

        if (fill_result_dict) {
            result.keyframe_updates[kf->id] = Tcw;
        } else {
            if (loop_kf_id == 0) {
                kf->update_pose(Tcw);
            } else {
                kf->Tcw_GBA = Tcw;
                kf->is_Tcw_GBA_valid = true;
                kf->GBA_kf_id = loop_kf_id;
            }
        }
    }

    // Put points back
    if (!fixed_points) {
        for (auto &p_pair : point_keys) {
            auto &p = p_pair.first;
            Key point_key = p_pair.second;
            Point3 point_position = final_values.at<Point3>(point_key);
            Eigen::Vector3d new_position = point_position;

            if (fill_result_dict) {
                result.point_updates[p->id] = new_position;
            } else {
                if (loop_kf_id == 0) {
                    p->update_position(new_position);
                    p->update_normal_and_depth(true);
                } else {
                    p->pt_GBA = new_position;
                    p->is_pt_GBA_valid = true;
                    p->GBA_kf_id = loop_kf_id;
                }
            }
        }
    }

    int num_active_edges = num_edges - num_bad_edges;
    result.mean_squared_error = graph.error(final_values) / std::max(num_active_edges, 1);

    if (verbose) {
        std::cout << "bundle_adjustment: mean_squared_error: " << result.mean_squared_error
                  << ", initial_mean_squared_error: " << initial_mean_squared_error
                  << ", num_edges: " << num_edges << ", num_bad_edges: " << num_bad_edges
                  << " (perc: " << (num_bad_edges * 100.0 / std::max(num_edges, 1)) << "%)"
                  << std::endl;
    }

    return result;
}

BundleAdjustmentResult OptimizerGTSAM::global_bundle_adjustment(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points, int rounds,
    int loop_kf_id, bool use_robust_kernel, bool *abort_flag, bool fill_result_dict, bool verbose) {

    const bool fixed_points = false;
    return bundle_adjustment(keyframes, points, std::nullopt, fixed_points, rounds, loop_kf_id,
                             use_robust_kernel, abort_flag, fill_result_dict, verbose);
}

BundleAdjustmentResult
OptimizerGTSAM::global_bundle_adjustment_map(MapPtr &map, int rounds, int loop_kf_id,
                                             bool use_robust_kernel, bool *abort_flag,
                                             bool fill_result_dict, bool verbose) {

    auto keyframes = map->get_keyframes_vector();
    auto points = map->get_points_vector();

    return global_bundle_adjustment(keyframes, points, rounds, loop_kf_id, use_robust_kernel,
                                    abort_flag, fill_result_dict, verbose);
}

// Pose optimization using a helper class (similar to Python version)
class PoseOptimizerGTSAM {
  public:
    FramePtr frame;
    NonlinearFactorGraph graph;
    Values initial;

    std::vector<std::tuple<boost::shared_ptr<gtsam_factors::ResectioningFactor>,
                           boost::shared_ptr<gtsam_factors::SwitchableRobustNoiseModel>, int>>
        mono_factor_tuples;
    std::vector<std::tuple<boost::shared_ptr<gtsam_factors::ResectioningFactorStereo>,
                           boost::shared_ptr<gtsam_factors::SwitchableRobustNoiseModel>, int>>
        stereo_factor_tuples;

    int num_factors = 0;
    bool use_robust_factors;

    boost::shared_ptr<Cal3_S2> K_mono;
    boost::shared_ptr<Cal3_S2Stereo> K_stereo;

    double thHuberMono;
    double thHuberStereo;

    PoseOptimizerGTSAM(FramePtr &frame, bool use_robust_factors = true)
        : frame(frame), use_robust_factors(use_robust_factors) {

        K_mono = boost::make_shared<Cal3_S2>(frame->camera->fx, frame->camera->fy, 0.0,
                                             frame->camera->cx, frame->camera->cy);

        if (frame->camera->b > 0) {
            K_stereo = boost::make_shared<Cal3_S2Stereo>(frame->camera->fx, frame->camera->fy, 0.0,
                                                         frame->camera->cx, frame->camera->cy,
                                                         frame->camera->b);
        }

        thHuberMono = std::sqrt(Parameters::kChi2Mono);
        thHuberStereo = std::sqrt(Parameters::kChi2Stereo);
    }

    void add_pose_node() {
        Eigen::Matrix4d frame_Twc = frame->Twc();
        Pose3 pose_initial(Rot3(frame_Twc.block<3, 3>(0, 0)), Point3(frame_Twc.block<3, 1>(0, 3)));
        initial.insert(X(0), pose_initial);
        // NOTE: there is no need to set a prior here
        // auto noise_prior = noiseModel::Isotropic::Sigma(6, 0.1);
        // graph.add(PriorFactorPose3(X(0), pose_initial, noise_prior));
    }

    void add_observations() {
        const auto &level_sigmas = FeatureSharedResources::level_sigmas;
        auto &outliers = frame->outliers;
        const auto &octaves = frame->octaves;

        // Lock is handled at a higher level or through MapPoint's public interface
        std::lock_guard<std::mutex> lock(MapPoint::global_lock);

        auto frame_points = frame->get_points();
        for (size_t idx = 0; idx < frame_points.size(); ++idx) {
            const auto &p = frame_points[idx];
            if (!p) {
                continue;
            }

            const Eigen::Vector2d kpu = Eigen::Vector2d(frame->kpsu(idx, 0), frame->kpsu(idx, 1));
            const auto &kps_ur = frame->kps_ur;
            const bool is_stereo_obs = (kps_ur.size() > idx && kps_ur[idx] >= 0);

            // Reset outlier flag
            outliers[idx] = false;

            double sigma = level_sigmas[octaves[idx]];

            auto noise_model = boost::make_shared<gtsam_factors::SwitchableRobustNoiseModel>(
                is_stereo_obs ? 3 : 2, sigma, is_stereo_obs ? thHuberStereo : thHuberMono);

            if (!use_robust_factors) {
                noise_model->setRobustModelActive(false);
            }

            gtsam::NonlinearFactor::shared_ptr factor;
            if (is_stereo_obs) {
                StereoPoint2 measurement(kpu.x(), kps_ur[idx], kpu.y());
                auto factor = boost::make_shared<gtsam_factors::ResectioningFactorStereo>(
                    noise_model, X(0), *K_stereo, measurement, Point3(p->pt()));
                stereo_factor_tuples.push_back({factor, noise_model, static_cast<int>(idx)});
                graph.add(factor);
            } else {
                Point2 measurement(kpu.x(), kpu.y());
                auto factor = boost::make_shared<gtsam_factors::ResectioningFactor>(
                    noise_model, X(0), *K_mono, measurement, Point3(p->pt()));
                mono_factor_tuples.push_back({factor, noise_model, static_cast<int>(idx)});
                graph.add(factor);
            }

            num_factors++;
        }
    }

    PoseOptimizationResult optimize(int rounds, bool verbose) {
        PoseOptimizationResult result;

        if (num_factors < 3) {
            MSG_RED_WARN("pose_optimization: not enough correspondences!");
            result.is_ok = false;
            return result;
        }

        const double chi2Mono = Parameters::kChi2Mono;
        const double chi2Stereo = Parameters::kChi2Stereo;

        // auto params = gtsam.LevenbergMarquardtParams().LegacyDefaults() # default ones
        auto params = LevenbergMarquardtParams().CeresDefaults();
        params.setlambdaInitial(1e-5);         // Matches g2o’s _tau
        params.setlambdaLowerBound(1e-8);      // Prevent over-reduction
        params.setlambdaUpperBound(1e3);       // Prevent excessive increase
        params.setlambdaFactor(2.0);           // Mimics g2o’s adaptive _ni
        params.setDiagonalDamping(true);       // Mimics g2o’s Hessian updates
        params.setUseFixedLambdaFactor(false); // Mimics g2o’s ni

        params.setMaxIterations(rounds);

        if (verbose) {
            params.setVerbosityLM("SUMMARY");
        }

        Values current_values = initial;
        double cost_prev = std::numeric_limits<double>::infinity();
        double cost = std::numeric_limits<double>::infinity();
        int num_inliers = 0;
        double total_inlier_error = 0.0;
        int num_bad_point_edges = 0;

        for (int it = 0; it < 4; ++it) {

            // LevenbergMarquardtOptimizer optimizer(graph, current_values, params);
            gtsam_factors::LevenbergMarquardtOptimizerG2o optimizer(graph, current_values, params);

            current_values = optimizer.optimize();

            cost_prev = cost;
            double cost = graph.error(current_values);

            num_bad_point_edges = 0;
            total_inlier_error = 0.0;
            num_inliers = 0;
            int num_edges = 0;

            for (auto &[factor, noise_model, idx] : mono_factor_tuples) {
                // reset weight to enable back the factor and its correct error computation
                factor->setWeight(1.0);
                // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
                // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
                double chi2 = 2.0 * factor->error(current_values);

                bool chi2_check_failure = (chi2 > chi2Mono);
                if (chi2_check_failure) {
                    frame->outliers[idx] = true;
                    factor->setWeight(kWeightForDisabledFactor); // disable the factor
                    num_bad_point_edges++;
                } else {
                    frame->outliers[idx] = false;
                    total_inlier_error += chi2; // Sum error only for inliers
                    num_inliers++;
                }

                if (it == 2) {
                    noise_model->setRobustModelActive(false);
                }

                num_edges++;
            }

            for (auto &[factor, noise_model, idx] : stereo_factor_tuples) {
                // reset weight to enable back the factor and its correct error computation
                factor->setWeight(1.0);
                // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
                // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
                double chi2 = 2.0 * factor->error(current_values);

                bool chi2_check_failure = (chi2 > chi2Stereo);
                if (chi2_check_failure) {
                    frame->outliers[idx] = true;
                    factor->setWeight(kWeightForDisabledFactor); // disable the factor
                    num_bad_point_edges++;
                } else {
                    frame->outliers[idx] = false;
                    total_inlier_error += chi2; // Sum error only for inliers
                    num_inliers++;
                }

                if (it == 2) {
                    noise_model->setRobustModelActive(false);
                }

                num_edges++;
            }

            if (num_edges < 10) {
                MSG_RED_WARN("pose_optimization: stopped - not enough edges!");
                break;
            }

            initial = current_values; // Use current result for next iteration
        }

        std::cout << "pose_optimization: available " << num_factors << " points, found "
                  << num_bad_point_edges << " bad points" << std::endl;

        int num_valid_points = num_factors - (num_factors - num_inliers);
        if (num_valid_points < 10) {
            MSG_RED_WARN("pose_optimization stopped - not enough edges!");
            result.is_ok = false;
            return result;
        }

        double ratio_bad_points =
            static_cast<double>(num_factors - num_inliers) / std::max(num_factors, 1);
        if (num_valid_points > 15 &&
            ratio_bad_points > Parameters::kMaxOutliersRatioInPoseOptimization) {
            MSG_RED_WARN_STREAM("pose_optimization: percentage of bad points is too high: "
                                << ratio_bad_points * 100.0 << "%");
            result.is_ok = false;
            return result;
        }

        // Update pose estimation
        if (current_values.exists(X(0))) {
            Pose3 pose_estimated = current_values.at<Pose3>(X(0));
            Eigen::Matrix4d Twc = OptimizerGTSAM::pose3_to_matrix(pose_estimated);
            frame->update_pose(inv_T(Twc));
            result.is_ok = true;
        } else {
            result.is_ok = false;
        }

        result.mean_squared_error = total_inlier_error / std::max(num_inliers, 1);
        result.num_valid_points = num_valid_points;

        return result;
    }
};

PoseOptimizationResult OptimizerGTSAM::pose_optimization(FramePtr &frame, bool verbose,
                                                         int rounds) {
    PoseOptimizerGTSAM optimizer(frame, true);
    optimizer.add_pose_node();
    optimizer.add_observations();
    return optimizer.optimize(rounds, verbose);
}

// Local bundle adjustment implementation
template <typename LockType>
std::pair<double, double> OptimizerGTSAM::local_bundle_adjustment(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    const std::vector<KeyFramePtr> &keyframes_ref, bool fixed_points, bool verbose, int rounds,
    bool *abort_flag, LockType *map_lock) {

    NonlinearFactorGraph graph;
    Values initial_estimates;

    const auto &level_sigmas = FeatureSharedResources::level_sigmas;

    const double th_huber_mono = std::sqrt(Parameters::kChi2Mono);
    const double th_huber_stereo = std::sqrt(Parameters::kChi2Stereo);

    std::vector<KeyFramePtr> good_keyframes;
    for (auto &kf : keyframes) {
        if (kf && !kf->is_bad()) {
            good_keyframes.push_back(kf);
        }
    }
    for (auto &kf : keyframes_ref) {
        if (kf && !kf->is_bad()) {
            good_keyframes.push_back(kf);
        }
    }

    std::unordered_map<KeyFramePtr, Key> keyframe_keys;
    std::unordered_map<MapPointPtr, Key> point_keys;
    // Use vector instead of unordered_map for graph_factors since pair doesn't have hash
    std::vector<std::tuple<boost::shared_ptr<gtsam_factors::WeightedGenericProjectionFactorCal3_S2>,
                           boost::shared_ptr<gtsam_factors::SwitchableRobustNoiseModel>,
                           MapPointPtr, KeyFramePtr, int>>
        graph_factors_mono;
    std::vector<
        std::tuple<boost::shared_ptr<gtsam_factors::WeightedGenericStereoProjectionFactor3D>,
                   boost::shared_ptr<gtsam_factors::SwitchableRobustNoiseModel>, MapPointPtr,
                   KeyFramePtr, int>>
        graph_factors_stereo;

    int num_edges = 0;
    int num_bad_edges = 0;

    // Add keyframe vertices
    for (auto &kf : good_keyframes) {
        Key pose_key = X(kf->kid);
        keyframe_keys[kf] = pose_key;

        Eigen::Matrix4d kf_Twc = kf->Twc();
        Pose3 pose(Rot3(kf_Twc.block<3, 3>(0, 0)), Point3(kf_Twc.block<3, 1>(0, 3)));
        initial_estimates.insert(pose_key, pose);

        if (kf->kid == 0 ||
            std::find(keyframes_ref.begin(), keyframes_ref.end(), kf) != keyframes_ref.end()) {
            auto prior_noise = noiseModel::Isotropic::Sigma(6, kSigmaForFixed);
            graph.add(PriorFactorPose3(pose_key, pose, prior_noise));
        }
    }

    // Add point vertices and edges
    for (auto &p : points) {
        if (!p || p->is_bad()) {
            continue;
        }
        Key point_key = L(p->id);
        point_keys[p] = point_key;
        Eigen::Vector3d pt = p->pt().head<3>();
        initial_estimates.insert(point_key, Point3(pt));

        if (fixed_points) {
            auto prior_noise = noiseModel::Isotropic::Sigma(3, kSigmaForFixed);
            graph.add(PriorFactorPoint3(point_key, Point3(pt), prior_noise));
        }

        auto observations = p->observations();
        for (const auto &obs : observations) {
            KeyFramePtr kf = obs.first;
            const int p_idx = obs.second;

            if (!kf || kf->is_bad()) {
                continue;
            }

            const auto &it = keyframe_keys.find(kf);
            if (it == keyframe_keys.end()) {
                continue;
            }

            assert(kf->get_point_match(p_idx) == p);

            const Eigen::Vector2d kpu = Eigen::Vector2d(kf->kpsu(p_idx, 0), kf->kpsu(p_idx, 1));
            const auto &kps_ur = kf->kps_ur;
            const bool is_stereo_obs = (kps_ur.size() > p_idx && kps_ur[p_idx] >= 0);

            double sigma = level_sigmas[kf->octaves[p_idx]];

            auto noise_model = boost::make_shared<gtsam_factors::SwitchableRobustNoiseModel>(
                is_stereo_obs ? 3 : 2, sigma, is_stereo_obs ? th_huber_stereo : th_huber_mono);

            Key pose_key = it->second;

            if (is_stereo_obs) {
                auto calib = boost::make_shared<Cal3_S2Stereo>(kf->camera->fx, kf->camera->fy, 0.0,
                                                               kf->camera->cx, kf->camera->cy,
                                                               kf->camera->b);
                StereoPoint2 measurement(kpu.x(), kps_ur[p_idx], kpu.y());
                auto factor =
                    boost::make_shared<gtsam_factors::WeightedGenericStereoProjectionFactor3D>(
                        measurement, noise_model, pose_key, point_key, calib);
                graph_factors_stereo.push_back(std::make_tuple(factor, noise_model, p, kf, p_idx));
                graph.add(factor);
            } else {
                auto calib = boost::make_shared<Cal3_S2>(kf->camera->fx, kf->camera->fy, 0.0,
                                                         kf->camera->cx, kf->camera->cy);
                Point2 measurement(kpu.x(), kpu.y());
                auto factor =
                    boost::make_shared<gtsam_factors::WeightedGenericProjectionFactorCal3_S2>(
                        measurement, noise_model, pose_key, point_key, calib);
                graph_factors_mono.push_back(std::make_tuple(factor, noise_model, p, kf, p_idx));
                graph.add(factor);
            }
            num_edges++;
        }
    }

    if (abort_flag && *abort_flag) {
        return {-1.0, 0.0};
    }

    const double chi2Mono = Parameters::kChi2Mono;
    const double chi2Stereo = Parameters::kChi2Stereo;

    // Initial optimization
    // LevenbergMarquardtParams params;
    auto params = gtsam::LevenbergMarquardtParams().CeresDefaults();
    params.setlambdaInitial(1e-5);         // Matches g2o’s _tau
    params.setlambdaLowerBound(1e-7);      // Prevent over-reduction
    params.setlambdaUpperBound(1e3);       // Prevent excessive increase
    params.setlambdaFactor(2.0);           // Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(true);       // Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(false); // Mimics g2o’s ni

    params.setMaxIterations(5);

    if (verbose) {
        params.setVerbosityLM("SUMMARY");
    }

    LevenbergMarquardtOptimizer optimizer(graph, initial_estimates, params);
    Values result = optimizer.optimize();

    if (!abort_flag || !*abort_flag) {

        // Check mono inliers
        for (auto &factor_tuple : graph_factors_mono) {
            auto factor = std::get<0>(factor_tuple);
            auto noise_model = std::get<1>(factor_tuple);
            auto p = std::get<2>(factor_tuple);
            auto kf = std::get<3>(factor_tuple);
            auto p_idx = std::get<4>(factor_tuple);

            // reset the factor weight to 1.0 to compute a meaningful chi2
            factor->setWeight(1.0);
            // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
            // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
            double chi2 = 2.0 * factor->error(result);

            bool chi2_check_failure = (chi2 > chi2Mono);

            if (!chi2_check_failure) {
                Key point_key = point_keys[p];
                Key pose_key = keyframe_keys[kf];
                Point3 point_position = result.at<Point3>(point_key);
                Pose3 pose_wc = result.at<Pose3>(pose_key);

                Eigen::Matrix4d Twc = pose3_to_matrix(pose_wc);
                Eigen::Matrix4d Tcw = inv_T(Twc);
                Eigen::Vector3d Pc =
                    (Tcw.block<3, 3>(0, 0) * point_position + Tcw.block<3, 1>(0, 3));
                double depth = Pc.z();
                chi2_check_failure = depth <= Parameters::kMinDepth;
            }

            if (chi2_check_failure) {
                num_bad_edges++;
                factor->setWeight(kWeightForDisabledFactor); // disable the factor
            } else {
                noise_model->setRobustModelActive(false);
            }
        }

        // Check stereo inliers
        for (auto &factor_tuple : graph_factors_stereo) {
            auto factor = std::get<0>(factor_tuple);
            auto noise_model = std::get<1>(factor_tuple);
            auto p = std::get<2>(factor_tuple);
            auto kf = std::get<3>(factor_tuple);
            auto p_idx = std::get<4>(factor_tuple);

            // reset the factor weight to 1.0 to compute a meaningful chi2
            factor->setWeight(1.0);
            // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
            // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
            double chi2 = 2.0 * factor->error(result);

            bool chi2_check_failure = (chi2 > chi2Stereo);

            if (!chi2_check_failure) {
                Key point_key = point_keys[p];
                Key pose_key = keyframe_keys[kf];
                Point3 point_position = result.at<Point3>(point_key);
                Pose3 pose_wc = result.at<Pose3>(pose_key);

                Eigen::Matrix4d Twc = pose3_to_matrix(pose_wc);
                Eigen::Matrix4d Tcw = inv_T(Twc);
                Eigen::Vector3d Pc =
                    (Tcw.block<3, 3>(0, 0) * point_position + Tcw.block<3, 1>(0, 3));
                double depth = Pc.z();
                chi2_check_failure = depth <= Parameters::kMinDepth;
            }

            if (chi2_check_failure) {
                num_bad_edges++;
                factor->setWeight(kWeightForDisabledFactor); // disable the factor
            } else {
                noise_model->setRobustModelActive(false);
            }
        }

        // Optimize again without outliers
        params.setMaxIterations(rounds);
        LevenbergMarquardtOptimizer optimizer2(graph, result, params);
        result = optimizer2.optimize();
    }

    // Final outlier check
    int num_bad_observations = 0;
    std::vector<std::tuple<MapPointPtr, KeyFramePtr, int, bool>> outliers_factors_data;

    double total_error = 0.0;
    int num_inlier_observations = 0;

    for (auto &factor_tuple : graph_factors_mono) {
        const bool is_stereo = false;
        auto factor = std::get<0>(factor_tuple);
        auto p = std::get<2>(factor_tuple);
        auto kf = std::get<3>(factor_tuple);
        auto p_idx = std::get<4>(factor_tuple);

        assert(kf->get_point_match(p_idx) == p);

        // reset the factor weight to 1.0 to compute a meaningful chi2
        factor->setWeight(1.0);
        // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
        // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
        double chi2 = 2.0 * factor->error(result);
        bool chi2_check_failure = (chi2 > chi2Mono);

        if (!chi2_check_failure) {
            Key point_key = point_keys[p];
            Key pose_key = keyframe_keys[kf];
            Point3 point_position = result.at<Point3>(point_key);
            Pose3 pose_wc = result.at<Pose3>(pose_key);

            Eigen::Matrix4d Twc = pose3_to_matrix(pose_wc);
            Eigen::Matrix4d Tcw = inv_T(Twc);
            Eigen::Vector3d Pc = (Tcw.block<3, 3>(0, 0) * point_position + Tcw.block<3, 1>(0, 3));
            double depth = Pc.z();
            chi2_check_failure = depth <= Parameters::kMinDepth;
        }

        if (chi2_check_failure) {
            num_bad_observations++;
            outliers_factors_data.push_back({p, kf, p_idx, is_stereo});
        } else {
            num_inlier_observations++;
            total_error += chi2;
        }
    }

    for (auto &factor_tuple : graph_factors_stereo) {
        const bool is_stereo = true;
        auto factor = std::get<0>(factor_tuple);
        auto p = std::get<2>(factor_tuple);
        auto kf = std::get<3>(factor_tuple);
        auto p_idx = std::get<4>(factor_tuple);

        assert(kf->get_point_match(p_idx) == p);

        // reset the factor weight to 1.0 to compute a meaningful chi2
        factor->setWeight(1.0);
        // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
        // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
        double chi2 = 2.0 * factor->error(result);

        bool chi2_check_failure = (chi2 > chi2Stereo);

        if (!chi2_check_failure) {
            Key point_key = point_keys[p];
            Key pose_key = keyframe_keys[kf];
            Point3 point_position = result.at<Point3>(point_key);
            Pose3 pose_wc = result.at<Pose3>(pose_key);

            Eigen::Matrix4d Twc = pose3_to_matrix(pose_wc);
            Eigen::Matrix4d Tcw = inv_T(Twc);
            Eigen::Vector3d Pc = (Tcw.block<3, 3>(0, 0) * point_position + Tcw.block<3, 1>(0, 3));
            double depth = Pc.z();
            chi2_check_failure = depth <= Parameters::kMinDepth;
        }

        if (chi2_check_failure) {
            num_bad_observations++;
            outliers_factors_data.push_back({p, kf, p_idx, is_stereo});
        } else {
            num_inlier_observations++;
            total_error += chi2;
        }
    }

    // Apply updates with map lock
    std::unique_ptr<pyslam::PyLockGuard<LockType>> lock;
    if (map_lock) {
        lock = std::make_unique<pyslam::PyLockGuard<LockType>>(map_lock);
    }
    const bool map_no_lock = (map_lock != nullptr);

    // Remove outlier observations
    for (const auto &[p, kf, p_idx, is_stereo] : outliers_factors_data) {
        const auto &p_f = kf->get_point_match(p_idx);
        if (p_f) {
            assert(p_f == p);
            p->remove_observation(kf, p_idx, map_no_lock);
            // the following instruction is now included in p.remove_observation()
            // f.remove_point(p)   # it removes multiple point instances (if these are present)
            // f.remove_point_match(p_idx) # this does not remove multiple point instances, but now
            // there cannot be multiple instances any more
        }
    }

    // Update keyframe poses
    for (auto &kf_pair : keyframe_keys) {
        auto &kf = kf_pair.first;
        Key pose_key = kf_pair.second;
        Pose3 pose_estimated = result.at<Pose3>(pose_key);
        Eigen::Matrix4d Twc = pose3_to_matrix(pose_estimated);
        kf->update_pose(inv_T(Twc));
        kf->lba_count++;
    }

    // Update point positions
    if (!fixed_points) {
        for (auto &p_pair : point_keys) {
            auto &p = p_pair.first;
            Key point_key = p_pair.second;
            Point3 point_position = result.at<Point3>(point_key);
            p->update_position(Eigen::Vector3d(point_position));
            p->update_normal_and_depth(true);
        }
    }

    int num_active_edges = num_inlier_observations;
    // double mean_squared_error = graph.error(result) / std::max(num_active_edges, 1);
    double mean_squared_error = total_error / std::max(num_active_edges, 1);
    double outlier_ratio = static_cast<double>(num_bad_observations) / std::max(1, num_edges);

    return {mean_squared_error, outlier_ratio};
}

// Explicit template instantiation
template std::pair<double, double> OptimizerGTSAM::local_bundle_adjustment<pyslam::MapMutex>(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    const std::vector<KeyFramePtr> &keyframes_ref, bool fixed_points, bool verbose, int rounds,
    bool *abort_flag, pyslam::MapMutex *map_lock);

#ifdef USE_PYTHON
template std::pair<double, double> OptimizerGTSAM::local_bundle_adjustment<pyslam::PyLock>(
    const std::vector<KeyFramePtr> &keyframes, const std::vector<MapPointPtr> &points,
    const std::vector<KeyFramePtr> &keyframes_ref, bool fixed_points, bool verbose, int rounds,
    bool *abort_flag, pyslam::PyLock *map_lock);
#endif

// Sim3 optimization
Sim3OptimizationResult OptimizerGTSAM::optimize_sim3(
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

    Cal3_S2 K1_mono(cam1->fx, cam1->fy, 0, cam1->cx, cam1->cy);
    Cal3_S2 K2_mono(cam2->fx, cam2->fy, 0, cam2->cx, cam2->cy);

    Eigen::Matrix4d kf1_Tcw = kf1->Tcw();
    Eigen::Matrix4d kf2_Tcw = kf2->Tcw();
    Eigen::Matrix3d R1w = kf1_Tcw.block<3, 3>(0, 0);
    Eigen::Vector3d t1w = kf1_Tcw.block<3, 1>(0, 3);
    Eigen::Matrix3d R2w = kf2_Tcw.block<3, 3>(0, 0);
    Eigen::Vector3d t2w = kf2_Tcw.block<3, 1>(0, 3);

    NonlinearFactorGraph graph;
    Values initial_estimate;

    // Initial Sim3 transformation
    Similarity3 sim3_init(Rot3(R12), Point3(t12), s12);
    insertSimilarity3(initial_estimate, X(0), sim3_init);

    if (fix_scale) {
        auto scale_prior = boost::make_shared<gtsam_factors::PriorFactorSimilarity3ScaleOnly>(
            X(0), s12, kSigmaForFixed);
        graph.add(scale_prior);
    }

    std::vector<MapPointPtr> actual_map_points1 = map_points1;
    if (actual_map_points1.empty()) {
        actual_map_points1 = kf1->get_points();
    }

    int num_matches = map_point_matches12.size();
    assert(static_cast<int>(actual_map_points1.size()) == num_matches);

    std::vector<
        std::tuple<boost::shared_ptr<gtsam_factors::SimResectioningFactor>, Eigen::Vector3d>>
        factors_12_data;
    std::vector<
        std::tuple<boost::shared_ptr<gtsam_factors::SimInvResectioningFactor>, Eigen::Vector3d>>
        factors_21_data;
    std::vector<int> match_idxs;

    double delta_huber = std::sqrt(th2);
    const auto &level_sigmas = FeatureSharedResources::level_sigmas;

    int num_correspondences = 0;
    for (int i = 0; i < num_matches; ++i) {
        auto mp1 = actual_map_points1[i];
        if (!mp1 || mp1->is_bad()) {
            continue;
        }
        auto mp2 = map_point_matches12[i];
        if (!mp2 || mp2->is_bad()) {
            continue;
        }

        int index2 = mp2->get_observation_idx(kf2);
        if (index2 >= 0) {
            double sigma2_12 = level_sigmas[kf1->octaves[i]];
            auto robust_noise_12 =
                noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(delta_huber),
                                           noiseModel::Isotropic::Sigma(2, sigma2_12));

            double sigma2_21 = level_sigmas[kf2->octaves[index2]];
            auto robust_noise_21 =
                noiseModel::Robust::Create(noiseModel::mEstimator::Huber::Create(delta_huber),
                                           noiseModel::Isotropic::Sigma(2, sigma2_21));

            // Factor 12
            Eigen::Vector3d p2_c2 = R2w * mp2->pt() + t2w;
            auto factor_12 = boost::make_shared<gtsam_factors::SimResectioningFactor>(
                X(0), K1_mono, Point2(kf1->kpsu(i, 0), kf1->kpsu(i, 1)), Point3(p2_c2),
                robust_noise_12);
            graph.add(factor_12);

            // Factor 21
            Eigen::Vector3d p1_c1 = R1w * mp1->pt() + t1w;
            auto factor_21 = boost::make_shared<gtsam_factors::SimInvResectioningFactor>(
                X(0), K2_mono, Point2(kf2->kpsu(index2, 0), kf2->kpsu(index2, 1)), Point3(p1_c1),
                robust_noise_21);
            graph.add(factor_21);

            factors_12_data.push_back({factor_12, p2_c2});
            factors_21_data.push_back({factor_21, p1_c1});
            match_idxs.push_back(i);
            num_correspondences++;
        }
    }

    if (verbose) {
        std::cout << "optimize_sim3: num_correspondences = " << num_correspondences << std::endl;
    }

    // if (num_correspondences < 10) {
    //     if (verbose) {
    //         std::cout << "optimize_sim3: Too few inliers, num_correspondences = "
    //                   << num_correspondences << std::endl;
    //     }
    //     return result;
    // }

    double initial_error = graph.error(initial_estimate);

    // Optimizer
    auto params = LevenbergMarquardtParams().CeresDefaults();
    params.setlambdaInitial(1e-5);         // Matches g2o’s _tau
    params.setlambdaLowerBound(1e-7);      // Prevent over-reduction
    params.setlambdaUpperBound(1e3);       // Prevent excessive increase
    params.setlambdaFactor(2.0);           // Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(true);       // Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(false); // Mimics g2o’s ni

    params.setMaxIterations(5);
    if (verbose) {
        params.setVerbosityLM("SUMMARY");
    }

    LevenbergMarquardtOptimizer optimizer(graph, initial_estimate, params);
    Values opt_result = optimizer.optimize();

    Similarity3 sim3_optimized = getSimilarity3(opt_result, X(0));
    Eigen::Matrix3d R12_opt = sim3_optimized.rotation().matrix();
    Eigen::Vector3d t12_opt = sim3_optimized.translation();
    double s12_opt = sim3_optimized.scale();

    Eigen::Matrix3d R21_opt = R12_opt.transpose() / s12_opt;
    Eigen::Vector3d t21_opt = -R21_opt * t12_opt;

    // Check inliers
    int num_bad = 0;
    for (size_t i = 0; i < factors_12_data.size(); ++i) {
        auto &[factor_12, p2_c2] = factors_12_data[i];
        auto &[factor_21, p1_c1] = factors_21_data[i];

        Eigen::Vector3d p2_c1 = s12_opt * R12_opt * p2_c2 + t12_opt;
        Eigen::Vector3d p1_c2 = R21_opt * p1_c1 + t21_opt;

        // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
        // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
        double chi2_12 = 2.0 * factor_12->error(opt_result);
        double chi2_21 = 2.0 * factor_21->error(opt_result);

        if (chi2_12 > th2 || p2_c1.z() <= 0 || chi2_21 > th2 || p1_c2.z() <= 0) {
            int index = match_idxs[i];
            actual_map_points1[index] = nullptr;
            factor_12->setWeight(kWeightForDisabledFactor);
            factor_21->setWeight(kWeightForDisabledFactor);
            factors_12_data[i] = {nullptr, Eigen::Vector3d::Zero()};
            factors_21_data[i] = {nullptr, Eigen::Vector3d::Zero()};
            num_bad++;
        }
    }

    if (num_correspondences - num_bad < 10) {
        if (verbose) {
            std::cout << "optimize_sim3: Too few inliers, num_correspondences = "
                      << num_correspondences << ", num_bad = " << num_bad << std::endl;
        }
        return result;
    }

    int num_more_iterations = (num_bad > 0) ? 10 : 5;
    params.setMaxIterations(num_more_iterations);
    LevenbergMarquardtOptimizer optimizer2(graph, opt_result, params);
    opt_result = optimizer2.optimize();

    double delta_err = graph.error(opt_result) - initial_error;

    sim3_optimized = getSimilarity3(opt_result, X(0));
    R12_opt = sim3_optimized.rotation().matrix();
    t12_opt = sim3_optimized.translation();
    s12_opt = sim3_optimized.scale();

    R21_opt = R12_opt.transpose() / s12_opt;
    t21_opt = -R21_opt * t12_opt;

    int num_inliers = 0;
    for (size_t i = 0; i < factors_12_data.size(); ++i) {
        auto &[factor_12, p2_c2] = factors_12_data[i];
        auto &[factor_21, p1_c1] = factors_21_data[i];

        if (!factor_12 || !factor_21) {
            continue;
        }

        Eigen::Vector3d p2_c1 = s12_opt * R12_opt * p2_c2 + t12_opt;
        Eigen::Vector3d p1_c2 = R21_opt * p1_c1 + t21_opt;

        // reset weight to 1.0 to activate the factor and correctly compute the error
        factor_12->setWeight(1.0);
        // reset weight to 1.0 to activate the factor and correctly compute the error
        factor_21->setWeight(1.0);

        // from the gtsam code comments, error() is typically equal to log-likelihood, e.g.
        // 0.5*(h(x)-z)^2/sigma^2  in case of Gaussian.
        double chi2_12 = 2.0 * factor_12->error(opt_result);
        double chi2_21 = 2.0 * factor_21->error(opt_result);

        if (chi2_12 > th2 || p2_c1.z() <= 0 || chi2_21 > th2 || p1_c2.z() <= 0) {
            int index = match_idxs[i];
            actual_map_points1[index] = nullptr;
        } else {
            num_inliers++;
        }
    }

    Similarity3 sim3_final = getSimilarity3(opt_result, X(0));
    double scale_out = fix_scale ? s12 : sim3_final.scale();

    result.num_inliers = num_inliers;
    result.R = sim3_final.rotation().matrix();
    result.t = sim3_final.translation();
    result.scale = scale_out;
    result.delta_error = delta_err;

    return result;
}

// Essential graph optimization
double OptimizerGTSAM::optimize_essential_graph(
    MapPtr map_object, KeyFramePtr loop_keyframe, KeyFramePtr current_keyframe,
    const std::unordered_map<KeyFramePtr, Sim3Pose> &non_corrected_sim3_map,
    const std::unordered_map<KeyFramePtr, Sim3Pose> &corrected_sim3_map,
    const std::unordered_map<KeyFramePtr, std::vector<KeyFramePtr>> &loop_connections,
    bool fix_scale, bool verbose) {

    NonlinearFactorGraph graph;
    Values initial_values;

    const double sigma_for_fixed = kSigmaForFixed;
    const double sigma_for_visual = 1.0;

    auto all_keyframes = map_object->get_keyframes_vector();
    auto all_map_points = map_object->get_points_vector();
    int max_keyframe_id = map_object->max_keyframe_id;

    std::vector<bool> vec_Scw_is_valid(max_keyframe_id + 1, false);
    std::vector<Sim3Pose> vec_Scw(max_keyframe_id + 1);
    std::vector<Sim3Pose> vec_corrected_Swc(max_keyframe_id + 1);

    const int min_number_features = 100;

    // Set KeyFrame initial values
    for (auto &keyframe : all_keyframes) {
        if (!keyframe || keyframe->is_bad()) {
            continue;
        }

        int keyframe_id = keyframe->kid;

        Sim3Pose Siw;
        auto it_kf_corrected_sim3 = corrected_sim3_map.find(keyframe);
        if (it_kf_corrected_sim3 != corrected_sim3_map.end()) {
            const auto &corrected_sim3 = it_kf_corrected_sim3->second;
            Siw = Sim3Pose(corrected_sim3.R(), corrected_sim3.t(), corrected_sim3.s());
        } else {
            Eigen::Matrix4d kf_Tcw = keyframe->Tcw();
            Siw = Sim3Pose(kf_Tcw.block<3, 3>(0, 0), kf_Tcw.block<3, 1>(0, 3), 1.0);
        }
        vec_Scw[keyframe_id] = Siw;
        vec_Scw_is_valid[keyframe_id] = true;

        Similarity3 Siw_gtsam(Rot3(Siw.R()), Point3(Siw.t()), Siw.s());
        insertSimilarity3(initial_values, X(keyframe_id), Siw_gtsam);

        if (keyframe == loop_keyframe) {
            auto fixed_sim3_prior = boost::make_shared<gtsam_factors::PriorFactorSimilarity3>(
                X(keyframe_id), Siw_gtsam, noiseModel::Isotropic::Sigma(7, sigma_for_fixed));
            graph.add(fixed_sim3_prior);
        }

        if (fix_scale) {
            auto fixed_scale_prior =
                boost::make_shared<gtsam_factors::PriorFactorSimilarity3ScaleOnly>(
                    X(keyframe_id), Siw.s(), sigma_for_fixed);
            graph.add(fixed_scale_prior);
        }
    }

    int num_graph_edges = 0;
    std::set<std::pair<int, int>> inserted_loop_edges;

    // Loop edges
    for (const auto &[keyframe, connections] : loop_connections) {
        const int keyframe_id = keyframe->kid;
        if (keyframe_id >= vec_Scw.size()) {
            std::cerr << "[optimize_essential_graph] keyframe_id " << keyframe_id
                      << " is out of bounds" << std::endl;
            continue;
        }
        if (!vec_Scw_is_valid[keyframe_id]) {
            MSG_RED_WARN_STREAM("optimize_essential_graph: SiW for keyframe " << keyframe_id
                                                                              << " is not valid");
            continue;
        }

        const Sim3Pose &Siw = vec_Scw[keyframe_id];
        const Sim3Pose Swi = Siw.inverse();

        for (auto &connected_keyframe : connections) {
            const int connected_id = connected_keyframe->kid;
            // accept (current_keyframe,loop_keyframe)
            // and all the other loop edges with weight >= min_number_features
            if ((keyframe_id != current_keyframe->kid || connected_id != loop_keyframe->kid) &&
                keyframe->get_weight(connected_keyframe) < min_number_features) {
                continue;
            }

            const Sim3Pose &Sjw = vec_Scw[connected_id];
            const Sim3Pose Sji = Sjw * Swi;

            Similarity3 Sji_gtsam(Rot3(Sji.R()), Point3(Sji.t()), Sji.s());

            auto edge = boost::make_shared<gtsam_factors::BetweenFactorSimilarity3Inverse>(
                X(connected_id), X(keyframe_id), Sji_gtsam,
                noiseModel::Isotropic::Sigma(7, sigma_for_visual));
            graph.add(edge);
            num_graph_edges++;
            inserted_loop_edges.insert(
                {std::min(keyframe_id, connected_id), std::max(keyframe_id, connected_id)});
        }
    }

    // Normal edges (spanning tree, loop edges, covisibility)
    for (auto &keyframe : all_keyframes) {
        if (!keyframe) { // || keyframe->is_bad()) {
            continue;
        }
        const int keyframe_id = keyframe->kid;

        Sim3Pose Swi;
        auto it_kf_non_corrected_sim3 = non_corrected_sim3_map.find(keyframe);
        if (it_kf_non_corrected_sim3 != non_corrected_sim3_map.end()) {
            Swi = it_kf_non_corrected_sim3->second.inverse();
        } else {
            Swi = vec_Scw[keyframe_id].inverse();
        }

        // Spanning tree edge
        auto parent_keyframe = keyframe->get_parent();
        if (parent_keyframe) {
            const int parent_id = parent_keyframe->kid;

            Sim3Pose Sjw;
            auto it_parent_non_corrected_sim3 = non_corrected_sim3_map.find(parent_keyframe);
            if (it_parent_non_corrected_sim3 != non_corrected_sim3_map.end()) {
                Sjw = it_parent_non_corrected_sim3->second;
            } else {
                Sjw = vec_Scw[parent_id];
            }

            const Sim3Pose Sji = Sjw * Swi;

            Similarity3 Sji_gtsam(Rot3(Sji.R()), Point3(Sji.t()), Sji.s());

            auto edge = boost::make_shared<gtsam_factors::BetweenFactorSimilarity3Inverse>(
                X(parent_id), X(keyframe_id), Sji_gtsam,
                noiseModel::Isotropic::Sigma(7, sigma_for_visual));
            graph.add(edge);
            num_graph_edges++;
        }

        // Loop edges
        for (auto &loop_edge : keyframe->get_loop_edges()) {
            if (loop_edge->kid < keyframe_id) {
                Sim3Pose Slw;
                const auto it_non_corrected_sim3_loop_edge = non_corrected_sim3_map.find(loop_edge);
                if (it_non_corrected_sim3_loop_edge != non_corrected_sim3_map.end()) {
                    Slw = it_non_corrected_sim3_loop_edge->second;
                } else {
                    Slw = vec_Scw[loop_edge->kid];
                }

                const Sim3Pose Sli = Slw * Swi;

                Similarity3 Sli_gtsam(Rot3(Sli.R()), Point3(Sli.t()), Sli.s());

                auto edge = boost::make_shared<gtsam_factors::BetweenFactorSimilarity3Inverse>(
                    X(loop_edge->kid), X(keyframe_id), Sli_gtsam,
                    noiseModel::Isotropic::Sigma(7, sigma_for_visual));
                graph.add(edge);
                num_graph_edges++;
            }
        }

        // Covisibility graph edges
        for (auto &connected_keyframe : keyframe->get_covisible_by_weight(min_number_features)) {
            if (connected_keyframe != parent_keyframe && !keyframe->has_child(connected_keyframe) &&
                connected_keyframe->kid < keyframe_id && !connected_keyframe->is_bad() &&
                inserted_loop_edges.find({std::min(keyframe_id, connected_keyframe->kid),
                                          std::max(keyframe_id, connected_keyframe->kid)}) ==
                    inserted_loop_edges.end()) {

                Sim3Pose Snw;
                auto it_non_corrected_sim3_connected_keyframe =
                    non_corrected_sim3_map.find(connected_keyframe);
                if (it_non_corrected_sim3_connected_keyframe != non_corrected_sim3_map.end()) {
                    Snw = it_non_corrected_sim3_connected_keyframe->second;
                } else {
                    Snw = vec_Scw[connected_keyframe->kid];
                }

                Sim3Pose Sni = Snw * Swi;

                Similarity3 Sni_gtsam(Rot3(Sni.R()), Point3(Sni.t()), Sni.s());

                auto edge = boost::make_shared<gtsam_factors::BetweenFactorSimilarity3Inverse>(
                    X(connected_keyframe->kid), X(keyframe_id), Sni_gtsam,
                    noiseModel::Isotropic::Sigma(7, sigma_for_visual));
                graph.add(edge);
                num_graph_edges++;
            }
        }
    }

    if (verbose) {
        std::cout << "[optimize_essential_graph]: Total number of graph edges: " << num_graph_edges
                  << std::endl;
    }

    // Optimize
    auto params = gtsam::LevenbergMarquardtParams().CeresDefaults();
    params.setlambdaInitial(1e-16);   // As in optimzer_g2o version of optimize_essential_graph()
    params.setlambdaLowerBound(1e-8); // Prevent over-reduction
    params.setlambdaUpperBound(1e3);  // Prevent excessive increase
    params.setlambdaFactor(2.0);      // Mimics g2o’s adaptive _ni
    params.setDiagonalDamping(true);  // Mimics g2o’s Hessian updates
    params.setUseFixedLambdaFactor(false); // Mimics g2o’s ni

    params.setMaxIterations(20);

    if (verbose) {
        params.setVerbosityLM("SUMMARY");
    }

    LevenbergMarquardtOptimizer optimizer(graph, initial_values, params);
    // gtsam_factors::LevenbergMarquardtOptimizerG2o optimizer(graph, initial_values, params);
    Values opt_result = optimizer.optimize();

    // SE3 Pose Recovering
    for (auto &keyframe : all_keyframes) {
        if (!keyframe) { // || keyframe->is_bad()) {
            continue;
        }

        int keyframe_id = keyframe->kid;
        Similarity3 corrected_Siw = getSimilarity3(opt_result, X(keyframe_id));

        Eigen::Matrix3d R = corrected_Siw.rotation().matrix();
        Eigen::Vector3d t = corrected_Siw.translation();
        double s = corrected_Siw.scale();
        Sim3Pose Siw(R, t, s);
        Sim3Pose Swi = Siw.inverse();

        vec_corrected_Swc[keyframe_id] = Swi;

        Eigen::Matrix4d Tiw = poseRt(R, t / s);
        keyframe->update_pose(Tiw);
    }

    // Correct points
    for (auto &map_point : all_map_points) {
        if (!map_point || map_point->is_bad()) {
            continue;
        }

        int reference_id;
        if (map_point->corrected_by_kf == current_keyframe->kid) {
            reference_id = map_point->corrected_reference;
        } else {
            auto reference_keyframe = map_point->get_reference_keyframe();
            reference_id = reference_keyframe->kid;
        }

        const Sim3Pose &Srw = vec_Scw[reference_id];
        const Sim3Pose &corrected_Swr = vec_corrected_Swc[reference_id];

        // Check if poses are valid
        // if (Srw.s() <= 0 || corrected_Swr.s() <= 0) {
        //     continue;
        // }

        const Eigen::Vector3d P3Dw = map_point->pt();
        const Eigen::Vector3d corrected_P3Dw = corrected_Swr.map(Srw.map(P3Dw));
        map_point->update_position(corrected_P3Dw);
        map_point->update_normal_and_depth();
    }

    double mean_squared_error = graph.error(opt_result) / std::max(num_graph_edges, 1);
    return mean_squared_error;
}

} // namespace pyslam
