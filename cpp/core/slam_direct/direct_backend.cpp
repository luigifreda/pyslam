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


#include "slam_direct/direct_backend.h"
#include "camera/camera_model_math.h"
#include "common/owned_g2o_optimizer.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/block_solver.h>
#include <g2o/core/optimization_algorithm_levenberg.h>
#include <g2o/core/robust_kernel_impl.h>
#include <g2o/solvers/eigen/linear_solver_eigen.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

#include <algorithm>
#include <cmath>
#include <map>
#include <set>

namespace pyslam {
namespace {
using PoseVertex = g2o::VertexSE3Expmap;
using PointVertex = g2o::VertexSBAPointXYZ;

// A separate right-camera edge preserves the measured vertical coordinate and
// also supports calibrated, non-rectified pinhole rigs, as in the vslam backend.
class ProjectionEdge : public g2o::BaseBinaryEdge<2, Vec2d, PointVertex, PoseVertex> {
  public:
    ProjectionEdge(const Camera &camera, const Eigen::Isometry3d &extrinsic)
        : camera_(camera), extrinsic_(extrinsic) {}
    void computeError() override {
        const auto *point = static_cast<const PointVertex *>(_vertices[0]);
        const auto *pose = static_cast<const PoseVertex *>(_vertices[1]);
        Vec2d uv;
        // Pose maps world to left camera; extrinsic_ then selects the observing camera.
        const Vec3d X = extrinsic_ * pose->estimate().map(point->estimate());
        if (X.z() > 1e-6 && camera_.project_with_jacobian(X, uv, nullptr))
            _error = _measurement - uv;
        else
            _error.setConstant(1e4);
    }
    void linearizeOplus() override {
        const auto *point = static_cast<const PointVertex *>(_vertices[0]);
        const auto *pose = static_cast<const PoseVertex *>(_vertices[1]);
        const Vec3d Xleft = pose->estimate().map(point->estimate());
        const Vec3d X = extrinsic_ * Xleft;
        Vec2d uv;
        Eigen::Matrix<double, 2, 3> J;
        if (X.z() <= 1e-6 || !camera_.project_with_jacobian(X, uv, &J)) {
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }
        Eigen::Matrix<double, 3, 6> chain;
        camera_model::g2o_left_se3_jacobian(Xleft, chain);
        _jacobianOplusXi = -J * extrinsic_.linear() * pose->estimate().rotation().toRotationMatrix();
        _jacobianOplusXj = -J * extrinsic_.linear() * chain;
    }
    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }
  private:
    const Camera &camera_;
    Eigen::Isometry3d extrinsic_;
};

// A measured RGBD z constraint fixes metric scale without a synthetic stereo baseline.
class DepthEdge : public g2o::BaseBinaryEdge<1, double, PointVertex, PoseVertex> {
  public:
    void computeError() override {
        const auto *point = static_cast<const PointVertex *>(_vertices[0]);
        const auto *pose = static_cast<const PoseVertex *>(_vertices[1]);
        _error[0] = _measurement - pose->estimate().map(point->estimate()).z();
    }
    void linearizeOplus() override {
        const auto *point = static_cast<const PointVertex *>(_vertices[0]);
        const auto *pose = static_cast<const PoseVertex *>(_vertices[1]);
        const Vec3d X = pose->estimate().map(point->estimate());
        Eigen::Matrix<double, 3, 6> chain;
        camera_model::g2o_left_se3_jacobian(X, chain);
        _jacobianOplusXi = -pose->estimate().rotation().toRotationMatrix().row(2);
        _jacobianOplusXj = -chain.row(2);
    }
    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }
};

// Inverse depth in a chosen camera. This is the monocular scale anchor.
class InverseDepthPriorEdge : public g2o::BaseBinaryEdge<1, double, PointVertex, PoseVertex> {
  public:
    void computeError() override {
        const auto *point = static_cast<const PointVertex *>(_vertices[0]);
        const auto *pose = static_cast<const PoseVertex *>(_vertices[1]);
        const double z = pose->estimate().map(point->estimate()).z();
        _error[0] = z > 1e-6 ? 1.0 / z - _measurement : 1e4;
    }
    void linearizeOplus() override {
        const auto *point = static_cast<const PointVertex *>(_vertices[0]);
        const auto *pose = static_cast<const PoseVertex *>(_vertices[1]);
        const Vec3d X = pose->estimate().map(point->estimate());
        if (!(X.z() > 1e-6)) {
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }
        Eigen::Matrix<double, 3, 6> chain;
        camera_model::g2o_left_se3_jacobian(X, chain);
        const double scale = -1.0 / (X.z() * X.z());
        _jacobianOplusXi = scale * pose->estimate().rotation().toRotationMatrix().row(2);
        _jacobianOplusXj = scale * chain.row(2);
    }
    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }
};

// What a residual edge measures. InversePrior is the monocular scale anchor, not an observation.
enum class EdgeKind { Left, Right, Depth, InversePrior };

struct ObservationEdge {
    g2o::OptimizableGraph::Edge *edge;
    DirectPoint *observation;
    PointVertex *point;
    PoseVertex *pose;
    Eigen::Isometry3d extrinsic;
    double threshold;
    EdgeKind kind;
    bool inlier() const {
        if (kind == EdgeKind::InversePrior) {
            return true; // Keep the scale anchor in the refinement.
        }
        const Vec3d X = extrinsic * pose->estimate().map(point->estimate());
        return X.allFinite() && X.z() > 1e-6 && std::isfinite(edge->chi2()) && edge->chi2() <= threshold;
    }
};
} // namespace

DirectBAResult optimize_direct_map(DirectMap &map, const Camera &camera,
                                   const DirectSlamParams &params) {
    DirectBAResult result;
    auto &keyframes = map.keyframes_mutable();
    if (keyframes.size() < 2) return result;
    const size_t begin = keyframes.size() > size_t(params.num_active_keyframes)
                             ? keyframes.size() - params.num_active_keyframes : 0;
    result.keyframes = int(keyframes.size() - begin);
    OwnedG2oOptimizer optimizer;
    auto linear = std::make_unique<g2o::LinearSolverEigen<g2o::BlockSolver_6_3::PoseMatrixType>>();
    auto block = std::make_unique<g2o::BlockSolver_6_3>(std::move(linear));
    optimizer.setAlgorithm(optimizer.make<g2o::OptimizationAlgorithmLevenberg>(std::move(block)));
    std::map<int, PoseVertex *> poses;
    std::map<int64_t, PointVertex *> points;
    std::map<int64_t, int> counts;
    std::map<int64_t, bool> metric;
    std::map<int64_t, std::vector<int>> views;
    auto &landmarks = map.landmarks_mutable();
    for (size_t i = begin; i < keyframes.size(); ++i) {
        for (const auto &obs : keyframes[i].points) {
            const auto it = landmarks.find(obs.id);
            if (obs.outlier || it == landmarks.end() || it->second.bad) continue;
            ++counts[obs.id];
            views[obs.id].push_back(keyframes[i].id);
            metric[obs.id] = metric[obs.id] || (obs.has_right && !obs.right_outlier) ||
                            (camera.sensor_type == SensorType::RGBD && obs.has_depth && !obs.depth_outlier);
        }
    }
    // Do not optimize a disconnected window or an unanchored monocular scale.
    // Such graphs have independent gauges even when one pose is fixed.
    std::set<int> connected{keyframes[begin].id};
    bool metric_constraint = false;
    std::map<int, int> shared_counts;
    for (const auto &[id, observers] : views) {
        if (observers.size() < 2) continue;
        metric_constraint = metric_constraint || metric[id];
        for (int frame : observers) ++shared_counts[frame];
    }
    for (size_t i = begin; i < keyframes.size(); ++i)
        if (shared_counts[keyframes[i].id] < 6) return result;
    if (camera.sensor_type == SensorType::MONOCULAR) {
        for (const auto &[id, landmark] : landmarks) {
            if (!landmark.bad && landmark.has_inverse_prior && landmark.inverse_depth_sigma2 > 0.0 &&
                counts.count(id) != 0) {
                metric_constraint = true;
                break;
            }
        }
    }
    if (!metric_constraint) return result;
    for (int pass = 0; pass < result.keyframes; ++pass) {
        for (const auto &[id, observers] : views) {
            const bool touches = std::any_of(observers.begin(), observers.end(),
                                             [&](int frame) { return connected.count(frame) != 0; });
            if (touches) connected.insert(observers.begin(), observers.end());
        }
    }
    if (connected.size() != keyframes.size() - begin) return result;
    int vertex_id = 0;
    for (size_t i = begin; i < keyframes.size(); ++i) {
        auto *vertex = optimizer.make<PoseVertex>();
        const auto &pose = keyframes[i].pose;
        vertex->setId(vertex_id++);
        vertex->setEstimate(g2o::SE3Quat(pose.Rcw(), pose.tcw()));
        vertex->setFixed(i == begin); // Fix the gauge; stereo/depth observations fix metric scale.
        optimizer.addVertex(vertex);
        poses[keyframes[i].id] = vertex;
    }
    for (const auto &[id, count] : counts) {
        if (count < 2 && !metric[id]) continue; // A single bearing does not constrain a 3D point.
        auto *vertex = optimizer.make<PointVertex>();
        vertex->setId(vertex_id++);
        vertex->setEstimate(landmarks.at(id).position_w);
        // Schur elimination handles 3D landmarks separately from the six-DOF pose blocks.
        vertex->setMarginalized(true);
        optimizer.addVertex(vertex);
        points[id] = vertex;
    }
    result.landmarks = int(points.size());
    // A window landmark is still observed by archived keyframes. Keep those
    // cameras fixed and include their measurements, otherwise each window can
    // pull shared world points away from the older map that the viewer retains.
    // Find them through the landmark -> observers index, so the cost scales with
    // the window landmarks rather than with the whole archived map.
    std::set<size_t> anchors;
    for (const auto &[id, vertex] : points) {
        const std::vector<int> *observers = map.observers(id);
        if (!observers) continue;
        for (int frame : *observers) {
            // Keyframe IDs are insertion indices (see DirectMap).
            if (frame >= 0 && size_t(frame) < begin) anchors.insert(size_t(frame));
        }
    }
    for (const size_t i : anchors) {
        const auto &keyframe = keyframes[i];
        const bool observes_active_point = std::any_of(
            keyframe.points.begin(), keyframe.points.end(), [&](const DirectPoint &obs) {
                return !obs.outlier && points.count(obs.id) != 0;
            });
        if (!observes_active_point) continue;
        auto *vertex = optimizer.make<PoseVertex>();
        vertex->setId(vertex_id++);
        vertex->setEstimate(g2o::SE3Quat(keyframe.pose.Rcw(), keyframe.pose.tcw()));
        vertex->setFixed(true);
        optimizer.addVertex(vertex);
        poses[keyframe.id] = vertex;
    }
    std::vector<ObservationEdge> edges;
    const Eigen::Isometry3d identity = Eigen::Isometry3d::Identity();
    // Rectified rigs derive their right-camera translation from bf = fx * baseline.
    Eigen::Isometry3d right_from_left = identity;
    if (camera.right_camera) right_from_left.matrix() = camera.T_right_left;
    else right_from_left.translation().x() = -camera.bf / camera.fx;
    const Camera &right_camera = camera.right_camera ? *camera.right_camera : camera;
    auto add_edge = [&](auto *edge, DirectPoint &obs, PointVertex *point, PoseVertex *pose,
                        const Eigen::Isometry3d &extrinsic, double threshold, EdgeKind kind) {
        edge->setVertex(0, point);
        edge->setVertex(1, pose);
        auto *kernel = optimizer.make<g2o::RobustKernelHuber>();
        kernel->setDelta(std::sqrt(threshold));
        edge->setRobustKernel(kernel);
        optimizer.addEdge(edge);
        edges.push_back({edge, &obs, point, pose, extrinsic, threshold, kind});
    };
    for (size_t i = 0; i < keyframes.size(); ++i) {
        const auto pose_it = poses.find(keyframes[i].id);
        if (pose_it == poses.end()) continue;
        auto *pose = pose_it->second;
        for (auto &obs : keyframes[i].points) {
            const auto it = points.find(obs.id);
            if (obs.outlier || it == points.end()) continue;
            auto *point = it->second;
            auto *left = optimizer.make<ProjectionEdge>(camera, identity);
            left->setMeasurement(obs.uv);
            left->setInformation(Eigen::Matrix2d::Identity());
            add_edge(left, obs, point, pose, identity, params.chi2_threshold, EdgeKind::Left);
            if (obs.has_right && !obs.right_outlier) {
                auto *right = optimizer.make<ProjectionEdge>(right_camera, right_from_left);
                right->setMeasurement(obs.uv_right);
                right->setInformation(Eigen::Matrix2d::Identity());
                add_edge(right, obs, point, pose, right_from_left, params.chi2_threshold, EdgeKind::Right);
            }
            if (camera.sensor_type == SensorType::RGBD && obs.has_depth && !obs.depth_outlier) {
                auto *depth = optimizer.make<DepthEdge>();
                depth->setMeasurement(obs.depth);
                const double sigma = params.ba_depth_sigma * std::max(1.0, obs.depth * obs.depth);
                depth->setInformation(Eigen::Matrix<double, 1, 1>::Constant(1.0 / (sigma * sigma)));
                add_edge(depth, obs, point, pose, identity, 3.841, EdgeKind::Depth);
            }
            const auto landmark = landmarks.find(obs.id);
            if (camera.sensor_type == SensorType::MONOCULAR && landmark != landmarks.end() &&
                landmark->second.has_inverse_prior && landmark->second.prior_keyframe == keyframes[i].id &&
                landmark->second.inverse_depth_sigma2 > 0.0) {
                auto *prior = optimizer.make<InverseDepthPriorEdge>();
                prior->setMeasurement(landmark->second.inverse_depth_mu);
                prior->setInformation(Eigen::Matrix<double, 1, 1>::Constant(
                    1.0 / landmark->second.inverse_depth_sigma2));
                add_edge(prior, obs, point, pose, identity, 1e9, EdgeKind::InversePrior);
            }
        }
    }
    result.observations = int(edges.size());
    if (edges.size() < 12 || points.size() < 6) return result;
    optimizer.initializeOptimization();
    optimizer.computeActiveErrors();
    result.initial_cost = optimizer.activeRobustChi2();
    if (optimizer.optimize(params.ba_iterations) <= 0) return result;
    // Classify residuals after robust LM, then refine using only level-0 (inlier) edges.
    int inliers = 0;
    for (auto &record : edges) {
        record.edge->computeError();
        const bool good = record.inlier();
        record.edge->setLevel(good ? 0 : 1);
        inliers += good;
    }
    if (inliers < 12 || inliers < int(edges.size() / 2)) return result;
    optimizer.initializeOptimization(0);
    optimizer.optimize(std::max(1, params.ba_iterations / 2));
    // Check the complete robust objective before mutating the map, including
    // excluded edges. A failed solve leaves poses, landmarks and observations intact.
    result.final_cost = 0.0;
    for (auto &record : edges) {
        record.edge->computeError();
        Eigen::Vector3d rho;
        record.edge->robustKernel()->robustify(record.edge->chi2(), rho);
        result.final_cost += rho[0];
    }
    if (!std::isfinite(result.final_cost) || result.final_cost > result.initial_cost * 1.05 + 1e-6)
        return result;
    for (const auto &[id, vertex] : poses)
        if (!vertex->estimate().to_homogeneous_matrix().allFinite()) return result;
    for (const auto &[id, vertex] : points)
        if (!vertex->estimate().allFinite()) return result;
    for (const auto &[id, vertex] : poses)
        keyframes.at(id).pose = CameraPose(vertex->estimate().to_homogeneous_matrix());
    for (const auto &[id, vertex] : points)
        landmarks.at(id).position_w = vertex->estimate();
    // Reject observation channels separately: a bad depth/right sample need not lose a left track.
    std::map<int64_t, int> good_left;
    for (auto &record : edges) {
        const bool bad = !record.inlier();
        if (record.kind == EdgeKind::Left) {
            record.observation->outlier = bad;
            good_left[record.observation->id] += !bad;
        } else if (record.kind == EdgeKind::Right) record.observation->right_outlier = bad;
        else if (record.kind == EdgeKind::Depth) record.observation->depth_outlier = bad;
        result.outliers += bad;
    }
    for (const auto &[id, count] : good_left)
        if (count == 0) landmarks.at(id).bad = true;
    map.update_trajectory();
    result.applied = true;
    return result;
}

} // namespace pyslam
