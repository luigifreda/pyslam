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

// g2o reprojection edges that call CameraT::project_with_jacobian. Shared by
// cpp_core (pyslam::Camera) and camera_models_g2o (camera_models::CameraModel).
// This header does not include gtsam.
//
// g2o stores Tcw and uses a left perturbation; its error is measurement - projection.
// An invalid projection yields a finite (2 fx) penalty and zero Jacobians.
// Cameras whose projects_raw_keypoints() is false keep the original g2o pinhole
// computation of the base edge. Stereo observations keep the (u, v, u_right) layout.

#include "camera_handle.h"
#include "camera_model_math.h"

#include <g2o/core/base_binary_edge.h>
#include <g2o/core/base_unary_edge.h>
#include <g2o/types/sba/types_six_dof_expmap.h>

namespace pyslam {
namespace camera_model {

template <typename CameraT> class EdgeCameraProjectXYZ : public g2o::EdgeSE3ProjectXYZ {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    template <typename Ptr> explicit EdgeCameraProjectXYZ(Ptr camera) : camera_(std::move(camera)) {
        fx = camera_->fx;
        fy = camera_->fy;
        cx = camera_->cx;
        cy = camera_->cy;
    }

    void computeError() override {
        if (!camera_->projects_raw_keypoints()) {
            g2o::EdgeSE3ProjectXYZ::computeError();
            return;
        }
        Eigen::Vector2d uv;
        const bool valid = camera_->project_with_jacobian(camera_point(), uv, nullptr);
        _error = valid ? Eigen::Vector2d(_measurement - uv) : Eigen::Vector2d::Constant(2.0 * camera_->fx);
    }

    bool isDepthPositive() const {
        if (!camera_->projects_raw_keypoints()) return camera_point().z() > 0.0;
        Eigen::Vector2d uv;
        return camera_->project_with_jacobian(camera_point(), uv, nullptr);
    }

    void linearizeOplus() override {
        if (!camera_->projects_raw_keypoints()) {
            g2o::EdgeSE3ProjectXYZ::linearizeOplus();
            return;
        }
        const auto *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        const g2o::SE3Quat T = pose->estimate();
        Eigen::Matrix<double, 2, 6> J_pose;
        Eigen::Matrix<double, 2, 3> J_point;
        if (!camera_->g2o_reprojection_error_jacobians(camera_point(), T.rotation().toRotationMatrix(),
                                                       J_pose, J_point)) {
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }
        _jacobianOplusXi = J_point;
        _jacobianOplusXj = J_pose;
    }

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

    const CameraT *camera() const { return camera_.get(); }

  private:
    Eigen::Vector3d camera_point() const {
        const auto *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        const auto *point = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        return pose->estimate().map(point->estimate());
    }

    CameraHandle<CameraT> camera_;
};

template <typename CameraT> class EdgeCameraProjectXYZOnlyPose : public g2o::EdgeSE3ProjectXYZOnlyPose {
  public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    template <typename Ptr>
    explicit EdgeCameraProjectXYZOnlyPose(Ptr camera) : camera_(std::move(camera)) {
        fx = camera_->fx;
        fy = camera_->fy;
        cx = camera_->cx;
        cy = camera_->cy;
    }

    void computeError() override {
        if (!camera_->projects_raw_keypoints()) {
            g2o::EdgeSE3ProjectXYZOnlyPose::computeError();
            return;
        }
        Eigen::Vector2d uv;
        const bool valid = camera_->project_with_jacobian(camera_point(), uv, nullptr);
        _error = valid ? Eigen::Vector2d(_measurement - uv) : Eigen::Vector2d::Constant(2.0 * camera_->fx);
    }

    bool isDepthPositive() const {
        if (!camera_->projects_raw_keypoints()) return camera_point().z() > 0.0;
        Eigen::Vector2d uv;
        return camera_->project_with_jacobian(camera_point(), uv, nullptr);
    }

    void linearizeOplus() override {
        if (!camera_->projects_raw_keypoints()) {
            g2o::EdgeSE3ProjectXYZOnlyPose::linearizeOplus();
            return;
        }
        const auto *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[0]);
        Eigen::Matrix<double, 2, 6> J_pose;
        Eigen::Matrix<double, 2, 3> J_point;
        if (!camera_->g2o_reprojection_error_jacobians(
                camera_point(), pose->estimate().rotation().toRotationMatrix(), J_pose, J_point)) {
            _jacobianOplusXi.setZero();
            return;
        }
        _jacobianOplusXi = J_pose;
    }

    bool read(std::istream &) override { return false; }
    bool write(std::ostream &) const override { return false; }

  private:
    Eigen::Vector3d camera_point() const {
        return static_cast<const g2o::VertexSE3Expmap *>(_vertices[0])->estimate().map(Xw);
    }

    CameraHandle<CameraT> camera_;
};

template <typename CameraT> class EdgeStereoCameraProjectXYZ : public g2o::EdgeStereoSE3ProjectXYZ {
  public:
    template <typename Ptr> explicit EdgeStereoCameraProjectXYZ(Ptr camera) : camera_(std::move(camera)) {}

    void computeError() override {
        if (!camera_->projects_raw_keypoints()) {
            g2o::EdgeStereoSE3ProjectXYZ::computeError();
            return;
        }
        Eigen::Vector3d uv;
        const bool valid = camera_->project_stereo_with_jacobian(camera_point(), uv);
        _error = valid ? Eigen::Vector3d(_measurement - uv) : Eigen::Vector3d::Constant(2 * camera_->fx);
    }

    void linearizeOplus() override {
        if (!camera_->projects_raw_keypoints()) {
            g2o::EdgeStereoSE3ProjectXYZ::linearizeOplus();
            return;
        }
        const auto *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        const Eigen::Vector3d X = camera_point();
        Eigen::Vector3d uv;
        Eigen::Matrix3d J;
        if (!camera_->project_stereo_with_jacobian(X, uv, &J)) {
            _jacobianOplusXi.setZero();
            _jacobianOplusXj.setZero();
            return;
        }
        Eigen::Matrix<double, 3, 6> chain;
        g2o_left_se3_jacobian(X, chain);
        _jacobianOplusXi = -J * pose->estimate().rotation().toRotationMatrix();
        _jacobianOplusXj = -J * chain;
    }

    bool isDepthPositive() const {
        if (!camera_->projects_raw_keypoints()) return camera_point().z() > 0;
        Eigen::Vector3d uv;
        return camera_->project_stereo_with_jacobian(camera_point(), uv);
    }

  private:
    Eigen::Vector3d camera_point() const {
        const auto *pose = static_cast<const g2o::VertexSE3Expmap *>(_vertices[1]);
        const auto *point = static_cast<const g2o::VertexSBAPointXYZ *>(_vertices[0]);
        return pose->estimate().map(point->estimate());
    }

    CameraHandle<CameraT> camera_;
};

template <typename CameraT>
class EdgeStereoCameraProjectXYZOnlyPose : public g2o::EdgeStereoSE3ProjectXYZOnlyPose {
  public:
    template <typename Ptr>
    explicit EdgeStereoCameraProjectXYZOnlyPose(Ptr camera) : camera_(std::move(camera)) {}

    void computeError() override {
        if (!camera_->projects_raw_keypoints()) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose::computeError();
            return;
        }
        Eigen::Vector3d uv;
        const bool valid = camera_->project_stereo_with_jacobian(camera_point(), uv);
        _error = valid ? Eigen::Vector3d(_measurement - uv) : Eigen::Vector3d::Constant(2 * camera_->fx);
    }

    void linearizeOplus() override {
        if (!camera_->projects_raw_keypoints()) {
            g2o::EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus();
            return;
        }
        const Eigen::Vector3d X = camera_point();
        Eigen::Vector3d uv;
        Eigen::Matrix3d J;
        if (!camera_->project_stereo_with_jacobian(X, uv, &J)) {
            _jacobianOplusXi.setZero();
            return;
        }
        Eigen::Matrix<double, 3, 6> chain;
        g2o_left_se3_jacobian(X, chain);
        _jacobianOplusXi = -J * chain;
    }

    bool isDepthPositive() const {
        if (!camera_->projects_raw_keypoints()) return camera_point().z() > 0;
        Eigen::Vector3d uv;
        return camera_->project_stereo_with_jacobian(camera_point(), uv);
    }

  private:
    Eigen::Vector3d camera_point() const {
        return static_cast<const g2o::VertexSE3Expmap *>(_vertices[0])->estimate().map(Xw);
    }

    CameraHandle<CameraT> camera_;
};

} // namespace camera_model
} // namespace pyslam
