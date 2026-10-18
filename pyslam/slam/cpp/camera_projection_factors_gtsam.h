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

// gtsam reprojection factors that call CameraT::project_with_jacobian. Shared by
// cpp_core (pyslam::Camera) and camera_models_gtsam (camera_models::CameraModel).
// This header does not include g2o.
//
// gtsam stores Twc and uses Pose3::retract; its error is projection - measurement.
// An invalid projection yields a finite (2 fx) error and zero Jacobians.

#include "camera_handle.h"

#include <gtsam/base/Matrix.h>
#include <gtsam/geometry/Pose3.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/linear/NoiseModel.h>
#include <gtsam/nonlinear/NonlinearFactor.h>
#include <gtsam_factors/weighted_projection_factors.h>

#include <boost/make_shared.hpp>
#include <boost/optional.hpp>

#include <cassert>
#include <limits>
#include <memory>
#include <stdexcept>

namespace pyslam {
namespace camera_model {

struct CameraWorldProjection {
    Eigen::Vector2d uv = Eigen::Vector2d::Zero();
    bool valid = false;
    Eigen::Matrix<double, 2, 6> J_pose = Eigen::Matrix<double, 2, 6>::Zero();
    Eigen::Matrix<double, 2, 3> J_point = Eigen::Matrix<double, 2, 3>::Zero();
};

template <typename CameraT>
CameraWorldProjection project_world_point(const CameraT &camera, const gtsam::Pose3 &pose_wc,
                                          const gtsam::Point3 &point_w, bool want_jacobian) {
    CameraWorldProjection out;
    gtsam::Matrix36 dXc_dpose;
    gtsam::Matrix33 dXc_dpoint;
    const gtsam::Point3 Xc = want_jacobian ? pose_wc.transformTo(point_w, dXc_dpose, dXc_dpoint)
                                           : pose_wc.transformTo(point_w);
    Eigen::Matrix<double, 2, 3> J_uv_Xc;
    out.valid = camera.project_with_jacobian(Xc, out.uv, want_jacobian ? &J_uv_Xc : nullptr);
    if (want_jacobian && out.valid) {
        out.J_pose = J_uv_Xc * dXc_dpose;
        out.J_point = J_uv_Xc * dXc_dpoint;
    }
    return out;
}

template <typename CameraT>
class CameraProjectionFactor : public gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3> {
  public:
    using Base = gtsam::NoiseModelFactor2<gtsam::Pose3, gtsam::Point3>;

    template <typename Ptr>
    CameraProjectionFactor(const gtsam::Point2 &measured, const gtsam::SharedNoiseModel &model,
                           gtsam::Key pose_key, gtsam::Key point_key, Ptr camera)
        : Base(model, pose_key, point_key), measured_(measured), camera_(std::move(camera)) {}

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new CameraProjectionFactor(*this)));
    }

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose, const gtsam::Point3 &point,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none) const override {
        if (weight_ <= std::numeric_limits<double>::epsilon()) {
            if (H1) *H1 = gtsam::Matrix::Zero(2, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(2, 3);
            return gtsam::Vector2::Zero();
        }
        const bool want_jacobian = static_cast<bool>(H1) || static_cast<bool>(H2);
        const CameraWorldProjection projected = project_world_point(*camera_, pose, point, want_jacobian);
        if (!projected.valid) {
            if (H1) *H1 = gtsam::Matrix::Zero(2, 6);
            if (H2) *H2 = gtsam::Matrix::Zero(2, 3);
            return gtsam::Vector2::Constant(2.0 * camera_->fx);
        }
        if (H1) *H1 = weight_ * projected.J_pose;
        if (H2) *H2 = weight_ * projected.J_point;
        gtsam::Vector2 error;
        error << projected.uv[0] - measured_.x(), projected.uv[1] - measured_.y();
        return weight_ * error;
    }

    void setWeight(double weight) {
        assert(weight > 0.0);
        weight_ = weight;
    }
    double getWeight() const { return weight_; }

  private:
    gtsam::Point2 measured_;
    CameraHandle<CameraT> camera_;
    double weight_ = 1.0;
};

template <typename CameraT> class CameraResectioningFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3> {
  public:
    using Base = gtsam::NoiseModelFactorN<gtsam::Pose3>;

    template <typename Ptr>
    CameraResectioningFactor(const gtsam::SharedNoiseModel &model, gtsam::Key pose_key, Ptr camera,
                             const gtsam::Point2 &measured, const gtsam::Point3 &world_point)
        : Base(model, pose_key), camera_(std::move(camera)), measured_(measured), world_point_(world_point) {}

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::static_pointer_cast<gtsam::NonlinearFactor>(
            gtsam::NonlinearFactor::shared_ptr(new CameraResectioningFactor(*this)));
    }

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose,
                                boost::optional<gtsam::Matrix &> H = boost::none) const override {
        if (weight_ <= std::numeric_limits<double>::epsilon()) {
            if (H) *H = gtsam::Matrix::Zero(2, 6);
            return gtsam::Vector2::Zero();
        }
        const CameraWorldProjection projected =
            project_world_point(*camera_, pose, world_point_, static_cast<bool>(H));
        if (!projected.valid) {
            if (H) *H = gtsam::Matrix::Zero(2, 6);
            return gtsam::Vector2::Constant(2.0 * camera_->fx);
        }
        if (H) *H = weight_ * projected.J_pose;
        gtsam::Vector2 error;
        error << projected.uv[0] - measured_.x(), projected.uv[1] - measured_.y();
        return weight_ * error;
    }

    void setWeight(double weight) {
        assert(weight > 0.0);
        weight_ = weight;
    }
    double getWeight() const { return weight_; }

  private:
    CameraHandle<CameraT> camera_;
    gtsam::Point2 measured_;
    gtsam::Point3 world_point_;
    double weight_ = 1.0;
};

// Three-component stereo observations keep the legacy (u, v, u_right) layout. The right
// horizontal coordinate uses the rig extrinsics and camera model, or a virtual baseline
// when the source is an RGB-D depth map.
template <typename CameraT>
gtsam::Vector3 stereo_world_error(const CameraT &camera, const gtsam::Pose3 &pose, const gtsam::Point3 &point,
                                  const gtsam::StereoPoint2 &measurement, double weight,
                                  boost::optional<gtsam::Matrix &> H1,
                                  boost::optional<gtsam::Matrix &> H2 = boost::none) {
    gtsam::Matrix36 dpose;
    gtsam::Matrix33 dpoint;
    const auto X = (H1 || H2) ? pose.transformTo(point, dpose, dpoint) : pose.transformTo(point);
    Eigen::Vector3d uv;
    Eigen::Matrix3d J;
    const bool valid = camera.project_stereo_with_jacobian(X, uv, (H1 || H2) ? &J : nullptr);
    if (!valid || weight <= std::numeric_limits<double>::epsilon()) {
        if (H1) *H1 = gtsam::Matrix::Zero(3, 6);
        if (H2) *H2 = gtsam::Matrix::Zero(3, 3);
        return weight <= std::numeric_limits<double>::epsilon() ? gtsam::Vector3::Zero().eval()
                                                                : gtsam::Vector3::Constant(2 * camera.fx).eval();
    }
    // GTSAM StereoPoint2 orders its components as (u_left, u_right, v).
    Eigen::Matrix3d ordered;
    if (H1 || H2) {
        ordered.row(0) = J.row(0);
        ordered.row(1) = J.row(2);
        ordered.row(2) = J.row(1);
    }
    if (H1) *H1 = weight * ordered * dpose;
    if (H2) *H2 = weight * ordered * dpoint;
    return weight * gtsam::Vector3(uv.x() - measurement.uL(), uv.z() - measurement.uR(), uv.y() - measurement.v());
}

// The base class only provides the weight and the Cal3_S2Stereo slot used by the pinhole
// path; a non-positive baseline (native rig) is stored as zero.
template <typename CameraT>
class StereoCameraProjectionFactor : public gtsam_factors::WeightedGenericStereoProjectionFactor3D {
  public:
    using Base = gtsam_factors::WeightedGenericStereoProjectionFactor3D;

    template <typename Ptr>
    StereoCameraProjectionFactor(const gtsam::StereoPoint2 &measured, const gtsam::SharedNoiseModel &noise,
                                 gtsam::Key pose, gtsam::Key point, Ptr camera)
        : Base(measured, noise, pose, point,
               boost::make_shared<gtsam::Cal3_S2Stereo>(camera->fx, camera->fy, 0.0, camera->cx, camera->cy,
                                                        camera->b > 0.0 ? camera->b : 0.0)),
          camera_(std::move(camera)), measured_(measured) {}

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose, const gtsam::Point3 &point,
                                boost::optional<gtsam::Matrix &> H1 = boost::none,
                                boost::optional<gtsam::Matrix &> H2 = boost::none) const override {
        return stereo_world_error(*camera_, pose, point, measured_, getWeight(), H1, H2);
    }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<StereoCameraProjectionFactor>(*this);
    }

  private:
    CameraHandle<CameraT> camera_;
    gtsam::StereoPoint2 measured_;
};

template <typename CameraT> class StereoCameraResectioningFactor : public gtsam::NoiseModelFactorN<gtsam::Pose3> {
  public:
    using Base = gtsam::NoiseModelFactorN<gtsam::Pose3>;

    template <typename Ptr>
    StereoCameraResectioningFactor(const gtsam::SharedNoiseModel &noise, gtsam::Key pose, Ptr camera,
                                   const gtsam::StereoPoint2 &measured, const gtsam::Point3 &point)
        : Base(noise, pose), camera_(std::move(camera)), point_(point), measured_(measured) {}

    gtsam::Vector evaluateError(const gtsam::Pose3 &pose,
                                boost::optional<gtsam::Matrix &> H = boost::none) const override {
        return stereo_world_error(*camera_, pose, point_, measured_, weight_, H);
    }

    void setWeight(double weight) {
        assert(weight > 0.0);
        weight_ = weight;
    }
    double getWeight() const { return weight_; }

    gtsam::NonlinearFactor::shared_ptr clone() const override {
        return boost::make_shared<StereoCameraResectioningFactor>(*this);
    }

  private:
    CameraHandle<CameraT> camera_;
    gtsam::Point3 point_;
    gtsam::StereoPoint2 measured_;
    double weight_ = 1.0;
};

// GTSAM's Python holders use std::shared_ptr, while the native build uses
// boost::shared_ptr. Share ownership without copying switchable noise state.
inline gtsam::SharedNoiseModel python_noise_model(const std::shared_ptr<gtsam::noiseModel::Base> &model) {
    if (!model) throw std::invalid_argument("A noise model is required");
    return gtsam::SharedNoiseModel(model.get(), [model](gtsam::noiseModel::Base *) {});
}

} // namespace camera_model
} // namespace pyslam
