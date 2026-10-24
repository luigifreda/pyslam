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

#include "camera_model.h"

#include "camera_model_math.h"
#include "camera_rig_geometry.h"
#include "config_parameters.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace camera_models {

CameraModel::~CameraModel() = default;

namespace {

template <typename Scalar> Scalar coeff(const std::vector<double> &D, size_t index) {
    return static_cast<Scalar>(index < D.size() ? D[index] : 0.0);
}

} // namespace

bool CameraModel::project_with_jacobian(const Eigen::Vector3d &Xc, Eigen::Vector2d &uv,
                                        Eigen::Matrix<double, 2, 3> *J_uv_Xc) const {
    using pyslam::camera_model::project_double_sphere;
    using pyslam::camera_model::project_kannala_brandt4;
    using pyslam::camera_model::project_pinhole;
    using pyslam::camera_model::project_pinhole_radtan8;
    switch (type) {
    case CameraType::KANNALA_BRANDT4:
        return project_kannala_brandt4(fx, fy, cx, cy, coeff<double>(D, 0), coeff<double>(D, 1),
                                       coeff<double>(D, 2), coeff<double>(D, 3), Xc, uv, J_uv_Xc);
    case CameraType::PINHOLE_RADTAN8:
        return project_pinhole_radtan8(fx, fy, cx, cy, coeff<double>(D, 0), coeff<double>(D, 1),
                                      coeff<double>(D, 2), coeff<double>(D, 3), coeff<double>(D, 4),
                                      coeff<double>(D, 5), coeff<double>(D, 6), coeff<double>(D, 7),
                                      Xc, uv, J_uv_Xc);
    case CameraType::DOUBLE_SPHERE:
        return project_double_sphere(fx, fy, cx, cy, coeff<double>(D, 0), coeff<double>(D, 1), Xc,
                                    uv, J_uv_Xc);
    default:
        return project_pinhole(fx, fy, cx, cy, Xc, uv, J_uv_Xc);
    }
}

bool CameraModel::unproject_bearing(const Eigen::Vector2d &uv, Eigen::Vector3d &bearing,
                                    Eigen::Matrix<double, 3, 2> *J_bearing_uv) const {
    using pyslam::camera_model::unproject_double_sphere_bearing;
    using pyslam::camera_model::unproject_kannala_brandt4_bearing;
    using pyslam::camera_model::unproject_pinhole_bearing;
    using pyslam::camera_model::unproject_pinhole_radtan8_bearing;
    switch (type) {
    case CameraType::KANNALA_BRANDT4:
        return unproject_kannala_brandt4_bearing(
            fx, fy, cx, cy, coeff<double>(D, 0), coeff<double>(D, 1), coeff<double>(D, 2),
            coeff<double>(D, 3), uv, bearing, J_bearing_uv);
    case CameraType::PINHOLE_RADTAN8:
        return unproject_pinhole_radtan8_bearing(
            fx, fy, cx, cy, coeff<double>(D, 0), coeff<double>(D, 1), coeff<double>(D, 2),
            coeff<double>(D, 3), coeff<double>(D, 4), coeff<double>(D, 5), coeff<double>(D, 6),
            coeff<double>(D, 7), uv, bearing, J_bearing_uv);
    case CameraType::DOUBLE_SPHERE:
        return unproject_double_sphere_bearing(fx, fy, cx, cy, coeff<double>(D, 0),
                                              coeff<double>(D, 1), uv, bearing, J_bearing_uv);
    default:
        return unproject_pinhole_bearing(fx, fy, cx, cy, uv, bearing, J_bearing_uv);
    }
}

bool CameraModel::project_stereo_with_jacobian(const Eigen::Vector3d &Xc, Eigen::Vector3d &uvur,
                                               Eigen::Matrix3d *J) const {
    return pyslam::camera_model::project_stereo_with_jacobian(*this, Xc, uvur, J);
}

bool CameraModel::g2o_reprojection_error_jacobians(
    const Eigen::Vector3d &Xc, const Eigen::Matrix3d &Rcw, Eigen::Matrix<double, 2, 6> &J_error_pose,
    Eigen::Matrix<double, 2, 3> &J_error_point) const {
    Eigen::Vector2d uv;
    Eigen::Matrix<double, 2, 3> J_uv_Xc;
    const bool valid = project_with_jacobian(Xc, uv, &J_uv_Xc);
    if (!valid) {
        J_error_pose.setZero();
        J_error_point.setZero();
        return false;
    }
    pyslam::camera_model::g2o_reprojection_error_jacobians(J_uv_Xc, Xc, Rcw, J_error_pose,
                                                          J_error_point);
    return true;
}

bool CameraModel::triangulate_stereo_point(const Eigen::Vector2d &ul, const Eigen::Vector2d &ur,
                                           double max_error, Eigen::Vector3d &X) const {
    return pyslam::camera_model::triangulate_stereo_point(*this, ul, ur, max_error, X);
}

void CameraModel::project_into(pyslam::MatNx3dRef xcs, Eigen::Ref<pyslam::MatNx2d> uv,
                               Eigen::Ref<pyslam::VecNd> depth, Eigen::Ref<pyslam::VecNd> valid) const {
    const int count = static_cast<int>(xcs.rows());
    if (uv.rows() != count || depth.size() != count || valid.size() != count) {
        throw std::invalid_argument("project_into: output buffers must have one row per input point");
    }
    for (int i = 0; i < count; ++i) {
        const Eigen::Vector3d X = xcs.row(i).transpose();
        Eigen::Vector2d projected;
        const bool ok = project_with_jacobian(X, projected, nullptr);
        uv(i, 0) = projected[0];
        uv(i, 1) = projected[1];
        depth[i] = X[2];
        valid[i] = ok ? 1.0 : 0.0;
    }
}

void CameraModel::project_stereo_into(pyslam::MatNx3dRef xcs, Eigen::Ref<pyslam::MatNx3d> uvur,
                                      Eigen::Ref<pyslam::VecNd> depth) const {
    const int count = static_cast<int>(xcs.rows());
    if (uvur.rows() != count || depth.size() != count) {
        throw std::invalid_argument("project_stereo_into: output buffers must have one row per input point");
    }
    for (int i = 0; i < count; ++i) {
        const Eigen::Vector3d X = xcs.row(i).transpose();
        Eigen::Vector3d projected;
        const bool ok = project_stereo_with_jacobian(X, projected, nullptr);
        uvur.row(i) = projected.transpose();
        depth[i] = ok ? X.z() : std::numeric_limits<double>::quiet_NaN();
    }
}

void CameraModel::unproject_bearings_into(pyslam::MatNx2dRef uvs, Eigen::Ref<pyslam::MatNx3d> bearings,
                                          Eigen::Ref<pyslam::VecNd> valid) const {
    const int count = static_cast<int>(uvs.rows());
    if (bearings.rows() != count || valid.size() != count) {
        throw std::invalid_argument(
            "unproject_bearings_into: output buffers must have one row per input point");
    }
    for (int i = 0; i < count; ++i) {
        Eigen::Vector3d bearing;
        const bool ok = unproject_bearing(uvs.row(i).transpose(), bearing, nullptr);
        bearings.row(i) = bearing.transpose();
        valid[i] = ok ? 1.0 : 0.0;
    }
}

void CameraModel::undistort_points_into(pyslam::MatNx2dRef uvs, Eigen::Ref<pyslam::MatNx2d> out) const {
    const int count = static_cast<int>(uvs.rows());
    if (out.rows() != count) {
        throw std::invalid_argument("undistort_points_into: output buffer must have one row per input point");
    }
    for (int i = 0; i < count; ++i) {
        Eigen::Vector3d bearing;
        const bool ok = unproject_bearing(uvs.row(i).transpose(), bearing, nullptr);
        if (!ok || std::abs(bearing.z()) <= 1e-12) {
            out(i, 0) = std::numeric_limits<double>::quiet_NaN();
            out(i, 1) = std::numeric_limits<double>::quiet_NaN();
        } else {
            out(i, 0) = fx * bearing.x() / bearing.z() + cx;
            out(i, 1) = fy * bearing.y() / bearing.z() + cy;
        }
    }
}

void CameraModel::unproject_points_3d_into(pyslam::MatNx2dRef uvs, pyslam::VecNdRef depths,
                                           Eigen::Ref<pyslam::MatNx3d> out) const {
    const int count = static_cast<int>(uvs.rows());
    if (out.rows() != count || depths.size() != count) {
        throw std::invalid_argument("unproject_points_3d_into: output buffer must have one row per input point");
    }
    for (int i = 0; i < count; ++i) {
        Eigen::Vector3d bearing;
        const bool ok = unproject_bearing(uvs.row(i).transpose(), bearing, nullptr);
        if (!ok || std::abs(bearing.z()) <= 1e-12) {
            out.row(i).setConstant(std::numeric_limits<double>::quiet_NaN());
        } else {
            const double depth = depths[i];
            out(i, 0) = depth * bearing.x() / bearing.z();
            out(i, 1) = depth * bearing.y() / bearing.z();
            out(i, 2) = depth;
        }
    }
}

void CameraModel::triangulate_stereo_into(pyslam::MatNx2dRef left, pyslam::MatNx2dRef right, double max_error,
                                          Eigen::Ref<pyslam::MatNx3d> points, Eigen::Ref<pyslam::VecNd> valid) const {
    if (!right_camera || left.rows() != right.rows()) {
        throw std::invalid_argument("triangulate_stereo_into requires a rig and paired pixel arrays");
    }
    const int count = static_cast<int>(left.rows());
    if (points.rows() != count || valid.size() != count) {
        throw std::invalid_argument("triangulate_stereo_into: output buffers must have one row per input point");
    }
    points.setZero();
    valid.setZero();
    for (int i = 0; i < count; ++i) {
        Eigen::Vector3d X;
        if (!triangulate_stereo_point(left.row(i).transpose(), right.row(i).transpose(), max_error, X)) {
            continue;
        }
        points.row(i) = X.transpose();
        valid[i] = 1.0;
    }
}

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>
search_stereo_rig(const CameraModel &camera, pyslam::MatNx2fRef kps, Eigen::Ref<const Eigen::VectorXi> octaves,
                  const cv::Mat &des, pyslam::MatNx2fRef kps_r, Eigen::Ref<const Eigen::VectorXi> octaves_r,
                  const cv::Mat &des_r, float max_descriptor_distance, float ratio_test,
                  Eigen::Ref<const Eigen::VectorXf> level_sigmas2) {
    const auto sigma2_of = [&level_sigmas2](int octave) -> double {
        return (octave >= 0 && octave < level_sigmas2.size()) ? static_cast<double>(level_sigmas2[octave]) : 1.0;
    };
    return pyslam::camera_model::search_stereo_rig(camera, kps, octaves, des, kps_r, octaves_r, des_r,
                                                   max_descriptor_distance, ratio_test, sigma2_of,
                                                   pyslam::Parameters::kMinDepth);
}

} // namespace camera_models
