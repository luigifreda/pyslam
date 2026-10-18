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

// Stereo-rig geometry shared by cpp_core (pyslam::Camera) and the camera_models
// module (camera_models::CameraModel). CameraT must provide fx, cx, bf,
// right_camera (pointer-like), T_right_left, project_with_jacobian and
// unproject_bearing.

#include "utils/descriptor_helpers.h"

#include <Eigen/Core>
#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <vector>

namespace pyslam {
namespace camera_model {

// (u_left, v_left, u_right). With a rig, u_right comes from the right camera and the
// extrinsics; otherwise it is the virtual pinhole coordinate fx*x/z + cx - bf/z.
template <typename CameraT>
bool project_stereo_with_jacobian(const CameraT &camera, const Eigen::Vector3d &Xc,
                                  Eigen::Vector3d &uvur, Eigen::Matrix3d *J) {
    Eigen::Vector2d uv;
    Eigen::Matrix<double, 2, 3> JL;
    bool valid = camera.project_with_jacobian(Xc, uv, J ? &JL : nullptr);
    uvur.template head<2>() = uv;
    if (camera.right_camera) {
        const Eigen::Matrix3d R = camera.T_right_left.template topLeftCorner<3, 3>();
        const Eigen::Vector3d Xr = R * Xc + camera.T_right_left.template topRightCorner<3, 1>();
        Eigen::Vector2d ur;
        Eigen::Matrix<double, 2, 3> JR;
        valid = camera.right_camera->project_with_jacobian(Xr, ur, J ? &JR : nullptr) && valid;
        uvur[2] = ur[0];
        if (J) J->row(2) = JR.row(0) * R;
    } else {
        valid = valid && Xc.z() > 1e-12 && camera.bf > 0.0;
        if (valid) {
            const double iz = 1.0 / Xc.z();
            uvur[2] = camera.fx * Xc.x() * iz + camera.cx - camera.bf * iz;
            if (J) J->row(2) << camera.fx * iz, 0.0, (camera.bf - camera.fx * Xc.x()) * iz * iz;
        }
    }
    if (J) {
        if (valid) J->template topRows<2>() = JL;
        else J->setZero();
    }
    if (!valid) uvur.setConstant(2.0 * camera.fx);
    return valid && uvur.allFinite();
}

// Intersects the left and right rays of a rig observation. X is in the left camera and
// must reproject within max_error pixels in both images.
template <typename CameraT>
bool triangulate_stereo_point(const CameraT &camera, const Eigen::Vector2d &ul,
                              const Eigen::Vector2d &ur, double max_error, Eigen::Vector3d &X) {
    if (!camera.right_camera) return false;
    const Eigen::Matrix3d R = camera.T_right_left.template topLeftCorner<3, 3>();
    const Eigen::Vector3d t = camera.T_right_left.template topRightCorner<3, 1>();
    Eigen::Vector3d l, r;
    if (!camera.unproject_bearing(ul, l, nullptr) ||
        !camera.right_camera->unproject_bearing(ur, r, nullptr))
        return false;
    l.normalize();
    r.normalize();
    const Eigen::Vector3d a = R * l;
    const double c = a.dot(r), det = 1.0 - c * c;
    if (!(det > 1e-6)) return false;
    const double dl = (-a.dot(t) + c * r.dot(t)) / det;
    const double dr = (-c * a.dot(t) + r.dot(t)) / det;
    if (!(dl > 0.0 && dr > 0.0)) return false;
    const Eigen::Vector3d Xl = dl * l, Xr = R * Xl + t;
    Eigen::Vector2d pl, pr;
    if (!camera.project_with_jacobian(Xl, pl, nullptr) ||
        !camera.right_camera->project_with_jacobian(Xr, pr, nullptr))
        return false;
    const double max_error2 = max_error * max_error;
    if ((pl - ul).squaredNorm() > max_error2 || (pr - ur).squaredNorm() > max_error2) return false;
    X = Xl;
    return true;
}

// First-order pixel distance between the image of a unit bearing and the image of the
// plane (through the optical center) with the given unit normal.
template <typename CameraT>
double bearing_to_plane_pixel_distance(const CameraT &camera, const Eigen::Vector3d &bearing,
                                       const Eigen::Vector3d &unit_normal) {
    Eigen::Vector2d uv;
    Eigen::Matrix<double, 2, 3> J;
    if (!camera.project_with_jacobian(bearing, uv, &J))
        return std::numeric_limits<double>::infinity();
    return std::abs(unit_normal.dot(bearing)) * (J * unit_normal).norm();
}

// Stereo matching for a calibrated rig on raw model pixels. A right keypoint is a
// candidate when its ray lies on the left ray's epipolar plane between the point at
// infinity and the right center (positive depth), within the 1-DoF chi2 band of its
// octave, and at most one octave apart. The best descriptor must pass the ratio test;
// matches are one-to-one and triangulated against the 2-DoF chi2 bound.
// Octaves are any indexable containers with size(); sigma2_of(octave) gives sigma^2.
template <typename CameraT, typename KpsT, typename OctavesT, typename Sigma2Fn>
std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>
search_stereo_rig(const CameraT &camera, const KpsT &kps, const OctavesT &octaves,
                  const cv::Mat &des, const KpsT &kps_r, const OctavesT &octaves_r,
                  const cv::Mat &des_r, float max_descriptor_distance, float ratio_test,
                  Sigma2Fn sigma2_of, double min_depth) {
    std::vector<int> idxs_left, idxs_right;
    std::vector<double> depths;
    const auto *right = camera.right_camera.get();
    const int num_left = static_cast<int>(kps.rows());
    const int num_right = static_cast<int>(kps_r.rows());
    if (!right || num_left == 0 || num_right == 0 || des.rows != num_left ||
        des_r.rows != num_right || static_cast<int>(octaves.size()) != num_left ||
        static_cast<int>(octaves_r.size()) != num_right) {
        return {std::move(idxs_left), std::move(idxs_right), std::move(depths)};
    }

    const NormType norm_type = des.type() == CV_8U ? NormType::Hamming : NormType::L2;
    constexpr double kChi2OneDof = 3.84;
    constexpr double kChi2TwoDof = 5.991;
    constexpr double kMaxBandSin = 0.1; // cheap prefilter, far wider than any accepted band

    const Eigen::Matrix3d R = camera.T_right_left.template topLeftCorner<3, 3>();
    const Eigen::Vector3d t = camera.T_right_left.template topRightCorner<3, 1>();

    std::vector<Eigen::Vector3d> bearings_r(num_right);
    std::vector<Eigen::Matrix<double, 2, 3>> jacobians_r(num_right);
    std::vector<char> valid_r(num_right, 0);
    for (int j = 0; j < num_right; ++j) {
        Eigen::Vector3d f;
        if (!right->unproject_bearing(Eigen::Vector2d(kps_r(j, 0), kps_r(j, 1)), f, nullptr) ||
            !(f.norm() > 0.0))
            continue;
        f.normalize();
        Eigen::Vector2d uv;
        if (!right->project_with_jacobian(f, uv, &jacobians_r[j])) continue;
        bearings_r[j] = f;
        valid_r[j] = 1;
    }

    std::vector<int> best_right(num_left, -1);
    std::vector<float> best_distance(num_left, std::numeric_limits<float>::infinity());
    for (int i = 0; i < num_left; ++i) {
        Eigen::Vector3d f;
        if (!camera.unproject_bearing(Eigen::Vector2d(kps(i, 0), kps(i, 1)), f, nullptr) ||
            !(f.norm() > 0.0))
            continue;
        // The right image of the left ray X = d * f, d > 0, spans the directions between
        // R * f (d -> inf) and t (d -> 0) on the epipolar plane with normal t x (R * f).
        const Eigen::Vector3d at_infinity = R * f.normalized();
        Eigen::Vector3d normal = t.cross(at_infinity);
        const double normal_norm = normal.norm();
        if (!(normal_norm > 1e-9)) continue;
        normal /= normal_norm;
        const Eigen::Vector3d lower_bound = normal.cross(at_infinity);
        const Eigen::Vector3d upper_bound = t.cross(normal);

        const cv::Mat descriptor = des.row(i);
        float best = std::numeric_limits<float>::infinity();
        float second = std::numeric_limits<float>::infinity();
        int best_j = -1;
        for (int j = 0; j < num_right; ++j) {
            if (!valid_r[j] || std::abs(octaves[i] - octaves_r[j]) > 1) continue;
            const Eigen::Vector3d &fr = bearings_r[j];
            const double sin_band = std::abs(normal.dot(fr));
            if (sin_band > kMaxBandSin) continue;
            // Scalar triple products: the two fixed cross products belong outside
            // this O(N_left * N_right) loop.
            if (fr.dot(lower_bound) > 0.0 || fr.dot(upper_bound) > 0.0) continue;
            const double distance_px = sin_band * (jacobians_r[j] * normal).norm();
            if (distance_px * distance_px > kChi2OneDof * sigma2_of(octaves_r[j])) continue;
            const float d = descriptor_distance(descriptor, des_r.row(j), norm_type);
            if (d < best) {
                second = best;
                best = d;
                best_j = j;
            } else if (d < second) {
                second = d;
            }
        }
        if (best_j < 0 || best > max_descriptor_distance) continue;
        if (std::isfinite(second) && !(best < ratio_test * second)) continue;
        best_right[i] = best_j;
        best_distance[i] = best;
    }

    std::vector<int> owner(num_right, -1);
    for (int i = 0; i < num_left; ++i) {
        const int j = best_right[i];
        if (j >= 0 && (owner[j] < 0 || best_distance[i] < best_distance[owner[j]])) owner[j] = i;
    }

    idxs_left.reserve(num_left);
    idxs_right.reserve(num_left);
    depths.reserve(num_left);
    for (int i = 0; i < num_left; ++i) {
        const int j = best_right[i];
        if (j < 0 || owner[j] != i) continue;
        const double max_error =
            std::sqrt(kChi2TwoDof * sigma2_of(std::max<int>(octaves[i], octaves_r[j])));
        Eigen::Vector3d X;
        if (!triangulate_stereo_point(camera, Eigen::Vector2d(kps(i, 0), kps(i, 1)),
                                      Eigen::Vector2d(kps_r(j, 0), kps_r(j, 1)), max_error, X) ||
            !(X.z() > min_depth))
            continue;
        idxs_left.push_back(i);
        idxs_right.push_back(j);
        depths.push_back(X.z());
    }
    return {std::move(idxs_left), std::move(idxs_right), std::move(depths)};
}

} // namespace camera_model
} // namespace pyslam
