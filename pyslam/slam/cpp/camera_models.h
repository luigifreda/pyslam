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

#include "camera.h"

namespace pyslam {

// Kannala-Brandt with four distortion coefficients k1..k4 (eight parameters
// in total, the model ORB-SLAM3 calls KannalaBrandt8). D = [k1, k2, k3, k4].
// unproject_bearing returns (sin(theta) * mx / thetad, sin(theta) * my / thetad, cos(theta)).
class KannalaBrandt4Camera : public Camera {
  public:
    KannalaBrandt4Camera();
    explicit KannalaBrandt4Camera(const ConfigDict &config);
    KannalaBrandt4Camera(int width, int height, double fx, double fy, double cx, double cy,
                         double k1, double k2, double k3, double k4);

    bool project_with_jacobian(Vec3dRef Xc, Vec2d &uv,
                               Eigen::Matrix<double, 2, 3> *J_uv_Xc) const override;
    bool project_with_jacobian(Vec3fRef Xc, Vec2f &uv,
                               Eigen::Matrix<float, 2, 3> *J_uv_Xc) const override;
    bool unproject_bearing(Vec2dRef uv, Vec3d &bearing,
                           Eigen::Matrix<double, 3, 2> *J_bearing_uv) const override;
    bool unproject_bearing(Vec2fRef uv, Vec3f &bearing,
                           Eigen::Matrix<float, 3, 2> *J_bearing_uv) const override;

    std::pair<Vec2<float>, float> project_point(Vec3Ref<float> xcs) const override;
    std::pair<Vec2<double>, double> project_point(Vec3Ref<double> xcs) const override;
    std::pair<MatNx2d, VecNd> project(MatNx3dRef xcs) const override;
    std::pair<MatNx2f, VecNf> project(MatNx3fRef xcs) const override;

    std::pair<Vec3<float>, float> project_point_stereo(Vec3Ref<float> xcs) const override;
    std::pair<Vec3<double>, double> project_point_stereo(Vec3Ref<double> xcs) const override;
    std::pair<MatNx3d, VecNd> project_stereo(MatNx3dRef xcs) const override;
    std::pair<MatNx3f, VecNf> project_stereo(MatNx3fRef xcs) const override;

    MatNx2d unproject_points(MatNx2dRef uvs) const override;
    MatNx2f unproject_points(MatNx2fRef uvs) const override;
    MatNx3d unproject_points_3d(MatNx2dRef uvs, VecNdRef depths) const override;
    MatNx3f unproject_points_3d(MatNx2fRef uvs, VecNfRef depths) const override;

    // Detected pixel -> pinhole image of the same ray. project() still predicts
    // the detected pixel.
    MatNx2d undistort_points(MatNx2dRef uvs) override;
    MatNx2f undistort_points(MatNx2fRef uvs) override;
    bool projects_raw_keypoints() const override { return true; }
    bool requires_positive_depth() const override { return false; }

    static KannalaBrandt4Camera from_json(const std::string &json_str);

#ifdef USE_PYTHON
    pybind11::tuple state_tuple() const;
    void restore_from_state(const pybind11::tuple &t);
#endif
};

// OpenCV rational-radial pinhole. D = [k1, k2, p1, p2, k3, k4, k5, k6].
// unproject_bearing returns the undistorted point on the plane z = 1.
class PinholeRadtan8Camera : public Camera {
  public:
    PinholeRadtan8Camera();
    explicit PinholeRadtan8Camera(const ConfigDict &config);
    PinholeRadtan8Camera(int width, int height, double fx, double fy, double cx, double cy,
                         double k1, double k2, double p1, double p2, double k3, double k4,
                         double k5, double k6);

    bool project_with_jacobian(Vec3dRef Xc, Vec2d &uv,
                               Eigen::Matrix<double, 2, 3> *J_uv_Xc) const override;
    bool project_with_jacobian(Vec3fRef Xc, Vec2f &uv,
                               Eigen::Matrix<float, 2, 3> *J_uv_Xc) const override;
    bool unproject_bearing(Vec2dRef uv, Vec3d &bearing,
                           Eigen::Matrix<double, 3, 2> *J_bearing_uv) const override;
    bool unproject_bearing(Vec2fRef uv, Vec3f &bearing,
                           Eigen::Matrix<float, 3, 2> *J_bearing_uv) const override;

    std::pair<Vec2<float>, float> project_point(Vec3Ref<float> xcs) const override;
    std::pair<Vec2<double>, double> project_point(Vec3Ref<double> xcs) const override;
    std::pair<MatNx2d, VecNd> project(MatNx3dRef xcs) const override;
    std::pair<MatNx2f, VecNf> project(MatNx3fRef xcs) const override;

    std::pair<Vec3<float>, float> project_point_stereo(Vec3Ref<float> xcs) const override;
    std::pair<Vec3<double>, double> project_point_stereo(Vec3Ref<double> xcs) const override;
    std::pair<MatNx3d, VecNd> project_stereo(MatNx3dRef xcs) const override;
    std::pair<MatNx3f, VecNf> project_stereo(MatNx3fRef xcs) const override;

    MatNx2d unproject_points(MatNx2dRef uvs) const override;
    MatNx2f unproject_points(MatNx2fRef uvs) const override;
    MatNx3d unproject_points_3d(MatNx2dRef uvs, VecNdRef depths) const override;
    MatNx3f unproject_points_3d(MatNx2fRef uvs, VecNfRef depths) const override;

    MatNx2d undistort_points(MatNx2dRef uvs) override;
    MatNx2f undistort_points(MatNx2fRef uvs) override;
    bool projects_raw_keypoints() const override { return true; }

    static PinholeRadtan8Camera from_json(const std::string &json_str);

#ifdef USE_PYTHON
    pybind11::tuple state_tuple() const;
    void restore_from_state(const pybind11::tuple &t);
#endif
};

// Double sphere. D = [xi, alpha].
// unproject_bearing returns the closed-form direction, which is not on z = 1.
class DoubleSphereCamera : public Camera {
  public:
    DoubleSphereCamera();
    explicit DoubleSphereCamera(const ConfigDict &config);
    DoubleSphereCamera(int width, int height, double fx, double fy, double cx, double cy, double xi,
                       double alpha);

    bool project_with_jacobian(Vec3dRef Xc, Vec2d &uv,
                               Eigen::Matrix<double, 2, 3> *J_uv_Xc) const override;
    bool project_with_jacobian(Vec3fRef Xc, Vec2f &uv,
                               Eigen::Matrix<float, 2, 3> *J_uv_Xc) const override;
    bool unproject_bearing(Vec2dRef uv, Vec3d &bearing,
                           Eigen::Matrix<double, 3, 2> *J_bearing_uv) const override;
    bool unproject_bearing(Vec2fRef uv, Vec3f &bearing,
                           Eigen::Matrix<float, 3, 2> *J_bearing_uv) const override;

    std::pair<Vec2<float>, float> project_point(Vec3Ref<float> xcs) const override;
    std::pair<Vec2<double>, double> project_point(Vec3Ref<double> xcs) const override;
    std::pair<MatNx2d, VecNd> project(MatNx3dRef xcs) const override;
    std::pair<MatNx2f, VecNf> project(MatNx3fRef xcs) const override;

    std::pair<Vec3<float>, float> project_point_stereo(Vec3Ref<float> xcs) const override;
    std::pair<Vec3<double>, double> project_point_stereo(Vec3Ref<double> xcs) const override;
    std::pair<MatNx3d, VecNd> project_stereo(MatNx3dRef xcs) const override;
    std::pair<MatNx3f, VecNf> project_stereo(MatNx3fRef xcs) const override;

    MatNx2d unproject_points(MatNx2dRef uvs) const override;
    MatNx2f unproject_points(MatNx2fRef uvs) const override;
    MatNx3d unproject_points_3d(MatNx2dRef uvs, VecNdRef depths) const override;
    MatNx3f unproject_points_3d(MatNx2fRef uvs, VecNfRef depths) const override;

    MatNx2d undistort_points(MatNx2dRef uvs) override;
    MatNx2f undistort_points(MatNx2fRef uvs) override;
    bool projects_raw_keypoints() const override { return true; }
    bool requires_positive_depth() const override { return false; }

    static DoubleSphereCamera from_json(const std::string &json_str);

#ifdef USE_PYTHON
    pybind11::tuple state_tuple() const;
    void restore_from_state(const pybind11::tuple &t);
#endif
};

} // namespace pyslam
