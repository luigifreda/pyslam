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

// Python-facing camera used when USE_CPP_CORE is false. Projection calls
// pyslam::camera_model in camera_model_math.h. This type is not the SLAM Camera
// owned by cpp_core.

#include "eigen_aliases.h"

#include <Eigen/Core>

#include <memory>
#include <tuple>
#include <utility>
#include <vector>

#include <opencv2/core.hpp>

namespace camera_models {

enum class CameraType {
    NONE = 0,
    PINHOLE = 1,
    KANNALA_BRANDT4 = 2,
    PINHOLE_RADTAN8 = 3,
    DOUBLE_SPHERE = 4
};

enum class SensorType { MONOCULAR = 0, STEREO = 1, RGBD = 2 };

class CameraModel : public std::enable_shared_from_this<CameraModel> {
  public:
    CameraModel() = default;
    virtual ~CameraModel();

    int width = 0;
    int height = 0;
    double fx = 0.0;
    double fy = 0.0;
    double cx = 0.0;
    double cy = 0.0;
    double bf = -1.0;
    double b = -1.0;
    std::vector<double> D;
    CameraType type = CameraType::NONE;
    SensorType sensor_type = SensorType::MONOCULAR;
    std::shared_ptr<CameraModel> right_camera;
    Eigen::Matrix4d T_right_left = Eigen::Matrix4d::Identity();

    void set_intrinsic_matrices() {}
    // Keypoints are raw model pixels; the shared optimizer edges never take the pinhole path.
    bool projects_raw_keypoints() const { return true; }

    bool project_with_jacobian(const Eigen::Vector3d &Xc, Eigen::Vector2d &uv,
                               Eigen::Matrix<double, 2, 3> *J_uv_Xc) const;
    bool unproject_bearing(const Eigen::Vector2d &uv, Eigen::Vector3d &bearing,
                           Eigen::Matrix<double, 3, 2> *J_bearing_uv) const;
    bool project_stereo_with_jacobian(const Eigen::Vector3d &Xc, Eigen::Vector3d &uvur,
                                      Eigen::Matrix3d *J_uvu_Xc = nullptr) const;
    bool g2o_reprojection_error_jacobians(const Eigen::Vector3d &Xc, const Eigen::Matrix3d &Rcw,
                                          Eigen::Matrix<double, 2, 6> &J_error_pose,
                                          Eigen::Matrix<double, 2, 3> &J_error_point) const;
    bool triangulate_stereo_point(const Eigen::Vector2d &left, const Eigen::Vector2d &right,
                                  double max_error, Eigen::Vector3d &X) const;

    // Batch entry points write into caller-owned buffers. The Python bindings
    // pass those buffers with noconvert(), so a C-contiguous array is not copied.
    void project_into(pyslam::MatNx3dRef xcs, Eigen::Ref<pyslam::MatNx2d> uv,
                      Eigen::Ref<pyslam::VecNd> depth, Eigen::Ref<pyslam::VecNd> valid) const;
    void project_stereo_into(pyslam::MatNx3dRef xcs, Eigen::Ref<pyslam::MatNx3d> uvur,
                             Eigen::Ref<pyslam::VecNd> depth) const;
    void unproject_bearings_into(pyslam::MatNx2dRef uvs, Eigen::Ref<pyslam::MatNx3d> bearings,
                                 Eigen::Ref<pyslam::VecNd> valid) const;
    void undistort_points_into(pyslam::MatNx2dRef uvs, Eigen::Ref<pyslam::MatNx2d> out) const;
    void unproject_points_3d_into(pyslam::MatNx2dRef uvs, pyslam::VecNdRef depths,
                                  Eigen::Ref<pyslam::MatNx3d> out) const;
    void triangulate_stereo_into(pyslam::MatNx2dRef left, pyslam::MatNx2dRef right, double max_error,
                                 Eigen::Ref<pyslam::MatNx3d> points, Eigen::Ref<pyslam::VecNd> valid) const;
};

class KannalaBrandt4Camera : public CameraModel {
  public:
    KannalaBrandt4Camera() { type = CameraType::KANNALA_BRANDT4; }
};

class PinholeRadtan8Camera : public CameraModel {
  public:
    PinholeRadtan8Camera() { type = CameraType::PINHOLE_RADTAN8; }
};

class DoubleSphereCamera : public CameraModel {
  public:
    DoubleSphereCamera() { type = CameraType::DOUBLE_SPHERE; }
};

std::tuple<std::vector<int>, std::vector<int>, std::vector<double>>
search_stereo_rig(const CameraModel &camera, pyslam::MatNx2fRef kps, Eigen::Ref<const Eigen::VectorXi> octaves,
                  const cv::Mat &des, pyslam::MatNx2fRef kps_r, Eigen::Ref<const Eigen::VectorXi> octaves_r,
                  const cv::Mat &des_r, float max_descriptor_distance, float ratio_test,
                  Eigen::Ref<const Eigen::VectorXf> level_sigmas2);

} // namespace camera_models
