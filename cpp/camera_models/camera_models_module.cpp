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

#include "opencv_type_casters.h"

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <cstring>
#include <tuple>
#include <vector>

namespace py = pybind11;

namespace {

template <typename CameraT> void bind_model(py::module &m, const char *name) {
    py::class_<CameraT, camera_models::CameraModel, std::shared_ptr<CameraT>>(m, name).def(py::init<>());
}

template <typename T> py::array_t<T> copied_array(const std::vector<T> &values) {
    py::array_t<T> out(values.size());
    if (!values.empty()) {
        std::memcpy(out.mutable_data(), values.data(), values.size() * sizeof(T));
    }
    return out;
}

} // namespace

PYBIND11_MODULE(camera_models, m) {
    m.doc() = "Nonlinear camera projection and stereo-rig matching. "
              "Shares camera_model_math.h with cpp_core. g2o edges and gtsam factors "
              "are separate modules and are not loaded from here.";

    py::enum_<camera_models::CameraType>(m, "CameraType")
        .value("NONE", camera_models::CameraType::NONE)
        .value("PINHOLE", camera_models::CameraType::PINHOLE)
        .value("KANNALA_BRANDT4", camera_models::CameraType::KANNALA_BRANDT4)
        .value("PINHOLE_RADTAN8", camera_models::CameraType::PINHOLE_RADTAN8)
        .value("DOUBLE_SPHERE", camera_models::CameraType::DOUBLE_SPHERE);

    py::enum_<camera_models::SensorType>(m, "SensorType")
        .value("MONOCULAR", camera_models::SensorType::MONOCULAR)
        .value("STEREO", camera_models::SensorType::STEREO)
        .value("RGBD", camera_models::SensorType::RGBD);

    py::class_<camera_models::CameraModel, std::shared_ptr<camera_models::CameraModel>>(m, "CameraModel")
        .def(py::init<>())
        .def_readwrite("width", &camera_models::CameraModel::width)
        .def_readwrite("height", &camera_models::CameraModel::height)
        .def_readwrite("fx", &camera_models::CameraModel::fx)
        .def_readwrite("fy", &camera_models::CameraModel::fy)
        .def_readwrite("cx", &camera_models::CameraModel::cx)
        .def_readwrite("cy", &camera_models::CameraModel::cy)
        .def_readwrite("bf", &camera_models::CameraModel::bf)
        .def_readwrite("b", &camera_models::CameraModel::b)
        .def_readwrite("D", &camera_models::CameraModel::D)
        .def_readwrite("type", &camera_models::CameraModel::type)
        .def_readwrite("sensor_type", &camera_models::CameraModel::sensor_type)
        .def_readwrite("right_camera", &camera_models::CameraModel::right_camera)
        .def_readwrite("T_right_left", &camera_models::CameraModel::T_right_left)
        .def("set_intrinsic_matrices", &camera_models::CameraModel::set_intrinsic_matrices)
        .def(
            "project_into",
            [](const camera_models::CameraModel &self, pyslam::MatNx3dRef xcs, Eigen::Ref<pyslam::MatNx2d> uv,
               Eigen::Ref<pyslam::VecNd> depth, Eigen::Ref<pyslam::VecNd> valid) {
                py::gil_scoped_release release;
                self.project_into(xcs, uv, depth, valid);
            },
            py::arg("xcs").noconvert(), py::arg("uv").noconvert(), py::arg("depth").noconvert(),
            py::arg("valid").noconvert())
        .def(
            "project_stereo_into",
            [](const camera_models::CameraModel &self, pyslam::MatNx3dRef xcs, Eigen::Ref<pyslam::MatNx3d> uvur,
               Eigen::Ref<pyslam::VecNd> depth) {
                py::gil_scoped_release release;
                self.project_stereo_into(xcs, uvur, depth);
            },
            py::arg("xcs").noconvert(), py::arg("uvur").noconvert(), py::arg("depth").noconvert())
        .def(
            "unproject_bearings_into",
            [](const camera_models::CameraModel &self, pyslam::MatNx2dRef uvs, Eigen::Ref<pyslam::MatNx3d> bearings,
               Eigen::Ref<pyslam::VecNd> valid) {
                py::gil_scoped_release release;
                self.unproject_bearings_into(uvs, bearings, valid);
            },
            py::arg("uvs").noconvert(), py::arg("bearings").noconvert(), py::arg("valid").noconvert())
        .def(
            "undistort_points_into",
            [](const camera_models::CameraModel &self, pyslam::MatNx2dRef uvs, Eigen::Ref<pyslam::MatNx2d> out) {
                py::gil_scoped_release release;
                self.undistort_points_into(uvs, out);
            },
            py::arg("uvs").noconvert(), py::arg("out").noconvert())
        .def(
            "unproject_points_3d_into",
            [](const camera_models::CameraModel &self, pyslam::MatNx2dRef uvs, pyslam::VecNdRef depths,
               Eigen::Ref<pyslam::MatNx3d> out) {
                py::gil_scoped_release release;
                self.unproject_points_3d_into(uvs, depths, out);
            },
            py::arg("uvs").noconvert(), py::arg("depths").noconvert(), py::arg("out").noconvert())
        .def(
            "triangulate_stereo_into",
            [](const camera_models::CameraModel &self, pyslam::MatNx2dRef left, pyslam::MatNx2dRef right,
               double max_error, Eigen::Ref<pyslam::MatNx3d> points, Eigen::Ref<pyslam::VecNd> valid) {
                py::gil_scoped_release release;
                self.triangulate_stereo_into(left, right, max_error, points, valid);
            },
            py::arg("left").noconvert(), py::arg("right").noconvert(), py::arg("max_error"),
            py::arg("points").noconvert(), py::arg("valid").noconvert())
        .def(
            "project_with_jacobian",
            [](const camera_models::CameraModel &self, pyslam::Vec3dRef point) {
                Eigen::Vector2d uv;
                Eigen::Matrix<double, 2, 3> jacobian;
                const bool valid = self.project_with_jacobian(point, uv, &jacobian);
                return std::make_tuple(uv, valid, jacobian);
            },
            py::arg("point").noconvert())
        .def(
            "unproject_bearing",
            [](const camera_models::CameraModel &self, pyslam::Vec2dRef uv) {
                Eigen::Vector3d bearing;
                Eigen::Matrix<double, 3, 2> jacobian;
                const bool valid = self.unproject_bearing(uv, bearing, &jacobian);
                return std::make_tuple(bearing, valid, jacobian);
            },
            py::arg("uv").noconvert())
        .def(
            "project_stereo_with_jacobian",
            [](const camera_models::CameraModel &self, pyslam::Vec3dRef point) {
                Eigen::Vector3d uv;
                Eigen::Matrix3d jacobian;
                const bool valid = self.project_stereo_with_jacobian(point, uv, &jacobian);
                return std::make_tuple(uv, valid, jacobian);
            },
            py::arg("point").noconvert());

    bind_model<camera_models::KannalaBrandt4Camera>(m, "KannalaBrandt4Camera");
    bind_model<camera_models::PinholeRadtan8Camera>(m, "PinholeRadtan8Camera");
    bind_model<camera_models::DoubleSphereCamera>(m, "DoubleSphereCamera");

    m.def("_shared_camera", [](const py::handle &obj) {
        auto *held = new std::shared_ptr<camera_models::CameraModel>(
            obj.cast<std::shared_ptr<camera_models::CameraModel>>());
        return py::capsule(held, [](void *pointer) {
            delete static_cast<std::shared_ptr<camera_models::CameraModel> *>(pointer);
        });
    });

    struct EpipolarMatcher {};
    py::class_<EpipolarMatcher>(m, "EpipolarMatcher")
        .def_static(
            "search_stereo_rig",
            [](const camera_models::CameraModel &camera, pyslam::MatNx2fRef kps,
               Eigen::Ref<const Eigen::VectorXi> octaves, const cv::Mat &des, pyslam::MatNx2fRef kps_r,
               Eigen::Ref<const Eigen::VectorXi> octaves_r, const cv::Mat &des_r, float max_descriptor_distance,
               float ratio_test, Eigen::Ref<const Eigen::VectorXf> level_sigmas2) {
                std::vector<int> left, right;
                std::vector<double> depths;
                {
                    py::gil_scoped_release release;
                    std::tie(left, right, depths) =
                        camera_models::search_stereo_rig(camera, kps, octaves, des, kps_r, octaves_r, des_r,
                                                         max_descriptor_distance, ratio_test, level_sigmas2);
                }
                return std::make_tuple(copied_array(left), copied_array(right), copied_array(depths));
            },
            py::arg("camera"), py::arg("kps").noconvert(), py::arg("octaves").noconvert(), py::arg("des"),
            py::arg("kps_r").noconvert(), py::arg("octaves_r").noconvert(), py::arg("des_r"),
            py::arg("max_descriptor_distance"), py::arg("ratio_test"),
            py::arg("level_sigmas2") = Eigen::VectorXf());
}
