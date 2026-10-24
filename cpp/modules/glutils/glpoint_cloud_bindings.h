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

#include "glpoint_cloud.h"
#include "glutils_bindings_utils.h"

namespace glutils_bindings_detail {

template <typename PointT, typename ColorT = float>
void BindGlPointCloud(py::module_ &m, const char *name) {
    using GlPointCloudT = glutils::GlPointCloudT<PointT, ColorT>;
    py::class_<GlPointCloudT>(m, name)
        .def(py::init<>())
        .def("draw", &GlPointCloudT::Draw, py::call_guard<py::gil_scoped_release>())
        .def("set", &GlPointCloudT::Set, py::arg("points"), py::arg("colors"),
             py::arg("point_count"), py::call_guard<py::gil_scoped_release>())
        .def("set_points", &GlPointCloudT::SetPoints, py::arg("points"), py::arg("point_count"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_colors",
             static_cast<void (GlPointCloudT::*)(const ColorT *)>(&GlPointCloudT::SetColors),
             py::arg("colors"), py::call_guard<py::gil_scoped_release>())
        // Zero-copy bindings: accept contiguous numpy arrays without dtype casting.
        .def(
            "set",
            [](GlPointCloudT &self, const py::array_t<PointT, py::array::c_style> &points,
               const py::array_t<ColorT, py::array::c_style> &colors) {
                ValidatePointsArray(points);
                ValidateColorsArray(colors);
                auto points_info = points.request();
                auto colors_info = colors.request();
                if (colors_info.shape[0] != points_info.shape[0]) {
                    throw std::runtime_error("points and colors must have the same length");
                }
                const auto *point_data = static_cast<const PointT *>(points_info.ptr);
                const auto *color_data = static_cast<const ColorT *>(colors_info.ptr);
                const auto point_count = static_cast<std::size_t>(points_info.shape[0]);
                py::gil_scoped_release release;
                self.Set(point_data, color_data, point_count);
            },
            py::arg("points"), py::arg("colors"))
        .def(
            "set_points",
            [](GlPointCloudT &self, const py::array_t<PointT, py::array::c_style> &points) {
                ValidatePointsArray(points);
                auto info = points.request();
                const auto *point_data = static_cast<const PointT *>(info.ptr);
                const auto point_count = static_cast<std::size_t>(info.shape[0]);
                py::gil_scoped_release release;
                self.SetPoints(point_data, point_count);
            },
            py::arg("points"))
        .def(
            "set_colors",
            [](GlPointCloudT &self, const py::array_t<ColorT, py::array::c_style> &colors) {
                ValidateColorsArray(colors);
                auto info = colors.request();
                const auto *color_data = static_cast<const ColorT *>(info.ptr);
                const auto color_count = static_cast<std::size_t>(info.shape[0]);
                py::gil_scoped_release release;
                self.SetColors(color_data, color_count);
            },
            py::arg("colors"));
}

template <typename PointT, typename ColorT = float>
void BindGlPointCloudDirect(py::module_ &m, const char *name) {
    using GlPointCloudDirectT = glutils::GlPointCloudDirectT<PointT, ColorT>;
    py::class_<GlPointCloudDirectT>(m, name)
        .def(py::init<>())
        .def("draw", &GlPointCloudDirectT::Draw, py::call_guard<py::gil_scoped_release>())
        .def("clear", &GlPointCloudDirectT::Clear, py::call_guard<py::gil_scoped_release>())
        .def("update", &GlPointCloudDirectT::Update, py::arg("points"), py::arg("colors"),
             py::arg("point_count"), py::call_guard<py::gil_scoped_release>())
        // Prefer zero-copy views; fall back to force-cast when needed.
        .def(
            "update",
            [](GlPointCloudDirectT &self, const py::array &points_obj,
               const py::object &colors_obj) {
                auto points_view = py::array_t<PointT, py::array::c_style>::ensure(points_obj);
                if (!points_view) {
                    points_view = py::array_t<PointT, py::array::c_style | py::array::forcecast>(
                        points_obj);
                }
                ValidatePointsArray(points_view);
                auto points_info = points_view.request();
                const auto *point_data = static_cast<const PointT *>(points_info.ptr);
                const auto point_count = static_cast<std::size_t>(points_info.shape[0]);

                const ColorT *color_data = nullptr;
                if (!colors_obj.is_none()) {
                    auto colors_view =
                        py::array_t<ColorT, py::array::c_style>::ensure(colors_obj);
                    if (!colors_view) {
                        colors_view = py::array_t<ColorT,
                                                  py::array::c_style | py::array::forcecast>(
                            colors_obj);
                    }
                    ValidateColorsArray(colors_view);
                    auto colors_info = colors_view.request();
                    if (colors_info.shape[0] != points_info.shape[0]) {
                        throw std::runtime_error("points and colors must have the same length");
                    }
                    color_data = static_cast<const ColorT *>(colors_info.ptr);
                }
                py::gil_scoped_release release;
                self.Update(point_data, color_data, point_count);
            },
            py::arg("points"), py::arg("colors") = py::none());
}

} // namespace glutils_bindings_detail
