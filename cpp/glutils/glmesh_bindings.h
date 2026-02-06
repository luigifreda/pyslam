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

#include "glmesh.h"
#include "glutils_bindings_utils.h"

namespace glutils_bindings_detail {

template <typename VertexT, typename ColorT = float>
void BindGlMesh(py::module_ &m, const char *name) {
    using GlMeshT = glutils::GlMeshT<VertexT, ColorT>;
    py::class_<GlMeshT>(m, name)
        .def(py::init<>())
        .def("draw", &GlMeshT::Draw, py::arg("wireframe") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("set_vertices", &GlMeshT::SetVertices, py::arg("vertices"), py::arg("vertex_count"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_colors", &GlMeshT::SetColors, py::arg("colors"),
             py::call_guard<py::gil_scoped_release>())
        .def("clear_colors", &GlMeshT::ClearColors, py::call_guard<py::gil_scoped_release>())
        .def("set_triangles", &GlMeshT::SetTriangles, py::arg("triangles"), py::arg("tri_count"),
             py::call_guard<py::gil_scoped_release>())
        .def("reserve_gpu", &GlMeshT::ReserveGPU, py::arg("max_vertices"), py::arg("max_indices"),
             py::call_guard<py::gil_scoped_release>())
        // Prefer zero-copy views; fall back to force-cast when needed.
        .def(
            "set",
            [](GlMeshT &self, const py::array &vertices_obj, const py::array &triangles_obj,
               const py::object &colors_obj) {
                auto vertices_view = py::array_t<VertexT, py::array::c_style>::ensure(vertices_obj);
                if (!vertices_view) {
                    vertices_view = py::array_t<VertexT, py::array::c_style | py::array::forcecast>(
                        vertices_obj);
                }
                auto triangles_view =
                    py::array_t<unsigned int, py::array::c_style>::ensure(triangles_obj);
                if (!triangles_view) {
                    triangles_view =
                        py::array_t<unsigned int, py::array::c_style | py::array::forcecast>(
                            triangles_obj);
                }
                ValidatePointsArray(vertices_view);
                ValidateTrianglesArray(triangles_view);
                auto vertices_info = vertices_view.request();
                auto triangles_info = triangles_view.request();

                const auto *vertex_data = static_cast<const VertexT *>(vertices_info.ptr);
                const auto *tri_data = static_cast<const unsigned int *>(triangles_info.ptr);
                const auto vertex_count = static_cast<std::size_t>(vertices_info.shape[0]);
                const auto tri_count = static_cast<std::size_t>(triangles_info.shape[0]);
#ifndef NDEBUG
                const auto tri_index_count = tri_count * 3;
                if (tri_index_count > 0 && vertex_count == 0) {
                    throw std::runtime_error("triangles provided but vertex_count is zero");
                }
                for (std::size_t i = 0; i < tri_index_count; ++i) {
                    if (tri_data[i] >= vertex_count) {
                        throw std::runtime_error("triangle index out of range");
                    }
                }
#endif

                const ColorT *color_data = nullptr;
                if (!colors_obj.is_none()) {
                    auto colors_view = py::array_t<ColorT, py::array::c_style>::ensure(colors_obj);
                    if (!colors_view) {
                        colors_view =
                            py::array_t<ColorT, py::array::c_style | py::array::forcecast>(
                                colors_obj);
                    }
                    ValidateColorsArray(colors_view);
                    auto colors_info = colors_view.request();
                    if (colors_info.shape[0] != vertices_info.shape[0]) {
                        throw std::runtime_error("vertices and colors must have the same length");
                    }
                    color_data = static_cast<const ColorT *>(colors_info.ptr);
                }

                py::gil_scoped_release release;
                self.SetVertices(vertex_data, vertex_count);
                self.SetTriangles(tri_data, tri_count);
                if (color_data) {
                    self.SetColors(color_data);
                } else {
                    self.ClearColors();
                }
            },
            py::arg("vertices"), py::arg("triangles"), py::arg("colors") = py::none())
        .def(
            "set_vertices",
            [](GlMeshT &self, const py::array &vertices_obj) {
                auto vertices_view = py::array_t<VertexT, py::array::c_style>::ensure(vertices_obj);
                if (!vertices_view) {
                    vertices_view = py::array_t<VertexT, py::array::c_style | py::array::forcecast>(
                        vertices_obj);
                }
                ValidatePointsArray(vertices_view);
                auto info = vertices_view.request();
                const auto *vertex_data = static_cast<const VertexT *>(info.ptr);
                const auto vertex_count = static_cast<std::size_t>(info.shape[0]);
                py::gil_scoped_release release;
                self.SetVertices(vertex_data, vertex_count);
            },
            py::arg("vertices"))
        .def(
            "set_colors",
            [](GlMeshT &self, const py::array &colors_obj) {
                auto colors_view = py::array_t<ColorT, py::array::c_style>::ensure(colors_obj);
                if (!colors_view) {
                    colors_view =
                        py::array_t<ColorT, py::array::c_style | py::array::forcecast>(colors_obj);
                }
                ValidateColorsArray(colors_view);
                auto info = colors_view.request();
                if (static_cast<std::size_t>(info.shape[0]) != self.VertexCount()) {
                    throw std::runtime_error("vertices and colors must have the same length");
                }
                const auto *color_data = static_cast<const ColorT *>(info.ptr);
                py::gil_scoped_release release;
                self.SetColors(color_data);
            },
            py::arg("colors"))
        .def(
            "set_triangles",
            [](GlMeshT &self, const py::array &triangles_obj) {
                auto triangles_view =
                    py::array_t<unsigned int, py::array::c_style>::ensure(triangles_obj);
                if (!triangles_view) {
                    triangles_view =
                        py::array_t<unsigned int, py::array::c_style | py::array::forcecast>(
                            triangles_obj);
                }
                ValidateTrianglesArray(triangles_view);
                auto info = triangles_view.request();
                const auto *tri_data = static_cast<const unsigned int *>(info.ptr);
                const auto tri_count = static_cast<std::size_t>(info.shape[0]);
#ifndef NDEBUG
                const auto tri_index_count = tri_count * 3;
                const auto vertex_count = self.VertexCount();
                if (tri_index_count > 0 && vertex_count == 0) {
                    throw std::runtime_error("triangles provided but vertex_count is zero");
                }
                for (std::size_t i = 0; i < tri_index_count; ++i) {
                    if (tri_data[i] >= vertex_count) {
                        throw std::runtime_error("triangle index out of range");
                    }
                }
#endif
                py::gil_scoped_release release;
                self.SetTriangles(tri_data, tri_count);
            },
            py::arg("triangles"));
}

template <typename VertexT, typename ColorT = float>
void BindGlMeshDirect(py::module_ &m, const char *name) {
    using GlMeshDirectT = glutils::GlMeshDirectT<VertexT, ColorT>;
    py::class_<GlMeshDirectT>(m, name)
        .def(py::init<>())
        .def("draw", &GlMeshDirectT::Draw, py::arg("wireframe") = false,
             py::call_guard<py::gil_scoped_release>())
        .def("clear", &GlMeshDirectT::Clear, py::call_guard<py::gil_scoped_release>())
        .def("update", &GlMeshDirectT::Update, py::arg("vertices"), py::arg("colors"),
             py::arg("triangles"), py::arg("vertex_count"), py::arg("tri_count"),
             py::call_guard<py::gil_scoped_release>())
        // Prefer zero-copy views; fall back to force-cast when needed.
        .def(
            "update",
            [](GlMeshDirectT &self, const py::array &vertices_obj, const py::array &triangles_obj,
               const py::object &colors_obj) {
                auto vertices_view = py::array_t<VertexT, py::array::c_style>::ensure(vertices_obj);
                if (!vertices_view) {
                    vertices_view = py::array_t<VertexT, py::array::c_style | py::array::forcecast>(
                        vertices_obj);
                }
                auto triangles_view =
                    py::array_t<unsigned int, py::array::c_style>::ensure(triangles_obj);
                if (!triangles_view) {
                    triangles_view =
                        py::array_t<unsigned int, py::array::c_style | py::array::forcecast>(
                            triangles_obj);
                }
                ValidatePointsArray(vertices_view);
                ValidateTrianglesArray(triangles_view);
                auto vertices_info = vertices_view.request();
                auto triangles_info = triangles_view.request();

                const auto *vertex_data = static_cast<const VertexT *>(vertices_info.ptr);
                const auto *tri_data = static_cast<const unsigned int *>(triangles_info.ptr);
                const auto vertex_count = static_cast<std::size_t>(vertices_info.shape[0]);
                const auto tri_count = static_cast<std::size_t>(triangles_info.shape[0]);
#ifndef NDEBUG
                const auto tri_index_count = tri_count * 3;
                if (tri_index_count > 0 && vertex_count == 0) {
                    throw std::runtime_error("triangles provided but vertex_count is zero");
                }
                for (std::size_t i = 0; i < tri_index_count; ++i) {
                    if (tri_data[i] >= vertex_count) {
                        throw std::runtime_error("triangle index out of range");
                    }
                }
#endif

                const ColorT *color_data = nullptr;
                if (!colors_obj.is_none()) {
                    auto colors_view = py::array_t<ColorT, py::array::c_style>::ensure(colors_obj);
                    if (!colors_view) {
                        colors_view =
                            py::array_t<ColorT, py::array::c_style | py::array::forcecast>(
                                colors_obj);
                    }
                    ValidateColorsArray(colors_view);
                    auto colors_info = colors_view.request();
                    if (colors_info.shape[0] != vertices_info.shape[0]) {
                        throw std::runtime_error("vertices and colors must have the same length");
                    }
                    color_data = static_cast<const ColorT *>(colors_info.ptr);
                }

                py::gil_scoped_release release;
                self.Update(vertex_data, color_data, tri_data, vertex_count, tri_count);
            },
            py::arg("vertices"), py::arg("triangles"), py::arg("colors") = py::none());
}

} // namespace glutils_bindings_detail
