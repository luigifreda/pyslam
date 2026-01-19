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

#include "glmesh.h"
#include "glpoint_cloud.h"
#include "glutils_bindings.h"
#include "glutils_camera.h"
#include "glutils_common.h"

namespace {

template <typename PointT, int Flags>
void ValidatePointsArray(const py::array_t<PointT, Flags> &points) {
    auto info = points.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("points must be an Nx3 array");
    }
}

template <typename ColorT, int Flags>
void ValidateColorsArray(const py::array_t<ColorT, Flags> &colors) {
    auto info = colors.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("colors must be an Nx3 array");
    }
}

template <typename IndexT, int Flags>
void ValidateTrianglesArray(const py::array_t<IndexT, Flags> &triangles) {
    auto info = triangles.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("triangles must be an Nx3 array");
    }
}

template <typename PointT, typename ColorT = float>
void BindGlPointCloud(py::module_ &m, const char *name) {
    using GlPointCloudT = glutils::GlPointCloudT<PointT, ColorT>;
    py::class_<GlPointCloudT>(m, name)
        .def(py::init<>())
        .def("draw", &GlPointCloudT::Draw)
        .def("set", &GlPointCloudT::Set, py::arg("points"), py::arg("colors"),
             py::arg("point_count"))
        .def("set_points", &GlPointCloudT::SetPoints, py::arg("points"), py::arg("point_count"))
        .def("set_colors",
             static_cast<void (GlPointCloudT::*)(const ColorT *)>(&GlPointCloudT::SetColors),
             py::arg("colors"))
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
                self.SetColors(color_data, color_count);
            },
            py::arg("colors"));
}

template <typename PointT, typename ColorT = float>
void BindGlPointCloudDirect(py::module_ &m, const char *name) {
    using GlPointCloudDirectT = glutils::GlPointCloudDirectT<PointT, ColorT>;
    py::class_<GlPointCloudDirectT>(m, name)
        .def(py::init<>())
        .def("draw", &GlPointCloudDirectT::Draw)
        .def("update_and_draw", &GlPointCloudDirectT::UpdateAndDraw, py::arg("points"),
             py::arg("colors"), py::arg("point_count"))
        // Zero-copy bindings: accept contiguous numpy arrays without dtype casting.
        .def(
            "update_and_draw",
            [](GlPointCloudDirectT &self, const py::array_t<PointT, py::array::c_style> &points,
               const py::object &colors_obj) {
                ValidatePointsArray(points);
                auto points_info = points.request();
                const auto *point_data = static_cast<const PointT *>(points_info.ptr);
                const auto point_count = static_cast<std::size_t>(points_info.shape[0]);

                const ColorT *color_data = nullptr;
                if (!colors_obj.is_none()) {
                    auto colors = colors_obj.cast<py::array_t<ColorT, py::array::c_style>>();
                    ValidateColorsArray(colors);
                    auto colors_info = colors.request();
                    if (colors_info.shape[0] != points_info.shape[0]) {
                        throw std::runtime_error("points and colors must have the same length");
                    }
                    color_data = static_cast<const ColorT *>(colors_info.ptr);
                }
                self.UpdateAndDraw(point_data, color_data, point_count);
            },
            py::arg("points"), py::arg("colors") = py::none())
        // Fallback binding: allow dtype conversion if needed (may copy).
        .def(
            "update_and_draw",
            [](GlPointCloudDirectT &self, const py::array &points_obj,
               const py::object &colors_obj) {
                auto points =
                    py::array_t<PointT, py::array::c_style | py::array::forcecast>(points_obj);
                ValidatePointsArray(points);
                auto points_info = points.request();
                const auto *point_data = static_cast<const PointT *>(points_info.ptr);
                const auto point_count = static_cast<std::size_t>(points_info.shape[0]);

                const ColorT *color_data = nullptr;
                if (!colors_obj.is_none()) {
                    auto colors =
                        py::array_t<ColorT, py::array::c_style | py::array::forcecast>(colors_obj);
                    ValidateColorsArray(colors);
                    auto colors_info = colors.request();
                    if (colors_info.shape[0] != points_info.shape[0]) {
                        throw std::runtime_error("points and colors must have the same length");
                    }
                    color_data = static_cast<const ColorT *>(colors_info.ptr);
                }
                self.UpdateAndDraw(point_data, color_data, point_count);
            },
            py::arg("points"), py::arg("colors") = py::none());
}

template <typename VertexT, typename ColorT = float>
void BindGlMesh(py::module_ &m, const char *name) {
    using GlMeshT = glutils::GlMeshT<VertexT, ColorT>;
    py::class_<GlMeshT>(m, name)
        .def(py::init<>())
        .def("draw", &GlMeshT::Draw, py::arg("wireframe") = false)
        .def("set_vertices", &GlMeshT::SetVertices, py::arg("vertices"), py::arg("vertex_count"))
        .def("set_colors", &GlMeshT::SetColors, py::arg("colors"))
        .def("clear_colors", &GlMeshT::ClearColors)
        .def("set_triangles", &GlMeshT::SetTriangles, py::arg("triangles"), py::arg("tri_count"))
        .def("reserve_gpu", &GlMeshT::ReserveGPU, py::arg("max_vertices"), py::arg("max_indices"))
        // Zero-copy bindings: accept contiguous numpy arrays without dtype casting.
        .def(
            "set",
            [](GlMeshT &self, const py::array_t<VertexT, py::array::c_style> &vertices,
               const py::array_t<unsigned int, py::array::c_style> &triangles,
               const py::object &colors_obj) {
                ValidatePointsArray(vertices);
                ValidateTrianglesArray(triangles);
                auto vertices_info = vertices.request();
                auto triangles_info = triangles.request();

                const auto *vertex_data = static_cast<const VertexT *>(vertices_info.ptr);
                const auto *tri_data = static_cast<const unsigned int *>(triangles_info.ptr);
                const auto vertex_count = static_cast<std::size_t>(vertices_info.shape[0]);
                const auto tri_count = static_cast<std::size_t>(triangles_info.shape[0]);

                self.SetVertices(vertex_data, vertex_count);
                self.SetTriangles(tri_data, tri_count);

                if (colors_obj.is_none()) {
                    self.ClearColors();
                    return;
                }

                auto colors = colors_obj.cast<py::array_t<ColorT, py::array::c_style>>();
                ValidateColorsArray(colors);
                auto colors_info = colors.request();
                if (colors_info.shape[0] != vertices_info.shape[0]) {
                    throw std::runtime_error("vertices and colors must have the same length");
                }
                const auto *color_data = static_cast<const ColorT *>(colors_info.ptr);
                self.SetColors(color_data);
            },
            py::arg("vertices"), py::arg("triangles"), py::arg("colors") = py::none())
        .def(
            "set_vertices",
            [](GlMeshT &self, const py::array_t<VertexT, py::array::c_style> &vertices) {
                ValidatePointsArray(vertices);
                auto info = vertices.request();
                const auto *vertex_data = static_cast<const VertexT *>(info.ptr);
                const auto vertex_count = static_cast<std::size_t>(info.shape[0]);
                self.SetVertices(vertex_data, vertex_count);
            },
            py::arg("vertices"))
        .def(
            "set_colors",
            [](GlMeshT &self, const py::array_t<ColorT, py::array::c_style> &colors) {
                ValidateColorsArray(colors);
                auto info = colors.request();
                const auto *color_data = static_cast<const ColorT *>(info.ptr);
                self.SetColors(color_data);
            },
            py::arg("colors"))
        .def(
            "set_triangles",
            [](GlMeshT &self, const py::array_t<unsigned int, py::array::c_style> &triangles) {
                ValidateTrianglesArray(triangles);
                auto info = triangles.request();
                const auto *tri_data = static_cast<const unsigned int *>(info.ptr);
                const auto tri_count = static_cast<std::size_t>(info.shape[0]);
                self.SetTriangles(tri_data, tri_count);
            },
            py::arg("triangles"));
}

template <typename VertexT, typename ColorT = float>
void BindGlMeshDirect(py::module_ &m, const char *name) {
    using GlMeshDirectT = glutils::GlMeshDirectT<VertexT, ColorT>;
    py::class_<GlMeshDirectT>(m, name)
        .def(py::init<>())
        .def("draw", &GlMeshDirectT::Draw, py::arg("wireframe") = false)
        .def("update_and_draw", &GlMeshDirectT::UpdateAndDraw, py::arg("vertices"),
             py::arg("colors"), py::arg("triangles"), py::arg("vertex_count"), py::arg("tri_count"),
             py::arg("wireframe"))
        // Zero-copy bindings: accept contiguous numpy arrays without dtype casting.
        .def(
            "update_and_draw",
            [](GlMeshDirectT &self, const py::array_t<VertexT, py::array::c_style> &vertices,
               const py::array_t<unsigned int, py::array::c_style> &triangles,
               const py::object &colors_obj, bool wireframe) {
                ValidatePointsArray(vertices);
                ValidateTrianglesArray(triangles);
                auto vertices_info = vertices.request();
                auto triangles_info = triangles.request();

                const auto *vertex_data = static_cast<const VertexT *>(vertices_info.ptr);
                const auto *tri_data = static_cast<const unsigned int *>(triangles_info.ptr);
                const auto vertex_count = static_cast<std::size_t>(vertices_info.shape[0]);
                const auto tri_count = static_cast<std::size_t>(triangles_info.shape[0]);

                const ColorT *color_data = nullptr;
                if (!colors_obj.is_none()) {
                    auto colors = colors_obj.cast<py::array_t<ColorT, py::array::c_style>>();
                    ValidateColorsArray(colors);
                    auto colors_info = colors.request();
                    if (colors_info.shape[0] != vertices_info.shape[0]) {
                        throw std::runtime_error("vertices and colors must have the same length");
                    }
                    color_data = static_cast<const ColorT *>(colors_info.ptr);
                }

                self.UpdateAndDraw(vertex_data, color_data, tri_data, vertex_count, tri_count,
                                   wireframe);
            },
            py::arg("vertices"), py::arg("triangles"), py::arg("colors") = py::none(),
            py::arg("wireframe") = false)
        // Fallback binding: allow dtype conversion if needed (may copy).
        .def(
            "update_and_draw",
            [](GlMeshDirectT &self, const py::array &vertices_obj, const py::array &triangles_obj,
               const py::object &colors_obj, bool wireframe) {
                auto vertices =
                    py::array_t<VertexT, py::array::c_style | py::array::forcecast>(vertices_obj);
                auto triangles =
                    py::array_t<unsigned int, py::array::c_style | py::array::forcecast>(
                        triangles_obj);
                ValidatePointsArray(vertices);
                ValidateTrianglesArray(triangles);
                auto vertices_info = vertices.request();
                auto triangles_info = triangles.request();

                const auto *vertex_data = static_cast<const VertexT *>(vertices_info.ptr);
                const auto *tri_data = static_cast<const unsigned int *>(triangles_info.ptr);
                const auto vertex_count = static_cast<std::size_t>(vertices_info.shape[0]);
                const auto tri_count = static_cast<std::size_t>(triangles_info.shape[0]);

                const ColorT *color_data = nullptr;
                if (!colors_obj.is_none()) {
                    auto colors =
                        py::array_t<ColorT, py::array::c_style | py::array::forcecast>(colors_obj);
                    ValidateColorsArray(colors);
                    auto colors_info = colors.request();
                    if (colors_info.shape[0] != vertices_info.shape[0]) {
                        throw std::runtime_error("vertices and colors must have the same length");
                    }
                    color_data = static_cast<const ColorT *>(colors_info.ptr);
                }

                self.UpdateAndDraw(vertex_data, color_data, tri_data, vertex_count, tri_count,
                                   wireframe);
            },
            py::arg("vertices"), py::arg("triangles"), py::arg("colors") = py::none(),
            py::arg("wireframe") = false);
}

} // namespace

PYBIND11_MODULE(glutils, m) {
    // optional module docstring
    m.doc() = "pybind11 plugin for glutils module";

    // DrawPoints has multiple overloads, so we need to disambiguate
    // Bind no-copy overloads first so they are preferred when possible.
    m.def("DrawPoints", static_cast<void (*)(DoubleArrayNoCopy)>(&glutils::DrawPoints), "points"_a);
    m.def("DrawPoints",
          static_cast<void (*)(DoubleArrayNoCopy, FloatArrayNoCopy)>(&glutils::DrawPoints),
          "points"_a, "colors"_a);
    m.def("DrawPoints", static_cast<void (*)(FloatArrayNoCopy)>(&glutils::DrawPoints), "points"_a);
    m.def("DrawPoints",
          static_cast<void (*)(FloatArrayNoCopy, FloatArrayNoCopy)>(&glutils::DrawPoints),
          "points"_a, "colors"_a);
    m.def("DrawPoints", static_cast<void (*)(DoubleArray)>(&glutils::DrawPoints), "points"_a);
    m.def("DrawPoints", static_cast<void (*)(DoubleArray, FloatArray)>(&glutils::DrawPoints),
          "points"_a, "colors"_a);

    // Single overload functions - pybind11 can infer the type
    // Draw meshes have multiple overloads, so we need to disambiguate
    // Bind no-copy overloads first so they are preferred when possible.
    m.def("DrawMesh",
          static_cast<void (*)(DoubleArrayNoCopy, IntArrayNoCopy, FloatArrayNoCopy, const bool)>(
              &glutils::DrawMesh),
          "vertices"_a, "triangles"_a, "colors"_a, "wireframe"_a = false);
    m.def("DrawMesh",
          static_cast<void (*)(FloatArrayNoCopy, IntArrayNoCopy, FloatArrayNoCopy, const bool)>(
              &glutils::DrawMesh),
          "vertices"_a, "triangles"_a, "colors"_a, "wireframe"_a = false);
    m.def("DrawMesh",
          static_cast<void (*)(DoubleArray, IntArray, FloatArray, const bool)>(&glutils::DrawMesh),
          "vertices"_a, "triangles"_a, "colors"_a, "wireframe"_a = false);
    m.def("DrawMesh",
          static_cast<void (*)(FloatArray, IntArray, FloatArray, const bool)>(&glutils::DrawMesh),
          "vertices"_a, "triangles"_a, "colors"_a, "wireframe"_a = false);
    m.def("DrawMonochromeMesh",
          static_cast<void (*)(DoubleArray, IntArray, const std::array<float, 3> &, const bool)>(
              &glutils::DrawMonochromeMesh),
          py::arg("vertices"), py::arg("triangles"), py::arg("color"),
          py::arg("wireframe") = false);
    m.def("DrawMonochromeMesh",
          static_cast<void (*)(FloatArray, IntArray, const std::array<float, 3> &, const bool)>(
              &glutils::DrawMonochromeMesh),
          py::arg("vertices"), py::arg("triangles"), py::arg("color"),
          py::arg("wireframe") = false);
    m.def("DrawMonochromeMesh",
          static_cast<void (*)(DoubleArrayNoCopy, IntArrayNoCopy, const std::array<float, 3> &,
                               const bool)>(&glutils::DrawMonochromeMesh),
          py::arg("vertices"), py::arg("triangles"), py::arg("color"),
          py::arg("wireframe") = false);
    m.def("DrawMonochromeMesh",
          static_cast<void (*)(FloatArrayNoCopy, IntArrayNoCopy, const std::array<float, 3> &,
                               const bool)>(&glutils::DrawMonochromeMesh),
          py::arg("vertices"), py::arg("triangles"), py::arg("color"),
          py::arg("wireframe") = false);

    m.def("DrawLine", &glutils::DrawLine, "points"_a, "point_size"_a = 0.0f);
    m.def("DrawLines", &glutils::DrawLines, "points"_a, "point_size"_a = 0.0f);
    m.def("DrawLines2", &glutils::DrawLines2, "points"_a, "points2"_a, "point_size"_a = 0.0f);

    m.def("DrawTrajectory", &glutils::DrawTrajectory, "points"_a, "point_size"_a = 0.0f);

    m.def("DrawCameras", &glutils::DrawCameras, "poses"_a, "w"_a = 1.0f, "h_ratio"_a = 0.75f,
          "z_ratio"_a = 0.6f);
    m.def("DrawCamera", &glutils::DrawCamera, "poses"_a, "w"_a = 1.0f, "h_ratio"_a = 0.75f,
          "z_ratio"_a = 0.6f);

    m.def("DrawBoxes", &glutils::DrawBoxes, "poses"_a, "sizes"_a, "line_width"_a = 1.0f);

    m.def("DrawPlane", &glutils::DrawPlane, "num_divs"_a = 200, "div_size"_a = 10.0f,
          "scale"_a = 1.0f);

    py::class_<glutils::CameraImage>(m, "CameraImage")
        .def(py::init([](const UByteArray &image, const DoubleArray &pose, const size_t id,
                         const float scale, const float h_ratio, const float z_ratio,
                         const std::array<float, 3> &color) {
                 return new glutils::CameraImage(image, pose, id, scale, h_ratio, z_ratio, color);
             }),
             "image"_a, "pose"_a, "id"_a, "scale"_a = 1.0, "h_ratio"_a = 0.75, "z_ratio"_a = 0.6,
             "color"_a = std::array<float, 3>{0.0, 1.0, 0.0})
        .def(py::init([](const UByteArray &image,
                         const py::array_t<float, py::array::c_style | py::array::forcecast> &pose,
                         const size_t id, const float scale, const float h_ratio,
                         const float z_ratio, const std::array<float, 3> &color) {
                 return new glutils::CameraImage(image, pose, id, scale, h_ratio, z_ratio, color);
             }),
             "image"_a, "pose"_a, "id"_a, "scale"_a = 1.0, "h_ratio"_a = 0.75, "z_ratio"_a = 0.6,
             "color"_a = std::array<float, 3>{0.0, 1.0, 0.0})
        .def("draw", &glutils::CameraImage::draw)
        .def("drawPose", &glutils::CameraImage::drawPose)
        .def("setPose",
             [](glutils::CameraImage &self,
                const py::array_t<float, py::array::c_style | py::array::forcecast> &pose) {
                 self.setPose(pose);
             })
        .def("setPose",
             [](glutils::CameraImage &self,
                const py::array_t<double, py::array::c_style | py::array::forcecast> &pose) {
                 self.setPose(pose);
             })
        .def("setTransparent", &glutils::CameraImage::setTransparent);

    py::class_<glutils::CameraImages>(m, "CameraImages")
        .def(py::init<>())
        .def(
            "add",
            [](glutils::CameraImages &self, const UByteArray &image,
               const py::array_t<float, py::array::c_style | py::array::forcecast> &pose,
               const size_t id, const float scale, const float h_ratio, const float z_ratio,
               const std::array<float, 3> &color) {
                self.add(image, pose, id, scale, h_ratio, z_ratio, color);
            },
            "image"_a, "pose"_a, "id"_a, "scale"_a = 1.0, "h_ratio"_a = 0.75, "z_ratio"_a = 0.6,
            "color"_a = std::array<float, 3>{0.0, 1.0, 0.0})
        .def(
            "add",
            [](glutils::CameraImages &self, const UByteArray &image, const DoubleArray &pose,
               const size_t id, const float scale, const float h_ratio, const float z_ratio,
               const std::array<float, 3> &color) {
                self.add(image, pose, id, scale, h_ratio, z_ratio, color);
            },
            "image"_a, "pose"_a, "id"_a, "scale"_a = 1.0, "h_ratio"_a = 0.75, "z_ratio"_a = 0.6,
            "color"_a = std::array<float, 3>{0.0, 1.0, 0.0})
        .def("drawPoses", &glutils::CameraImages::drawPoses)
        .def("draw", &glutils::CameraImages::draw)
        .def("clear", &glutils::CameraImages::clear)
        .def("erase", &glutils::CameraImages::erase)
        .def("size", &glutils::CameraImages::size)
        .def("setTransparent", &glutils::CameraImages::setTransparent)
        .def("setAllTransparent", &glutils::CameraImages::setAllTransparent)
        .def("__getitem__", &glutils::CameraImages::operator[], py::return_value_policy::reference)
        .def("__len__", &glutils::CameraImages::size);

    BindGlPointCloud<double>(m, "GlPointCloudD");
    BindGlPointCloud<float>(m, "GlPointCloudF");

    BindGlPointCloudDirect<double>(m, "GlPointCloudDirectD");
    BindGlPointCloudDirect<float>(m, "GlPointCloudDirectF");

    BindGlMesh<double>(m, "GlMeshD");
    BindGlMesh<float>(m, "GlMeshF");

    BindGlMeshDirect<double>(m, "GlMeshDirectD");
    BindGlMeshDirect<float>(m, "GlMeshDirectF");
}
