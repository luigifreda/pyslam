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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

// CRITICAL: Include opaque types BEFORE stl.h to ensure opaque declarations take effect
#include "glutils_opaque_types.h"

#include <pybind11/stl.h>

#include "glutils_gl_includes.h"

#include "glmesh_bindings.h"
#include "globject_bindings.h"
#include "glpoint_cloud_bindings.h"
#include "glutils_bindings.h"
#include "glutils_camera.h"

namespace glutils_bindings = glutils_bindings_detail;

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

    m.def("DrawObjectData", &glutils_bindings::DrawObjectDataNoCopy, "object_data"_a);

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

    glutils_bindings::BindGlPointCloud<double>(m, "GlPointCloudD");
    glutils_bindings::BindGlPointCloud<float>(m, "GlPointCloudF");

    glutils_bindings::BindGlPointCloudDirect<double>(m, "GlPointCloudDirectD");
    glutils_bindings::BindGlPointCloudDirect<float>(m, "GlPointCloudDirectF");

    glutils_bindings::BindGlMesh<double>(m, "GlMeshD");
    glutils_bindings::BindGlMesh<float>(m, "GlMeshF");

    glutils_bindings::BindGlMeshDirect<double>(m, "GlMeshDirectD");
    glutils_bindings::BindGlMeshDirect<float>(m, "GlMeshDirectF");

    glutils_bindings::BindGlObject<double>(m, "GlObjectD");
    glutils_bindings::BindGlObject<float>(m, "GlObjectF");

    glutils_bindings::BindGlObjectSet<double>(m, "GlObjectSetD");
    glutils_bindings::BindGlObjectSet<float>(m, "GlObjectSetF");

    glutils_bindings::bind_object_color_draw_mode(m);
}
