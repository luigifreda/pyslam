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

#include "glutils_bindings.h"
#include "glutils_drawing.h"

namespace glutils {

void DrawPoints(DoubleArray points) {
    auto info = points.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("points must be an Nx3 array");
    }

    const auto *point_data = static_cast<const double *>(info.ptr);
    const std::size_t point_count = static_cast<std::size_t>(info.shape[0]);

    py::gil_scoped_release release;
    glutils_detail::DrawPointCloud(point_data, point_count);
}

void DrawPoints(DoubleArray points, DoubleArray colors) {
    auto points_info = points.request();
    if (points_info.ndim != 2 || points_info.shape[1] != 3) {
        throw std::runtime_error("points must be an Nx3 array");
    }

    auto colors_info = colors.request();
    if (colors_info.ndim != 2 || colors_info.shape[1] != 3) {
        throw std::runtime_error("colors must be an Nx3 array");
    }
    if (colors_info.shape[0] != points_info.shape[0]) {
        throw std::runtime_error("points and colors must have the same length");
    }

    const auto *point_data = static_cast<const double *>(points_info.ptr);
    const auto *color_data = static_cast<const double *>(colors_info.ptr);
    const std::size_t point_count = static_cast<std::size_t>(points_info.shape[0]);

    py::gil_scoped_release release;
    glutils_detail::DrawColoredPointCloud(point_data, color_data, point_count);
}

void DrawMesh(DoubleArray vertices, IntArray triangles, DoubleArray colors, const bool wireframe) {
    auto vertices_info = vertices.request();
    if (vertices_info.ndim != 2 || vertices_info.shape[1] != 3) {
        throw std::runtime_error("vertices must be an Nx3 array");
    }

    auto triangles_info = triangles.request();
    if (triangles_info.ndim != 2 || triangles_info.shape[1] != 3) {
        throw std::runtime_error("triangles must be an Mx3 array");
    }

    auto colors_info = colors.request();
    if (colors_info.ndim != 2 || colors_info.shape[1] != 3) {
        throw std::runtime_error("colors must be an Nx3 array");
    }

    if (colors_info.shape[0] != vertices_info.shape[0]) {
        throw std::runtime_error("colors and vertices must have the same length");
    }

    const auto *vertex_data = static_cast<const double *>(vertices_info.ptr);
    const auto *color_data = static_cast<const double *>(colors_info.ptr);
    const auto *index_data = reinterpret_cast<const unsigned int *>(triangles_info.ptr);
    const std::size_t vertex_count = static_cast<std::size_t>(vertices_info.shape[0]);
    const std::size_t index_count =
        static_cast<std::size_t>(triangles_info.shape[0] * triangles_info.shape[1]);

    py::gil_scoped_release release;
    glutils_detail::DrawMesh(vertex_data, color_data, vertex_count, index_data, index_count,
                             wireframe);
}

void DrawMonochromeMesh(DoubleArray vertices, IntArray triangles, const std::array<float, 3> &color,
                        const bool wireframe) {
    auto vertices_info = vertices.request();
    if (vertices_info.ndim != 2 || vertices_info.shape[1] != 3) {
        throw std::runtime_error("vertices must be an Nx3 array");
    }

    auto triangles_info = triangles.request();
    if (triangles_info.ndim != 2 || triangles_info.shape[1] != 3) {
        throw std::runtime_error("triangles must be an Mx3 array");
    }

    const auto *vertex_data = static_cast<const double *>(vertices_info.ptr);
    const auto *index_data = reinterpret_cast<const unsigned int *>(triangles_info.ptr);
    const std::size_t vertex_count = static_cast<std::size_t>(vertices_info.shape[0]);
    const std::size_t index_count =
        static_cast<std::size_t>(triangles_info.shape[0] * triangles_info.shape[1]);

    py::gil_scoped_release release;
    glutils_detail::DrawMonochromeMesh(vertex_data, vertex_count, index_data, index_count, color,
                                       wireframe);
}

void DrawCameras(DoubleArray cameras, const float w = 1.0f, const float h_ratio = 0.75f,
                 const float z_ratio = 0.6f) {
    auto info = cameras.request();
    if (info.ndim != 3 || info.shape[1] != 4 || info.shape[2] != 4) {
        throw std::runtime_error("poses must be an Nx4x4 array");
    }

    const auto *pose_data = static_cast<const double *>(info.ptr);
    const std::size_t camera_count = static_cast<std::size_t>(info.shape[0]);

    py::gil_scoped_release release;
    glutils_detail::DrawCameraSet(pose_data, camera_count, w, h_ratio, z_ratio);
}

void DrawCamera(DoubleArray camera, const float w = 1.0f, const float h_ratio = 0.75f,
                const float z_ratio = 0.6f) {
    auto info = camera.request();
    if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 4) {
        throw std::runtime_error("pose must be a 4x4 array");
    }

    const auto *pose_data = static_cast<const double *>(info.ptr);

    py::gil_scoped_release release;
    glutils_detail::DrawCameraMatrix(pose_data, w, h_ratio, z_ratio);
}

void DrawLine(DoubleArray points, const float point_size = 0.0f) {
    auto info = points.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("points must be an Nx3 array");
    }

    const auto *point_data = static_cast<const double *>(info.ptr);
    const std::size_t point_count = static_cast<std::size_t>(info.shape[0]);

    py::gil_scoped_release release;
    glutils_detail::DrawLineStrip(point_data, point_count);
    glutils_detail::DrawPointMarkers(point_data, point_count, point_size);
}

void DrawLines(DoubleArray points, const float point_size = 0.0f) {
    auto info = points.request();
    if (info.ndim != 2 || info.shape[1] != 6) {
        throw std::runtime_error("points must be an Nx6 array");
    }

    const auto *segment_data = static_cast<const double *>(info.ptr);
    const std::size_t segment_count = static_cast<std::size_t>(info.shape[0]);

    py::gil_scoped_release release;
    glutils_detail::DrawSegments(segment_data, segment_count);
    glutils_detail::DrawSegmentMarkers(segment_data, segment_count, point_size);
}

void DrawLines2(DoubleArray points, DoubleArray points2, const float point_size = 0.0f) {
    auto info1 = points.request();
    if (info1.ndim != 2 || info1.shape[1] != 3) {
        throw std::runtime_error("points must be an Nx3 array");
    }

    auto info2 = points2.request();
    if (info2.ndim != 2 || info2.shape[1] != 3) {
        throw std::runtime_error("points2 must be an Mx3 array");
    }

    const auto *points1_data = static_cast<const double *>(info1.ptr);
    const auto *points2_data = static_cast<const double *>(info2.ptr);
    const std::size_t count = static_cast<std::size_t>(std::min(info1.shape[0], info2.shape[0]));

    py::gil_scoped_release release;
    glutils_detail::DrawLinesBetweenSets(points1_data, points2_data, count);
    glutils_detail::DrawPairMarkers(points1_data, points2_data, count, point_size);
}

void DrawTrajectory(DoubleArray points, const float point_size = 0.0f) {
    auto info = points.request();
    if (info.ndim != 2 || info.shape[1] != 3) {
        throw std::runtime_error("points must be an Nx3 array");
    }

    const auto *point_data = static_cast<const double *>(info.ptr);
    const std::size_t point_count = static_cast<std::size_t>(info.shape[0]);

    py::gil_scoped_release release;
    glutils_detail::DrawLineStrip(point_data, point_count);
    glutils_detail::DrawPointMarkers(point_data, point_count, point_size);
}

void DrawBoxes(DoubleArray poses, DoubleArray sizes, const float line_width = 1.0f) {
    auto pose_info = poses.request();
    if (pose_info.ndim != 3 || pose_info.shape[1] != 4 || pose_info.shape[2] != 4) {
        throw std::runtime_error("poses must be an Nx4x4 array");
    }

    auto size_info = sizes.request();
    if (size_info.ndim != 2 || size_info.shape[1] != 3) {
        throw std::runtime_error("sizes must be an Nx3 array");
    }

    if (size_info.shape[0] != pose_info.shape[0]) {
        throw std::runtime_error("poses and sizes must have the same length");
    }

    const auto *pose_data = static_cast<const double *>(pose_info.ptr);
    const auto *size_data = static_cast<const double *>(size_info.ptr);
    const std::size_t box_count = static_cast<std::size_t>(pose_info.shape[0]);

    py::gil_scoped_release release;
    glutils_detail::DrawBoxes(pose_data, size_data, box_count, line_width);
}

void DrawPlane(const int num_divs = 200, const float div_size = 10.0f, const float scale = 1.0f) {
    py::gil_scoped_release release;
    glutils_detail::DrawPlane(num_divs, div_size, scale);
}

} // namespace glutils
