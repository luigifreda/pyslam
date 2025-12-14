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
#include <pybind11/stl.h>

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#include <GL/gl.h>
#endif

#include <algorithm>
#include <array>
#include <iostream>
#include <stdexcept>
#include <string>

namespace py = pybind11;
using namespace pybind11::literals;

using DoubleArray = py::array_t<double, py::array::c_style | py::array::forcecast>;
using IntArray = py::array_t<int, py::array::c_style | py::array::forcecast>;
using UByteArray = py::array_t<unsigned char, py::array::c_style | py::array::forcecast>;

constexpr std::size_t kMatrixElementCount = 16;

namespace glutils_detail {

inline void DrawPointCloud(const double *points, std::size_t point_count) {
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_DOUBLE, 0, points);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count));
    glDisableClientState(GL_VERTEX_ARRAY);
}

inline void DrawColoredPointCloud(const double *points, const double *colors,
                                  std::size_t point_count) {
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GL_DOUBLE, 0, points);
    glColorPointer(3, GL_DOUBLE, 0, colors);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count));
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

inline void DrawMesh(const double *vertices, const double *colors, std::size_t vertex_count,
                     const unsigned int *indices, std::size_t index_count, bool wireframe) {
    (void)vertex_count;
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GL_DOUBLE, 0, vertices);
    glColorPointer(3, GL_DOUBLE, 0, colors);

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(index_count), GL_UNSIGNED_INT, indices);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

inline void DrawMonochromeMesh(const double *vertices, std::size_t vertex_count,
                               const unsigned int *indices, std::size_t index_count,
                               const std::array<float, 3> &color, bool wireframe) {
    (void)vertex_count; // silence unused warnings if asserts removed

    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    glColor3f(color[0], color[1], color[2]);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_DOUBLE, 0, vertices);

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(index_count), GL_UNSIGNED_INT, indices);

    glDisableClientState(GL_VERTEX_ARRAY);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

inline void DrawCameraFrustum(float w, float h, float z) {
    glBegin(GL_LINES);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(w, h, z);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(w, -h, z);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(-w, -h, z);
    glVertex3f(0.0f, 0.0f, 0.0f);
    glVertex3f(-w, h, z);

    glVertex3f(w, h, z);
    glVertex3f(w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(-w, -h, z);

    glVertex3f(-w, h, z);
    glVertex3f(w, h, z);

    glVertex3f(-w, -h, z);
    glVertex3f(w, -h, z);
    glEnd();
}

inline void DrawCameraMatrix(const double *pose, float w, float h_ratio, float z_ratio) {
    glPushMatrix();
    glMultTransposeMatrixd(pose);
    DrawCameraFrustum(w, w * h_ratio, w * z_ratio);
    glPopMatrix();
}

inline void DrawCameraSet(const double *poses, std::size_t count, float w, float h_ratio,
                          float z_ratio) {
    const float h = w * h_ratio;
    const float z = w * z_ratio;
    for (std::size_t i = 0; i < count; ++i) {
        glPushMatrix();
        glMultTransposeMatrixd(poses + i * kMatrixElementCount);
        DrawCameraFrustum(w, h, z);
        glPopMatrix();
    }
}

inline void DrawLineStrip(const double *points, std::size_t point_count) {
    glBegin(GL_LINE_STRIP);
    for (std::size_t i = 0; i < point_count; ++i) {
        const double *p = points + i * 3;
        glVertex3d(p[0], p[1], p[2]);
    }
    glEnd();
}

inline void DrawPointMarkers(const double *points, std::size_t point_count, float point_size) {
    if (point_size <= 0.0f) {
        return;
    }

    glPointSize(point_size);
    glBegin(GL_POINTS);
    for (std::size_t i = 0; i < point_count; ++i) {
        const double *p = points + i * 3;
        glVertex3d(p[0], p[1], p[2]);
    }
    glEnd();
}

inline void DrawSegments(const double *segments, std::size_t segment_count) {
    glBegin(GL_LINES);
    for (std::size_t i = 0; i < segment_count; ++i) {
        const double *segment = segments + i * 6;
        glVertex3d(segment[0], segment[1], segment[2]);
        glVertex3d(segment[3], segment[4], segment[5]);
    }
    glEnd();
}

inline void DrawSegmentMarkers(const double *segments, std::size_t segment_count,
                               float point_size) {
    if (point_size <= 0.0f) {
        return;
    }

    glPointSize(point_size);
    glBegin(GL_POINTS);
    for (std::size_t i = 0; i < segment_count; ++i) {
        const double *segment = segments + i * 6;
        glVertex3d(segment[0], segment[1], segment[2]);
        glVertex3d(segment[3], segment[4], segment[5]);
    }
    glEnd();
}

inline void DrawLinesBetweenSets(const double *points1, const double *points2,
                                 std::size_t point_count) {
    glBegin(GL_LINES);
    for (std::size_t i = 0; i < point_count; ++i) {
        const double *p1 = points1 + i * 3;
        const double *p2 = points2 + i * 3;
        glVertex3d(p1[0], p1[1], p1[2]);
        glVertex3d(p2[0], p2[1], p2[2]);
    }
    glEnd();
}

inline void DrawPairMarkers(const double *points1, const double *points2, std::size_t point_count,
                            float point_size) {
    if (point_size <= 0.0f) {
        return;
    }

    glPointSize(point_size);
    glBegin(GL_POINTS);
    for (std::size_t i = 0; i < point_count; ++i) {
        const double *p1 = points1 + i * 3;
        const double *p2 = points2 + i * 3;
        glVertex3d(p1[0], p1[1], p1[2]);
        glVertex3d(p2[0], p2[1], p2[2]);
    }
    glEnd();
}

inline void DrawBoxWireframe(float half_w, float half_h, float half_z) {
    glBegin(GL_LINES);
    glVertex3f(-half_w, -half_h, -half_z);
    glVertex3f(half_w, -half_h, -half_z);
    glVertex3f(-half_w, -half_h, -half_z);
    glVertex3f(-half_w, half_h, -half_z);
    glVertex3f(-half_w, -half_h, -half_z);
    glVertex3f(-half_w, -half_h, half_z);

    glVertex3f(half_w, half_h, -half_z);
    glVertex3f(-half_w, half_h, -half_z);
    glVertex3f(half_w, half_h, -half_z);
    glVertex3f(half_w, -half_h, -half_z);
    glVertex3f(half_w, half_h, -half_z);
    glVertex3f(half_w, half_h, half_z);

    glVertex3f(-half_w, half_h, half_z);
    glVertex3f(half_w, half_h, half_z);
    glVertex3f(-half_w, half_h, half_z);
    glVertex3f(-half_w, -half_h, half_z);
    glVertex3f(-half_w, half_h, half_z);
    glVertex3f(-half_w, half_h, -half_z);

    glVertex3f(half_w, -half_h, half_z);
    glVertex3f(-half_w, -half_h, half_z);
    glVertex3f(half_w, -half_h, half_z);
    glVertex3f(half_w, half_h, half_z);
    glVertex3f(half_w, -half_h, half_z);
    glVertex3f(half_w, -half_h, -half_z);
    glEnd();
}

inline void DrawBoxes(const double *poses, const double *sizes, std::size_t box_count) {
    for (std::size_t i = 0; i < box_count; ++i) {
        glPushMatrix();
        glMultTransposeMatrixd(poses + i * kMatrixElementCount);

        const double *size = sizes + i * 3;
        const float half_w = static_cast<float>(size[0] * 0.5);
        const float half_h = static_cast<float>(size[1] * 0.5);
        const float half_z = static_cast<float>(size[2] * 0.5);

        DrawBoxWireframe(half_w, half_h, half_z);

        glPopMatrix();
    }
}

inline void DrawPlane(int num_divs = 200, float div_size = 10.0f, float scale = 1.0f) {
    glLineWidth(0.1f);
    // Plane parallel to x-z at origin with normal -y
    const float scaled_div_size = scale * div_size;
    const float minx = -static_cast<float>(num_divs) * scaled_div_size;
    const float minz = -static_cast<float>(num_divs) * scaled_div_size;
    const float maxx = static_cast<float>(num_divs) * scaled_div_size;
    const float maxz = static_cast<float>(num_divs) * scaled_div_size;

    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);
    const int line_count = 2 * num_divs;
    for (int n = 0; n < line_count; ++n) {
        const float x_pos = minx + scaled_div_size * static_cast<float>(n);
        const float z_pos = minz + scaled_div_size * static_cast<float>(n);
        // Vertical lines (parallel to z-axis)
        glVertex3f(x_pos, 0.0f, minz);
        glVertex3f(x_pos, 0.0f, maxz);
        // Horizontal lines (parallel to x-axis)
        glVertex3f(minx, 0.0f, z_pos);
        glVertex3f(maxx, 0.0f, z_pos);
    }
    glEnd();
    glLineWidth(1.0f);
}

} // namespace glutils_detail

template <typename T, int Flags>
std::array<double, kMatrixElementCount> ExtractPoseMatrix(const py::array_t<T, Flags> &pose) {
    auto info = pose.request();
    if (info.ndim != 2 || info.shape[0] != 4 || info.shape[1] != 4) {
        throw std::runtime_error("pose must be a 4x4 array");
    }

    const auto *src = static_cast<const T *>(info.ptr);
    std::array<double, kMatrixElementCount> matrix{};
    for (std::size_t i = 0; i < kMatrixElementCount; ++i) {
        matrix[i] = static_cast<double>(src[i]);
    }
    return matrix;
}

inline std::size_t ComputeAlignment(const py::buffer_info &info) {
    const ssize_t row_stride = info.strides[0] >= 0 ? info.strides[0] : -info.strides[0];
    for (int align : {8, 4, 2, 1}) {
        if (row_stride % align == 0) {
            return static_cast<std::size_t>(align);
        }
    }
    return 1;
}

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

void DrawBoxes(DoubleArray cameras, DoubleArray sizes) {
    auto pose_info = cameras.request();
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
    glutils_detail::DrawBoxes(pose_data, size_data, box_count);
}

void DrawPlane(const int num_divs = 200, const float div_size = 10.0f, const float scale = 1.0f) {
    py::gil_scoped_release release;
    glutils_detail::DrawPlane(num_divs, div_size, scale);
}

class CameraImage {
  public:
    using Ptr = std::shared_ptr<CameraImage>;

    template <typename T, int Flags>
    CameraImage(const UByteArray &image, const py::array_t<T, Flags> &pose, const size_t id,
                const float w = 1.0f, const float h_ratio = 0.75f, const float z_ratio = 0.6f,
                const std::array<float, 3> &color = {0.0f, 1.0f, 0.0f})
        : texture(0), imageWidth(0), imageHeight(0), w(w), h_ratio(h_ratio), z_ratio(z_ratio),
          color(color), id(id) {
        auto image_info = image.request();
        bool is_color = true;
        if (image_info.ndim == 3) {
            if (image_info.shape[2] != 3) {
                throw std::invalid_argument("Image must have 3 channels");
            }
        } else if (image_info.ndim == 2) {
            is_color = false;
        } else {
            throw std::invalid_argument("Image must have 2 or 3 dimensions");
        }

        imageWidth = static_cast<int>(image_info.shape[1]);
        imageHeight = static_cast<int>(image_info.shape[0]);

        const auto unpack_alignment = ComputeAlignment(image_info);
        const auto *image_data = static_cast<const unsigned char *>(image_info.ptr);

        const auto pose_matrix = ExtractPoseMatrix(pose);
        for (std::size_t i = 0; i < kMatrixElementCount; ++i) {
            matrix_[i] = static_cast<GLdouble>(pose_matrix[i]);
        }

        py::gil_scoped_release release;
        glGenTextures(1, &texture);
        glBindTexture(GL_TEXTURE_2D, texture);
        glPixelStorei(GL_UNPACK_ALIGNMENT, static_cast<GLint>(unpack_alignment));

        const GLenum format = is_color ? GL_RGB : GL_LUMINANCE;
        glTexImage2D(GL_TEXTURE_2D, 0, format, imageWidth, imageHeight, 0, format, GL_UNSIGNED_BYTE,
                     image_data);

        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);

        GLenum err;
        while ((err = glGetError()) != GL_NO_ERROR) {
            std::cerr << "OpenGL error: " << err << std::endl;
        }
    }

    ~CameraImage() {
        std::cout << "CameraImage " << this->id << " deleted" << std::endl;
        if (texture != 0) {
            py::gil_scoped_release release;
            glDeleteTextures(1, &texture);
        }
    }

    void draw() const { drawMatrix(matrix_.data()); }

    void drawPose(DoubleArray pose) const {
        const auto pose_matrix = ExtractPoseMatrix(pose);
        drawMatrix(pose_matrix.data());
    }

    void drawMatrix(const GLdouble *poseMatrix) const {
        py::gil_scoped_release release;
        drawMatrixNoGIL(poseMatrix);
    }

    template <typename T, int Flags> void setPose(const py::array_t<T, Flags> &pose) {
        const auto pose_matrix = ExtractPoseMatrix(pose);
        for (std::size_t i = 0; i < kMatrixElementCount; ++i) {
            matrix_[i] = static_cast<GLdouble>(pose_matrix[i]);
        }
    }

    void setColor(const std::array<float, 3> &new_color) { this->color = new_color; }

    void setTransparent(bool transparent) { this->isTransparent = transparent; }

  private:
    void drawMatrixNoGIL(const GLdouble *poseMatrix) const {
        glPushMatrix();
        glMultTransposeMatrixd(poseMatrix);
        draw_();
        glPopMatrix();
    }

    void draw_() const {
        const float h = w * h_ratio;
        const float z = w * z_ratio;

        glColor3f(color[0], color[1], color[2]);
        glutils_detail::DrawCameraFrustum(w, h, z);

        if (!isTransparent) {
            glColor3f(1.0f, 1.0f, 1.0f);

            glEnable(GL_TEXTURE_2D);
            glBindTexture(GL_TEXTURE_2D, texture);

            const GLboolean isCullFaceEnabled = glIsEnabled(GL_CULL_FACE);
            if (isCullFaceEnabled) {
                glDisable(GL_CULL_FACE);
            }

            glBegin(GL_QUADS);
            glTexCoord2f(0.0f, 0.0f);
            glVertex3f(-w, -h, z);
            glTexCoord2f(1.0f, 0.0f);
            glVertex3f(w, -h, z);
            glTexCoord2f(1.0f, 1.0f);
            glVertex3f(w, h, z);
            glTexCoord2f(0.0f, 1.0f);
            glVertex3f(-w, h, z);
            glEnd();

            if (isCullFaceEnabled) {
                glEnable(GL_CULL_FACE);
            }

            glDisable(GL_TEXTURE_2D);
        }
    }

  private:
    GLuint texture;
    int imageWidth;
    int imageHeight;
    float w;
    float h_ratio;
    float z_ratio;
    bool isTransparent = false;
    std::array<float, 3> color = {0.0f, 1.0f, 0.0f};
    std::array<GLdouble, kMatrixElementCount> matrix_{1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                                      0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};

  public:
    size_t id = 0;
};

// Explicitly instantiate the setPose function for float and double
// template CameraImage::CameraImage(py::array_t<unsigned char> image, py::array_t<float> pose,
// size_t id, float w, float h_ratio, float z_ratio, std::array<float, 3> color); template
// CameraImage::CameraImage(py::array_t<unsigned char> image, py::array_t<double> pose, size_t id,
// float w, float h_ratio, float z_ratio, std::array<float, 3> color); template void
// CameraImage::setPose<float>(py::array_t<float> pose); template void
// CameraImage::setPose<double>(py::array_t<double> pose);

class CameraImages {
  public:
    CameraImages() = default;

    // Add a camera image from numpy array
    template <typename T, int Flags>
    void add(const UByteArray &image, const py::array_t<T, Flags> &poseMatrix, const size_t id,
             const float w = 1.0f, const float h_ratio = 0.75f, const float z_ratio = 0.6f,
             const std::array<float, 3> &color = {0.0f, 1.0f, 0.0f}) {
        auto dtype = poseMatrix.dtype();
        // std::cout << "Pose matrix dtype: " << dtype << std::endl;
        const std::string dtype_str = py::str(dtype);
        if (dtype.is(py::dtype::of<float>()) || dtype_str == "float32") {
            // Ensure float32
            // cams.emplace_back(image, poseMatrix.template cast<py::array_t<float>>(), id, w,
            // h_ratio, z_ratio, color);
            const auto cam =
                std::make_shared<CameraImage>(image, poseMatrix, id, w, h_ratio, z_ratio, color);
            cams.push_back(cam);
        } else if (dtype.is(py::dtype::of<double>()) || dtype_str == "float64") {
            // Ensure float64
            // cams.emplace_back(image, poseMatrix.template cast<py::array_t<double>>(), id, w,
            // h_ratio, z_ratio, color);
            const auto cam =
                std::make_shared<CameraImage>(image, poseMatrix, id, w, h_ratio, z_ratio, color);
            cams.push_back(cam);
        } else {
            std::cout << "unmanaged dtype: " << dtype << std::endl;
            throw std::runtime_error("Pose matrix must be float32 or float64.");
        }
    }

    void drawPoses(DoubleArray cameras) const {
        auto info = cameras.request();
        if (info.ndim != 3 || info.shape[1] != 4 || info.shape[2] != 4) {
            throw std::runtime_error("poses must be an Nx4x4 array");
        }

        const auto *pose_data = static_cast<const double *>(info.ptr);
        const std::size_t matrix_count = static_cast<std::size_t>(info.shape[0]);
        if (matrix_count != cams.size()) {
            throw std::runtime_error("poses length must match stored camera images");
        }

        for (std::size_t i = 0; i < matrix_count; ++i) {
            cams[i]->drawMatrix(pose_data + i * kMatrixElementCount);
        }
    }

    void draw() const {
        for (const auto &cam : cams) {
            cam->draw();
        }
    }

    CameraImage::Ptr &operator[](size_t i) { return cams[i]; }

    void clear() { cams.clear(); }

    size_t size() const { return cams.size(); }

    void erase(size_t id) {
        auto it = std::find_if(cams.begin(), cams.end(),
                               [id](const CameraImage::Ptr &cam) { return cam->id == id; });
        if (it != cams.end()) {
            cams.erase(it);
        }
    }

    void setTransparent(size_t id, bool isTransparent) {
        auto it = std::find_if(cams.begin(), cams.end(),
                               [id](const CameraImage::Ptr &cam) { return cam->id == id; });
        if (it != cams.end()) {
            (*it)->setTransparent(isTransparent);
        }
    }

    void setAllTransparent(bool isTransparent) {
        for (auto &cam : cams) {
            cam->setTransparent(isTransparent);
        }
    }

  private:
    std::vector<CameraImage::Ptr> cams;
};

PYBIND11_MODULE(glutils, m) {
    // optional module docstring
    m.doc() = "pybind11 plugin for glutils module";

    m.def("DrawPoints", static_cast<void (*)(DoubleArray)>(&DrawPoints), "points"_a);
    m.def("DrawPoints", static_cast<void (*)(DoubleArray, DoubleArray)>(&DrawPoints), "points"_a,
          "colors"_a);

    m.def("DrawMesh",
          static_cast<void (*)(DoubleArray, IntArray, DoubleArray, const bool)>(&DrawMesh),
          "vertices"_a, "triangles"_a, "colors"_a, "wireframe"_a = false);
    m.def("DrawMonochromeMesh",
          static_cast<void (*)(DoubleArray, IntArray, const std::array<float, 3> &, const bool)>(
              &DrawMonochromeMesh),
          py::arg("vertices"), py::arg("triangles"), py::arg("color"),
          py::arg("wireframe") = false);

    m.def("DrawLine", static_cast<void (*)(DoubleArray, const float)>(&DrawLine), "points"_a,
          "point_size"_a = 0.0f);
    m.def("DrawLines", static_cast<void (*)(DoubleArray, const float)>(&DrawLines), "points"_a,
          "point_size"_a = 0.0f);
    m.def("DrawLines2", static_cast<void (*)(DoubleArray, DoubleArray, const float)>(&DrawLines2),
          "points"_a, "points2"_a, "point_size"_a = 0.0f);

    m.def("DrawTrajectory", static_cast<void (*)(DoubleArray, const float)>(&DrawTrajectory),
          "points"_a, "point_size"_a = 0.0f);

    m.def("DrawCameras", static_cast<void (*)(DoubleArray, float, float, float)>(&DrawCameras),
          "poses"_a, "w"_a = 1.0f, "h_ratio"_a = 0.75f, "z_ratio"_a = 0.6f);
    m.def("DrawCamera", static_cast<void (*)(DoubleArray, float, float, float)>(&DrawCamera),
          "poses"_a, "w"_a = 1.0f, "h_ratio"_a = 0.75f, "z_ratio"_a = 0.6f);

    m.def("DrawBoxes", static_cast<void (*)(DoubleArray, DoubleArray)>(&DrawBoxes), "poses"_a,
          "sizes"_a);

    m.def("DrawPlane", static_cast<void (*)(int, float, float)>(&DrawPlane), "num_divs"_a = 200,
          "div_size"_a = 10.0f, "scale"_a = 1.0f);

    py::class_<CameraImage>(m, "CameraImage")
        .def(py::init([](const UByteArray &image, const DoubleArray &pose, const size_t id,
                         const float scale, const float h_ratio, const float z_ratio,
                         const std::array<float, 3> &color) {
                 return new CameraImage(image, pose, id, scale, h_ratio, z_ratio, color);
             }),
             "image"_a, "pose"_a, "id"_a, "scale"_a = 1.0, "h_ratio"_a = 0.75, "z_ratio"_a = 0.6,
             "color"_a = std::array<float, 3>{0.0, 1.0, 0.0})
        .def(py::init([](const UByteArray &image,
                         const py::array_t<float, py::array::c_style | py::array::forcecast> &pose,
                         const size_t id, const float scale, const float h_ratio,
                         const float z_ratio, const std::array<float, 3> &color) {
                 return new CameraImage(image, pose, id, scale, h_ratio, z_ratio, color);
             }),
             "image"_a, "pose"_a, "id"_a, "scale"_a = 1.0, "h_ratio"_a = 0.75, "z_ratio"_a = 0.6,
             "color"_a = std::array<float, 3>{0.0, 1.0, 0.0})
        .def("draw", &CameraImage::draw)
        .def("drawPose", &CameraImage::drawPose)
        .def("setPose",
             [](CameraImage &self,
                const py::array_t<float, py::array::c_style | py::array::forcecast> &pose) {
                 self.setPose(pose);
             })
        .def("setPose",
             [](CameraImage &self,
                const py::array_t<double, py::array::c_style | py::array::forcecast> &pose) {
                 self.setPose(pose);
             })
        .def("setTransparent", &CameraImage::setTransparent);

    py::class_<CameraImages>(m, "CameraImages")
        .def(py::init<>())
        //.def("add", &CameraImages::add, "image"_a, "pose"_a, "id"_a, "scale"_a=1.0,
        //"h_ratio"_a=0.75, "z_ratio"_a=0.6, "color"_a=std::array<float, 3>{0.0, 1.0, 0.0})
        .def(
            "add",
            [](CameraImages &self, const UByteArray &image,
               const py::array_t<float, py::array::c_style | py::array::forcecast> &pose,
               const size_t id, const float scale, const float h_ratio, const float z_ratio,
               const std::array<float, 3> &color) {
                self.add(image, pose, id, scale, h_ratio, z_ratio, color);
            },
            "image"_a, "pose"_a, "id"_a, "scale"_a = 1.0, "h_ratio"_a = 0.75, "z_ratio"_a = 0.6,
            "color"_a = std::array<float, 3>{0.0, 1.0, 0.0})
        .def(
            "add",
            [](CameraImages &self, const UByteArray &image, const DoubleArray &pose,
               const size_t id, const float scale, const float h_ratio, const float z_ratio,
               const std::array<float, 3> &color) {
                self.add(image, pose, id, scale, h_ratio, z_ratio, color);
            },
            "image"_a, "pose"_a, "id"_a, "scale"_a = 1.0, "h_ratio"_a = 0.75, "z_ratio"_a = 0.6,
            "color"_a = std::array<float, 3>{0.0, 1.0, 0.0})
        .def("drawPoses", &CameraImages::drawPoses)
        .def("draw", &CameraImages::draw)
        .def("clear", &CameraImages::clear)
        .def("erase", &CameraImages::erase)
        .def("size", &CameraImages::size)
        .def("setTransparent", &CameraImages::setTransparent)
        .def("setAllTransparent", &CameraImages::setAllTransparent)
        .def("__getitem__", &CameraImages::operator[], py::return_value_policy::reference)
        .def("__len__", &CameraImages::size);
}
