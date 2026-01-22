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

#include "glutils_gl_includes.h"

#include <type_traits>

namespace glutils_detail {

template <typename ScalarT> constexpr GLenum GlArrayType() {
    static_assert(std::is_same<ScalarT, float>::value || std::is_same<ScalarT, double>::value,
                  "glutils_detail supports float or double only");
    return std::is_same<ScalarT, float>::value ? GL_FLOAT : GL_DOUBLE;
}

// ================================================================
// Draw points
// ================================================================

// Draw a point cloud
template <typename PointT> void DrawPointCloud(const PointT *points, std::size_t point_count);

// Draw a colored point cloud
template <typename PointT, typename ColorT>
void DrawColoredPointCloud(const PointT *points, const ColorT *colors, std::size_t point_count);

// Draw point markers
void DrawPointMarkers(const double *points, std::size_t point_count, float point_size);

// Draw pair markers
void DrawPairMarkers(const double *points1, const double *points2, std::size_t point_count,
                     float point_size);

// ================================================================
// Draw meshes
// ================================================================

// Draw a colored mesh
template <typename VertexT, typename ColorT>
void DrawMesh(const VertexT *vertices, const ColorT *colors, std::size_t vertex_count,
              const unsigned int *indices, std::size_t index_count, bool wireframe);

// Draw a monochrome mesh
template <typename VertexT>
void DrawMonochromeMesh(const VertexT *vertices, std::size_t vertex_count,
                        const unsigned int *indices, std::size_t index_count,
                        const std::array<float, 3> &color, bool wireframe);

// ================================================================
// Draw cameras
// ================================================================

// Draw a camera frustum
void DrawCameraFrustum(const float w, const float h, const float z);

// Draw a camera pose
template <typename ScalarT>
void DrawCameraMatrix(const ScalarT *pose, const float w, const float h_ratio, const float z_ratio);

// Draw a set of camera poses
template <typename ScalarT>
void DrawCameraSet(const ScalarT *poses, const std::size_t count, const float w,
                   const float h_ratio, const float z_ratio);

// ================================================================
// Draw lines
// ================================================================

// Draw a line strip
template <typename ScalarT>
void DrawLineStrip(const ScalarT *points, const std::size_t point_count);

template <typename ScalarT>
void DrawSegments(const ScalarT *segments, const std::size_t segment_count);

template <typename ScalarT>
void DrawSegmentMarkers(const ScalarT *segments, const std::size_t segment_count,
                        const float point_size);

template <typename ScalarT>
void DrawLinesBetweenSets(const ScalarT *points1, const ScalarT *points2,
                          const std::size_t point_count);

// ================================================================
// Draw boxes
// ================================================================

// Draw a box wireframe
void DrawBoxWireframe(const float half_w, const float half_h, const float half_z,
                      const float line_width = 1.0f);

// Draw multiple boxes
template <typename ScalarT, typename SizeT>
void DrawBoxes(const ScalarT *poses, const SizeT *sizes, const std::size_t box_count,
               float line_width = 1.0f);

// ================================================================
// Draw a planes
// ================================================================

// Draw a plane grid
void DrawPlane(const int num_divs = 200, const float div_size = 10.0f, const float scale = 1.0f);

} // namespace glutils_detail
