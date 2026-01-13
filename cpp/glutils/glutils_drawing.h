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

#include "glutils_common.h"

namespace glutils_detail {

// ================================================================
// Draw points
// ================================================================

// Draw a point cloud
inline void DrawPointCloud(const double *points, std::size_t point_count) {
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GL_DOUBLE, 0, points);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count));
    glDisableClientState(GL_VERTEX_ARRAY);
}

// Draw a colored point cloud
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

// Draw point markers
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

// Draw pair markers
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

// ================================================================
// Draw meshes
// ================================================================

// Draw a colored mesh
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

// Draw a monochrome mesh
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

// ================================================================
// Draw cameras
// ================================================================

// Draw a camera frustum
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

// Draw a camera pose
inline void DrawCameraMatrix(const double *pose, float w, float h_ratio, float z_ratio) {
    glPushMatrix();
    glMultTransposeMatrixd(pose);
    DrawCameraFrustum(w, w * h_ratio, w * z_ratio);
    glPopMatrix();
}

// Draw a set of camera poses
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

// ================================================================
// Draw lines
// ================================================================

// Draw a line strip
inline void DrawLineStrip(const double *points, std::size_t point_count) {
    glBegin(GL_LINE_STRIP);
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

// ================================================================
// Draw boxes
// ================================================================

// Draw a box wireframe
inline void DrawBoxWireframe(const float half_w, const float half_h, const float half_z,
                             const float line_width = 1.0f) {
    glLineWidth(line_width);
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
    glLineWidth(1.0f);
}

// Draw multiple boxes
inline void DrawBoxes(const double *poses, const double *sizes, std::size_t box_count,
                      const float line_width = 1.0f) {
    for (std::size_t i = 0; i < box_count; ++i) {
        glPushMatrix();
        glMultTransposeMatrixd(poses + i * kMatrixElementCount);

        const double *size = sizes + i * 3;
        const float half_w = static_cast<float>(size[0] * 0.5);
        const float half_h = static_cast<float>(size[1] * 0.5);
        const float half_z = static_cast<float>(size[2] * 0.5);

        DrawBoxWireframe(half_w, half_h, half_z, line_width);

        glPopMatrix();
    }
}

// ================================================================
// Draw a planes
// ================================================================

// Draw a plane grid
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
