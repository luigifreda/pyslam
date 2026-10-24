#include "glutils_drawing.h"

namespace glutils_detail {

// ================================================================
// Draw points
// ================================================================

template <typename PointT> void DrawPointCloud(const PointT *points, std::size_t point_count) {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GlArrayType<PointT>(), 0, points);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count));
    glDisableClientState(GL_VERTEX_ARRAY);
}

template void DrawPointCloud<float>(const float *points, std::size_t point_count);
template void DrawPointCloud<double>(const double *points, std::size_t point_count);

// Draw a colored point cloud
template <typename PointT, typename ColorT>
void DrawColoredPointCloud(const PointT *points, const ColorT *colors, std::size_t point_count) {
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glVertexPointer(3, GlArrayType<PointT>(), 0, points);
    glColorPointer(3, GlArrayType<ColorT>(), 0, colors);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count));
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);
}

template void DrawColoredPointCloud<float, float>(const float *points, const float *colors,
                                                  std::size_t point_count);
template void DrawColoredPointCloud<double, float>(const double *points, const float *colors,
                                                   std::size_t point_count);
template void DrawColoredPointCloud<float, double>(const float *points, const double *colors,
                                                   std::size_t point_count);
template void DrawColoredPointCloud<double, double>(const double *points, const double *colors,
                                                    std::size_t point_count);

void DrawPointMarkers(const double *points, std::size_t point_count, float point_size) {
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

void DrawPairMarkers(const double *points1, const double *points2, std::size_t point_count,
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
template <typename VertexT, typename ColorT>
void DrawMesh(const VertexT *vertices, const ColorT *colors, std::size_t vertex_count,
              const unsigned int *indices, std::size_t index_count, bool wireframe) {
    (void)vertex_count;
    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);

    glVertexPointer(3, GlArrayType<VertexT>(), 0, vertices);
    glColorPointer(3, GlArrayType<ColorT>(), 0, colors);

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(index_count), GL_UNSIGNED_INT, indices);

    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_VERTEX_ARRAY);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

template void DrawMesh<float, float>(const float *vertices, const float *colors,
                                     std::size_t vertex_count, const unsigned int *indices,
                                     std::size_t index_count, bool wireframe);
template void DrawMesh<double, float>(const double *vertices, const float *colors,
                                      std::size_t vertex_count, const unsigned int *indices,
                                      std::size_t index_count, bool wireframe);
template void DrawMesh<float, double>(const float *vertices, const double *colors,
                                      std::size_t vertex_count, const unsigned int *indices,
                                      std::size_t index_count, bool wireframe);
template void DrawMesh<double, double>(const double *vertices, const double *colors,
                                       std::size_t vertex_count, const unsigned int *indices,
                                       std::size_t index_count, bool wireframe);

// Draw a monochrome mesh
template <typename VertexT>
void DrawMonochromeMesh(const VertexT *vertices, std::size_t vertex_count,
                        const unsigned int *indices, std::size_t index_count,
                        const std::array<float, 3> &color, bool wireframe) {
    (void)vertex_count; // silence unused warnings if asserts removed

    glBindBuffer(GL_ARRAY_BUFFER, 0);
    glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
    glPolygonMode(GL_FRONT_AND_BACK, wireframe ? GL_LINE : GL_FILL);

    glColor3f(color[0], color[1], color[2]);

    glEnableClientState(GL_VERTEX_ARRAY);
    glVertexPointer(3, GlArrayType<VertexT>(), 0, vertices);

    glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(index_count), GL_UNSIGNED_INT, indices);

    glDisableClientState(GL_VERTEX_ARRAY);

    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
}

template void DrawMonochromeMesh<float>(const float *vertices, std::size_t vertex_count,
                                        const unsigned int *indices, std::size_t index_count,
                                        const std::array<float, 3> &color, bool wireframe);
template void DrawMonochromeMesh<double>(const double *vertices, std::size_t vertex_count,
                                         const unsigned int *indices, std::size_t index_count,
                                         const std::array<float, 3> &color, bool wireframe);

// ================================================================
// Draw cameras
// ================================================================

void DrawCameraFrustum(const float w, const float h, const float z) {
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

template <typename ScalarT>
void DrawCameraMatrix(const ScalarT *pose, const float w, const float h_ratio,
                      const float z_ratio) {
    glPushMatrix();
    if constexpr (std::is_same<ScalarT, float>::value) {
        glMultTransposeMatrixf(pose);
    } else if constexpr (std::is_same<ScalarT, double>::value) {
        glMultTransposeMatrixd(pose);
    } else {
        static_assert(std::is_same<ScalarT, float>::value || std::is_same<ScalarT, double>::value,
                      "DrawCameraMatrix supports float or double only");
    }
    DrawCameraFrustum(w, w * h_ratio, w * z_ratio);
    glPopMatrix();
}

template void DrawCameraMatrix<float>(const float *pose, const float w, const float h_ratio,
                                      const float z_ratio);
template void DrawCameraMatrix<double>(const double *pose, const float w, const float h_ratio,
                                       const float z_ratio);

template <typename ScalarT>
void DrawCameraSet(const ScalarT *poses, const std::size_t count, const float w,
                   const float h_ratio, const float z_ratio) {
    const float h = w * h_ratio;
    const float z = w * z_ratio;
    for (std::size_t i = 0; i < count; ++i) {
        glPushMatrix();
        if constexpr (std::is_same<ScalarT, float>::value) {
            glMultTransposeMatrixf(poses + i * kMatrixElementCount);
        } else if constexpr (std::is_same<ScalarT, double>::value) {
            glMultTransposeMatrixd(poses + i * kMatrixElementCount);
        } else {
            static_assert(std::is_same<ScalarT, float>::value ||
                              std::is_same<ScalarT, double>::value,
                          "DrawCameraSet supports float or double only");
        }
        DrawCameraFrustum(w, h, z);
        glPopMatrix();
    }
}

template void DrawCameraSet<float>(const float *poses, const std::size_t count, const float w,
                                   const float h_ratio, const float z_ratio);
template void DrawCameraSet<double>(const double *poses, const std::size_t count, const float w,
                                    const float h_ratio, const float z_ratio);

// ================================================================
// Draw lines
// ================================================================

template <typename ScalarT>
void DrawLineStrip(const ScalarT *points, const std::size_t point_count) {
    glBegin(GL_LINE_STRIP);
    for (std::size_t i = 0; i < point_count; ++i) {
        const ScalarT *p = points + i * 3;
        if constexpr (std::is_same<ScalarT, float>::value) {
            glVertex3f(p[0], p[1], p[2]);
        } else if constexpr (std::is_same<ScalarT, double>::value) {
            glVertex3d(p[0], p[1], p[2]);
        } else {
            static_assert(std::is_same<ScalarT, float>::value ||
                              std::is_same<ScalarT, double>::value,
                          "DrawLineStrip supports float or double only");
        }
    }
    glEnd();
}

template void DrawLineStrip<float>(const float *points, const std::size_t point_count);
template void DrawLineStrip<double>(const double *points, const std::size_t point_count);

template <typename ScalarT>
void DrawSegments(const ScalarT *segments, const std::size_t segment_count) {
    glBegin(GL_LINES);
    for (std::size_t i = 0; i < segment_count; ++i) {
        const ScalarT *segment = segments + i * 6;
        if constexpr (std::is_same<ScalarT, float>::value) {
            glVertex3f(segment[0], segment[1], segment[2]);
            glVertex3f(segment[3], segment[4], segment[5]);
        } else if constexpr (std::is_same<ScalarT, double>::value) {
            glVertex3d(segment[0], segment[1], segment[2]);
            glVertex3d(segment[3], segment[4], segment[5]);
        } else {
            static_assert(std::is_same<ScalarT, float>::value ||
                              std::is_same<ScalarT, double>::value,
                          "DrawSegments supports float or double only");
        }
    }
    glEnd();
}

template void DrawSegments<float>(const float *segments, const std::size_t segment_count);
template void DrawSegments<double>(const double *segments, const std::size_t segment_count);

template <typename ScalarT>
void DrawSegmentMarkers(const ScalarT *segments, const std::size_t segment_count,
                        const float point_size) {
    if (point_size <= 0.0f) {
        return;
    }

    glPointSize(point_size);
    glBegin(GL_POINTS);
    for (std::size_t i = 0; i < segment_count; ++i) {
        const ScalarT *segment = segments + i * 6;
        if constexpr (std::is_same<ScalarT, float>::value) {
            glVertex3f(segment[0], segment[1], segment[2]);
            glVertex3f(segment[3], segment[4], segment[5]);
        } else if constexpr (std::is_same<ScalarT, double>::value) {
            glVertex3d(segment[0], segment[1], segment[2]);
            glVertex3d(segment[3], segment[4], segment[5]);
        } else {
            static_assert(std::is_same<ScalarT, float>::value ||
                              std::is_same<ScalarT, double>::value,
                          "DrawSegmentMarkers supports float or double only");
        }
    }
    glEnd();
}

template void DrawSegmentMarkers<float>(const float *segments, const std::size_t segment_count,
                                        const float point_size);
template void DrawSegmentMarkers<double>(const double *segments, const std::size_t segment_count,
                                         const float point_size);

template <typename ScalarT>
void DrawLinesBetweenSets(const ScalarT *points1, const ScalarT *points2,
                          const std::size_t point_count) {
    glBegin(GL_LINES);
    for (std::size_t i = 0; i < point_count; ++i) {
        const ScalarT *p1 = points1 + i * 3;
        const ScalarT *p2 = points2 + i * 3;
        if constexpr (std::is_same<ScalarT, float>::value) {
            glVertex3f(p1[0], p1[1], p1[2]);
            glVertex3f(p2[0], p2[1], p2[2]);
        } else if constexpr (std::is_same<ScalarT, double>::value) {
            glVertex3d(p1[0], p1[1], p1[2]);
            glVertex3d(p2[0], p2[1], p2[2]);
        } else {
            static_assert(std::is_same<ScalarT, float>::value ||
                              std::is_same<ScalarT, double>::value,
                          "DrawLinesBetweenSets supports float or double only");
        }
    }
    glEnd();
}

template void DrawLinesBetweenSets<float>(const float *points1, const float *points2,
                                          const std::size_t point_count);
template void DrawLinesBetweenSets<double>(const double *points1, const double *points2,
                                           const std::size_t point_count);

// ================================================================
// Draw boxes
// ================================================================

void DrawBoxWireframe(const float half_w, const float half_h, const float half_z,
                      const float line_width) {
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

template <typename ScalarT, typename SizeT>
void DrawBoxes(const ScalarT *poses, const SizeT *sizes, const std::size_t box_count,
               const float line_width) {
    for (std::size_t i = 0; i < box_count; ++i) {
        glPushMatrix();
        if constexpr (std::is_same<ScalarT, float>::value) {
            glMultTransposeMatrixf(poses + i * kMatrixElementCount);
        } else if constexpr (std::is_same<ScalarT, double>::value) {
            glMultTransposeMatrixd(poses + i * kMatrixElementCount);
        } else {
            static_assert(std::is_same<ScalarT, float>::value ||
                              std::is_same<ScalarT, double>::value,
                          "DrawBoxes supports float or double only");
        }

        const SizeT *size = sizes + i * 3;
        const float half_w = static_cast<float>(size[0] * 0.5);
        const float half_h = static_cast<float>(size[1] * 0.5);
        const float half_z = static_cast<float>(size[2] * 0.5);

        DrawBoxWireframe(half_w, half_h, half_z, line_width);

        glPopMatrix();
    }
}

template void DrawBoxes<float, float>(const float *poses, const float *sizes,
                                      const std::size_t box_count, const float line_width);
template void DrawBoxes<double, float>(const double *poses, const float *sizes,
                                       const std::size_t box_count, const float line_width);
template void DrawBoxes<float, double>(const float *poses, const double *sizes,
                                       const std::size_t box_count, const float line_width);
template void DrawBoxes<double, double>(const double *poses, const double *sizes,
                                        const std::size_t box_count, const float line_width);

// ================================================================
// Draw a plane
// ================================================================

void DrawPlane(const int num_divs, const float div_size, const float scale) {
    glLineWidth(0.1f);
    // Plane parallel to x-z at origin with normal -y
    const float scaled_div_size = scale * div_size;
    const float minx = -static_cast<float>(num_divs) * scaled_div_size;
    const float minz = -static_cast<float>(num_divs) * scaled_div_size;
    const float maxx = static_cast<float>(num_divs) * scaled_div_size;
    const float maxz = static_cast<float>(num_divs) * scaled_div_size;

    glColor3f(0.7f, 0.7f, 0.7f);
    glBegin(GL_LINES);
    const int line_count = 2 * num_divs + 1;
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
