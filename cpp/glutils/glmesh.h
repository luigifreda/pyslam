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

#if defined(__APPLE__)
#include <OpenGL/gl.h>
#else
#ifndef GL_GLEXT_PROTOTYPES
#define GL_GLEXT_PROTOTYPES
#endif
#include <GL/gl.h>
#include <GL/glext.h>
#endif

#include <cstddef>
#include <type_traits>
#include <vector>

namespace glutils {

// ================================================================
// Draw mesh by using VBOs and a VAO.
// The vertices and colors are stored in CPU memory and uploaded to GPU when UpdateGPU() is called.
// ================================================================
template <typename VertexT, typename ColorT = VertexT> class GlMeshT {
  public:
    GlMeshT()
        : vao_(0), vbo_pos_(0), vbo_col_(0), ebo_(0), vertex_count_(0), index_count_(0),
          pos_capacity_(0), col_capacity_(0), idx_capacity_(0), pos_dirty_(false),
          col_dirty_(false), idx_dirty_(false), has_colors_(false) {

        // If you can use VAO (recommended). If you're on very old GL, you can omit VAO.
        glGenVertexArrays(1, &vao_);
        glGenBuffers(1, &vbo_pos_);
        glGenBuffers(1, &vbo_col_);
        glGenBuffers(1, &ebo_);
    }

    ~GlMeshT() {
        if (ebo_)
            glDeleteBuffers(1, &ebo_);
        if (vbo_col_)
            glDeleteBuffers(1, &vbo_col_);
        if (vbo_pos_)
            glDeleteBuffers(1, &vbo_pos_);
        if (vao_)
            glDeleteVertexArrays(1, &vao_);
    }

    void SetVertices(const VertexT *vertices_xyz, std::size_t vertex_count) {
        if (vertex_count > 0 && vertices_xyz == nullptr) {
            vertex_count_ = 0;
            vertices_.clear();
            pos_dirty_ = false;
            ClearColors();
            return;
        }

        const bool size_changed = (vertex_count != vertex_count_);
        vertices_.assign(vertices_xyz, vertices_xyz + vertex_count * 3);
        vertex_count_ = vertex_count;

        if (size_changed && has_colors_) {
            ClearColors();
        }

        pos_dirty_ = true;
    }

    void SetColors(const ColorT *colors_rgb) {
        if (vertex_count_ == 0 || colors_rgb == nullptr) {
            ClearColors();
            return;
        }
        colors_.assign(colors_rgb, colors_rgb + vertex_count_ * 3);
        has_colors_ = true;
        col_dirty_ = true;
    }

    void ClearColors() {
        colors_.clear();
        has_colors_ = false;
        col_dirty_ = false;
    }

    void SetTriangles(const unsigned int *tri_idx, std::size_t tri_count) {
        // tri_count = number of triangles, indices are tri_count*3
        if (tri_count > 0 && tri_idx == nullptr) {
            index_count_ = 0;
            indices_.clear();
            idx_dirty_ = false;
            return;
        }
        indices_.assign(tri_idx, tri_idx + tri_count * 3);
        index_count_ = tri_count * 3;
        idx_dirty_ = true;
    }

    // Optional: call when you know you'll reuse the same size often.
    void ReserveGPU(std::size_t max_vertices, std::size_t max_indices) {
        EnsureBuffers(true);
        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
        pos_capacity_ = max_vertices * 3 * sizeof(VertexT);
        glBufferData(GL_ARRAY_BUFFER, pos_capacity_, nullptr, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_col_);
        col_capacity_ = max_vertices * 3 * sizeof(ColorT);
        glBufferData(GL_ARRAY_BUFFER, col_capacity_, nullptr, GL_DYNAMIC_DRAW);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
        idx_capacity_ = max_indices * sizeof(unsigned int);
        glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx_capacity_, nullptr, GL_DYNAMIC_DRAW);
    }

    void UpdateGPU() {
        // VAO caches the binding + pointer setup (best if you can do it once)
        glBindVertexArray(vao_);

        if (pos_dirty_) {
            EnsureBuffers(false);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
            const std::size_t bytes = vertices_.size() * sizeof(VertexT);
            if (bytes > pos_capacity_) {
                glBufferData(GL_ARRAY_BUFFER, bytes, vertices_.data(), GL_DYNAMIC_DRAW);
                pos_capacity_ = bytes;
            } else if (bytes > 0) {
                glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, vertices_.data());
            }

            // Legacy client state:
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(3, GLType<VertexT>(), 0, nullptr);

            pos_dirty_ = false;
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
            glEnableClientState(GL_VERTEX_ARRAY);
            glVertexPointer(3, GLType<VertexT>(), 0, nullptr);
        }

        if (has_colors_) {
            EnsureBuffers(true);
            glBindBuffer(GL_ARRAY_BUFFER, vbo_col_);
            if (col_dirty_) {
                const std::size_t bytes = colors_.size() * sizeof(ColorT);
                if (bytes > col_capacity_) {
                    glBufferData(GL_ARRAY_BUFFER, bytes, colors_.data(), GL_DYNAMIC_DRAW);
                    col_capacity_ = bytes;
                } else if (bytes > 0) {
                    glBufferSubData(GL_ARRAY_BUFFER, 0, bytes, colors_.data());
                }
                col_dirty_ = false;
            }
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GLType<ColorT>(), 0, nullptr);
        } else {
            glDisableClientState(GL_COLOR_ARRAY);
        }

        if (idx_dirty_) {
            EnsureBuffers(false);
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
            const std::size_t bytes = indices_.size() * sizeof(unsigned int);
            if (bytes > idx_capacity_) {
                glBufferData(GL_ELEMENT_ARRAY_BUFFER, bytes, indices_.data(), GL_DYNAMIC_DRAW);
                idx_capacity_ = bytes;
            } else if (bytes > 0) {
                glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, bytes, indices_.data());
            }
            idx_dirty_ = false;
        } else {
            glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
        }

        glBindVertexArray(0);
    }

    void Draw(bool wireframe) {
        if (vertex_count_ == 0 || index_count_ == 0)
            return;

        UpdateGPU();

        if (wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glBindVertexArray(vao_);
        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(index_count_), GL_UNSIGNED_INT, nullptr);
        glBindVertexArray(0);

        if (wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        if (has_colors_) {
            glDisableClientState(GL_COLOR_ARRAY);
        }
        glDisableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    std::size_t VertexCount() const { return vertex_count_; }
    std::size_t ColorCount() const { return has_colors_ ? (colors_.size() / 3) : 0; }
    std::size_t TriangleCount() const { return index_count_ / 3; }

  private:
    void EnsureBuffers(bool need_colors) {
        if (!vbo_pos_)
            glGenBuffers(1, &vbo_pos_);
        if (!ebo_)
            glGenBuffers(1, &ebo_);
        if (need_colors && !vbo_col_)
            glGenBuffers(1, &vbo_col_);
    }

    GLuint vao_;
    GLuint vbo_pos_;
    GLuint vbo_col_;
    GLuint ebo_;

    std::size_t vertex_count_;
    std::size_t index_count_;
    std::size_t pos_capacity_;
    std::size_t col_capacity_;
    std::size_t idx_capacity_;

    bool pos_dirty_;
    bool col_dirty_;
    bool idx_dirty_;
    bool has_colors_;

    std::vector<VertexT> vertices_;     // xyzxyz...
    std::vector<ColorT> colors_;        // rgb rgb...
    std::vector<unsigned int> indices_; // triangle indices

    template <typename T> static constexpr GLenum GLType() {
        if constexpr (std::is_same_v<T, float>) {
            return GL_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            return GL_DOUBLE;
        } else {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "GlMeshT supports only float or double vertex/color types.");
            return GL_FLOAT;
        }
    }
};

// ================================================================
// Draw mesh by using VBOs and a VAO.
// The vertices and colors are directly uploaded to GPU memory when Update() is called.
// ================================================================
template <typename VertexT, typename ColorT = float> class GlMeshDirectT {
  public:
    GlMeshDirectT()
        : vao_(0), vbo_pos_(0), vbo_col_(0), ebo_(0), vertex_count_(0), index_count_(0),
          pos_capacity_(0), col_capacity_(0), idx_capacity_(0), has_colors_(false) {
        glGenVertexArrays(1, &vao_);
    }

    GlMeshDirectT(const GlMeshDirectT &) = delete;
    GlMeshDirectT &operator=(const GlMeshDirectT &) = delete;

    GlMeshDirectT(GlMeshDirectT &&other) noexcept
        : vao_(other.vao_), vbo_pos_(other.vbo_pos_), vbo_col_(other.vbo_col_), ebo_(other.ebo_),
          vertex_count_(other.vertex_count_), index_count_(other.index_count_),
          pos_capacity_(other.pos_capacity_), col_capacity_(other.col_capacity_),
          idx_capacity_(other.idx_capacity_), has_colors_(other.has_colors_) {
        other.vao_ = 0;
        other.vbo_pos_ = 0;
        other.vbo_col_ = 0;
        other.ebo_ = 0;
        other.vertex_count_ = 0;
        other.index_count_ = 0;
        other.pos_capacity_ = 0;
        other.col_capacity_ = 0;
        other.idx_capacity_ = 0;
        other.has_colors_ = false;
    }

    GlMeshDirectT &operator=(GlMeshDirectT &&other) noexcept {
        if (this == &other)
            return *this;

        if (ebo_)
            glDeleteBuffers(1, &ebo_);
        if (vbo_col_)
            glDeleteBuffers(1, &vbo_col_);
        if (vbo_pos_)
            glDeleteBuffers(1, &vbo_pos_);
        if (vao_)
            glDeleteVertexArrays(1, &vao_);

        vao_ = other.vao_;
        vbo_pos_ = other.vbo_pos_;
        vbo_col_ = other.vbo_col_;
        ebo_ = other.ebo_;
        vertex_count_ = other.vertex_count_;
        index_count_ = other.index_count_;
        pos_capacity_ = other.pos_capacity_;
        col_capacity_ = other.col_capacity_;
        idx_capacity_ = other.idx_capacity_;
        has_colors_ = other.has_colors_;

        other.vao_ = 0;
        other.vbo_pos_ = 0;
        other.vbo_col_ = 0;
        other.ebo_ = 0;
        other.vertex_count_ = 0;
        other.index_count_ = 0;
        other.pos_capacity_ = 0;
        other.col_capacity_ = 0;
        other.idx_capacity_ = 0;
        other.has_colors_ = false;

        return *this;
    }

    ~GlMeshDirectT() {
        if (ebo_)
            glDeleteBuffers(1, &ebo_);
        if (vbo_col_)
            glDeleteBuffers(1, &vbo_col_);
        if (vbo_pos_)
            glDeleteBuffers(1, &vbo_pos_);
        if (vao_)
            glDeleteVertexArrays(1, &vao_);
    }

    void Draw(bool wireframe) {
        if (vertex_count_ == 0 || index_count_ == 0)
            return;
        if (!vbo_pos_ || !ebo_)
            return;

        glBindVertexArray(vao_);

        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GLType<VertexT>(), 0, nullptr);

        if (has_colors_ && vbo_col_) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_col_);
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GLType<ColorT>(), 0, nullptr);
        } else {
            glDisableClientState(GL_COLOR_ARRAY);
        }

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);

        if (wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);

        glDrawElements(GL_TRIANGLES, static_cast<GLsizei>(index_count_), GL_UNSIGNED_INT, nullptr);

        if (wireframe)
            glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

        if (has_colors_ && vbo_col_) {
            glDisableClientState(GL_COLOR_ARRAY);
        }
        glDisableClientState(GL_VERTEX_ARRAY);

        glBindVertexArray(0);
        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, 0);
        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void Update(const VertexT *vertices_xyz, const ColorT *colors_rgb, const unsigned int *tri_idx,
                std::size_t vertex_count, std::size_t tri_count) {
        vertex_count_ = vertex_count;
        index_count_ = tri_count * 3;

        if (vertex_count_ == 0 || index_count_ == 0) {
            has_colors_ = false;
            return;
        }

        if (!vertices_xyz || !tri_idx) {
            has_colors_ = false;
            vertex_count_ = 0;
            index_count_ = 0;
            return;
        }

        EnsureBuffers(false);
        if (!vbo_pos_ || !ebo_) {
            has_colors_ = false;
            vertex_count_ = 0;
            index_count_ = 0;
            return;
        }

        glBindBuffer(GL_ARRAY_BUFFER, vbo_pos_);
        const std::size_t pos_bytes = vertex_count_ * 3 * sizeof(VertexT);
        if (pos_bytes > pos_capacity_) {
            glBufferData(GL_ARRAY_BUFFER, pos_bytes, vertices_xyz, GL_DYNAMIC_DRAW);
            pos_capacity_ = pos_bytes;
        } else {
            glBufferSubData(GL_ARRAY_BUFFER, 0, pos_bytes, vertices_xyz);
        }

        glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, ebo_);
        const std::size_t idx_bytes = index_count_ * sizeof(unsigned int);
        if (idx_bytes > idx_capacity_) {
            glBufferData(GL_ELEMENT_ARRAY_BUFFER, idx_bytes, tri_idx, GL_DYNAMIC_DRAW);
            idx_capacity_ = idx_bytes;
        } else {
            glBufferSubData(GL_ELEMENT_ARRAY_BUFFER, 0, idx_bytes, tri_idx);
        }

        has_colors_ = (colors_rgb != nullptr);
        if (has_colors_) {
            EnsureBuffers(true);
            if (vbo_col_) {
                glBindBuffer(GL_ARRAY_BUFFER, vbo_col_);
                const std::size_t col_bytes = vertex_count_ * 3 * sizeof(ColorT);
                if (col_bytes > col_capacity_) {
                    glBufferData(GL_ARRAY_BUFFER, col_bytes, colors_rgb, GL_DYNAMIC_DRAW);
                    col_capacity_ = col_bytes;
                } else {
                    glBufferSubData(GL_ARRAY_BUFFER, 0, col_bytes, colors_rgb);
                }
            } else {
                has_colors_ = false;
            }
        }
    }

    void Clear() {
        vertex_count_ = 0;
        index_count_ = 0;
        pos_capacity_ = 0;
        col_capacity_ = 0;
        idx_capacity_ = 0;
        has_colors_ = false;
        if (vbo_pos_) {
            glDeleteBuffers(1, &vbo_pos_);
            vbo_pos_ = 0;
        }
        if (vbo_col_) {
            glDeleteBuffers(1, &vbo_col_);
            vbo_col_ = 0;
        }
        if (ebo_) {
            glDeleteBuffers(1, &ebo_);
            ebo_ = 0;
        }
    }

  private:
    void EnsureBuffers(bool need_colors) {
        if (!vbo_pos_)
            glGenBuffers(1, &vbo_pos_);
        if (!ebo_)
            glGenBuffers(1, &ebo_);
        if (need_colors && !vbo_col_)
            glGenBuffers(1, &vbo_col_);
    }

    template <typename T> static constexpr GLenum GLType() {
        if constexpr (std::is_same_v<T, float>) {
            return GL_FLOAT;
        } else if constexpr (std::is_same_v<T, double>) {
            return GL_DOUBLE;
        } else {
            static_assert(std::is_same_v<T, float> || std::is_same_v<T, double>,
                          "GlMeshDirectT supports only float or double vertex/color types.");
            return GL_FLOAT;
        }
    }

    GLuint vao_;
    GLuint vbo_pos_;
    GLuint vbo_col_;
    GLuint ebo_;

    std::size_t vertex_count_;
    std::size_t index_count_;
    std::size_t pos_capacity_;
    std::size_t col_capacity_;
    std::size_t idx_capacity_;
    bool has_colors_;
};

// ================================================================

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using GlMeshD = GlMeshT<double, float>;
using GlMeshF = GlMeshT<float, float>;
using GlMeshDirectD = GlMeshDirectT<double, float>;
using GlMeshDirectF = GlMeshDirectT<float, float>;

} // namespace glutils
