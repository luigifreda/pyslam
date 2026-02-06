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
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace glutils {

// ================================================================
// Draw point cloud by using VBOs.
// The points and colors are stored in CPU memory and uploaded to GPU when UpdateGPU() is called.
// ================================================================

// Class to draw point clouds. A VAO is used to store the points and colors in GPU memory.
// The points and colors are stored in CPU memory and uploaded to GPU when UpdateGPU() is called.
template <typename PointT, typename ColorT = float> class GlPointCloudT {
  public:
    GlPointCloudT()
        : vbo_positions_(0), vbo_colors_(0), point_count_(0), positions_capacity_(0),
          colors_capacity_(0), positions_dirty_(false), colors_dirty_(false), has_colors_(false) {}

    GlPointCloudT(const GlPointCloudT &) = delete;
    GlPointCloudT &operator=(const GlPointCloudT &) = delete;

    GlPointCloudT(GlPointCloudT &&other) noexcept
        : vbo_positions_(other.vbo_positions_), vbo_colors_(other.vbo_colors_),
          point_count_(other.point_count_), positions_capacity_(other.positions_capacity_),
          colors_capacity_(other.colors_capacity_), positions_dirty_(other.positions_dirty_),
          colors_dirty_(other.colors_dirty_), has_colors_(other.has_colors_),
          positions_(std::move(other.positions_)), colors_(std::move(other.colors_)) {
        other.vbo_positions_ = 0;
        other.vbo_colors_ = 0;
        other.point_count_ = 0;
        other.positions_capacity_ = 0;
        other.colors_capacity_ = 0;
        other.positions_dirty_ = false;
        other.colors_dirty_ = false;
        other.has_colors_ = false;
    }

    GlPointCloudT &operator=(GlPointCloudT &&other) noexcept {
        if (this == &other)
            return *this;

        if (vbo_positions_)
            glDeleteBuffers(1, &vbo_positions_);
        if (vbo_colors_)
            glDeleteBuffers(1, &vbo_colors_);

        vbo_positions_ = other.vbo_positions_;
        vbo_colors_ = other.vbo_colors_;
        point_count_ = other.point_count_;
        positions_capacity_ = other.positions_capacity_;
        colors_capacity_ = other.colors_capacity_;
        positions_dirty_ = other.positions_dirty_;
        colors_dirty_ = other.colors_dirty_;
        has_colors_ = other.has_colors_;
        positions_ = std::move(other.positions_);
        colors_ = std::move(other.colors_);

        other.vbo_positions_ = 0;
        other.vbo_colors_ = 0;
        other.point_count_ = 0;
        other.positions_capacity_ = 0;
        other.colors_capacity_ = 0;
        other.positions_dirty_ = false;
        other.colors_dirty_ = false;
        other.has_colors_ = false;

        return *this;
    }

    ~GlPointCloudT() {
        if (vbo_positions_)
            glDeleteBuffers(1, &vbo_positions_);
        if (vbo_colors_)
            glDeleteBuffers(1, &vbo_colors_);
    }

    /* ----------- Setters ----------- */

    void Set(const PointT *points, const ColorT *colors, std::size_t point_count) {
        SetPoints(points, point_count);
        if (colors)
            SetColors(colors);
        else
            ClearColors();
    }

    void SetPoints(const PointT *points, std::size_t point_count) {
        if (point_count > 0 && points == nullptr) {
            // Choose one: throw, assert, or just clear.
            // assert(false && "SetPoints: points is null but point_count > 0");
            point_count_ = 0;
            positions_.clear();
            positions_dirty_ = false;
            has_colors_ = false;
            colors_.clear();
            colors_dirty_ = false;
            return;
        }

        const bool size_changed = (point_count != point_count_);
        if (size_changed) {
            has_colors_ = false;
            colors_dirty_ = false;
        }
        positions_.assign(points, points + point_count * 3);
        point_count_ = point_count;
        positions_dirty_ = true;
    }

    void SetColors(const ColorT *colors) { SetColors(colors, point_count_); }

    void SetColors(const ColorT *colors, std::size_t color_count) {
        if (color_count != point_count_) {
            throw std::runtime_error("colors must have the same length as points");
        }
        if (point_count_ == 0 || colors == nullptr) {
            ClearColors();
            return;
        }
        colors_.assign(colors, colors + point_count_ * 3);
        has_colors_ = true;
        colors_dirty_ = true;
    }

    void ClearColors() {
        colors_.clear();
        has_colors_ = false;
        colors_dirty_ = false;
    }

    /* ----------- GPU upload ----------- */

    void UpdateGPU() {
        if (positions_dirty_) {
            EnsureBuffers(false);
            const std::size_t bytes = positions_.size() * sizeof(PointT);
            if (vbo_positions_) {
                glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_);
                glBufferData(GL_ARRAY_BUFFER, bytes, positions_.data(), GL_DYNAMIC_DRAW);
                positions_capacity_ = bytes;
            }
            positions_dirty_ = false;
        }

        if (has_colors_ && colors_dirty_) {
            EnsureBuffers(true);
            const std::size_t bytes = colors_.size() * sizeof(ColorT);
            if (vbo_colors_) {
                glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
                glBufferData(GL_ARRAY_BUFFER, bytes, colors_.data(), GL_DYNAMIC_DRAW);
                colors_capacity_ = bytes;
            }
            colors_dirty_ = false;
        }
    }

    /* ----------- Draw ----------- */

    void Draw() {
        if (point_count_ == 0)
            return;

        UpdateGPU();

        /* Positions */
        glEnableClientState(GL_VERTEX_ARRAY);
        if (vbo_positions_) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_);
            glVertexPointer(3, GlType<PointT>(), 0, nullptr);
        } else {
            glBindBuffer(GL_ARRAY_BUFFER, 0);
            glVertexPointer(3, GlType<PointT>(), 0, positions_.data());
        }

        /* Colors (optional) */
        if (has_colors_) {
            glEnableClientState(GL_COLOR_ARRAY);
            if (vbo_colors_) {
                glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
                glColorPointer(3, GlType<ColorT>(), 0, nullptr);
            } else {
                glBindBuffer(GL_ARRAY_BUFFER, 0);
                glColorPointer(3, GlType<ColorT>(), 0, colors_.data());
            }
        }

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count_));

        if (has_colors_) {
            glDisableClientState(GL_COLOR_ARRAY);
        }
        glDisableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

  private:
    void EnsureBuffers(bool need_colors) {
        if (!vbo_positions_)
            glGenBuffers(1, &vbo_positions_);
        if (need_colors && !vbo_colors_)
            glGenBuffers(1, &vbo_colors_);
    }

    template <typename ScalarT> static constexpr GLenum GlType() {
        static_assert(std::is_same<ScalarT, float>::value || std::is_same<ScalarT, double>::value,
                      "GlPointCloudT supports float or double only");
        return std::is_same<ScalarT, float>::value ? GL_FLOAT : GL_DOUBLE;
    }

    GLuint vbo_positions_; // Vertex buffer object for positions
    GLuint vbo_colors_;    // Vertex buffer object for colors

    std::size_t point_count_;
    std::size_t positions_capacity_;
    std::size_t colors_capacity_;

    bool positions_dirty_;
    bool colors_dirty_;
    bool has_colors_;

    std::vector<PointT> positions_;
    std::vector<ColorT> colors_;
};

// ================================================================
// Draw point cloud by using VBOs.
// The points and colors are directly uploaded to GPU memory when Update() is called.
// ================================================================
template <typename PointT, typename ColorT = float> class GlPointCloudDirectT {
  public:
    GlPointCloudDirectT()
        : vbo_positions_(0), vbo_colors_(0), point_count_(0), positions_capacity_(0),
          colors_capacity_(0), has_colors_(false) {}

    GlPointCloudDirectT(const GlPointCloudDirectT &) = delete;
    GlPointCloudDirectT &operator=(const GlPointCloudDirectT &) = delete;

    GlPointCloudDirectT(GlPointCloudDirectT &&other) noexcept
        : vbo_positions_(other.vbo_positions_), vbo_colors_(other.vbo_colors_),
          point_count_(other.point_count_), positions_capacity_(other.positions_capacity_),
          colors_capacity_(other.colors_capacity_), has_colors_(other.has_colors_) {
        other.vbo_positions_ = 0;
        other.vbo_colors_ = 0;
        other.point_count_ = 0;
        other.positions_capacity_ = 0;
        other.colors_capacity_ = 0;
        other.has_colors_ = false;
    }

    GlPointCloudDirectT &operator=(GlPointCloudDirectT &&other) noexcept {
        if (this == &other)
            return *this;

        if (vbo_positions_)
            glDeleteBuffers(1, &vbo_positions_);
        if (vbo_colors_)
            glDeleteBuffers(1, &vbo_colors_);

        vbo_positions_ = other.vbo_positions_;
        vbo_colors_ = other.vbo_colors_;
        point_count_ = other.point_count_;
        positions_capacity_ = other.positions_capacity_;
        colors_capacity_ = other.colors_capacity_;
        has_colors_ = other.has_colors_;

        other.vbo_positions_ = 0;
        other.vbo_colors_ = 0;
        other.point_count_ = 0;
        other.positions_capacity_ = 0;
        other.colors_capacity_ = 0;
        other.has_colors_ = false;

        return *this;
    }

    ~GlPointCloudDirectT() {
        if (vbo_positions_)
            glDeleteBuffers(1, &vbo_positions_);
        if (vbo_colors_)
            glDeleteBuffers(1, &vbo_colors_);
    }

    /* ----------- Draw ----------- */

    void Draw() {
        if (point_count_ == 0)
            return;
        if (!vbo_positions_)
            return;

        /* Positions */
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_);
        glEnableClientState(GL_VERTEX_ARRAY);
        glVertexPointer(3, GlType<PointT>(), 0, nullptr);

        /* Colors (optional) */
        if (has_colors_) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GlType<ColorT>(), 0, nullptr);
        }

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count_));

        if (has_colors_) {
            glDisableClientState(GL_COLOR_ARRAY);
        }
        glDisableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, 0);
    }

    void Update(const PointT *points, const ColorT *colors, std::size_t point_count) {
        point_count_ = point_count;
        if (point_count_ == 0) {
            has_colors_ = false;
            return;
        }

        if (!points) {
            // assert(false && "Update: points is null but point_count > 0");
            has_colors_ = false;
            point_count_ = 0;
            return;
        }

        EnsureBuffers(false);
        if (!vbo_positions_) {
            has_colors_ = false;
            point_count_ = 0;
            return;
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_);
        const std::size_t pos_bytes = point_count_ * 3 * sizeof(PointT);
        glBufferData(GL_ARRAY_BUFFER, pos_bytes, points, GL_DYNAMIC_DRAW);
        positions_capacity_ = pos_bytes;

        has_colors_ = (colors != nullptr);
        if (has_colors_) {
            EnsureBuffers(true);
            if (!vbo_colors_) {
                has_colors_ = false;
            } else {
                glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
                const std::size_t col_bytes = point_count_ * 3 * sizeof(ColorT);
                glBufferData(GL_ARRAY_BUFFER, col_bytes, colors, GL_DYNAMIC_DRAW);
                colors_capacity_ = col_bytes;
            }
        }
    }

    void Clear() {
        point_count_ = 0;
        has_colors_ = false;
        positions_capacity_ = 0;
        colors_capacity_ = 0;
        if (vbo_positions_) {
            glDeleteBuffers(1, &vbo_positions_);
            vbo_positions_ = 0;
        }
        if (vbo_colors_) {
            glDeleteBuffers(1, &vbo_colors_);
            vbo_colors_ = 0;
        }
    }

  private:
    void EnsureBuffers(bool need_colors) {
        if (!vbo_positions_)
            glGenBuffers(1, &vbo_positions_);
        if (need_colors && !vbo_colors_)
            glGenBuffers(1, &vbo_colors_);
    }

    template <typename ScalarT> static constexpr GLenum GlType() {
        static_assert(std::is_same<ScalarT, float>::value || std::is_same<ScalarT, double>::value,
                      "GlPointCloudDirectT supports float or double only");
        return std::is_same<ScalarT, float>::value ? GL_FLOAT : GL_DOUBLE;
    }

    GLuint vbo_positions_; // Vertex buffer object for positions
    GLuint vbo_colors_;    // Vertex buffer object for colors

    std::size_t point_count_;
    std::size_t positions_capacity_;
    std::size_t colors_capacity_;
    bool has_colors_;
};

// ================================================================

// NOTE: We use float for colors since it is the most common type and it is easy and convenient to
// handle.
using GlPointCloudD = GlPointCloudT<double, float>;
using GlPointCloudF = GlPointCloudT<float, float>;
using GlPointCloudDirectD = GlPointCloudDirectT<double, float>;
using GlPointCloudDirectF = GlPointCloudDirectT<float, float>;

} // namespace glutils
