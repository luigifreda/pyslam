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

#include <array>
#include <cassert>
#include <cstddef>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <unordered_map>
#include <vector>

#include "glutils_bindings_utils.h"
#include "glutils_drawing.h"

#include "../volumetric/voxel_grid_data.h"

namespace glutils {

enum class ObjectColorDrawMode : int {
    POINTS = 0,
    CLASS = 1,
    OBJECT_ID = 2,
};

// ================================================================
// Draw object point cloud by using VBOs.
// The points and colors are directly uploaded to GPU memory when Update() is called.
// ================================================================
template <typename PointT, typename ColorT = float> class GlObjectT {
  public:
    using Color3f = std::array<float, 3>;
    using Ptr = std::shared_ptr<GlObjectT<PointT, ColorT>>;

  public:
    GlObjectT()
        : vbo_positions_(0), vbo_colors_(0), point_count_(0), positions_capacity_(0),
          colors_capacity_(0), has_point_colors_(false) {}

    GlObjectT(const GlObjectT &) = delete;
    GlObjectT &operator=(const GlObjectT &) = delete;

    GlObjectT(GlObjectT &&other) noexcept
        : vbo_positions_(other.vbo_positions_), vbo_colors_(other.vbo_colors_),
          point_count_(other.point_count_), positions_capacity_(other.positions_capacity_),
          colors_capacity_(other.colors_capacity_), has_point_colors_(other.has_point_colors_) {
        other.vbo_positions_ = 0;
        other.vbo_colors_ = 0;
        other.point_count_ = 0;
        other.positions_capacity_ = 0;
        other.colors_capacity_ = 0;
        other.has_point_colors_ = false;
    }

    GlObjectT &operator=(GlObjectT &&other) noexcept {
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
        has_point_colors_ = other.has_point_colors_;

        other.vbo_positions_ = 0;
        other.vbo_colors_ = 0;
        other.point_count_ = 0;
        other.positions_capacity_ = 0;
        other.colors_capacity_ = 0;
        other.has_point_colors_ = false;

        return *this;
    }

    ~GlObjectT() {
        if (vbo_positions_)
            glDeleteBuffers(1, &vbo_positions_);
        if (vbo_colors_)
            glDeleteBuffers(1, &vbo_colors_);
    }

    static void SetColorDrawMode(const ObjectColorDrawMode color_draw_mode) {
        color_draw_mode_ = color_draw_mode;
    }

    static void EnableBoundingBoxes(const bool enable_bounding_boxes) {
        enable_bounding_boxes_ = enable_bounding_boxes;
    }

    static void SetBoundingBoxLineWidth(const int bounding_box_line_width) {
        bounding_box_line_width_ = bounding_box_line_width;
    }

    void SetObjectIDColor(const Color3f &object_id_color) { object_id_color_ = object_id_color; }
    void SetClassIDColor(const Color3f &class_id_color) { class_id_color_ = class_id_color; }

    void SetBoundingBox(const double *box_matrix, const double *box_size) {
        static_assert(sizeof(GLdouble) == sizeof(double), "GLdouble must be double");
        use_bounding_box_ = true;
        std::memcpy(box_matrix_, box_matrix, 16 * sizeof(GLdouble));
        std::memcpy(box_size_, box_size, 3 * sizeof(GLdouble));
    }

    // If you want to enable/disable this bounding box (for instance for background points), you
    // can call this function
    void SetUseBoundingBox(const bool use_bounding_box) { use_bounding_box_ = use_bounding_box; }

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
        const bool use_point_colors =
            (color_draw_mode_ == ObjectColorDrawMode::POINTS && has_point_colors_);
        if (use_point_colors) {
            glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
            glEnableClientState(GL_COLOR_ARRAY);
            glColorPointer(3, GlType<ColorT>(), 0, nullptr);
        } else if (color_draw_mode_ == ObjectColorDrawMode::CLASS) {
            glColor3f(class_id_color_[0], class_id_color_[1], class_id_color_[2]);
        } else if (color_draw_mode_ == ObjectColorDrawMode::OBJECT_ID) {
            glColor3f(object_id_color_[0], object_id_color_[1], object_id_color_[2]);
        }

        glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(point_count_));

        if (use_point_colors) {
            glDisableClientState(GL_COLOR_ARRAY);
        }
        glDisableClientState(GL_VERTEX_ARRAY);

        glBindBuffer(GL_ARRAY_BUFFER, 0);

        if (use_bounding_box_ && enable_bounding_boxes_) {
            glColor3f(object_id_color_[0], object_id_color_[1], object_id_color_[2]);

            glPushMatrix();
            glMultMatrixd(box_matrix_);
            glutils_detail::DrawBoxWireframe(box_size_[0] * 0.5, box_size_[1] * 0.5,
                                             box_size_[2] * 0.5, bounding_box_line_width_);
            glPopMatrix();
        }
    }

    void Update(const volumetric::ObjectData::Ptr &object_data, const Color3f &class_id_color,
                const Color3f &object_id_color) {
        point_count_ = static_cast<std::size_t>(object_data->points.size());
        if (point_count_ == 0) {
            has_point_colors_ = false;
            return;
        }
        if (!object_data->colors.empty() && object_data->colors.size() != point_count_) {
            has_point_colors_ = false;
            throw std::runtime_error("object_data colors must have the same length as points");
        }
        const PointT *point_data;
        std::vector<std::array<float, 3>>
            converted_float_points; // we need to keep this temporary vector here to avoid deletion
                                    // of the vector before we use the data in the function Update()

        if constexpr (std::is_same<PointT, double>::value) {
            point_data = glutils_bindings_detail::GetPackedVectorData<double>(object_data->points);
        } else if constexpr (std::is_same<PointT, float>::value) {
            point_data = glutils_bindings_detail::GetPackedVectorDataConverted<float, double>(
                object_data->points, converted_float_points);
        } else {
            throw std::runtime_error("Unsupported point type");
        }
        const ColorT *color_data =
            object_data->colors.empty()
                ? nullptr
                : glutils_bindings_detail::GetPackedVectorData(object_data->colors);
        Update(point_data, color_data, point_count_);
        SetClassIDColor(class_id_color);
        SetObjectIDColor(object_id_color);
        SetBoundingBox(object_data->oriented_bounding_box.get_matrix().data(),
                       object_data->oriented_bounding_box.size.data());
    }

    void Update(const PointT *points, const ColorT *colors, const std::size_t point_count) {
        point_count_ = point_count;

        if (point_count_ == 0) {
            has_point_colors_ = false;
            return;
        }

        if (!points) {
            // assert(false && "Update: points is null but point_count > 0");
            has_point_colors_ = false;
            point_count_ = 0;
            return;
        }

        EnsureBuffers(false);
        if (!vbo_positions_) {
            has_point_colors_ = false;
            point_count_ = 0;
            return;
        }
        glBindBuffer(GL_ARRAY_BUFFER, vbo_positions_);
        const std::size_t pos_bytes = point_count_ * 3 * sizeof(PointT);
        glBufferData(GL_ARRAY_BUFFER, pos_bytes, points, GL_DYNAMIC_DRAW);
        positions_capacity_ = pos_bytes;

        // NOTE: We only upload point colors when they are provided.
        has_point_colors_ = (colors != nullptr);
        if (has_point_colors_) {
            EnsureBuffers(true);
            if (!vbo_colors_) {
                has_point_colors_ = false;
            } else {
                glBindBuffer(GL_ARRAY_BUFFER, vbo_colors_);
                const std::size_t col_bytes = point_count_ * 3 * sizeof(ColorT);
                glBufferData(GL_ARRAY_BUFFER, col_bytes, colors, GL_DYNAMIC_DRAW);
                colors_capacity_ = col_bytes;
            }
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
                      "GlObjectT supports float or double only");
        return std::is_same<ScalarT, float>::value ? GL_FLOAT : GL_DOUBLE;
    }

    GLuint vbo_positions_; // Vertex buffer object for positions
    GLuint vbo_colors_;    // Vertex buffer object for colors

    std::size_t point_count_;
    std::size_t positions_capacity_;
    std::size_t colors_capacity_;
    bool has_point_colors_;

    Color3f object_id_color_ = {0.0f, 0.0f, 0.0f};
    Color3f class_id_color_ = {0.0f, 0.0f, 0.0f};

    bool use_bounding_box_ = false;
    // NOTE: We use the identity matrix and a default size of 1.0 for the bounding box.
    // box_matrix_ is the transformation matrix from the object-attached coordinate system to the
    // world coordinate system.
    GLdouble box_matrix_[16] = {1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0,
                                0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0};
    GLdouble box_size_[3] = {1.0, 1.0, 1.0};

    static bool enable_bounding_boxes_;
    static int bounding_box_line_width_;
    static ObjectColorDrawMode color_draw_mode_;
};

template <typename PointT, typename ColorT>
ObjectColorDrawMode GlObjectT<PointT, ColorT>::color_draw_mode_ = ObjectColorDrawMode::POINTS;

template <typename PointT, typename ColorT>
int GlObjectT<PointT, ColorT>::bounding_box_line_width_ = 1;

template <typename PointT, typename ColorT>
bool GlObjectT<PointT, ColorT>::enable_bounding_boxes_ = false;

// ================================================================
// Draw a set of objects.
// ================================================================

template <typename PointT, typename ColorT = float> class GlObjectSetT {
  public:
    using GlObjectType = GlObjectT<PointT, ColorT>;
    using Color3f = typename GlObjectType::Color3f;
    using Ptr = std::shared_ptr<GlObjectSetT<PointT, ColorT>>;

  public:
    GlObjectSetT() {}

    ~GlObjectSetT() { objects_.clear(); }

    void Add(const size_t id, const GlObjectType::Ptr &object) { objects_[id] = object; }

    typename GlObjectType::Ptr &operator[](const size_t id) { return objects_[id]; }

    void Remove(const size_t id) { objects_.erase(id); }

    void Clear() { objects_.clear(); }

    void Draw() {
        for (const auto &object : objects_) {
            object.second->Draw();
        }
    }

    void Update(std::vector<std::shared_ptr<volumetric::ObjectData>> &object_data_list,
                std::vector<Color3f> &class_id_colors, std::vector<Color3f> &object_id_colors) {
        assert(object_data_list.size() == class_id_colors.size() &&
               object_data_list.size() == object_id_colors.size());
        for (size_t i = 0; i < object_data_list.size(); ++i) {
            const auto &object_data = object_data_list[i];
            const auto &id = object_data->object_id;
            const auto &class_id_color = class_id_colors[i];
            const auto &object_id_color = object_id_colors[i];

            const auto it = objects_.find(id);
            if (it != objects_.end()) {
                it->second->Update(object_data, class_id_color, object_id_color);
            } else {
                auto object = std::make_shared<GlObjectType>();
                object->Update(object_data, class_id_color, object_id_color);
                objects_[id] = object;
            }
        }
    }

    void Update(std::vector<std::shared_ptr<volumetric::ObjectData>> &object_data_list,
                const Color3f *class_id_colors, const Color3f *object_id_colors,
                const std::size_t color_count) {
        if (!class_id_colors) {
            throw std::runtime_error("class_id_colors must be provided");
        }
        assert(object_data_list.size() == color_count);
        const Color3f *object_colors = object_id_colors ? object_id_colors : class_id_colors;
        for (size_t i = 0; i < object_data_list.size(); ++i) {
            const auto &object_data = object_data_list[i];
            const auto &id = object_data->object_id;
            const auto &class_id_color = class_id_colors[i];
            const auto &object_id_color = object_colors[i];

            const auto it = objects_.find(id);
            if (it != objects_.end()) {
                it->second->Update(object_data, class_id_color, object_id_color);
            } else {
                auto object = std::make_shared<GlObjectType>();
                object->Update(object_data, class_id_color, object_id_color);
                objects_[id] = object;
            }
        }
    }

    void Update(std::vector<std::shared_ptr<volumetric::ObjectData>> &object_data_list,
                const float *class_id_colors_flat, const float *object_id_colors_flat,
                const std::size_t color_count) {
        if (!class_id_colors_flat) {
            throw std::runtime_error("class_id_colors must be provided");
        }
        assert(object_data_list.size() == color_count);
        const float *object_colors =
            object_id_colors_flat ? object_id_colors_flat : class_id_colors_flat;
        for (size_t i = 0; i < object_data_list.size(); ++i) {
            const auto &object_data = object_data_list[i];
            const auto &id = object_data->object_id;
            const Color3f class_id_color = {class_id_colors_flat[i * 3],
                                            class_id_colors_flat[i * 3 + 1],
                                            class_id_colors_flat[i * 3 + 2]};
            const Color3f object_id_color = {object_colors[i * 3], object_colors[i * 3 + 1],
                                             object_colors[i * 3 + 2]};

            const auto it = objects_.find(id);
            if (it != objects_.end()) {
                it->second->Update(object_data, class_id_color, object_id_color);
            } else {
                auto object = std::make_shared<GlObjectType>();
                object->Update(object_data, class_id_color, object_id_color);
                objects_[id] = object;
            }
        }
    }

    void Update(const size_t id, const PointT *object_points, const ColorT *object_colors,
                const std::size_t object_point_count, const Color3f &class_id_color,
                const Color3f &object_id_color, const double *object_box_matrix,
                const double *object_box_size, const bool use_bounding_box = true) {
        const auto it = objects_.find(id);
        if (it != objects_.end()) {
            it->second->Update(object_points, object_colors, object_point_count);
            it->second->SetClassIDColor(class_id_color);
            it->second->SetObjectIDColor(object_id_color);
            it->second->SetBoundingBox(object_box_matrix, object_box_size);
            it->second->SetUseBoundingBox(use_bounding_box);
        } else {
            auto object = std::make_shared<GlObjectType>();
            object->Update(object_points, object_colors, object_point_count);
            object->SetClassIDColor(class_id_color);
            object->SetObjectIDColor(object_id_color);
            object->SetBoundingBox(object_box_matrix, object_box_size);
            object->SetUseBoundingBox(use_bounding_box);
            objects_[id] = object;
        }
    }

    static void SetColorDrawMode(const ObjectColorDrawMode color_draw_mode) {
        GlObjectType::SetColorDrawMode(color_draw_mode);
    }

    static void SetBoundingBoxLineWidth(const int bounding_box_line_width) {
        GlObjectType::SetBoundingBoxLineWidth(bounding_box_line_width);
    }

    static void EnableBoundingBoxes(const bool enable_bounding_boxes) {
        GlObjectType::EnableBoundingBoxes(enable_bounding_boxes);
    }

  private:
    std::unordered_map<size_t, typename GlObjectType::Ptr> objects_;
};

// NOTE: We use float for colors since it is the most common type and it is easy and convenient
// to handle.
using GlObjectF = GlObjectT<float, float>;
using GlObjectD = GlObjectT<double, float>;

using GlObjectSetF = GlObjectSetT<float, float>;
using GlObjectSetD = GlObjectSetT<double, float>;

} // namespace glutils
