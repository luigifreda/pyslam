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

#include "globject.h"
#include "glutils_bindings_utils.h"
#include "glutils_drawing.h"

#include "../volumetric/voxel_grid_data.h"

#include <memory>
#include <string>

namespace glutils_bindings_detail {

struct ColorBuffer {
    const std::array<float, 3> *data = nullptr;
    const float *flat_data = nullptr;
    std::size_t size = 0;
    std::vector<std::array<float, 3>> owned;
    py::array array_owner;
};

// Convert a Python sequence of RGB rows into std::vector<std::array<float, 3>> (copies).
inline std::vector<std::array<float, 3>>
ColorsFromSequence(const py::sequence &seq, std::size_t expected_size, const char *name) {
    const auto count = static_cast<std::size_t>(seq.size());
    if (expected_size != 0 && count != expected_size) {
        throw std::runtime_error(std::string(name) + " size must match object_data_list size");
    }
    std::vector<std::array<float, 3>> colors;
    colors.reserve(count);
    for (std::size_t i = 0; i < count; ++i) {
        py::sequence row = seq[static_cast<py::ssize_t>(i)].cast<py::sequence>();
        if (row.size() != 3) {
            throw std::runtime_error(std::string(name) + " rows must have length 3");
        }
        colors.push_back({row[0].cast<float>(), row[1].cast<float>(), row[2].cast<float>()});
    }
    return colors;
}

// Build a ColorBuffer from Python colors:
// - None -> zeroed owned buffer
// - float32 C-contiguous numpy array -> zero-copy view
// - otherwise -> copy via sequence conversion
inline ColorBuffer ColorsFromPyObject(const py::object &colors_obj, std::size_t expected_size,
                                      const char *name) {
    ColorBuffer buffer;
    if (colors_obj.is_none()) {
        buffer.owned.assign(expected_size, {0.0f, 0.0f, 0.0f});
        buffer.data = buffer.owned.data();
        buffer.size = expected_size;
        return buffer;
    }

    if (py::isinstance<py::array>(colors_obj)) {
        const auto array = py::reinterpret_borrow<py::array>(colors_obj);
        const auto is_float = array.dtype().is(py::dtype::of<float>());
        if (is_float && (array.flags() & py::array::c_style)) {
            const auto colors = py::reinterpret_borrow<py::array_t<float>>(array);
            ValidateColorsArray(colors);
            const auto rows = static_cast<std::size_t>(colors.shape(0));
            if (expected_size != 0 && rows != expected_size) {
                throw std::runtime_error(std::string(name) +
                                         " size must match object_data_list size");
            }
            buffer.array_owner = array;
            buffer.flat_data = colors.data();
            buffer.size = rows;
            return buffer;
        }
    }

    buffer.owned = ColorsFromSequence(colors_obj.cast<py::sequence>(), expected_size, name);
    buffer.data = buffer.owned.data();
    buffer.size = buffer.owned.size();
    return buffer;
}

inline void DrawObjectDataNoCopy(const py::object &object) {

    py::module_::import("volumetric");

    const auto object_ptr = object.cast<std::shared_ptr<volumetric::ObjectData>>();
    if (!object_ptr) {
        throw std::runtime_error("object_data must be a valid volumetric.ObjectData");
    }
    const auto &object_data = *object_ptr;

    const auto point_count = static_cast<std::size_t>(object_data.points.size());
    if (point_count == 0) {
        return;
    }

    if (!object_data.colors.empty() && object_data.colors.size() != point_count) {
        throw std::runtime_error("object_data colors must have the same length as points");
    }

    const auto *point_data = GetPackedVectorData(object_data.points);
    const auto *color_data =
        object_data.colors.empty() ? nullptr : GetPackedVectorData(object_data.colors);

    py::gil_scoped_release release;
    if (color_data) {
        glutils_detail::DrawColoredPointCloud(point_data, color_data, point_count);
    } else {
        glutils_detail::DrawPointCloud(point_data, point_count);
    }
}

template <typename PointT, typename ColorT = float>
void BindGlObject(py::module_ &m, const char *name) {
    using GlObjectT = glutils::GlObjectT<PointT, ColorT>;
    py::class_<GlObjectT>(m, name)
        .def(py::init<>())
        .def("draw", &GlObjectT::Draw, py::call_guard<py::gil_scoped_release>())
        .def("update",
             static_cast<void (GlObjectT::*)(const PointT *, const ColorT *, std::size_t)>(
                 &GlObjectT::Update),
             py::arg("points"), py::arg("colors"), py::arg("point_count"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_object_id_color", &GlObjectT::SetObjectIDColor, py::arg("object_id_color"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_class_id_color", &GlObjectT::SetClassIDColor, py::arg("class_id_color"),
             py::call_guard<py::gil_scoped_release>())
        .def("set_bounding_box", &GlObjectT::SetBoundingBox, py::arg("box_matrix"),
             py::arg("box_size"), py::call_guard<py::gil_scoped_release>())
        .def("set_use_bounding_box", &GlObjectT::SetUseBoundingBox, py::arg("use_bounding_box"),
             py::call_guard<py::gil_scoped_release>())
        .def_static("set_color_draw_mode", &GlObjectT::SetColorDrawMode, py::arg("color_draw_mode"))
        .def_static("enable_bounding_boxes", &GlObjectT::EnableBoundingBoxes,
                    py::arg("enable_bounding_boxes"))
        .def_static("set_bounding_box_line_width", &GlObjectT::SetBoundingBoxLineWidth,
                    py::arg("bounding_box_line_width"));
}

template <typename PointT, typename ColorT = float>
void BindGlObjectSet(py::module_ &m, const char *name) {
    using GlObjectSetT = glutils::GlObjectSetT<PointT, ColorT>;
    py::class_<GlObjectSetT>(m, name)
        .def(py::init<>())
        .def("draw", &GlObjectSetT::Draw, py::call_guard<py::gil_scoped_release>())
        .def("clear", &GlObjectSetT::Clear, py::call_guard<py::gil_scoped_release>())
        .def(
            "update",
            [](GlObjectSetT &self, volumetric::ObjectDataGroup &object_data_group,
               const py::object &class_id_colors, const py::object &object_id_colors) {
                auto &object_data_list = object_data_group.object_vector;
                auto class_colors =
                    ColorsFromPyObject(class_id_colors, object_data_list.size(), "class_id_colors");
                ColorBuffer *object_colors_ptr = nullptr;
                ColorBuffer object_colors;
                if (object_id_colors.is_none()) {
                    object_colors_ptr = &class_colors;
                } else {
                    object_colors = ColorsFromPyObject(object_id_colors, object_data_list.size(),
                                                       "object_id_colors");
                    object_colors_ptr = &object_colors;
                }
                py::gil_scoped_release release;
                const bool class_flat = class_colors.flat_data != nullptr;
                const bool object_flat =
                    object_colors_ptr && object_colors_ptr->flat_data != nullptr;
                if (class_flat && object_flat) {
                    self.Update(object_data_list, class_colors.flat_data,
                                object_colors_ptr ? object_colors_ptr->flat_data : nullptr,
                                object_data_list.size());
                } else if (class_flat || object_flat) {
                    const std::size_t count = object_data_list.size();
                    std::vector<float> class_flat_owned;
                    std::vector<float> object_flat_owned;
                    const float *class_flat_ptr = class_colors.flat_data;
                    const float *object_flat_ptr =
                        object_colors_ptr ? object_colors_ptr->flat_data : nullptr;

                    if (!class_flat) {
                        class_flat_owned.resize(count * 3);
                        for (std::size_t i = 0; i < count; ++i) {
                            class_flat_owned[i * 3] = class_colors.data[i][0];
                            class_flat_owned[i * 3 + 1] = class_colors.data[i][1];
                            class_flat_owned[i * 3 + 2] = class_colors.data[i][2];
                        }
                        class_flat_ptr = class_flat_owned.data();
                    }

                    if (!object_flat && object_colors_ptr) {
                        object_flat_owned.resize(count * 3);
                        for (std::size_t i = 0; i < count; ++i) {
                            object_flat_owned[i * 3] = object_colors_ptr->data[i][0];
                            object_flat_owned[i * 3 + 1] = object_colors_ptr->data[i][1];
                            object_flat_owned[i * 3 + 2] = object_colors_ptr->data[i][2];
                        }
                        object_flat_ptr = object_flat_owned.data();
                    }

                    self.Update(object_data_list, class_flat_ptr, object_flat_ptr, count);
                } else {
                    self.Update(object_data_list, class_colors.data, object_colors_ptr->data,
                                object_data_list.size());
                }
            },
            py::arg("object_data_list"), py::arg("class_id_colors"), py::arg("object_id_colors"))
        .def(
            "update_from_volumetric_objects",
            [](GlObjectSetT &self, const py::sequence &object_list,
               const py::object &class_id_colors, const py::object &object_id_colors,
               const bool use_background_bounding_box = false) {
                const auto object_count = static_cast<std::size_t>(object_list.size());
                auto class_colors =
                    ColorsFromPyObject(class_id_colors, object_count, "class_id_colors");
                const ColorBuffer *object_colors_ptr = nullptr;
                ColorBuffer object_colors;
                if (object_id_colors.is_none()) {
                    object_colors_ptr = &class_colors;
                } else {
                    object_colors =
                        ColorsFromPyObject(object_id_colors, object_count, "object_id_colors");
                    object_colors_ptr = &object_colors;
                }

                for (std::size_t i = 0; i < object_count; ++i) {
                    const auto object = py::reinterpret_borrow<py::object>(
                        object_list[static_cast<py::ssize_t>(i)]);

                    const auto points_obj =
                        py::reinterpret_borrow<py::array>(object.attr("points"));
                    const auto points =
                        py::array_t<PointT, py::array::c_style | py::array::forcecast>(points_obj);
                    ValidatePointsArray(points);
                    const auto point_count = static_cast<std::size_t>(points.shape(0));
                    const auto *point_data = points.data();

                    const auto colors_obj =
                        py::reinterpret_borrow<py::array>(object.attr("colors"));
                    const ColorT *color_data = nullptr;
                    if (colors_obj.size() > 0) {
                        const auto colors =
                            py::array_t<ColorT, py::array::c_style | py::array::forcecast>(
                                colors_obj);
                        ValidateColorsArray(colors);
                        if (static_cast<std::size_t>(colors.shape(0)) != point_count) {
                            throw std::runtime_error(
                                "object colors must have the same length as points");
                        }
                        color_data = colors.data();
                    }

                    const auto box = object.attr("oriented_bounding_box");
                    const auto box_matrix_obj =
                        py::reinterpret_borrow<py::array>(box.attr("box_matrix"));
                    const auto box_matrix =
                        py::array_t<double, py::array::c_style | py::array::forcecast>(
                            box_matrix_obj);
                    if (box_matrix.size() != 16) {
                        throw std::runtime_error("object box_matrix must have 16 elements");
                    }

                    const auto box_size_obj =
                        py::reinterpret_borrow<py::array>(box.attr("box_size"));
                    const auto box_size =
                        py::array_t<double, py::array::c_style | py::array::forcecast>(
                            box_size_obj);
                    if (box_size.size() != 3) {
                        throw std::runtime_error("object box_size must have 3 elements");
                    }

                    const auto class_id = object.attr("class_id").cast<std::size_t>();
                    const auto object_id = object.attr("object_id").cast<std::size_t>();

                    const auto class_color =
                        class_colors.flat_data
                            ? std::array<float, 3>{class_colors.flat_data[i * 3],
                                                   class_colors.flat_data[i * 3 + 1],
                                                   class_colors.flat_data[i * 3 + 2]}
                            : class_colors.data[i];

                    const auto object_color =
                        object_colors_ptr->flat_data
                            ? std::array<float, 3>{object_colors_ptr->flat_data[i * 3],
                                                   object_colors_ptr->flat_data[i * 3 + 1],
                                                   object_colors_ptr->flat_data[i * 3 + 2]}
                            : object_colors_ptr->data[i];

                    {
                        py::gil_scoped_release release;
                        bool use_bounding_box =
                            (class_id != 0 && object_id != 0) ? true : use_background_bounding_box;
                        self.Update(object_id, point_data, color_data, point_count, class_color,
                                    object_color, box_matrix.data(), box_size.data(),
                                    use_bounding_box);
                    }
                }
            },
            py::arg("object_list"), py::arg("class_id_colors"), py::arg("object_id_colors"),
            py::arg("use_background_bounding_box") = false)
        .def_static("set_color_draw_mode", &GlObjectSetT::SetColorDrawMode,
                    py::arg("color_draw_mode"))
        .def_static("enable_bounding_boxes", &GlObjectSetT::EnableBoundingBoxes,
                    py::arg("enable_bounding_boxes"))
        .def_static("set_bounding_box_line_width", &GlObjectSetT::SetBoundingBoxLineWidth,
                    py::arg("bounding_box_line_width"));
}

void bind_object_color_draw_mode(py::module_ &m) {
    py::enum_<glutils::ObjectColorDrawMode>(m, "ObjectColorDrawMode")
        .value("POINTS", glutils::ObjectColorDrawMode::POINTS)
        .value("CLASS", glutils::ObjectColorDrawMode::CLASS)
        .value("OBJECT_ID", glutils::ObjectColorDrawMode::OBJECT_ID);
}

} // namespace glutils_bindings_detail
