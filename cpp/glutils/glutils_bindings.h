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

#include "glutils_bindings_common.h"

namespace glutils {

// Forward declarations for Python binding wrapper functions

void DrawPoints(DoubleArray points);
void DrawPoints(DoubleArray points, FloatArray colors);
void DrawPoints(DoubleArrayNoCopy points);
void DrawPoints(DoubleArrayNoCopy points, FloatArrayNoCopy colors);
void DrawPoints(FloatArrayNoCopy points);
void DrawPoints(FloatArrayNoCopy points, FloatArrayNoCopy colors);

void DrawMesh(DoubleArray vertices, IntArray triangles, FloatArray colors, const bool wireframe);
void DrawMesh(FloatArray vertices, IntArray triangles, FloatArray colors, const bool wireframe);
void DrawMesh(DoubleArrayNoCopy vertices, IntArrayNoCopy triangles, FloatArrayNoCopy colors,
              const bool wireframe);
void DrawMesh(FloatArrayNoCopy vertices, IntArrayNoCopy triangles, FloatArrayNoCopy colors,
              const bool wireframe);

void DrawMonochromeMesh(DoubleArray vertices, IntArray triangles, const std::array<float, 3> &color,
                        const bool wireframe);
void DrawMonochromeMesh(FloatArray vertices, IntArray triangles, const std::array<float, 3> &color,
                        const bool wireframe);
void DrawMonochromeMesh(DoubleArrayNoCopy vertices, IntArrayNoCopy triangles,
                        const std::array<float, 3> &color, const bool wireframe);
void DrawMonochromeMesh(FloatArrayNoCopy vertices, IntArrayNoCopy triangles,
                        const std::array<float, 3> &color, const bool wireframe);

void DrawCameras(DoubleArray cameras, const float w, const float h_ratio, const float z_ratio);
void DrawCamera(DoubleArray camera, const float w, const float h_ratio, const float z_ratio);

void DrawLine(DoubleArray points, const float point_size);
void DrawLines(DoubleArray points, const float point_size);
void DrawLines2(DoubleArray points, DoubleArray points2, const float point_size);

void DrawTrajectory(DoubleArray points, const float point_size);

void DrawBoxes(DoubleArray poses, DoubleArray sizes, const float line_width);

void DrawPlane(const int num_divs, const float div_size, const float scale);

} // namespace glutils
