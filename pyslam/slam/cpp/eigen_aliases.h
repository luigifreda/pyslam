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
#include <Eigen/Core>

namespace pyslam {

using Vec3b = Eigen::Matrix<unsigned char, 3, 1>; // 3-channel unsigned char vector
using Vec3i = Eigen::Matrix<int, 3, 1>;           // 3-channel int vector

using Vec3d = Eigen::Vector3d;
using Vec2d = Eigen::Vector2d;
using VecNd = Eigen::VectorXd;

using Mat2d = Eigen::Matrix<double, 2, 2>;
using Mat3d = Eigen::Matrix<double, 3, 3>;

// Row-major dynamic matrices we exchange with NumPy
using MatNx2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;              // (N,2)
using MatNx3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;              // (N,3)
using MatNxMd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // (N,M)

// Convenient Refs (for const input and const output)
// NOTE: These Ref types are:
// - Lightweight view
// - Can bind to many compatible Eigen objects without copying
// - Good for function arguments in APIs.

using MatNx2dRef = Eigen::Ref<const MatNx2d>;
using MatNx3dRef = Eigen::Ref<const MatNx3d>;
using MatNxMdRef = Eigen::Ref<const MatNxMd>;

using Mat2dRef = Eigen::Ref<const Mat2d>;
using Mat3dRef = Eigen::Ref<const Mat3d>;

using Vec3bRef = Eigen::Ref<const Vec3b>;
using Vec3iRef = Eigen::Ref<const Vec3i>;

using Vec2dRef = Eigen::Ref<const Vec2d>;
using Vec3dRef = Eigen::Ref<const Vec3d>;
using VecNdRef = Eigen::Ref<const VecNd>;

} // namespace pyslam