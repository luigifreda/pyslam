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

// -- 3D and 3-channel vectors

using Vec2f = Eigen::Vector2f;
using Vec2d = Eigen::Vector2d;

using Vec3b = Eigen::Matrix<unsigned char, 3, 1>; // 3-channel unsigned char vector
using Vec3i = Eigen::Matrix<int, 3, 1>;           // 3-channel int vector
using Vec3f = Eigen::Vector3f;
using Vec3d = Eigen::Vector3d;

using Vec6f = Eigen::Matrix<float, 6, 1>;
using Vec6d = Eigen::Matrix<double, 6, 1>;

using VecNf = Eigen::VectorXf;
using VecNd = Eigen::VectorXd;

template <typename Scalar> using Vec2 = Eigen::Matrix<Scalar, 2, 1>;
template <typename Scalar> using Vec3 = Eigen::Matrix<Scalar, 3, 1>;
template <typename Scalar> using Vec6 = Eigen::Matrix<Scalar, 6, 1>;
template <typename Scalar> using VecN = Eigen::Matrix<Scalar, Eigen::Dynamic, 1>;

// -- 2D and 3D matrices

using Mat2f = Eigen::Matrix<float, 2, 2>;
using Mat3f = Eigen::Matrix<float, 3, 3>;
using Mat4f = Eigen::Matrix<float, 4, 4>;

using Mat2d = Eigen::Matrix<double, 2, 2>;
using Mat3d = Eigen::Matrix<double, 3, 3>;
using Mat4d = Eigen::Matrix<double, 4, 4>;

template <typename Scalar> using Mat2 = Eigen::Matrix<Scalar, 2, 2>;
template <typename Scalar> using Mat3 = Eigen::Matrix<Scalar, 3, 3>;
template <typename Scalar> using Mat4 = Eigen::Matrix<Scalar, 4, 4>;

// -- Row-major dynamic matrices we exchange with NumPy

using MatNx2f = Eigen::Matrix<float, Eigen::Dynamic, 2, Eigen::RowMajor>;              // (N,2)
using MatNx3f = Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>;              // (N,3)
using MatNxMf = Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // (N,M)

using MatNx2d = Eigen::Matrix<double, Eigen::Dynamic, 2, Eigen::RowMajor>;              // (N,2)
using MatNx3d = Eigen::Matrix<double, Eigen::Dynamic, 3, Eigen::RowMajor>;              // (N,3)
using MatNxMd = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // (N,M)

template <typename Scalar>
using MatNx2 = Eigen::Matrix<Scalar, Eigen::Dynamic, 2, Eigen::RowMajor>; // (N,2)
template <typename Scalar>
using MatNx3 = Eigen::Matrix<Scalar, Eigen::Dynamic, 3, Eigen::RowMajor>; // (N,3)
template <typename Scalar>
using MatNxM = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>; // (N,M)

// -- Convenient Refs for const input

// NOTE: These Ref types are:
// - Lightweight const view
// - Can bind to many compatible Eigen objects without copying
// - Good for function arguments in APIs.

using Vec2fRef = Eigen::Ref<const Vec2f>;
using Vec2dRef = Eigen::Ref<const Vec2d>;

using Vec3bRef = Eigen::Ref<const Vec3b>;
using Vec3iRef = Eigen::Ref<const Vec3i>;
using Vec3fRef = Eigen::Ref<const Vec3f>;
using Vec3dRef = Eigen::Ref<const Vec3d>;

using Vec6fRef = Eigen::Ref<const Vec6f>;
using Vec6dRef = Eigen::Ref<const Vec6d>;

using VecNfRef = Eigen::Ref<const VecNf>;
using VecNdRef = Eigen::Ref<const VecNd>;

using Mat2fRef = Eigen::Ref<const Mat2f>;
using Mat2dRef = Eigen::Ref<const Mat2d>;

using Mat3fRef = Eigen::Ref<const Mat3f>;
using Mat3dRef = Eigen::Ref<const Mat3d>;

using Mat4fRef = Eigen::Ref<const Mat4f>;
using Mat4dRef = Eigen::Ref<const Mat4d>;

using MatNx2fRef = Eigen::Ref<const MatNx2f>;
using MatNx3fRef = Eigen::Ref<const MatNx3f>;
using MatNxMfRef = Eigen::Ref<const MatNxMf>;

using MatNx2dRef = Eigen::Ref<const MatNx2d>;
using MatNx3dRef = Eigen::Ref<const MatNx3d>;
using MatNxMdRef = Eigen::Ref<const MatNxMd>;

template <typename Scalar> using Vec2Ref = Eigen::Ref<const Vec2<Scalar>>;
template <typename Scalar> using Vec3Ref = Eigen::Ref<const Vec3<Scalar>>;
template <typename Scalar> using Vec6Ref = Eigen::Ref<const Vec6<Scalar>>;
template <typename Scalar> using VecNRef = Eigen::Ref<const VecN<Scalar>>;

template <typename Scalar> using Mat2Ref = Eigen::Ref<const Mat2<Scalar>>;
template <typename Scalar> using Mat3Ref = Eigen::Ref<const Mat3<Scalar>>;
template <typename Scalar> using Mat4Ref = Eigen::Ref<const Mat4<Scalar>>;

template <typename Scalar> using MatNx2Ref = Eigen::Ref<const MatNx2<Scalar>>;
template <typename Scalar> using MatNx3Ref = Eigen::Ref<const MatNx3<Scalar>>;
template <typename Scalar> using MatNxMRef = Eigen::Ref<const MatNxM<Scalar>>;

} // namespace pyslam