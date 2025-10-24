#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include "eigen_aliases.h"
namespace pyslam {

// Helper functions

// Create a 4x4 matrix from a 3x3 rotation matrix and a 3x1 translation vector
template <typename Scalar>
inline Mat4<Scalar> poseRt(const Mat3<Scalar> &R, const Vec3<Scalar> &t) {
    Mat4<Scalar> T = Mat4<Scalar>::Identity();
    T.template block<3, 3>(0, 0) = R;
    T.template block<3, 1>(0, 3) = t;
    return T;
}

// Overload for Eigen expressions
template <typename DerivedR, typename DerivedT>
inline Mat4d poseRt(const Eigen::MatrixBase<DerivedR> &R, const Eigen::MatrixBase<DerivedT> &t) {
    static_assert(DerivedR::RowsAtCompileTime == 3 && DerivedR::ColsAtCompileTime == 3,
                  "R must be 3x3");
    static_assert(DerivedT::RowsAtCompileTime == 3 && DerivedT::ColsAtCompileTime == 1,
                  "t must be 3x1");
    Mat4d T = Mat4d::Identity();
    T.template block<3, 3>(0, 0) = R;
    T.template block<3, 1>(0, 3) = t;
    return T;
}

template <> inline Mat4<double> poseRt(const Mat3d &R, const Vec3d &t) {
    Mat4d T = Mat4d::Identity();
    T.template block<3, 3>(0, 0) = R;
    T.template block<3, 1>(0, 3) = t;
    return T;
}
template <> inline Mat4<float> poseRt(const Mat3f &R, const Vec3f &t) {
    Mat4f T = Mat4f::Identity();
    T.template block<3, 3>(0, 0) = R;
    T.template block<3, 1>(0, 3) = t;
    return T;
}

// Extract a 3x3 rotation matrix and a 3x1 translation vector from a 4x4 matrix
template <typename Scalar>
inline std::pair<Mat3<Scalar>, Vec3<Scalar>> extractRt(const Mat4<Scalar> &T) {
    return {T.template block<3, 3>(0, 0), T.template block<3, 1>(0, 3)};
}

template <> inline std::pair<Mat3<double>, Vec3<double>> extractRt(const Mat4d &T) {
    return {T.template block<3, 3>(0, 0), T.template block<3, 1>(0, 3)};
}
template <> inline std::pair<Mat3<float>, Vec3<float>> extractRt(const Mat4f &T) {
    return {T.template block<3, 3>(0, 0), T.template block<3, 1>(0, 3)};
}

/**
 * Compute skew-symmetric matrix from 3D vector
 * w in IR^3 -> [0,-wz,wy],
 *              [wz,0,-wx],
 *              [-wy,wx,0]
 */
template <typename Scalar> inline Mat3<Scalar> skew(const Vec3<Scalar> &w) {
    Mat3<Scalar> S;
    S << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
    return S;
}

// Overload for Eigen expressions
template <typename Derived> inline Mat3d skew(const Eigen::MatrixBase<Derived> &w) {
    static_assert(Derived::RowsAtCompileTime == 3 && Derived::ColsAtCompileTime == 1,
                  "w must be 3x1");
    Mat3d S;
    S << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
    return S;
}

template <> inline Mat3<double> skew(const Vec3d &w) {
    Mat3d S;
    S << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
    return S;
}
template <> inline Mat3<float> skew(const Vec3f &w) {
    Mat3f S;
    S << 0, -w(2), w(1), w(2), 0, -w(0), -w(1), w(0), 0;
    return S;
}

} // namespace pyslam