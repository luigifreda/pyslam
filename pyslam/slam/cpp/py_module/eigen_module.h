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
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

namespace py = pybind11;
using namespace pybind11::literals;

template <typename _Scalar, int _Dim, int _Mode, int _Options = Eigen::AutoAlign>
void templatedEigenIsometry(py::module &m, const std::string &name) {
    using Transform = Eigen::Transform<_Scalar, _Dim, _Mode, _Options>;

    typedef typename Eigen::Matrix<_Scalar, Transform::Rows, Transform::HDim, Transform::Options>
        MatrixType;
    typedef typename Eigen::Matrix<_Scalar, Transform::Dim, Transform::HDim, Transform::Options>
        CompactMatrixType;
    typedef typename Eigen::Matrix<_Scalar, Transform::Dim, Transform::Dim, Transform::Options>
        RotationMatrixType;
    typedef typename Eigen::Matrix<_Scalar, Transform::Dim, Eigen::Dynamic> TranslationMatrixType;

    py::class_<Transform>(m, name.c_str(), py::module_local())
        .def(py::init([]() { return Transform::Identity(); }))

        .def(py::init<Transform &>(), "other"_a)

        .def(py::init([](MatrixType &m) { return Transform(m); }))

        .def(py::init([](CompactMatrixType &m) {
            MatrixType matrix = MatrixType::Identity();
            matrix.block(0, 0, Transform::Dim, Transform::HDim) = m;
            return Transform(matrix);
        }))

        .def(py::init([](RotationMatrixType &r, TranslationMatrixType &t) {
            MatrixType matrix = MatrixType::Identity();
            matrix.block(0, 0, Transform::Dim, Transform::Dim) = r;
            matrix.block(0, Transform::Dim, Transform::Dim, 1) = t;
            return Transform(matrix);
        }))

        .def(py::init([](Eigen::Quaterniond &q, TranslationMatrixType &t) {
            MatrixType matrix = MatrixType::Identity();
            Eigen::Matrix3d r = q.toRotationMatrix();
            matrix.block(0, 0, Transform::Dim, Transform::Dim) =
                r.block(0, 0, Transform::Dim, Transform::Dim);
            matrix.block(0, Transform::Dim, Transform::Dim, 1) = t;
            return Transform(matrix);
        }))

        .def("set_rotation",
             [](Transform &trans, RotationMatrixType &r) {
                 trans.matrix().block(0, 0, Transform::Dim, Transform::Dim) = r;
             })
        .def("set_rotation",
             [](Transform &trans, Eigen::Quaterniond &q) {
                 Eigen::Matrix3d r = q.toRotationMatrix();
                 trans.matrix().block(0, 0, Transform::Dim, Transform::Dim) = r;
             })

        .def("set_translation",
             [](Transform &trans, TranslationMatrixType &t) { trans.translation() = t; })

        .def(py::self * py::self)

        .def("__mul__",
             [](Transform &trans, Eigen::Matrix<_Scalar, Transform::Dim, 1> &t) {
                 Eigen::Matrix<_Scalar, Transform::Dim, 1> result = trans * t;
                 return result;
             })
        .def("__mul__", [](Transform &trans, TranslationMatrixType &t) { return trans * t; })

        .def("rows", &Transform::rows)
        .def("cols", &Transform::cols)

        .def("__call__", [](Transform &trans, int row, int col) { return trans(row, col); })

        .def("matrix", (MatrixType & (Transform::*)()) & Transform::matrix)
        .def("set_identity", &Transform::setIdentity)
        .def_static("identity", &Transform::Identity)
        .def("rotation_matrix",
             [](Transform &trans) {
                 MatrixType matrix = trans.matrix();
                 RotationMatrixType r = matrix.block(0, 0, Transform::Dim, Transform::Dim);
                 return r;
             })
        .def_property_readonly("R",
                               [](Transform &trans) {
                                   MatrixType matrix = trans.matrix();
                                   RotationMatrixType r =
                                       matrix.block(0, 0, Transform::Dim, Transform::Dim);
                                   return r;
                               })

        .def("quaternion",
             [](Transform &trans) {
                 MatrixType matrix = trans.matrix();
                 Eigen::Matrix<_Scalar, 3, 3> r = Eigen::Matrix<_Scalar, 3, 3>::Identity();
                 r.block(0, 0, Transform::Dim, Transform::Dim) =
                     matrix.block(0, 0, Transform::Dim, Transform::Dim);
                 return Eigen::Quaterniond(r);
             })
        .def("rotation",
             [](Transform &trans) {
                 MatrixType matrix = trans.matrix();
                 Eigen::Matrix<_Scalar, 3, 3> r = Eigen::Matrix<_Scalar, 3, 3>::Identity();
                 r.block(0, 0, Transform::Dim, Transform::Dim) =
                     matrix.block(0, 0, Transform::Dim, Transform::Dim);
                 return Eigen::Quaterniond(r);
             })
        .def("orientation",
             [](Transform &trans) {
                 MatrixType matrix = trans.matrix();
                 Eigen::Matrix<_Scalar, 3, 3> r = Eigen::Matrix<_Scalar, 3, 3>::Identity();
                 r.block(0, 0, Transform::Dim, Transform::Dim) =
                     matrix.block(0, 0, Transform::Dim, Transform::Dim);
                 return Eigen::Quaterniond(r);
             })
        .def("translation",
             [](Transform &trans) {
                 MatrixType matrix = trans.matrix();
                 Eigen::Matrix<_Scalar, Transform::Dim, 1> t =
                     matrix.block(0, Transform::Dim, Transform::Dim, 1);
                 return t;
             })
        .def("position",
             [](Transform &trans) {
                 MatrixType matrix = trans.matrix();
                 Eigen::Matrix<_Scalar, Transform::Dim, 1> t =
                     matrix.block(0, Transform::Dim, Transform::Dim, 1);
                 return t;
             })
        .def_property_readonly("t",
                               [](Transform &trans) {
                                   MatrixType matrix = trans.matrix();
                                   Eigen::Matrix<_Scalar, Transform::Dim, 1> t =
                                       matrix.block(0, Transform::Dim, Transform::Dim, 1);
                                   return t;
                               })

        .def("inverse", &Transform::inverse, "traits"_a = (Eigen::TransformTraits)(Transform::Mode))
        .def("make_affine", &Transform::makeAffine);
}

void bind_eigen(py::module &m) {

    py::class_<Eigen::Quaterniond>(m, "Quaternion", py::module_local())
        .def(py::init([]() { return Eigen::Quaterniond::Identity(); }))
        .def(py::init<const Eigen::Quaterniond &>())
        .def(py::init<const Eigen::AngleAxisd &>())
        .def(py::init<const Eigen::Matrix<double, 3, 3> &>())
        .def(py::init<const double &, const double &, const double &, const double &>(), "w"_a,
             "x"_a, "y"_a, "z"_a)

        .def(py::init([](const Eigen::Matrix<double, 4, 1> &m) {
            return std::make_unique<Eigen::Quaterniond>(m(0, 0), m(1, 0), m(2, 0), m(3, 0));
        }))
        .def_static("from_two_vectors",
                    [](Eigen::Matrix<double, 3, 1> &a, Eigen::Matrix<double, 3, 1> &b) {
                        return Eigen::Quaterniond::FromTwoVectors(a, b);
                    })
#if EIGEN_VERSION_AT_LEAST(3, 3, 7)
        .def("x", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::x)
        .def("y", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::y)
        .def("z", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::z)
        .def("w", (const double &(Eigen::Quaterniond::*)() const) & Eigen::Quaterniond::w)
#else
        .def("x", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::x)
        .def("y", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::y)
        .def("z", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::z)
        .def("w", (double (Eigen::Quaterniond::*) () const) &Eigen::Quaterniond::w)
#endif
        .def("vec", (const Eigen::VectorBlock<const Eigen::Quaterniond::Coefficients, 3> (
                        Eigen::Quaterniond::*)() const) &
                        Eigen::Quaterniond::vec)
        .def_static("identity", &Eigen::Quaterniond::Identity)
        .def("set_identity", [](Eigen::Quaterniond &q) { q.setIdentity(); })

        .def("rotation_matrix", &Eigen::Quaterniond::toRotationMatrix)
        .def("matrix", &Eigen::Quaterniond::toRotationMatrix)
        .def_property_readonly("R", &Eigen::Quaterniond::toRotationMatrix)

        .def("squared_norm", &Eigen::Quaterniond::squaredNorm)
        .def("norm", &Eigen::Quaterniond::norm)
        .def("normalize", &Eigen::Quaterniond::normalize)
        .def("normalized", &Eigen::Quaterniond::normalized)
        .def("dot", [](Eigen::Quaterniond &q1, Eigen::Quaterniond &q2) { return q1.dot(q2); })
        .def("angular_distance",
             [](Eigen::Quaterniond &q1, Eigen::Quaterniond &q2) { return q1.angularDistance(q2); })
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def("inverse", &Eigen::Quaterniond::inverse)
        .def("conjugate", &Eigen::Quaterniond::conjugate)
        .def("coeffs", (Eigen::Quaterniond::Coefficients & (Eigen::Quaterniond::*)()) &
                           Eigen::Quaterniond::coeffs) // x, y, z, w
        .def("__mul__", [](Eigen::Quaterniond &q, Eigen::Matrix<double, 3, 1> &t) {
            // Eigen::Matrix<double,3,1> result = q * t;
            Eigen::Matrix<double, 3, 1> result = q._transformVector(t);
            return result;
        });

    py::class_<Eigen::Rotation2Dd>(m, "Rotation2d", py::module_local())
        .def(py::init([]() { return Eigen::Rotation2Dd::Identity(); }))
        .def(py::init<Eigen::Rotation2Dd &>())
        .def(py::init<const double &>())
        .def(py::init<const Eigen::Matrix<double, 2, 2> &>())
        .def("angle", (double &(Eigen::Rotation2Dd::*)())&Eigen::Rotation2Dd::angle)
        .def("smallest_positive_angle", &Eigen::Rotation2Dd::smallestPositiveAngle)
        .def("smallest_angle", &Eigen::Rotation2Dd::smallestAngle)
        .def("inverse", &Eigen::Rotation2Dd::inverse)
        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self * Eigen::Matrix<double, 2, 1>())
        .def("from_rotation_matrix",
             [](Eigen::Rotation2Dd &r, const Eigen::Matrix<double, 2, 2> &R) {
                 r.fromRotationMatrix(R);
             })
        .def("to_rotation_matrix", &Eigen::Rotation2Dd::toRotationMatrix)
        .def("rotation_matrix", &Eigen::Rotation2Dd::toRotationMatrix)
        .def("matrix", &Eigen::Rotation2Dd::toRotationMatrix)
        .def_property_readonly("R", &Eigen::Rotation2Dd::toRotationMatrix)
        .def("slerp", &Eigen::Rotation2Dd::slerp)
        .def_static("identity", &Eigen::Rotation2Dd::Identity);

    py::class_<Eigen::AngleAxisd>(m, "AngleAxis", py::module_local())
        .def(py::init([]() { return Eigen::AngleAxisd::Identity(); }))
        .def(py::init<const double &, const Eigen::Matrix<double, 3, 1> &>())
        .def(py::init<const Eigen::AngleAxisd &>())
        .def(py::init<const Eigen::Quaterniond &>())
        .def(py::init<const Eigen::Matrix<double, 3, 3> &>())
        .def("angle", (double &(Eigen::AngleAxisd::*)())&Eigen::AngleAxisd::angle)
        .def("axis",
             (Eigen::Matrix<double, 3, 1> & (Eigen::AngleAxisd::*)()) & Eigen::AngleAxisd::axis)
        .def(py::self * py::self)
        .def(py::self * Eigen::Quaterniond())
        .def(Eigen::Quaterniond() * py::self)
        .def("inverse", &Eigen::AngleAxisd::inverse)
        .def("from_rotation_matrix",
             [](Eigen::AngleAxisd &r, const Eigen::Matrix<double, 3, 3> &R) {
                 r.fromRotationMatrix(R);
             })
        .def("to_rotation_matrix", &Eigen::AngleAxisd::toRotationMatrix)
        .def("rotation_matrix", &Eigen::AngleAxisd::toRotationMatrix)
        .def("matrix", &Eigen::AngleAxisd::toRotationMatrix)
        .def_property_readonly("R", &Eigen::AngleAxisd::toRotationMatrix)
        .def_static("identity", &Eigen::AngleAxisd::Identity);

    py::enum_<Eigen::TransformTraits>(m, "TransformTraits", py::module_local())
        .value("Isometry", Eigen::Isometry)
        .value("Affine", Eigen::Affine)
        .value("AffineCompact", Eigen::AffineCompact)
        .value("Projective", Eigen::Projective)
        .export_values();

    templatedEigenIsometry<double, 2, Eigen::Isometry>(m, "Isometry2d");
    templatedEigenIsometry<double, 3, Eigen::Isometry>(m, "Isometry3d");
}
