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

#include <pybind11/eigen.h>
#include <pybind11/numpy.h>
#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "Sim3PointRegistrationSolver.h"
#include "Sim3Solver.h"
#include "opencv_type_casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace utils;

PYBIND11_MODULE(sim3solver, m) {
    // optional module docstring
    m.doc() = "pybind11 plugin for sim3solver module";

    py::class_<Sim3SolverInput>(m, "Sim3SolverInput")
        .def(py::init<>())
        .def_readwrite("points_3d_w1", &Sim3SolverInput::mvX3Dw1)
        .def_readwrite("points_3d_w2", &Sim3SolverInput::mvX3Dw2)
        .def_readwrite("sigmas2_1", &Sim3SolverInput::mvSigmaSquare1)
        .def_readwrite("sigmas2_2", &Sim3SolverInput::mvSigmaSquare2)
        .def_readwrite("K1", &Sim3SolverInput::K1)
        .def_readwrite("Rcw1", &Sim3SolverInput::Rcw1)
        .def_readwrite("tcw1", &Sim3SolverInput::tcw1)
        .def_readwrite("K2", &Sim3SolverInput::K2)
        .def_readwrite("Rcw2", &Sim3SolverInput::Rcw2)
        .def_readwrite("tcw2", &Sim3SolverInput::tcw2)
        .def_readwrite("fix_scale", &Sim3SolverInput::bFixScale);

    py::class_<Sim3SolverInput2>(m, "Sim3SolverInput2")
        .def(py::init<>())
        .def_readwrite("points_3d_c1", &Sim3SolverInput2::mvX3Dc1)
        .def_readwrite("points_3d_c2", &Sim3SolverInput2::mvX3Dc2)
        .def_readwrite("sigmas2_1", &Sim3SolverInput2::mvSigmaSquare1)
        .def_readwrite("sigmas2_2", &Sim3SolverInput2::mvSigmaSquare2)
        .def_readwrite("K1", &Sim3SolverInput2::K1)
        .def_readwrite("K2", &Sim3SolverInput2::K2)
        .def_readwrite("fix_scale", &Sim3SolverInput2::bFixScale);

    py::class_<Sim3PointRegistrationSolverInput>(m, "Sim3PointRegistrationSolverInput")
        .def(py::init<>())
        .def_readwrite("points_3d_w1", &Sim3PointRegistrationSolverInput::mvX3Dw1)
        .def_readwrite("points_3d_w2", &Sim3PointRegistrationSolverInput::mvX3Dw2)
        .def_readwrite("sigma2", &Sim3PointRegistrationSolverInput::mSigma2)
        .def_readwrite("fix_scale", &Sim3PointRegistrationSolverInput::bFixScale);

    py::class_<Sim3Solver>(m, "Sim3Solver")
        .def(py::init<const Sim3SolverInput &>())
        .def(py::init<const Sim3SolverInput2 &>())
        .def("set_ransac_parameters", &Sim3Solver::SetRansacParameters, "probability"_a = 0.99,
             "minInliers"_a = 6, "maxIterations"_a = 300)
        .def("find",
             [](Sim3Solver &s) {
                 std::vector<uint8_t> vbInliers;
                 int nInliers;
                 bool bConverged;
                 Eigen::Matrix4f transformation;
                 {
                     py::gil_scoped_release release;
                     transformation = s.find(vbInliers, nInliers, bConverged);
                 }
                 return std::make_tuple(transformation, vbInliers, nInliers, bConverged);
             })
        .def(
            "iterate",
            [](Sim3Solver &s, const int nIterations) {
                std::vector<uint8_t> vbInliers;
                int nInliers;
                bool bNoMore;
                bool bConverged;
                Eigen::Matrix4f transformation;
                {
                    py::gil_scoped_release release;
                    transformation =
                        s.iterate(nIterations, bNoMore, vbInliers, nInliers, bConverged);
                }
                return std::make_tuple(transformation, bNoMore, vbInliers, nInliers, bConverged);
            },
            "nIterations"_a)
        .def("get_estimated_transformation", &Sim3Solver::GetEstimatedTransformation)
        .def("get_estimated_scale", &Sim3Solver::GetEstimatedScale)
        .def("get_estimated_rotation", &Sim3Solver::GetEstimatedRotation)
        .def("get_estimated_translation", &Sim3Solver::GetEstimatedTranslation)
        .def("compute_3d_registration_error", &Sim3Solver::Compute3dRegistrationError);

    py::class_<Sim3PointRegistrationSolver>(m, "Sim3PointRegistrationSolver")
        .def(py::init<const Sim3PointRegistrationSolverInput &>())
        .def("set_ransac_parameters", &Sim3PointRegistrationSolver::SetRansacParameters,
             "probability"_a = 0.99, "minInliers"_a = 6, "maxIterations"_a = 300)
        .def("find",
             [](Sim3PointRegistrationSolver &s) {
                 std::vector<uint8_t> vbInliers;
                 int nInliers;
                 bool bConverged;
                 Eigen::Matrix4f transformation;
                 {
                     py::gil_scoped_release release;
                     transformation = s.find(vbInliers, nInliers, bConverged);
                 }
                 return std::make_tuple(transformation, vbInliers, nInliers, bConverged);
             })
        .def(
            "iterate",
            [](Sim3PointRegistrationSolver &s, const int nIterations) {
                std::vector<uint8_t> vbInliers;
                int nInliers;
                bool bNoMore;
                bool bConverged;
                Eigen::Matrix4f transformation;
                {
                    py::gil_scoped_release release;
                    transformation =
                        s.iterate(nIterations, bNoMore, vbInliers, nInliers, bConverged);
                }
                return std::make_tuple(transformation, bNoMore, vbInliers, nInliers, bConverged);
            },
            "nIterations"_a)
        .def("get_estimated_transformation",
             &Sim3PointRegistrationSolver::GetEstimatedTransformation)
        .def("get_estimated_scale", &Sim3PointRegistrationSolver::GetEstimatedScale)
        .def("get_estimated_rotation", &Sim3PointRegistrationSolver::GetEstimatedRotation)
        .def("get_estimated_translation", &Sim3PointRegistrationSolver::GetEstimatedTranslation)
        .def("compute_3d_registration_error",
             &Sim3PointRegistrationSolver::Compute3dRegistrationError);
}
