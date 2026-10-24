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

#include "MLPnPsolver.h"
#include "PnPsolver.h"
#include "opencv_type_casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace utils;

PYBIND11_MODULE(pnpsolver, m) {
    // optional module docstring
    m.doc() = "pybind11 plugin for pnpsolver module";

    py::class_<PnPsolverInput>(m, "PnPsolverInput")
        .def(py::init<>())
        .def_readwrite("points_2d", &PnPsolverInput::mvP2D)
        .def_readwrite("points_3d", &PnPsolverInput::mvP3Dw)
        .def_readwrite("sigmas2", &PnPsolverInput::mvSigma2)
        .def_readwrite("fx", &PnPsolverInput::fx)
        .def_readwrite("fy", &PnPsolverInput::fy)
        .def_readwrite("cx", &PnPsolverInput::cx)
        .def_readwrite("cy", &PnPsolverInput::cy);

    py::class_<PnPsolver>(m, "PnPsolver")
        .def(py::init<const PnPsolverInput &>())
        .def("set_ransac_parameters", &PnPsolver::SetRansacParameters, "probability"_a = 0.99,
             "minInliers"_a = 8, "maxIterations"_a = 300, "minSet"_a = 4, "epsilon"_a = 0.4,
             "th2"_a = 5.991)
        .def("find",
             [](PnPsolver &s) {
                 std::vector<uint8_t> vbInliers;
                 int nInliers;
                 cv::Mat transformation;
                 {
                     py::gil_scoped_release release;
                     transformation = s.find(vbInliers, nInliers);
                 }
                 return std::make_tuple(transformation, vbInliers, nInliers);
             })
        .def(
            "iterate",
            [](PnPsolver &s, const int nIterations) {
                std::vector<uint8_t> vbInliers;
                int nInliers;
                bool bNoMore;
                cv::Mat Tout;
                {
                    py::gil_scoped_release release;
                    Tout = s.iterate(nIterations, bNoMore, vbInliers, nInliers);
                }
                bool ok = nInliers > 0;
                return std::make_tuple(ok, Tout, bNoMore, vbInliers, nInliers);
            },
            "nIterations"_a);

    py::class_<MLPnPsolver>(m, "MLPnPsolver")
        .def(py::init<const PnPsolverInput &>())
        .def("set_ransac_parameters", &MLPnPsolver::SetRansacParameters, "probability"_a = 0.99,
             "minInliers"_a = 8, "maxIterations"_a = 300, "minSet"_a = 6, "epsilon"_a = 0.4,
             "th2"_a = 5.991)
        .def(
            "iterate",
            [](MLPnPsolver &s, const int nIterations) {
                std::vector<uint8_t> vbInliers;
                int nInliers;
                bool bNoMore;
                Eigen::Matrix4f Tout;
                bool ok;
                {
                    py::gil_scoped_release release;
                    ok = s.iterate(nIterations, bNoMore, vbInliers, nInliers, Tout);
                }
                return std::make_tuple(ok, Tout, bNoMore, vbInliers, nInliers);
            },
            "nIterations"_a);
}
