/**
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

#include <pybind11/pybind11.h>

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "ORBextractor.h"
#include "opencv_type_casters.h"

namespace py = pybind11;
using namespace pybind11::literals;

using namespace ORB_SLAM2;

// Create typedefs for easier binding
using ORBextractorNonDeterministic = ORBextractor<false>; // original non-deterministic version
using ORBextractorDeterministic = ORBextractor<true>;     // deterministic version

// macro to bind the class
#define BIND_ORBEXTRACTOR(CLASS, NAME)                                                             \
    py::class_<CLASS>(m, NAME)                                                                     \
        .def(py::init<int, float, int, int, int>(), "nfeatures"_a, "scaleFactor"_a, "nlevels"_a,   \
             "iniThFAST"_a = 20, "minThFAST"_a = 7)                                                \
        .def("GetNumFeatures", &CLASS::GetNumFeatures)                                             \
        .def("GetLevels", &CLASS::GetLevels)                                                       \
        .def("GetScaleFactor", &CLASS::GetScaleFactor)                                             \
        .def("SetNumFeatures", &CLASS::SetNumFeatures)                                             \
        .def("detectAndCompute",                                                                   \
             [](CLASS &o, cv::Mat &image) {                                                        \
                 cv::Mat mask = cv::Mat();                                                         \
                 std::vector<cv::KeyPoint> keypoints;                                              \
                 cv::Mat descriptors;                                                              \
                 {                                                                                 \
                     py::gil_scoped_release release;                                               \
                     o.detectAndCompute(image, mask, keypoints, descriptors);                      \
                 }                                                                                 \
                 return std::make_tuple(keypoints, descriptors);                                   \
             })                                                                                    \
        .def(                                                                                      \
            "detect",                                                                              \
            [](CLASS &o, cv::Mat &image, bool bComputeOrientation = true) {                        \
                cv::Mat mask = cv::Mat();                                                          \
                std::vector<cv::KeyPoint> keypoints;                                               \
                {                                                                                  \
                    py::gil_scoped_release release;                                                \
                    o.detect(image, mask, keypoints, bComputeOrientation);                         \
                }                                                                                  \
                return keypoints;                                                                  \
            },                                                                                     \
            "image"_a, "bComputeOrientation"_a = true)                                             \
        .def_static("DistributeOctTree", &CLASS::DistributeOctTree, "vToDistributeKeys"_a,         \
                    "minX"_a, "maxX"_a, "minY"_a, "maxY"_a, "nFeatures"_a, "level"_a = 0)          \
        .def("__repr__", [](const CLASS &o) {                                                      \
            std::stringstream ss;                                                                  \
            ss << "<orb_features." << NAME << " - #features: " << o.GetNumFeatures()               \
               << ", #levels: " << o.GetLevels() << ", factor: " << o.GetScaleFactor() << ">";     \
            return ss.str();                                                                       \
        });

PYBIND11_MODULE(orbslam2_features, m) {
    m.doc() = "pybind11 plugin for ORBSLAM2 features";

    // bindings to ORBextractor class
    BIND_ORBEXTRACTOR(ORBextractorNonDeterministic, "ORBextractor");
    BIND_ORBEXTRACTOR(ORBextractorDeterministic, "ORBextractorDeterministic");
}