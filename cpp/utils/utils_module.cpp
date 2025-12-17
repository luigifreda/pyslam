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

#include <pybind11/pybind11.h>

#include <sstream>

#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>

#include "opencv_type_casters.h"
#include "utils.h"

namespace py = pybind11;
using namespace pybind11::literals;

PYBIND11_MODULE(pyslam_utils, m) {
    // optional module docstring
    m.doc() = "pybind11 plugin for pyslam_utils module";

    m.def(
        "extract_patches",
        [](cv::Mat &image, const std::vector<cv::KeyPoint> &kps, const int patch_size,
           const bool use_orientation, const float scale_factor, const int warp_flags) {
            std::vector<cv::Mat> patches;
            utils::extractPatches(image, kps, patch_size, patches, use_orientation, scale_factor,
                                  warp_flags);
            return patches;
        },
        "image"_a, "kps"_a, "patch_size"_a, "use_orientation"_a = true, "scale_factor"_a = 1.0,
        "warp_flags"_a = cv::WARP_INVERSE_MAP + cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS);

    m.def("good_matches_one_to_one", &utils::goodMatchesOneToOne, py::arg("matches"),
          py::arg("ratio_test") = 0.7f, "Filter good one-to-one matches with ratio test");

    m.def("good_matches_simple", &utils::goodMatchesSimple, py::arg("matches"),
          py::arg("ratio_test") = 0.7f, "Filter good one-to-one matches with ratio test");

    m.def("row_matches", &utils::rowMatches, py::arg("kps1"), py::arg("kps2"), py::arg("matches"),
          py::arg("max_distance"), py::arg("max_row_distance"), py::arg("max_disparity"),
          "Filter raw matches based on row and disparity constraints");

    m.def("row_matches_np", &utils::rowMatches_np, py::arg("kps1_np"), py::arg("kps2_np"),
          py::arg("matches"), py::arg("max_distance"), py::arg("max_row_distance"),
          py::arg("max_disparity"), "Filter raw matches based on row and disparity constraints");

    m.def("row_matches_with_ratio_test", &utils::rowMatchesWithRatioTest, py::arg("kps1"),
          py::arg("kps2"), py::arg("knn_matches"), py::arg("max_distance"),
          py::arg("max_row_distance"), py::arg("max_disparity"), py::arg("ratio_test"),
          "KNN match with ratio test and epipolar filtering");

    m.def("row_matches_with_ratio_test_np", &utils::rowMatchesWithRatioTest_np, py::arg("kps1_np"),
          py::arg("kps2_np"), py::arg("knn_matches"), py::arg("max_distance"),
          py::arg("max_row_distance"), py::arg("max_disparity"), py::arg("ratio_test"),
          "KNN match with ratio test and epipolar filtering");

    m.def("filter_non_row_matches", &utils::filterNonRowMatches, py::arg("kps1"), py::arg("kps2"),
          py::arg("idxs1"), py::arg("idxs2"), py::arg("max_row_distance"), py::arg("max_disparity"),
          "Post-filter matches by epipolar constraints");

    m.def("filter_non_row_matches_np", &utils::filterNonRowMatches_np, py::arg("kps1_np"),
          py::arg("kps2_np"), py::arg("idxs1_np"), py::arg("idxs2_np"), py::arg("max_row_distance"),
          py::arg("max_disparity"), "Post-filter matches by epipolar constraints");

    m.def("extract_mean_colors", &utils::extractMeanColors, py::arg("img"), py::arg("img_coords"),
          py::arg("delta"), py::arg("default_color"), "Extract mean colors from image");
}
