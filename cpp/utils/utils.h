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

#ifndef UTILS_MODULE_H
#define UTILS_MODULE_H

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>


namespace py = pybind11;

namespace utils
{

void extractPatch( const cv::Mat& image, const cv::KeyPoint& kp,
                          const int& patch_size, cv::Mat& patch,
                          const bool use_orientation=true,
                          const float scale_factor=1.0,
                          const int warp_flags= cv::WARP_INVERSE_MAP + cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS);


void extractPatches( const cv::Mat& image, const std::vector<cv::KeyPoint>& kps,
                          const int& patch_size, std::vector<cv::Mat>& patches,
                          const bool use_orientation=true,
                          const float scale_factor=1.0,
                          const int warp_flags= cv::WARP_INVERSE_MAP + cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS);


std::pair<py::array_t<int>, py::array_t<int>>
goodMatchesSimple(const std::vector<std::pair<cv::DMatch, cv::DMatch>>& matches, float ratio_test = 0.7f);

std::pair<py::array_t<int>, py::array_t<int>>
goodMatchesOneToOne(const std::vector<std::vector<cv::DMatch>>& matches, float ratio_test = 0.7f);


py::tuple rowMatches(
    const std::vector<cv::KeyPoint>& kps1,
    const std::vector<cv::KeyPoint>& kps2,
    const std::vector<cv::DMatch>& matches,
    float max_distance,
    float max_row_distance,
    float max_disparity);

std::pair<std::vector<int>, std::vector<int>> rowMatches_np(
    py::array_t<float, py::array::c_style | py::array::forcecast> kps1_np,
    py::array_t<float, py::array::c_style | py::array::forcecast> kps2_np,
    const std::vector<cv::DMatch>& matches,
    float max_distance,
    float max_row_distance,
    float max_disparity);


py::tuple rowMatchesWithRatioTest(
    const std::vector<cv::KeyPoint>& kps1,
    const std::vector<cv::KeyPoint>& kps2,
    const std::vector<std::vector<cv::DMatch>>& matches,
    float max_distance,
    float max_row_distance,
    float max_disparity,
    float ratio_test);

std::pair<std::vector<int>, std::vector<int>> rowMatchesWithRatioTest_np(
    py::array_t<float, py::array::c_style | py::array::forcecast> kps1_np,
    py::array_t<float, py::array::c_style | py::array::forcecast> kps2_np,
    const std::vector<std::vector<cv::DMatch>>& knn_matches,
    float max_distance,
    float max_row_distance,
    float max_disparity,
    float ratio_test);    


py::tuple filterNonRowMatches(
    const std::vector<cv::KeyPoint>& kps1,
    const std::vector<cv::KeyPoint>& kps2,
    const std::vector<int>& idxs1,
    const std::vector<int>& idxs2,
    float max_row_distance,
    float max_disparity);

std::pair<std::vector<int>, std::vector<int>> filterNonRowMatches_np(
    py::array_t<float, py::array::c_style | py::array::forcecast> kps1_np,
    py::array_t<float, py::array::c_style | py::array::forcecast> kps2_np,
    py::array_t<int, py::array::c_style | py::array::forcecast> idxs1_np,
    py::array_t<int, py::array::c_style | py::array::forcecast> idxs2_np,
    float max_row_distance,
    float max_disparity);


py::array_t<float> extractMeanColors(
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> img,
    py::array_t<int, py::array::c_style | py::array::forcecast> img_coords,
    int delta,
    std::array<float, 3> default_color
);


} // namespace utils 


#endif 