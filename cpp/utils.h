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

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <vector>
#include <iostream>

namespace utils
{

void extract_patch( const cv::Mat& image, const cv::KeyPoint& kp,
                          const int& patch_size, cv::Mat& patch,
                          const bool use_orientation=true,
                          const float scale_factor=1.0,
                          const int warp_flags= cv::WARP_INVERSE_MAP + cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS);


void extract_patches( const cv::Mat& image, const std::vector<cv::KeyPoint>& kps,
                          const int& patch_size, std::vector<cv::Mat>& patches,
                          const bool use_orientation=true,
                          const float scale_factor=1.0,
                          const int warp_flags= cv::WARP_INVERSE_MAP + cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS);


} // namespace utils 


#endif 