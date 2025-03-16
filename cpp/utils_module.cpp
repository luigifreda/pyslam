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


PYBIND11_MODULE(pyslam_utils, m) 
{
    // optional module docstring
    m.doc() = "pybind11 plugin for pyslam_utils module";
    
    // void extract_patches( const cv::Mat& image, const std::vector<cv::KeyPoint>& kps,
    //                         const int& patchSize, std::vector<cv::Mat>& patches,
    //                         const bool use_orientation=true,
    //                         const float scale_factor=1.0,
    //                         const int warp_flags= cv::WARP_INVERSE_MAP + cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS);
    m.def("extract_patches", 
            [](cv::Mat& image, const std::vector<cv::KeyPoint>& kps,
               const int patch_size,
               const bool use_orientation,
               const float scale_factor,
               const int warp_flags
               ) 
            { 
                std::vector<cv::Mat> patches;
                utils::extract_patches(image,kps,patch_size,patches,use_orientation,scale_factor,warp_flags);
                return patches; 
            },
            "image"_a,"kps"_a,"patch_size"_a,"use_orientation"_a=true,"scale_factor"_a=1.0,"warp_flags"_a=cv::WARP_INVERSE_MAP + cv::INTER_CUBIC + cv::WARP_FILL_OUTLIERS
            );          

}
