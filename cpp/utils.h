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