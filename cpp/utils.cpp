
#include "utils.h"

namespace utils
{

void extract_patch( const cv::Mat& image, const cv::KeyPoint& kp,
                          const int& patch_size, cv::Mat& patch,
                          const bool use_orientation,
                          const float scale_factor,
                          const int warp_flags)
{
    cv::Mat M;
    if ( use_orientation )
    {
        const float s = scale_factor * (float) kp.size / (float) patch_size;

        const float cosine = (kp.angle>=0) ? cos(kp.angle*(float)CV_PI/180.0f) : 1.f;
        const float sine   = (kp.angle>=0) ? sin(kp.angle*(float)CV_PI/180.0f) : 0.f;

        float M_[] = {
            s*cosine, -s*sine,   (-s*cosine + s*sine  ) * patch_size/2.0f + kp.pt.x,
            s*sine,    s*cosine, (-s*sine   - s*cosine) * patch_size/2.0f + kp.pt.y
        };
        M = cv::Mat( 2, 3, CV_32FC1, M_ ).clone();
    }
    else
    {
        const float s = scale_factor * (float)kp.size / (float)patch_size;
        float M_[] = {
        s,  0.f, -s * patch_size/2.0f + kp.pt.x,
        0.f,  s, -s * patch_size/2.0f + kp.pt.y
        };
        M = cv::Mat( 2, 3, CV_32FC1, M_ ).clone();
    }

    cv::warpAffine( image, patch, M, cv::Size( patch_size, patch_size ), warp_flags);
}


void extract_patches( const cv::Mat& image, const std::vector<cv::KeyPoint>& kps,
                          const int& patch_size, std::vector<cv::Mat>& patches,
                          const bool use_orientation,
                          const float scale_factor,
                          const int warp_flags)
{
    patches.resize(kps.size());
    for(size_t ii=0,iiEnd=kps.size();ii<iiEnd;ii++)
    {
        cv::Mat& patchii = patches[ii];
        const cv::KeyPoint& kpii = kps[ii];
        extract_patch(image, kpii, patch_size, patchii, use_orientation, scale_factor, warp_flags);        
    }

}


} // namespace utils 