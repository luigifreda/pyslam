#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <iostream>
#include <vector>

#include "ORBextractor.h"


int main(int argc, char** argv) {

    // Read image
    cv::Mat img = cv::imread("../kitti06-436.png", cv::IMREAD_COLOR);
    if (img.empty()) {
        std::cout << "Error opening image" << std::endl;
        return -1;
    }

    // Main settings
    int num_features = 2000;
    int num_levels = 8;
    float scale_factor = 1.2f;

    // Declare ORB extractor
    ORB_SLAM2::ORBextractor orb_extractor(num_features, scale_factor, num_levels);

    // Detect and compute keypoints
    std::vector<cv::KeyPoint> kps;
    cv::Mat des;

    for(int i = 0; i < 20; i++){
        
        orb_extractor.detectAndCompute(img.clone(), cv::Mat(), kps, des);

        // Print keypoint information
        std::cout << "# keypoints: " << kps.size();
        if (!des.empty()) {
            std::cout << ", descriptor shape: " << des.size();
        }    
        std::cout << std::endl;    
    }

    // Draw keypoints on image
    cv::Mat img_draw = img.clone();
    cv::drawKeypoints(img_draw, kps, img_draw, cv::Scalar(0, 255, 0), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

    // Show image
    cv::imshow("Image with Keypoints", img_draw);
    cv::waitKey(0);

    return 0;
}