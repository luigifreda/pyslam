/**
* This file is part of obindex2.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* obindex2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* obindex2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with obindex2. If not, see <http://www.gnu.org/licenses/>.
*/

#include <opencv2/features2d/features2d.hpp>
#ifdef HAVE_OPENCV_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif
#include "obindex2/binary_index.h"

int main() 
{
  // Creating feature detector and descriptor
  cv::Ptr<cv::FastFeatureDetector> det =
          cv::FastFeatureDetector::create();
#ifdef HAVE_OPENCV_CONTRIB          
  cv::Ptr<cv::xfeatures2d::BriefDescriptorExtractor> des =
          cv::xfeatures2d::BriefDescriptorExtractor::create();
#else 
  cv::Ptr<cv::Feature2D> des = cv::ORB::create();
#endif   

  // Loading the test image
  cv::Mat img = cv::imread("image00.jpg");

  // Computing keypoints and descriptors
  std::vector<cv::KeyPoint> kps;
  det->detect(img, kps);

  cv::Mat descs;
  des->compute(img, kps, descs);

  // Creating a new index of images
  obindex2::ImageIndex index(16, 150, 4);

  index.addImage(0, kps, descs);

  // Searching the same descriptor on the index
  std::vector<std::vector<cv::DMatch> > matches;
  index.searchDescriptors(descs, &matches, 2, 64);

  // Plotting the results
  unsigned bad_assoc = 0;
  for (unsigned i = 0; i < matches.size(); i++) {
    std::cout << "---" << std::endl;
    std::cout << "Descriptor " << i << std::endl;
    for (unsigned j = 0; j < matches[i].size(); j++) {
      std::cout << "Query: " << matches[i][j].queryIdx << ", ";
      std::cout << "Train: " << matches[i][j].trainIdx << ", ";
      std::cout << "Image: " << matches[i][j].imgIdx << ", ";
      std::cout << "Dist: " << matches[i][j].distance << std::endl;
    }

    if (static_cast<int>(i) != matches[i][0].trainIdx) {
      bad_assoc++;
    }
  }

  std::cout << "Incorrect associations: " << bad_assoc << std::endl;

  return 0;  // Correct test
}
