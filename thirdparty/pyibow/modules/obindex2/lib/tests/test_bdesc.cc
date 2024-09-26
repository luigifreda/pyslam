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

#define CATCH_CONFIG_MAIN

#include <opencv2/features2d/features2d.hpp>
#ifdef HAVE_OPENCV_CONTRIB
#include <opencv2/xfeatures2d/nonfree.hpp>
#include <opencv2/xfeatures2d.hpp>
#endif

#include "catch/catch.hpp"

#include "obindex2/binary_descriptor.h"

TEST_CASE("BD: self-created descriptors", "[bdesc]") {
  obindex2::BinaryDescriptor d1, d2;

  REQUIRE(d1 == d2);
  REQUIRE(d1.size() == 256);
  REQUIRE(d2.size() == 256);

  // Modifying several bits
  for (int i = 0; i < 256; i++) {
    if (i % 2 == 0) {
      d1.set(i);
    } else {
      d2.set(i);
    }
  }
  REQUIRE(d1 != d2);

  // Validating the Hamming distance
  int dist = static_cast<int>(
                obindex2::BinaryDescriptor::distHamming(d1, d2));
  REQUIRE(dist == 256);

  // Resetting original bits
  for (int i = 0; i < 256; i++) {
    if (i % 2 == 0) {
      d1.reset(i);
    } else {
      d2.reset(i);
    }
  }
  REQUIRE(d1 == d2);
}

TEST_CASE("BD: assigning descriptors", "[bdesc]") {
  obindex2::BinaryDescriptor d1, d2;

  // Modifying several bits
  for (int i = 0; i < 256; i++) {
    if (i % 2 == 0) {
      d1.set(i);
    } else {
      d2.set(i);
    }
  }
  REQUIRE(d1 != d2);

  d1 = d2;

  REQUIRE(d1 == d2);
  REQUIRE(&(d1.bits_) != &(d2.bits_));
}

TEST_CASE("BD: create descriptors from cv::Mat", "[bdesc]") {
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

  REQUIRE(kps.size());
  REQUIRE(descs.type() == CV_8U);

  // Selecting the first descriptor
  cv::Mat desc = descs.row(0);

  // Creating a BinaryDescriptor class using this descriptor
  obindex2::BinaryDescriptor d(desc);

  // Retranslate the descriptor
  cv::Mat m = d.toCvMat();

  // Assesing if the two descriptors are the same
  REQUIRE(m.cols == desc.cols);
  REQUIRE(m.rows == desc.rows);
  REQUIRE(m.ptr() != desc.ptr());

  cv::Mat diff;
  cv::compare(m, desc, diff, cv::CMP_NE);
  int nz = cv::countNonZero(diff);
  REQUIRE(nz == 0);
}
