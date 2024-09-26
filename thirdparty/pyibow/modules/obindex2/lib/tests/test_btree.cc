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

#include "obindex2/binary_tree.h"

int main() {
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

  // Creating a set of descriptors
  obindex2::BinaryDescriptorSet set;
  for (int i = 0; i < descs.rows; i++) {
    cv::Mat desc = descs.row(i);
    obindex2::BinaryDescriptorPtr d =
      std::make_shared<obindex2::BinaryDescriptor>(desc);
    set.insert(d);
  }

  obindex2::BinaryTree tree1(std::make_shared<
                             obindex2::BinaryDescriptorSet>(set));

  tree1.deleteTree();
  tree1.buildTree();

  // Deleting descriptors
  for (auto it = set.begin(); it != set.end(); it++) {
    obindex2::BinaryDescriptorPtr d = *it;
    tree1.deleteDescriptor(d);
  }

  tree1.printTree();

  tree1.buildTree();

  // Searching in the tree
  for (auto it = set.begin(); it != set.end(); it++) {
    obindex2::BinaryDescriptorPtr q = *it;
    obindex2::NodeQueuePtr pq = std::make_shared<
                                          obindex2::NodeQueue>();
    obindex2::DescriptorQueuePtr r = std::make_shared<
                                          obindex2::DescriptorQueue>();

    tree1.traverseFromRoot(q, pq, r);
    r->sort();

    std::cout << "---" << std::endl;

    for (unsigned i = 0; i < r->size(); i++) {
      obindex2::DescriptorQueueItem n = r->get(i);
      std::cout << n.dist << " " << n.desc << " vs " << q << std::endl;
    }
  }

  return 0;  // Correct test
}
