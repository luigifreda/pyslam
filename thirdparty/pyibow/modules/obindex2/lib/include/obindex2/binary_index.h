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

#ifndef LIB_INCLUDE_OBINDEX2_BINARY_INDEX_H_
#define LIB_INCLUDE_OBINDEX2_BINARY_INDEX_H_

#include <list>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "obindex2/binary_tree.h"

namespace obindex2 {

enum MergePolicy {
  MERGE_POLICY_NONE,
  MERGE_POLICY_AND,
  MERGE_POLICY_OR
};

struct InvIndexItem {
  InvIndexItem() :
      image_id(0),
      pt(0.0f, 0.0f),
      dist(DBL_MAX),
      kp_ind(-1) {}

  InvIndexItem(const int id,
               const cv::Point2f kp,
               const double d,
               const int kp_i = -1) :
  image_id(id),
  pt(kp),
  dist(d),
  kp_ind(kp_i)
  {}

  unsigned image_id;
  cv::Point2f pt;
  double dist;
  int kp_ind;
};

struct ImageMatch {
  ImageMatch() :
      image_id(-1),
      score(0.0) {}

  explicit ImageMatch(const int id, const double sc = 0.0) :
      image_id(id),
      score(sc) {}

  int image_id;
  double score;

  bool operator<(const ImageMatch &lcr) const { return score > lcr.score; }
};

struct PointMatches {
  std::vector<cv::Point2f> query;
  std::vector<cv::Point2f> train;
};

class ImageIndex {
 public:
  // Constructors
  explicit ImageIndex(const unsigned k = 16,
                      const unsigned s = 150,
                      const unsigned t = 4,
                      const MergePolicy merge_policy = MERGE_POLICY_NONE,
                      const bool purge_descriptors = true,
                      const unsigned min_feat_apps = 3);

  // Methods
  void addImage(const unsigned image_id,
                const std::vector<cv::KeyPoint>& kps,
                const cv::Mat& descs);
  void addImage(const unsigned image_id,
                const std::vector<cv::KeyPoint>& kps,
                const cv::Mat& descs,
                const std::vector<cv::DMatch>& matches);
  void searchImages(const cv::Mat& descs,
                    const std::vector<cv::DMatch>& gmatches,
                    std::vector<ImageMatch>* img_matches,
                    bool sort = true);
  void searchDescriptors(const cv::Mat& descs,
                         std::vector<std::vector<cv::DMatch> >* matches,
                         const unsigned knn = 2,
                         const unsigned checks = 32);
  void deleteDescriptor(const unsigned desc_id);
  void getMatchings(const std::vector<cv::KeyPoint>& query_kps,
                    const std::vector<cv::DMatch>& matches,
                    std::unordered_map<unsigned, PointMatches>* point_matches);
  inline unsigned numImages() {
    return nimages_;
  }

  inline unsigned numDescriptors() {
    return dset_.size();
  }

  inline void rebuild() {
    if (init_) {
      trees_.clear();
      initTrees();
    }
  }

 private:
  BinaryDescriptorSet dset_;
  unsigned k_;
  unsigned s_;
  unsigned t_;
  unsigned init_;
  unsigned nimages_;
  unsigned ndesc_;
  MergePolicy merge_policy_;
  bool purge_descriptors_;
  unsigned min_feat_apps_;

  std::vector<BinaryTreePtr> trees_;
  std::unordered_map<BinaryDescriptorPtr,
                     std::vector<InvIndexItem> > inv_index_;
  std::unordered_map<BinaryDescriptorPtr, unsigned> desc_to_id_;
  std::unordered_map<unsigned, BinaryDescriptorPtr> id_to_desc_;
  std::list<BinaryDescriptorPtr> recently_added_;

  void initTrees();
  void searchDescriptor(BinaryDescriptorPtr q,
                        std::vector<BinaryDescriptorPtr>* neigh,
                        std::vector<double>* distances,
                        unsigned knn = 2,
                        unsigned checks = 32);
  void insertDescriptor(BinaryDescriptorPtr q);
  void deleteDescriptor(BinaryDescriptorPtr q);
  void purgeDescriptors(const unsigned curr_img);
};

}  // namespace obindex2

#endif  // LIB_INCLUDE_OBINDEX2_BINARY_INDEX_H_
