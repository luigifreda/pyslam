/**
* This file is part of ibow-lcd.
*
* Copyright (C) 2017 Emilio Garcia-Fidalgo <emilio.garcia@uib.es> (University of the Balearic Islands)
*
* ibow-lcd is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ibow-lcd is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ibow-lcd. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef INCLUDE_IBOW_LCD_LCDETECTOR_H_
#define INCLUDE_IBOW_LCD_LCDETECTOR_H_

#include <algorithm>
#include <chrono>
#include <fstream>
#include <iostream>
#include <queue>
#include <memory>
#include <string>
#include <sstream>
#include <vector>

#include "ibow-lcd/island.h"
#include "obindex2/binary_index.h"

namespace ibow_lcd {

// LCDetectorParams
struct LCDetectorParams {
  LCDetectorParams() :
    k(16),
    s(150),
    t(4),
    merge_policy(obindex2::MERGE_POLICY_NONE),
    purge_descriptors(true),
    min_feat_apps(2),
    p(250),
    nndr(0.8f),
    nndr_bf(0.8f),
    ep_dist(2.0),
    conf_prob(0.985),
    min_score(0.3),
    island_size(7),
    min_inliers(22),
    nframes_after_lc(3),
    min_consecutive_loops(5) {}

  // Image index params
  unsigned k;  // Branching factor for the image index
  unsigned s;  // Maximum leaf size for the image index
  unsigned t;  // Number of trees to search in parallel
  obindex2::MergePolicy merge_policy;  // Merging policy
  bool purge_descriptors;  // Delete descriptors from index?
  unsigned min_feat_apps;  // Min apps of a feature to be a visual word

  // Loop Closure Params
  unsigned p;  // Previous images to be discarded when searching for a loop
  float nndr;  // Nearest neighbour distance ratio
  float nndr_bf;  // NNDR when matching with brute force
  double ep_dist;  // Distance to epipolar lines
  double conf_prob;  // Confidence probability
  double min_score;  // Min score to consider an image matching as correct
  unsigned island_size;  // Max number of images of an island
  unsigned min_inliers;  // Minimum number of inliers to consider a loop
  unsigned nframes_after_lc;  // Number of frames after a lc to wait for new lc
  int min_consecutive_loops;  // Min consecutive loops to avoid ep. geometry
};

// LCDetectorStatus
enum LCDetectorStatus {
  LC_DETECTED,
  LC_NOT_DETECTED,
  LC_NOT_ENOUGH_IMAGES,
  LC_NOT_ENOUGH_ISLANDS,
  LC_NOT_ENOUGH_INLIERS,
  LC_TRANSITION
};

// LCDetectorResult
struct LCDetectorResult {
  LCDetectorResult() :
    status(LC_NOT_DETECTED),
    query_id(1),
    train_id(-1),
    score{NAN} {}

  inline bool isLoop() {
    return status == LC_DETECTED;
  }

  LCDetectorStatus status;
  unsigned query_id;
  unsigned train_id;
  unsigned inliers;
  double score;  

  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & status;
    ar & query_id;
    ar & train_id;
    ar & inliers;
    ar & score;
  }
};

class LCDetector {
 public:
  explicit LCDetector(const LCDetectorParams& params);
  virtual ~LCDetector();

  void process(const unsigned image_id,
               const std::vector<cv::KeyPoint>& kps,
               const cv::Mat& descs,
               LCDetectorResult* result);

  void process(const unsigned image_id,
               const std::vector<cv::KeyPoint>& kps,
               const cv::Mat& descs,
               const bool add_to_index,
               LCDetectorResult* result);

  void debug(const unsigned image_id,
             const std::vector<cv::KeyPoint>& kps,
             const cv::Mat& descs,
             std::ofstream& out_file);

  // Number of images pushed to the LCDetector 
  size_t numPushedImages() const { return num_pushed_images; }

  // Number of images in the index 
  size_t numImages() const { return (index_? index_->numImages(): 0); }
  // Number of descriptors in the index
  size_t numDescriptors() const { return (index_? index_->numDescriptors(): 0); }

  void clear();

  void save(const std::string& filename) const;
  void load(const std::string& filename);

  void printStatus() const;

  friend std::ostream& operator<<(std::ostream &os, const LCDetector &db);

private:

  // Parameters
  unsigned p_;
  float nndr_;
  float nndr_bf_;
  double ep_dist_;
  double conf_prob_;
  double min_score_;
  unsigned island_size_;
  unsigned island_offset_;
  unsigned min_inliers_;
  unsigned nframes_after_lc_;

  // Last loop closure detected
  LCDetectorResult last_lc_result_;
  Island last_lc_island_;
  int min_consecutive_loops_;
  int consecutive_loops_ = 0;

  int num_pushed_images = 0;

  // Image Index
  std::shared_ptr<obindex2::ImageIndex> index_;

  // Queues to delay the publication of hypothesis
  std::queue<unsigned> queue_ids_;
  std::queue<std::vector<cv::KeyPoint> > queue_kps_;
  std::queue<cv::Mat> queue_descs_;

  std::vector<std::vector<cv::KeyPoint> > prev_kps_;
  std::vector<cv::Mat> prev_descs_;

private: 

  void addImage(const unsigned image_id,
                const std::vector<cv::KeyPoint>& kps,
                const cv::Mat& descs);
  void filterMatches(
      const std::vector<std::vector<cv::DMatch> >& matches_feats,
      std::vector<cv::DMatch>* matches);
  void filterCandidates(
      const std::vector<obindex2::ImageMatch>& image_matches,
      std::vector<obindex2::ImageMatch>* image_matches_filt);
  void buildIslands(
      const std::vector<obindex2::ImageMatch>& image_matches,
      std::vector<Island>* islands);
  void getPriorIslands(
      const Island& island,
      const std::vector<Island>& islands,
      std::vector<Island>* p_islands);
  unsigned checkEpipolarGeometry(
      const std::vector<cv::Point2f>& query,
      const std::vector<cv::Point2f>& train);
  void ratioMatchingBF(const cv::Mat& query,
                     const cv::Mat& train,
                     std::vector<cv::DMatch>* matches);
  void convertPoints(const std::vector<cv::KeyPoint>& query_kps,
                     const std::vector<cv::KeyPoint>& train_kps,
                     const std::vector<cv::DMatch>& matches,
                     std::vector<cv::Point2f>* query,
                     std::vector<cv::Point2f>* train);


protected:
  friend class boost::serialization::access;
  template <class Archive>
  void serialize(Archive & ar, const unsigned int version) {
    ar & p_;
    ar & nndr_;
    ar & nndr_bf_;
    ar & ep_dist_;
    ar & conf_prob_;
    ar & min_score_;
    ar & island_size_;
    ar & island_offset_;
    ar & min_inliers_;
    ar & nframes_after_lc_;

    ar & last_lc_result_;
    ar & last_lc_island_;
    ar & min_consecutive_loops_;
    ar & consecutive_loops_;

    ar & num_pushed_images;

    ar & index_;

    ar & queue_ids_;
    ar & queue_kps_;
    ar & queue_descs_;

    ar & prev_kps_;
    ar & prev_descs_;
  }

};

}  // namespace ibow_lcd

#endif  // INCLUDE_IBOW_LCD_LCDETECTOR_H_
