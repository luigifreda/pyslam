/*
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

#include "ibow-lcd/lcdetector.h"

#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/filesystem.hpp>

namespace ibow_lcd {

LCDetector::LCDetector(const LCDetectorParams& params) :
      last_lc_island_(-1, 0.0, -1, -1) {
  // Creating the image index
  index_ = std::make_shared<obindex2::ImageIndex>(params.k,
                                                  params.s,
                                                  params.t,
                                                  params.merge_policy,
                                                  params.purge_descriptors,
                                                  params.min_feat_apps);
  // Storing the remaining parameters
  p_ = params.p;
  nndr_ = params.nndr;
  nndr_bf_ = params.nndr_bf;
  ep_dist_ = params.ep_dist;
  conf_prob_ = params.conf_prob;
  min_score_ = params.min_score;
  island_size_ = params.island_size;
  island_offset_ = island_size_ / 2;
  min_inliers_ = params.min_inliers;
  nframes_after_lc_ = params.nframes_after_lc;
  last_lc_result_.status = LC_NOT_DETECTED;
  min_consecutive_loops_ = params.min_consecutive_loops;
  consecutive_loops_ = 0;
}

LCDetector::~LCDetector() {}

void LCDetector::process(const unsigned image_id,
                         const std::vector<cv::KeyPoint>& kps,
                         const cv::Mat& descs,
                         LCDetectorResult* result){

  process(image_id, kps, descs, true/*add_to_index*/, result);
}

void LCDetector::process(const unsigned image_id,
                         const std::vector<cv::KeyPoint>& kps,
                         const cv::Mat& descs,
                         const bool add_to_index,
                         LCDetectorResult* result) {
  result->query_id = image_id;

  if(add_to_index) {
    // Storing the keypoints and descriptors
    prev_kps_.push_back(kps);
    prev_descs_.push_back(descs);

    // Adding the current image to the queue to be added in the future
    queue_ids_.push(image_id);

    num_pushed_images++;
  }

  // Assessing if, at least, p images have arrived
  //if (queue_ids_.size() < p_) {
  if(num_pushed_images < p_) {
    result->status = LC_NOT_ENOUGH_IMAGES;
    result->train_id = 0;
    result->inliers = 0;
    last_lc_result_.status = LC_NOT_ENOUGH_IMAGES;
    return;
  }

  // Adding new hypothesis
  if(!queue_ids_.empty())
  {
    unsigned newimg_id = queue_ids_.front();
    queue_ids_.pop();

    addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id]);
  }
  else
  {
    std::cout << "LCDetector::process(): Queue is empty" << std::endl;
  }

  // Searching similar images in the index
  // Matching the descriptors agains the current visual words
  std::vector<std::vector<cv::DMatch> > matches_feats;

  // Searching the query descriptors against the features
  index_->searchDescriptors(descs, &matches_feats, 2, 64);

  // Filtering matches according to the ratio test
  std::vector<cv::DMatch> matches;
  filterMatches(matches_feats, &matches);

  std::vector<obindex2::ImageMatch> image_matches;

  // We look for similar images according to the filtered matches found
  index_->searchImages(descs, matches, &image_matches, true);

  // Filtering the resulting image matchings
  std::vector<obindex2::ImageMatch> image_matches_filt;
  filterCandidates(image_matches, &image_matches_filt);

  std::vector<Island> islands;
  buildIslands(image_matches_filt, &islands); 

  if (!islands.size()) {
    // No resulting islands
    result->status = LC_NOT_ENOUGH_ISLANDS;
    result->train_id = 0;
    result->inliers = 0;
    last_lc_result_.status = LC_NOT_ENOUGH_ISLANDS;
    return;
  }

  // std::cout << "Resulting Islands:" << std::endl;
  // for (unsigned i = 0; i < islands.size(); i++) {
  //   std::cout << islands[i].toString();
  // }

  // Selecting the corresponding island to be processed
  Island island = islands[0];
  std::vector<Island> p_islands;
  getPriorIslands(last_lc_island_, islands, &p_islands);
  if (p_islands.size()) {
    island = p_islands[0];
  }

  bool overlap = island.overlaps(last_lc_island_);
  last_lc_island_ = island;

  // if () {
  //   consecutive_loops_++;
  // } else {
  //   consecutive_loops_ = 1;
  // }

  unsigned best_img = island.img_id; 

  // Assessing the loop
  if (consecutive_loops_ > min_consecutive_loops_ && overlap) {
    // LOOP can be considered as detected
    result->status = LC_DETECTED;
    result->train_id = best_img;
    result->inliers = 0;
    result->score = island.score;    
    // Store the last result
    last_lc_result_ = *result;
    consecutive_loops_++;
  } else {
    // We obtain the image matchings, since we need them for compute F
    std::vector<cv::DMatch> tmatches;
    std::vector<cv::Point2f> tquery;
    std::vector<cv::Point2f> ttrain;
    ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
    convertPoints(kps, prev_kps_[best_img], tmatches, &tquery, &ttrain);
    unsigned inliers = checkEpipolarGeometry(tquery, ttrain);

    if (inliers > min_inliers_) {
      // LOOP detected
      result->status = LC_DETECTED;
      result->train_id = best_img;
      result->inliers = inliers;
      result->score = island.score;
      // Store the last result
      last_lc_result_ = *result;
      consecutive_loops_++;
    } else {
      result->status = LC_NOT_ENOUGH_INLIERS;
      result->train_id = best_img;
      result->inliers = inliers;
      result->score = island.score;
      last_lc_result_.status = LC_NOT_ENOUGH_INLIERS;
      consecutive_loops_ = 0;
    }
  }
  // else {
  //   result->status = LC_NOT_DETECTED;
  //   last_lc_result_.status = LC_NOT_DETECTED;
  // }
}

void LCDetector::debug(const unsigned image_id,
             const std::vector<cv::KeyPoint>& kps,
             const cv::Mat& descs,
             std::ofstream& out_file) {
  auto start = std::chrono::steady_clock::now();
  // Storing the keypoints and descriptors
  prev_kps_.push_back(kps);
  prev_descs_.push_back(descs);

  // Adding the current image to the queue to be added in the future
  queue_ids_.push(image_id);

  // Assessing if, at least, p images have arrived
  if (queue_ids_.size() < p_) {
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    out_file << 0 << "\t";  // min_id
    out_file << 0 << "\t";  // max_id
    out_file << 0 << "\t";  // img_id
    out_file << 0 << "\t";  // overlap
    out_file << 0 << "\t";  // Inliers
    out_file << index_->numDescriptors() << "\t";  // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t";  // Time
    out_file << std::endl;
    return;
  }

  // Adding new hypothesis
  unsigned newimg_id = queue_ids_.front();
  queue_ids_.pop();

  addImage(newimg_id, prev_kps_[newimg_id], prev_descs_[newimg_id]);

  // Searching similar images in the index
  // Matching the descriptors agains the current visual words
  std::vector<std::vector<cv::DMatch> > matches_feats;

  // Searching the query descriptors against the features
  index_->searchDescriptors(descs, &matches_feats, 2, 64);

  // Filtering matches according to the ratio test
  std::vector<cv::DMatch> matches;
  filterMatches(matches_feats, &matches);

  std::vector<obindex2::ImageMatch> image_matches;

  // We look for similar images according to the filtered matches found
  index_->searchImages(descs, matches, &image_matches, true);

  // Filtering the resulting image matchings
  std::vector<obindex2::ImageMatch> image_matches_filt;
  filterCandidates(image_matches, &image_matches_filt);

  std::vector<Island> islands;
  buildIslands(image_matches_filt, &islands);

  if (!islands.size()) {
    // No resulting islands
    auto end = std::chrono::steady_clock::now();
    auto diff = end - start;
    out_file << 0 << "\t";  // min_id
    out_file << 0 << "\t";  // max_id
    out_file << 0 << "\t";  // img_id
    out_file << 0 << "\t";  // overlap
    out_file << 0 << "\t";  // Inliers
    out_file << index_->numDescriptors() << "\t";  // Voc. Size
    out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t";  // Time
    out_file << std::endl;
    return;
  }

  // std::cout << "Resulting Islands:" << std::endl;
  // for (unsigned i = 0; i < islands.size(); i++) {
  //   std::cout << islands[i].toString();
  // }

  // Selecting the corresponding island to be processed
  Island island = islands[0];
  std::vector<Island> p_islands;
  getPriorIslands(last_lc_island_, islands, &p_islands);
  if (p_islands.size()) {
    island = p_islands[0];
  }

  bool overlap = island.overlaps(last_lc_island_);
  last_lc_island_ = island;

  unsigned best_img = island.img_id;

  // We obtain the image matchings, since we need them for compute F
  std::vector<cv::DMatch> tmatches;
  std::vector<cv::Point2f> tquery;
  std::vector<cv::Point2f> ttrain;
  ratioMatchingBF(descs, prev_descs_[best_img], &tmatches);
  convertPoints(kps, prev_kps_[best_img], tmatches, &tquery, &ttrain);
  unsigned inliers = checkEpipolarGeometry(tquery, ttrain);

  auto end = std::chrono::steady_clock::now();
  auto diff = end - start;

  // Writing results
  out_file << island.min_img_id << "\t";          // min_id
  out_file << island.max_img_id << "\t";          // max_id
  out_file << best_img << "\t";                   // img_id
  out_file << overlap << "\t";                    // overlap
  out_file << inliers << "\t";                    // Inliers
  out_file << index_->numDescriptors() << "\t";   // Voc. Size
  out_file << std::chrono::duration<double, std::milli>(diff).count() << "\t";  // Time
  out_file << std::endl;
}

void LCDetector::clear() {
  index_->clear();
  last_lc_result_.status = LC_NOT_DETECTED;
  consecutive_loops_ = 0;
  num_pushed_images = 0;
}

void LCDetector::addImage(const unsigned image_id,
                          const std::vector<cv::KeyPoint>& kps,
                          const cv::Mat& descs) {
  if (index_->numImages() == 0) {
    // This is the first image that is inserted into the index
    index_->addImage(image_id, kps, descs);
  } else {
    // We have to search the descriptor and filter them before adding descs
    // Matching the descriptors
    std::vector<std::vector<cv::DMatch> > matches_feats;

    // Searching the query descriptors against the features
    index_->searchDescriptors(descs, &matches_feats, 2, 64);

    // Filtering matches according to the ratio test
    std::vector<cv::DMatch> matches;
    filterMatches(matches_feats, &matches);

    // Finally, we add the image taking into account the correct matchings
    index_->addImage(image_id, kps, descs, matches);
  }
}

void LCDetector::filterMatches(
      const std::vector<std::vector<cv::DMatch> >& matches_feats,
      std::vector<cv::DMatch>* matches) {
  // Clearing the current matches vector
  matches->clear();

  // Filtering matches according to the ratio test
  for (unsigned m = 0; m < matches_feats.size(); m++) {
    if (matches_feats[m][0].distance <= matches_feats[m][1].distance * nndr_) {
      matches->push_back(matches_feats[m][0]);
    }
  }
}

void LCDetector::filterCandidates(
      const std::vector<obindex2::ImageMatch>& image_matches,
      std::vector<obindex2::ImageMatch>* image_matches_filt) {
  image_matches_filt->clear();

  double max_score = image_matches[0].score;
  double min_score = image_matches[image_matches.size() - 1].score;

  for (unsigned i = 0; i < image_matches.size(); i++) {
    // Computing the new score
    double new_score = (image_matches[i].score - min_score) /
                       (max_score - min_score);
    // Assessing if this image match is higher than a threshold
    if (new_score > min_score_) {
      obindex2::ImageMatch match = image_matches[i];
      match.score = new_score;
      image_matches_filt->push_back(match);
    } else {
      break;
    }
  }
}

void LCDetector::buildIslands(
      const std::vector<obindex2::ImageMatch>& image_matches,
      std::vector<Island>* islands) {
  islands->clear();

  // We process each of the resulting image matchings
  for (unsigned i = 0; i < image_matches.size(); i++) {
    // Getting information about this match
    unsigned curr_img_id = static_cast<unsigned>(image_matches[i].image_id);
    double curr_score = image_matches[i].score;

    // Theoretical island limits
    unsigned min_id = static_cast<unsigned>
                              (std::max((int)curr_img_id - (int)island_offset_,
                               0));
    unsigned max_id = curr_img_id + island_offset_;

    // We search for the closest island
    bool found = false;
    for (unsigned j = 0; j < islands->size(); j++) {
      if (islands->at(j).fits(curr_img_id)) {
        islands->at(j).incrementScore(curr_score);
        found = true;
        break;
      } else {
        // We adjust the limits of a future island
        islands->at(j).adjustLimits(curr_img_id, &min_id, &max_id);
      }
    }

    // Creating a new island if required
    if (!found) {
      Island new_island(curr_img_id,
                        curr_score,
                        min_id,
                        max_id);
      islands->push_back(new_island);
    }
  }

  // Normalizing the final scores according to the number of images
  for (unsigned j = 0; j < islands->size(); j++) {
    islands->at(j).normalizeScore();
  }

  std::sort(islands->begin(), islands->end());
}

void LCDetector::getPriorIslands(
      const Island& island,
      const std::vector<Island>& islands,
      std::vector<Island>* p_islands) {
  p_islands->clear();

  // We search for overlapping islands
  for (unsigned i = 0; i < islands.size(); i++) {
    Island tisl = islands[i];
    if (island.overlaps(tisl)) {
      p_islands->push_back(tisl);
    }
  }
}

unsigned LCDetector::checkEpipolarGeometry(
                                      const std::vector<cv::Point2f>& query,
                                      const std::vector<cv::Point2f>& train) {
  std::vector<uchar> inliers(query.size(), 0);
  if (query.size() > 7) {
    cv::Mat F =
      cv::findFundamentalMat(
        cv::Mat(query), cv::Mat(train),      // Matching points
        cv::FM_RANSAC,                        // RANSAC method
        ep_dist_,                                 // Distance to epipolar line
        conf_prob_,                              // Confidence probability
        inliers);                            // Match status (inlier or outlier)
  }

  // Extract the surviving (inliers) matches
  auto it = inliers.begin();
  unsigned total_inliers = 0;
  for (; it != inliers.end(); it++) {
    if (*it)
      total_inliers++;
  }

  return total_inliers;
}

void LCDetector::ratioMatchingBF(const cv::Mat& query,
                                 const cv::Mat& train,
                                 std::vector<cv::DMatch>* matches) {
  matches->clear();
  cv::BFMatcher matcher(cv::NORM_HAMMING);

  // Matching descriptors
  std::vector<std::vector<cv::DMatch> > matches12;
  matcher.knnMatch(query, train, matches12, 2);

  // Filtering the resulting matchings according to the given ratio
  for (unsigned m = 0; m < matches12.size(); m++) {
    if (matches12[m][0].distance <= matches12[m][1].distance * nndr_bf_) {
      matches->push_back(matches12[m][0]);
    }
  }
}

void LCDetector::convertPoints(const std::vector<cv::KeyPoint>& query_kps,
                               const std::vector<cv::KeyPoint>& train_kps,
                               const std::vector<cv::DMatch>& matches,
                               std::vector<cv::Point2f>* query,
                               std::vector<cv::Point2f>* train) {
  query->clear();
  train->clear();
  for (auto it = matches.begin(); it != matches.end(); it++) {
    // Get the position of query keypoints
    float x = query_kps[it->queryIdx].pt.x;
    float y = query_kps[it->queryIdx].pt.y;
    query->push_back(cv::Point2f(x, y));

    // Get the position of train keypoints
    x = train_kps[it->trainIdx].pt.x;
    y = train_kps[it->trainIdx].pt.y;
    train->push_back(cv::Point2f(x, y));
  }
}

void LCDetector::save(const std::string& filename) const
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error("Could not open file for writing: " + filename);

    boost::archive::binary_oarchive oa(ofs);
    oa << *this;
    std::cout << "ibow_lcd::LCDetector saved to " << filename;
    if(index_) std::cout << "\t num pushed images: " << num_pushed_images << " (Index with: " << index_->numImages() << " images, " << index_->numDescriptors() << " descriptors)" << std::endl;

}

void LCDetector::load(const std::string& filename)
{
if (!boost::filesystem::exists(filename))
        throw std::runtime_error("File does not exist: " + filename);

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open file for reading: " + filename);

    boost::archive::binary_iarchive ia(ifs);
    ia >> *this;

    // Reset variables
    last_lc_result_.status = LC_NOT_DETECTED;
    consecutive_loops_ = 0;

    std::cout << "ibow_lcd::LCDetector loaded from " << filename;
    if(index_) std::cout << "\t num pushed images: " << num_pushed_images << " (Index with: " << index_->numImages() << " images, " << index_->numDescriptors() << " descriptors)" << std::endl;    

}


void LCDetector::printStatus() const
{
  std::cout << *this << std::endl;
}

std::ostream& operator<<(std::ostream &os, const LCDetector &db)
{
  os << "ibow_lcd::LCDetector: num pushed images: " << db.numPushedImages();
  if(db.index_) os << ", Index with: " << db.index_->numImages() << " images, " << db.index_->numDescriptors() << " descriptors";
  else os << "No index";
  return os;
}

}  // namespace ibow_lcd
