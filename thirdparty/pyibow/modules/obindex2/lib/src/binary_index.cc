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

#include "obindex2/binary_index.h"

namespace obindex2 {

ImageIndex::ImageIndex(const unsigned k,
                       const unsigned s,
                       const unsigned t,
                       const MergePolicy merge_policy,
                       const bool purge_descriptors,
                       const unsigned min_feat_apps) :
    k_(k),
    s_(s),
    t_(t),
    init_(false),
    nimages_(0),
    ndesc_(0),
    merge_policy_(merge_policy),
    purge_descriptors_(purge_descriptors),
    min_feat_apps_(min_feat_apps) {
      // Validating the corresponding parameters
      assert(k_ > 1);
      assert(k_ < s_);
      assert(min_feat_apps > 0);
}

void ImageIndex::addImage(const unsigned image_id,
                          const std::vector<cv::KeyPoint>& kps,
                          const cv::Mat& descs) {
  // Creating the set of BinaryDescriptors
  for (int i = 0; i < descs.rows; i++) {
    // Creating the corresponding descriptor
    cv::Mat desc = descs.row(i);
    BinaryDescriptorPtr d = std::make_shared<BinaryDescriptor>(desc);
    insertDescriptor(d);

    // Creating the inverted index item
    InvIndexItem item;
    item.image_id = image_id;
    item.pt = kps[i].pt;
    item.dist = 0.0;
    item.kp_ind = i;
    inv_index_[d].push_back(item);
  }

  // If the trees are not initialized, we build them
  if (!init_) {
    assert(static_cast<int>(k_) < descs.rows);
    initTrees();
    init_ = true;
  }

  // Deleting unstable features
  if (purge_descriptors_) {
    purgeDescriptors(image_id);
  }

  nimages_++;
}

void ImageIndex::addImage(const unsigned image_id,
                const std::vector<cv::KeyPoint>& kps,
                const cv::Mat& descs,
                const std::vector<cv::DMatch>& matches) {
  // --- Adding new features
  // All features
  std::set<int> points;
  for (unsigned feat_ind = 0; feat_ind < kps.size(); feat_ind++) {
    points.insert(feat_ind);
  }

  // Matched features
  std::set<int> matched_points;
  for (unsigned match_ind = 0; match_ind < matches.size(); match_ind++) {
    matched_points.insert(matches[match_ind].queryIdx);
  }

  // Computing the difference
  std::set<int> diff;
  std::set_difference(points.begin(), points.end(),
                      matched_points.begin(), matched_points.end(),
                      std::inserter(diff, diff.end()));

  // Inserting new features into the index.
  for (auto it = diff.begin(); it != diff.end(); it++) {
    int index = *it;
    cv::Mat desc = descs.row(index);
    BinaryDescriptorPtr d = std::make_shared<BinaryDescriptor>(desc);
    insertDescriptor(d);

    // Creating the inverted index item
    InvIndexItem item;
    item.image_id = image_id;
    item.pt = kps[index].pt;
    item.dist = 0.0;
    item.kp_ind = index;
    inv_index_[d].push_back(item);
  }

  // --- Updating the matched descriptors into the index
  for (unsigned match_ind = 0; match_ind < matches.size(); match_ind++) {
    int qindex = matches[match_ind].queryIdx;
    int tindex = matches[match_ind].trainIdx;

    BinaryDescriptorPtr q_d = std::make_shared<BinaryDescriptor>
                                                            (descs.row(qindex));
    BinaryDescriptorPtr t_d = id_to_desc_[tindex];

    // Merge and replace according to the merging policy
    if (merge_policy_ == MERGE_POLICY_AND) {
      *t_d &= *q_d;
    } else if (merge_policy_ == MERGE_POLICY_OR) {
      *t_d |= *q_d;
    }

    // Creating the inverted index item
    InvIndexItem item;
    item.image_id = image_id;
    item.pt = kps[qindex].pt;
    item.dist = matches[match_ind].distance;
    item.kp_ind = qindex;
    inv_index_[t_d].push_back(item);
  }

  // Deleting unstable features
  if (purge_descriptors_) {
    purgeDescriptors(image_id);
  }

  nimages_++;
}

void ImageIndex::searchImages(const cv::Mat& descs,
                              const std::vector<cv::DMatch>& gmatches,
                              std::vector<ImageMatch>* img_matches,
                              bool sort) {
  // Initializing the resulting structure
  img_matches->resize(nimages_);
  for (unsigned i = 0; i < nimages_; i++) {
    img_matches->at(i).image_id = i;
  }

  // Counting the number of each word in the current document
  std::unordered_map<int, int> nwi_map;
  for (unsigned match_index = 0; match_index < gmatches.size(); match_index++) {
    int train_idx = gmatches[match_index].trainIdx;
    // Updating nwi_map, number of occurrences of a word in an image.
    if (nwi_map.count(train_idx)) {
      nwi_map[train_idx]++;
    } else {
      nwi_map[train_idx] = 1;
    }
  }

  // We process all the matchings again to increase the scores
  for (unsigned match_index = 0; match_index < gmatches.size(); match_index++) {
    int train_idx = gmatches[match_index].trainIdx;
    BinaryDescriptorPtr desc = id_to_desc_[train_idx];

    // Computing the TF term
    double tf = static_cast<double>(nwi_map[train_idx]) / descs.rows;

    // Computing the IDF term
    std::unordered_set<unsigned> nw;
    for (unsigned i = 0; i < inv_index_[desc].size(); i++) {
      nw.insert(inv_index_[desc][i].image_id);
    }
    double idf = log(static_cast<double>(nimages_) / nw.size());

    // Computing the final TF-IDF weighting term
    double tfidf = tf * idf;

    for (unsigned i = 0; i < inv_index_[desc].size(); i++) {
        int im = inv_index_[desc][i].image_id;
        img_matches->at(im).score += tfidf;
    }
  }

  if (sort) {
    std::sort(img_matches->begin(), img_matches->end());
  }
}

void ImageIndex::initTrees() {
  // Creating the trees
  BinaryDescriptorSetPtr dset_ptr =
                std::make_shared<BinaryDescriptorSet>(dset_);

  for (unsigned i = 0; i < t_; i++) {
    BinaryTreePtr tree_ptr =
              std::make_shared<BinaryTree>(dset_ptr, i, k_, s_);
    trees_.push_back(tree_ptr);
  }
}

void ImageIndex::searchDescriptors(
                              const cv::Mat& descs,
                              std::vector<std::vector<cv::DMatch> >* matches,
                              const unsigned knn,
                              const unsigned checks) {
  matches->clear();
  for (int i = 0; i < descs.rows; i++) {
    // Creating the corresponding descriptor
    cv::Mat desc = descs.row(i);
    BinaryDescriptorPtr d = std::make_shared<BinaryDescriptor>(desc);

    // Searching the descriptor in the index
    std::vector<BinaryDescriptorPtr> neighs;
    std::vector<double> dists;
    searchDescriptor(d, &neighs, &dists, knn, checks);

    // Translating the resulting matches to CV structures
    std::vector<cv::DMatch> des_match;
    for (unsigned j = 0; j < neighs.size(); j++) {
      cv::DMatch match;
      match.queryIdx = i;
      match.trainIdx = static_cast<int>(desc_to_id_[neighs[j]]);
      match.imgIdx = static_cast<int>(inv_index_[neighs[j]][0].image_id);
      match.distance = dists[j];
      des_match.push_back(match);
    }
    matches->push_back(des_match);
  }
}

void ImageIndex::deleteDescriptor(const unsigned desc_id) {
  BinaryDescriptorPtr d = id_to_desc_[desc_id];
  // Clearing the descriptor
  deleteDescriptor(d);
}

void ImageIndex::searchDescriptor(BinaryDescriptorPtr q,
                                  std::vector<BinaryDescriptorPtr>* neigh,
                                  std::vector<double>* distances,
                                  unsigned knn,
                                  unsigned checks) {
  unsigned points_searched = 0;
  NodePriorityQueue pq;
  DescriptorQueue r;

  // Initializing search structures
  std::vector<NodeQueuePtr> pqs;
  std::vector<DescriptorQueuePtr> rs;
  for (unsigned i = 0; i < trees_.size(); i++) {
    NodeQueuePtr tpq = std::make_shared<NodeQueue>();
    pqs.push_back(tpq);

    DescriptorQueuePtr tr = std::make_shared<DescriptorQueue>();
    rs.push_back(tr);
  }

  // Searching in the trees
  //#pragma omp parallel for
  for (unsigned i = 0; i < trees_.size(); i++) {
    trees_[i]->traverseFromRoot(q, pqs[i], rs[i]);
  }

  //  Gathering results from each individual search
  std::unordered_set<BinaryDescriptorPtr> already_added;
  for (unsigned i = 0; i < trees_.size(); i++) {
    // Obtaining descriptor nodes
    unsigned r_size = rs[i]->size();
    for (unsigned j = 0; j < r_size; j++) {
      DescriptorQueueItem r_item = rs[i]->get(j);
      std::pair<std::unordered_set<BinaryDescriptorPtr>::iterator,
                bool > result;
      result = already_added.insert(r_item.desc);
      if (result.second) {
        r.push(r_item);
        points_searched++;
      }
    }
  }

  // Continuing the search if not enough descriptors have been checked
  if (points_searched < checks) {
    // Gathering the next nodes to search
    for (unsigned i = 0; i < trees_.size(); i++) {
      // Obtaining priority queue nodes
      unsigned pq_size = pqs[i]->size();
      for (unsigned j = 0; j < pq_size; j++) {
        pq.push(pqs[i]->get(j));
      }
    }

    NodePriorityQueuePtr pq_ptr = std::make_shared<NodePriorityQueue>(pq);
    while (points_searched < checks && !pq.empty()) {
      // Get the closest node to continue the search
      NodeQueueItem n = pq.top();
      pq.pop();

      // Searching in the node
      NodeQueuePtr tpq = std::make_shared<NodeQueue>();
      DescriptorQueuePtr tr = std::make_shared<DescriptorQueue>();
      trees_[n.tree_id]->traverseFromNode(q, n.node, tpq, tr);

      // Adding new nodes to search to PQ
      for (unsigned i = 0; i < tpq->size(); i++) {
        pq.push(tpq->get(i));
      }

      for (unsigned j = 0; j < tr->size(); j++) {
        DescriptorQueueItem r_item = tr->get(j);
        std::pair<std::unordered_set<BinaryDescriptorPtr>::iterator,
                bool > result;
        result = already_added.insert(r_item.desc);
        if (result.second) {
          r.push(r_item);
          points_searched++;
        }
      }
    }
  }
  r.sort();

  // Returning the required number of descriptors descriptors
  neigh->clear();
  distances->clear();
  unsigned ndescs = std::min(knn, r.size());
  for (unsigned i = 0; i < ndescs; i++) {
    DescriptorQueueItem d = r.get(i);

    neigh->push_back(d.desc);
    distances->push_back(d.dist);
  }
}

void ImageIndex::insertDescriptor(BinaryDescriptorPtr q) {
  dset_.insert(q);
  desc_to_id_[q] = ndesc_;
  id_to_desc_[ndesc_] = q;
  ndesc_++;
  recently_added_.push_back(q);

  // Indexing the descriptor inside each tree
  if (init_) {
    //#pragma omp parallel for
    for (unsigned i = 0; i < trees_.size(); i++) {
      trees_[i]->addDescriptor(q);
    }
  }
}

void ImageIndex::deleteDescriptor(BinaryDescriptorPtr q) {
  // Deleting the descriptor from each tree
  if (init_) {
    //#pragma omp parallel for
    for (unsigned i = 0; i < trees_.size(); i++) {
      trees_[i]->deleteDescriptor(q);
    }
  }

  dset_.erase(q);
  unsigned desc_id = desc_to_id_[q];
  desc_to_id_.erase(q);
  id_to_desc_.erase(desc_id);
  inv_index_.erase(q);
}

void ImageIndex::getMatchings(
      const std::vector<cv::KeyPoint>& query_kps,
      const std::vector<cv::DMatch>& matches,
      std::unordered_map<unsigned, PointMatches>* point_matches) {
  for (unsigned i = 0; i < matches.size(); i++) {
    // Getting the query point
    int qid = matches[i].queryIdx;
    cv::Point2f qpoint = query_kps[qid].pt;

    // Processing the train points
    int tid = matches[i].trainIdx;
    BinaryDescriptorPtr desc_ptr = id_to_desc_[static_cast<unsigned>(tid)];
    for (unsigned j = 0; j < inv_index_[desc_ptr].size(); j++) {
      InvIndexItem item = inv_index_[desc_ptr][j];
      unsigned im_id = item.image_id;
      cv::Point2f tpoint = item.pt;

      (*point_matches)[im_id].query.push_back(qpoint);
      (*point_matches)[im_id].train.push_back(tpoint);
    }
  }
}

void ImageIndex::purgeDescriptors(const unsigned curr_img) {
  auto it = recently_added_.begin();

  while (it != recently_added_.end()) {
    BinaryDescriptorPtr desc = *it;
    // We assess if at least three images have passed since creation
    if ((curr_img - inv_index_[desc][0].image_id) > 1) {
      // If so, we assess if the feature has been seen at least twice
      if (inv_index_[desc].size() < min_feat_apps_) {
        deleteDescriptor(desc);
      }

      it = recently_added_.erase(it);
    } else {
      // This descriptor should be maintained in the list
      it++;
    }
  }
}

}  // namespace obindex2
