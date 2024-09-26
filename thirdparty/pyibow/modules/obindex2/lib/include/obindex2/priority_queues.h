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

#ifndef LIB_INCLUDE_OBINDEX2_PRIORITY_QUEUES_H_
#define LIB_INCLUDE_OBINDEX2_PRIORITY_QUEUES_H_

#include <algorithm>
#include <queue>
#include <string>
#include <sstream>
#include <vector>

#include "obindex2/binary_descriptor.h"
#include "obindex2/binary_tree_node.h"

namespace obindex2 {

struct NodeQueueItem {
 public:
  inline explicit NodeQueueItem(const double d,
                                const unsigned id,
                                BinaryTreeNodePtr n) :
    dist(d),
    tree_id(id),
    node(n) {}

  double dist;
  unsigned tree_id;
  BinaryTreeNodePtr node;

  inline bool operator<(const NodeQueueItem& item) const {
    return dist < item.dist;
  }
};

class NodeQueue {
 public:
  inline void push(const NodeQueueItem& item) {
    items.push_back(item);
  }

  inline NodeQueueItem get(unsigned index) {
    return items[index];
  }

  inline void sort() {
    std::sort(items.begin(), items.end());
  }

  inline unsigned size() {
    return items.size();
  }

 private:
  std::vector<NodeQueueItem> items;
};

typedef std::shared_ptr<NodeQueue> NodeQueuePtr;

class CompareNodeQueueItem {
 public:
  inline bool operator()(const NodeQueueItem& a, const NodeQueueItem& b) {
    return a.dist > b.dist;
  }
};

typedef std::priority_queue<NodeQueueItem,
                       std::vector<NodeQueueItem>,
                       CompareNodeQueueItem> NodePriorityQueue;
typedef std::shared_ptr<NodePriorityQueue> NodePriorityQueuePtr;

struct DescriptorQueueItem {
 public:
  inline explicit DescriptorQueueItem(const double d, BinaryDescriptorPtr bd) :
    dist(d),
    desc(bd) {}

  double dist;
  BinaryDescriptorPtr desc;

  inline bool operator<(const DescriptorQueueItem& item) const {
    return dist < item.dist;
  }
};

class DescriptorQueue {
 public:
  inline void push(const DescriptorQueueItem& item) {
    items.push_back(item);
  }

  inline DescriptorQueueItem get(unsigned index) {
    return items[index];
  }

  inline void sort() {
    std::sort(items.begin(), items.end());
  }

  inline unsigned size() {
    return items.size();
  }

 private:
  std::vector<DescriptorQueueItem> items;
};

typedef std::shared_ptr<DescriptorQueue> DescriptorQueuePtr;

}  // namespace obindex2

#endif  // LIB_INCLUDE_OBINDEX2_PRIORITY_QUEUES_H_
