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

#ifndef LIB_INCLUDE_OBINDEX2_BINARY_TREE_H_
#define LIB_INCLUDE_OBINDEX2_BINARY_TREE_H_

#include <stdlib.h>
#include <time.h>

#include <limits>
#include <unordered_set>

#include "obindex2/binary_descriptor.h"
#include "obindex2/priority_queues.h"

namespace obindex2 {

class BinaryTree {
 public:
  // Constructors
  explicit BinaryTree(BinaryDescriptorSetPtr dset,
                      const unsigned tree_id = 0,
                      const unsigned k = 16,
                      const unsigned s = 150);
  virtual ~BinaryTree();

  // Methods
  void buildTree();
  void deleteTree();
  unsigned traverseFromRoot(BinaryDescriptorPtr q,
                            NodeQueuePtr pq,
                            DescriptorQueuePtr r);
  void traverseFromNode(BinaryDescriptorPtr q,
                        BinaryTreeNodePtr n,
                        NodeQueuePtr pq,
                        DescriptorQueuePtr r);
  BinaryTreeNodePtr searchFromRoot(BinaryDescriptorPtr q);
  BinaryTreeNodePtr searchFromNode(BinaryDescriptorPtr q,
                                   BinaryTreeNodePtr n);
  void addDescriptor(BinaryDescriptorPtr q);
  void deleteDescriptor(BinaryDescriptorPtr q);
  void printTree();
  inline unsigned numDegradedNodes() {
    return degraded_nodes_;
  }

  inline unsigned numNodes() {
    return nset_.size();
  }

 private:
  BinaryDescriptorSetPtr dset_;
  unsigned tree_id_;
  BinaryTreeNodePtr root_;
  unsigned k_;
  unsigned s_;
  unsigned k_2_;
  NodeSet nset_;
  std::unordered_map<BinaryDescriptorPtr, BinaryTreeNodePtr> desc_to_node_;

  // Tree statistics
  unsigned degraded_nodes_;
  unsigned nvisited_nodes_;

  void buildNode(BinaryDescriptorSet d, BinaryTreeNodePtr root);
  void printNode(BinaryTreeNodePtr n);
  void deleteNodeRecursive(BinaryTreeNodePtr n);
};

typedef std::shared_ptr<BinaryTree> BinaryTreePtr;

}  // namespace obindex2

#endif  // LIB_INCLUDE_OBINDEX2_BINARY_TREE_H_
