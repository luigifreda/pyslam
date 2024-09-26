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

#ifndef LIB_INCLUDE_OBINDEX2_BINARY_TREE_NODE_H_
#define LIB_INCLUDE_OBINDEX2_BINARY_TREE_NODE_H_

#include <vector>
#include <unordered_set>

#include "obindex2/binary_descriptor.h"

namespace obindex2 {

class BinaryTreeNode;
typedef std::shared_ptr<BinaryTreeNode> BinaryTreeNodePtr;

class BinaryTreeNode {
 public:
  // Constructors
  BinaryTreeNode();
  explicit BinaryTreeNode(const bool leaf,
                          BinaryDescriptorPtr desc = nullptr,
                          BinaryTreeNodePtr root = nullptr);

  // Methods
  inline bool isLeaf() {
    return is_leaf_;
  }

  inline void setLeaf(const bool leaf) {
    is_leaf_ = leaf;
  }

  inline bool isBad() {
    return is_bad_;
  }

  inline void setBad(const bool bad) {
    is_bad_ = bad;
  }

  inline BinaryDescriptorPtr getDescriptor() {
    return desc_;
  }

  inline void setDescriptor(BinaryDescriptorPtr desc) {
    desc_ = desc;
  }

  inline BinaryTreeNodePtr getRoot() {
    return root_;
  }

  inline void setRoot(BinaryTreeNodePtr root) {
    root_ = root;
  }

  inline double distance(BinaryDescriptorPtr desc) {
    return obindex2::BinaryDescriptor::distHamming(*desc_, *desc);
  }

  inline void addChildNode(BinaryTreeNodePtr child) {
    ch_nodes_.insert(child);
  }

  inline void deleteChildNode(BinaryTreeNodePtr child) {
    ch_nodes_.erase(child);
  }

  inline std::unordered_set<BinaryTreeNodePtr>* getChildrenNodes() {
    return &ch_nodes_;
  }

  inline unsigned childNodeSize() {
    return ch_nodes_.size();
  }

  inline void addChildDescriptor(BinaryDescriptorPtr child) {
    ch_descs_.insert(child);
  }

  inline void deleteChildDescriptor(BinaryDescriptorPtr child) {
    ch_descs_.erase(child);
  }

  inline std::unordered_set<BinaryDescriptorPtr>* getChildrenDescriptors() {
    return &ch_descs_;
  }

  inline unsigned childDescriptorSize() {
    return ch_descs_.size();
  }

  inline void selectNewCenter() {
    desc_ = *std::next(ch_descs_.begin(), rand() % ch_descs_.size());
  }

 private:
  bool is_leaf_;
  bool is_bad_;
  BinaryDescriptorPtr desc_;
  BinaryTreeNodePtr root_;
  std::unordered_set<BinaryTreeNodePtr> ch_nodes_;
  std::unordered_set<BinaryDescriptorPtr> ch_descs_;
};

typedef std::unordered_set<BinaryTreeNodePtr> NodeSet;

}  // namespace obindex2

#endif  // LIB_INCLUDE_OBINDEX2_BINARY_TREE_NODE_H_
