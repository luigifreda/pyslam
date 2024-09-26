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

#include "obindex2/binary_tree.h"

namespace obindex2 {

  BinaryTreeNode::BinaryTreeNode() :
    is_leaf_(false),
    is_bad_(false),
    desc_(nullptr),
    root_(nullptr) {
  }

  BinaryTreeNode::BinaryTreeNode(const bool leaf,
                                 BinaryDescriptorPtr desc,
                                 BinaryTreeNodePtr root) :
    is_leaf_(leaf),
    is_bad_(false),
    desc_(desc),
    root_(root) {
  }

}  // namespace obindex2
