/** 
* This file is part of PYSLAM 
*
* Copyright (C) 2016-present Luigi Freda <luigi dot freda at gmail dot com> 
*
* PYSLAM is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* PYSLAM is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with PYSLAM. If not, see <http://www.gnu.org/licenses/>.
*/
/**
* This file is part of ORB-SLAM2.
*
* Copyright (C) 2014-2016 Raúl Mur-Artal <raulmur at unizar dot es> (University of Zaragoza)
* For more information see <https://github.com/raulmur/ORB_SLAM2>
*
* ORB-SLAM2 is free software: you can redistribute it and/or modify
* it under the terms of the GNU General Public License as published by
* the Free Software Foundation, either version 3 of the License, or
* (at your option) any later version.
*
* ORB-SLAM2 is distributed in the hope that it will be useful,
* but WITHOUT ANY WARRANTY; without even the implied warranty of
* MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
* GNU General Public License for more details.
*
* You should have received a copy of the GNU General Public License
* along with ORB-SLAM2. If not, see <http://www.gnu.org/licenses/>.
*/

#ifndef DATABASE_H
#define DATABASE_H

#include <vector>
#include <list>
#include <set>
#include <unordered_map>
#include <map>

#include <mutex>

#include "TemplatedVocabulary.h"
#include "FORB.h"


namespace DBoW2
{

typedef long unsigned int FrameId;
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
typedef std::list<pair<float,FrameId>> QueryResult;


class KeyFrameOrbDatabase
{
public:

    KeyFrameOrbDatabase(const ORBVocabulary &voc);

   void add(const FrameId& id, const BowVector& bowVector);

   void erase(const FrameId& id, const BowVector& bowVector);

   void clear();

   // Loop Detection
   QueryResult detectLoopCandidates(const FrameId& id, const BowVector& bowVector,
                                    const set<FrameId>& spConnectedKeyFrames,
                                    const float minScore);

   // Relocalization
   QueryResult detectRelocalizationCandidates(const FrameId& id, const BowVector& bowVector);

protected:

  // Associated vocabulary
  const ORBVocabulary* mpVoc;

  // Inverted file
  std::vector<list<FrameId> > mvInvertedFile;
  std::unordered_map<FrameId, BowVector> mmapFrameIdToBowVector;

  // Mutex
  std::mutex mMutex;
};

} //namespace DBoW2

#endif
