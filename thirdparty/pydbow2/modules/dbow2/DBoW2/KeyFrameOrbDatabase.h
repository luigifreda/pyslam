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

#ifndef DATABASE_H
#define DATABASE_H

#include <vector>
#include <list>
#include <set>
#include <unordered_map>
#include <map>

#include <mutex>

#include "TemplatedVocabulary.h"
#include "TemplatedDatabase.h"
#include "FORB.h"



#include <boost/archive/binary_oarchive.hpp>
#include <boost/archive/binary_iarchive.hpp>
#include <boost/serialization/vector.hpp>
#include <boost/serialization/map.hpp>
#include <boost/serialization/unordered_map.hpp>
#include <boost/serialization/list.hpp>  
#include <boost/serialization/utility.hpp>
#include <boost/filesystem.hpp>

namespace DBoW2
{

typedef long unsigned int FrameId;
typedef DBoW2::TemplatedVocabulary<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBVocabulary;
typedef std::list<pair<float,FrameId>> QueryResult;

typedef DBoW2::TemplatedDatabase<DBoW2::FORB::TDescriptor, DBoW2::FORB> ORBDatabase;

class KeyFrameOrbDatabase
{
public:

    KeyFrameOrbDatabase();
    KeyFrameOrbDatabase(ORBVocabulary &voc);

    void setVocabulary(ORBVocabulary &voc);

    void add(const FrameId& id, const BowVector& bowVector);

    void erase(const FrameId& id, const BowVector& bowVector);

    void clear();

    size_t size() const { return mNumEntries; }

    void save(const std::string& filename);
    void load(const std::string& filename);

    // Loop Detection
    QueryResult detectLoopCandidates(const FrameId& id, const BowVector& bowVector,
                                    const set<FrameId>& spConnectedKeyFrames,
                                    const float minScore);

    // Relocalization
    QueryResult detectRelocalizationCandidates(const FrameId& id, const BowVector& bowVector);


    void printStatus() const;

    friend std::ostream& operator<<(std::ostream &os,const KeyFrameOrbDatabase &db);

protected: 

    friend class boost::serialization::access;

    template<typename Archive>
    void serialize(Archive & ar, const unsigned int version) {
      ar & mvInvertedFile;
      ar & mmapFrameIdToBowVector;
      ar & mNumEntries;
    }

protected:

    // Associated vocabulary
    ORBVocabulary* mpVoc = nullptr;

    // Inverted file
    std::vector<std::list<FrameId> > mvInvertedFile;
    std::unordered_map<FrameId, BowVector> mmapFrameIdToBowVector;

    size_t mNumEntries = 0;

    // Mutex
    std::mutex mMutex;
};

} //namespace DBoW2

#endif
