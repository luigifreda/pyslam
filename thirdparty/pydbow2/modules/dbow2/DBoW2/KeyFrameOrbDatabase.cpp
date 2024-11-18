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

#include "KeyFrameOrbDatabase.h"

#include "BowVector.h"

#include <filesystem>
#include<mutex>

using namespace std;

namespace
{
    struct QueryData
    {
        //long unsigned int mnQueryId;
        int mnWords = 0;
        float mScore = 0;
    };
}


namespace DBoW2
{

KeyFrameOrbDatabase::KeyFrameOrbDatabase(): mpVoc(nullptr) {}

KeyFrameOrbDatabase::KeyFrameOrbDatabase(ORBVocabulary &voc):
    mpVoc(&voc)
{
    mvInvertedFile.resize(voc.size());
}

void KeyFrameOrbDatabase::setVocabulary(ORBVocabulary &voc)
{
    mpVoc = &voc;
    mvInvertedFile.clear();
    mvInvertedFile.resize(voc.size());    
}


void KeyFrameOrbDatabase::add(const FrameId& id, const BowVector& bowVector)
{
    unique_lock<mutex> lock(mMutex);

    for(DBoW2::BowVector::const_iterator vit=bowVector.begin(), vend=bowVector.end(); vit!=vend; vit++)
        mvInvertedFile[vit->first].push_back(id);
    mmapFrameIdToBowVector[id]=bowVector;

    mNumEntries++;
}

void KeyFrameOrbDatabase::erase(const FrameId& id, const BowVector& bowVector)
{
    unique_lock<mutex> lock(mMutex);

    // Erase elements in the Inverse File for the entry
    for(DBoW2::BowVector::const_iterator vit=bowVector.begin(), vend=bowVector.end(); vit!=vend; vit++)
    {
        // List of keyframes that share the word
        auto &lFrameIds = mvInvertedFile[vit->first];

        for(auto lit=lFrameIds.begin(), lend=lFrameIds.end(); lit!=lend;)
        {
            if(id==*lit)
            {
                lit = lFrameIds.erase(lit);
                break;
            }
            else 
            {
                lit++;
            }
        }
    }
    mmapFrameIdToBowVector.erase(id);

    mNumEntries--;
}

void KeyFrameOrbDatabase::clear()
{
    unique_lock<mutex> lock(mMutex);    
    mvInvertedFile.clear();
    mvInvertedFile.resize(mpVoc->size());

    mmapFrameIdToBowVector.clear();

    mNumEntries = 0;
}


void KeyFrameOrbDatabase::save(const std::string& filename) 
{
    unique_lock<mutex> lock(mMutex); 

    std::cout << "Saving KeyFrameOrbDatabase to " << filename << " (" << size() << " entries)" << std::endl;

    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs) throw std::runtime_error("Could not open file for writing: " + filename);

    boost::archive::binary_oarchive oa(ofs);
    oa << *this;
    std::cout << "KeyFrameOrbDatabase saved to " << filename << " (" << size() << " entries)" << std::endl;
}

void KeyFrameOrbDatabase::load(const std::string& filename)
{
    unique_lock<mutex> lock(mMutex); 

    std::cout << "Loading KeyFrameOrbDatabase from " << filename << std::endl;

    if (!boost::filesystem::exists(filename))
        throw std::runtime_error("File does not exist: " + filename);

    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs) throw std::runtime_error("Could not open file for reading: " + filename);

    boost::archive::binary_iarchive ia(ifs);
    ia >> *this;
    std::cout << "KeyFrameOrbDatabase loaded from " << filename << " (" << size() << " entries)" << std::endl;    
    
}

QueryResult KeyFrameOrbDatabase::detectLoopCandidates(const FrameId& id, const BowVector& bowVector,
                                                       const set<FrameId>& spConnectedKeyFrames,
                                                       const float minScore)
{
    QueryResult result;
    
    list<FrameId> lKFsSharingWords;
    std::map<FrameId, QueryData> queryData; // we expect a few entries here so we can use std::map


    // std::cout << "[KeyFrameOrbDatabase::detectLoopCandidates] id: " << id << ", connected keyframes: "; 
    // for(auto id: spConnectedKeyFrames)
    //     std::cout << id << ",";
    // std::cout << std::endl;

    // Search all registered frames that share a word with current query frame
    // Discard frames connected to the query keyframe
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=bowVector.begin(), vend=bowVector.end(); vit != vend; vit++)
        {
            auto &lFrameIds = mvInvertedFile[vit->first];

            for(auto lit=lFrameIds.begin(), lend=lFrameIds.end(); lit!=lend; lit++)
            {
                const FrameId fid=*lit;
                // if(pKFi->mnLoopQuery!=id)
                // {
                //     pKFi->mnLoopWords=0;
                //     if(!spConnectedKeyFrames.count(fid))
                //     {
                //         pKFi->mnLoopQuery=id;
                //         lKFsSharingWords.push_back(fid);
                //     }
                // }
                // pKFi->mnLoopWords++;
                auto it = queryData.find(fid); 
                if(it == queryData.end())
                {
                    if(!spConnectedKeyFrames.count(fid) && fid!=id)
                    {
                        QueryData data{1,0};
                        queryData.insert({fid, data});
                        lKFsSharingWords.push_back(fid);
                    }
                }
                else
                {
                    it->second.mnWords++;
                }
            }
        }
    }

    if(lKFsSharingWords.empty())
        return result;

    //list<pair<float,FrameId> > lScoreAndMatch;
    auto& lScoreAndMatch = result;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(auto lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        // if((*lit)->mnLoopWords>maxCommonWords)
        //     maxCommonWords=(*lit)->mnLoopWords;
        const FrameId fid = *lit;
        const int fidNumWords = queryData[fid].mnWords;
        if(fidNumWords>maxCommonWords)
            maxCommonWords=fidNumWords;
    }

    const int minCommonWords = maxCommonWords*0.8f;

    int nscores=0;

    // Compute similarity score. Retain the matches whose score is higher than minScore
    for(auto lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        const FrameId fid = *lit;
        auto& fidQueryData = queryData[fid];
        if(fidQueryData.mnWords>minCommonWords)
        {
            nscores++;

            const auto& fidBowVector = mmapFrameIdToBowVector[fid];
            const float si = mpVoc->score(bowVector,fidBowVector);
            //std::cout << "[KeyFrameOrbDatabase::detectLoopCandidates] - score ( " << id << " , " << fid << " ): " << si << std::endl;
            fidQueryData.mScore = si;
            if(si>=minScore)
                lScoreAndMatch.push_back(make_pair(si,fid));
        }
    }

    return result;

#if 0
    list<pair<float,FrameId> > lAccScoreAndMatch;
    float bestAccScore = minScore;

    // Lets now accumulate score by covisibility
    for(list<pair<float,FrameId> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        FrameId pKFi = it->second;
        vector<FrameId> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = it->first;
        FrameId pBestKF = pKFi;
        for(vector<FrameId>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            FrameId pKF2 = *vit;
            if(pKF2->mnLoopQuery==pKF->mnId && pKF2->mnLoopWords>minCommonWords)
            {
                accScore+=pKF2->mLoopScore;
                if(pKF2->mLoopScore>bestScore)
                {
                    pBestKF=pKF2;
                    bestScore = pKF2->mLoopScore;
                }
            }
        }

        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;

    set<FrameId> spAlreadyAddedKF;
    vector<FrameId> vpLoopCandidates;
    vpLoopCandidates.reserve(lAccScoreAndMatch.size());

    for(list<pair<float,FrameId> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        if(it->first>minScoreToRetain)
        {
            FrameId pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpLoopCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpLoopCandidates;
#endif 
}

QueryResult KeyFrameOrbDatabase::detectRelocalizationCandidates(const FrameId& id, const BowVector& bowVector)
{
    QueryResult result;
    
    list<FrameId> lKFsSharingWords;
    std::map<FrameId, QueryData> queryData; // we expect a few entries here so we can use std::map


    // Search all keyframes that share a word with current frame
    {
        unique_lock<mutex> lock(mMutex);

        for(DBoW2::BowVector::const_iterator vit=bowVector.begin(), vend=bowVector.end(); vit != vend; vit++)
        {
            auto &lFrameIds = mvInvertedFile[vit->first];

            for(auto lit=lFrameIds.begin(), lend=lFrameIds.end(); lit!=lend; lit++)
            {
                FrameId fid=*lit;
                // if(pKFi->mnRelocQuery!=F->mnId)
                // {
                //     pKFi->mnRelocWords=0;
                //     pKFi->mnRelocQuery=F->mnId;
                //     lKFsSharingWords.push_back(pKFi);
                // }
                // pKFi->mnRelocWords++;
                auto it = queryData.find(fid); 
                if(it == queryData.end())
                {
                    if(fid!=id)
                    {                    
                        QueryData data{1,0};
                        queryData.insert({fid, data});
                        lKFsSharingWords.push_back(fid);
                    }
                }
                else
                {
                    it->second.mnWords++;
                }
            }
        }
    }
    if(lKFsSharingWords.empty())
        return result;

    // Only compare against those keyframes that share enough words
    int maxCommonWords=0;
    for(auto lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        // if((*lit)->mnRelocWords>maxCommonWords)
        //     maxCommonWords=(*lit)->mnRelocWords;
        const FrameId fid = *lit;
        const int fidNumWords = queryData[fid].mnWords;
        if(fidNumWords>maxCommonWords)
            maxCommonWords=fidNumWords;
    }

    const int minCommonWords = maxCommonWords*0.8f;

    //list<pair<float,FrameId> > lScoreAndMatch;
    auto& lScoreAndMatch = result;    

    int nscores=0;

    // Compute similarity score.
    for(auto lit=lKFsSharingWords.begin(), lend= lKFsSharingWords.end(); lit!=lend; lit++)
    {
        const FrameId fid = *lit;
        auto& fidQueryData = queryData[fid];
        if(fidQueryData.mnWords>minCommonWords)
        {
            nscores++;

            const auto& fidBowVector = mmapFrameIdToBowVector[fid];            
            const float si = mpVoc->score(bowVector,fidBowVector);
            fidQueryData.mScore = si;
            lScoreAndMatch.push_back(make_pair(si,fid));
        }
    }

    return result;

#if 0
    list<pair<float,FrameId> > lAccScoreAndMatch;
    float bestAccScore = 0;

    // Lets now accumulate score by covisibility
    for(list<pair<float,FrameId> >::iterator it=lScoreAndMatch.begin(), itend=lScoreAndMatch.end(); it!=itend; it++)
    {
        FrameId pKFi = it->second;
        vector<FrameId> vpNeighs = pKFi->GetBestCovisibilityKeyFrames(10);

        float bestScore = it->first;
        float accScore = bestScore;
        FrameId pBestKF = pKFi;
        for(vector<FrameId>::iterator vit=vpNeighs.begin(), vend=vpNeighs.end(); vit!=vend; vit++)
        {
            FrameId pKF2 = *vit;
            if(pKF2->mnRelocQuery!=F->mnId)
                continue;

            accScore+=pKF2->mRelocScore;
            if(pKF2->mRelocScore>bestScore)
            {
                pBestKF=pKF2;
                bestScore = pKF2->mRelocScore;
            }

        }
        lAccScoreAndMatch.push_back(make_pair(accScore,pBestKF));
        if(accScore>bestAccScore)
            bestAccScore=accScore;
    }

    // Return all those keyframes with a score higher than 0.75*bestScore
    float minScoreToRetain = 0.75f*bestAccScore;
    set<FrameId> spAlreadyAddedKF;
    vector<FrameId> vpRelocCandidates;
    vpRelocCandidates.reserve(lAccScoreAndMatch.size());
    for(list<pair<float,FrameId> >::iterator it=lAccScoreAndMatch.begin(), itend=lAccScoreAndMatch.end(); it!=itend; it++)
    {
        const float &si = it->first;
        if(si>minScoreToRetain)
        {
            FrameId pKFi = it->second;
            if(!spAlreadyAddedKF.count(pKFi))
            {
                vpRelocCandidates.push_back(pKFi);
                spAlreadyAddedKF.insert(pKFi);
            }
        }
    }

    return vpRelocCandidates;
#endif     
}

void KeyFrameOrbDatabase::printStatus() const
{
  std::cout << *this << std::endl;
}

std::ostream& operator<<(std::ostream &os, const KeyFrameOrbDatabase &db)
{
  os << "KeyFrameOrbDatabase: #Entries = " << db.size();
  return os;
}


} //namespace 
