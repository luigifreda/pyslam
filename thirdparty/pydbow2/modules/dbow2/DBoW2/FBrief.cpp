/**
 * File: FBrief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: functions for BRIEF descriptors
 * License: see the LICENSE.txt file
 *
 */
 
#include <vector>
#include <string>
#include <sstream>

#include "FBrief.h"

using namespace std;

namespace DBoW2 {

// --------------------------------------------------------------------------

void FBrief::meanValue(const std::vector<FBrief::pDescriptor> &descriptors, 
  FBrief::TDescriptor &mean)
{
  mean.reset();
  
  if(descriptors.empty()) return;
  
  const int N2 = descriptors.size() / 2;
  
  vector<int> counters(FBrief::L, 0);

  vector<FBrief::pDescriptor>::const_iterator it;
  for(it = descriptors.begin(); it != descriptors.end(); ++it)
  {
    const FBrief::TDescriptor &desc = **it;
    for(int i = 0; i < FBrief::L; ++i)
    {
      if(desc[i]) counters[i]++;
    }
  }
  
  for(int i = 0; i < FBrief::L; ++i)
  {
    if(counters[i] > N2) mean.set(i);
  }
  
}

// --------------------------------------------------------------------------
  
double FBrief::distance(const FBrief::TDescriptor &a, 
  const FBrief::TDescriptor &b)
{
  return (double)(a^b).count();
}

// --------------------------------------------------------------------------
  
std::string FBrief::toString(const FBrief::TDescriptor &a)
{
  return a.to_string(); // reversed
}

// --------------------------------------------------------------------------
  
void FBrief::fromString(FBrief::TDescriptor &a, const std::string &s)
{
  stringstream ss(s);
  ss >> a;
}

// --------------------------------------------------------------------------

void FBrief::toMat32F(const std::vector<TDescriptor> &descriptors, 
  cv::Mat &mat)
{
  if(descriptors.empty())
  {
    mat.release();
    return;
  }
  
  const int N = descriptors.size();
  
  mat.create(N, FBrief::L, CV_32F);
  
  for(int i = 0; i < N; ++i)
  {
    const TDescriptor& desc = descriptors[i];
    float *p = mat.ptr<float>(i);
    for(int j = 0; j < FBrief::L; ++j, ++p)
    {
      *p = (desc[j] ? 1.f : 0.f);
    }
  } 
}

// --------------------------------------------------------------------------

} // namespace DBoW2

