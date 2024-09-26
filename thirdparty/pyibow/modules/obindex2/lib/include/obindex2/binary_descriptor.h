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

#ifndef LIB_INCLUDE_OBINDEX2_BINARY_DESCRIPTOR_H_
#define LIB_INCLUDE_OBINDEX2_BINARY_DESCRIPTOR_H_

#include <bitset>
#include <memory>
#include <string>
#include <sstream>
#include <unordered_set>

#include <boost/dynamic_bitset.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/core/hal/hal.hpp>

namespace obindex2 {

class BinaryDescriptor {
 public:
  // Constructors
  explicit BinaryDescriptor(const unsigned nbits = 256);
  explicit BinaryDescriptor(const unsigned char* bits, unsigned nbytes);
  explicit BinaryDescriptor(const cv::Mat& desc);
  explicit BinaryDescriptor(const BinaryDescriptor& bd);
  // Destructor
  virtual ~BinaryDescriptor();

  // Methods
  inline void set(int nbit) {
    // Detecting the correct byte
    int nbyte = nbit / 8;
    int nb = 7 - (nbit % 8);

    // Setting the bit
    bits_[nbyte] |= 1 << nb;
  }

  inline void reset(int nbit) {
    // Detecting the correct byte
    int nbyte = nbit / 8;
    int nb = 7 - (nbit % 8);

    // Setting the bit
    bits_[nbyte] &= ~(1 << nb);
  }

  inline int size() {
    return static_cast<int>(size_in_bits_);
  }

  inline static double distHamming(const BinaryDescriptor& a,
                                   const BinaryDescriptor& b) {
    int hamming = cv::hal::normHamming(a.bits_, b.bits_, a.size_in_bytes_);
    return static_cast<double>(hamming);
  }

  // Operator overloading
  inline bool operator==(const BinaryDescriptor& d) {
    int hamming = cv::hal::normHamming(bits_, d.bits_, size_in_bytes_);
    return hamming == 0;
  }

  inline bool operator!=(const BinaryDescriptor& d) {
    int hamming = cv::hal::normHamming(bits_, d.bits_, size_in_bytes_);
    return hamming != 0;
  }

  inline BinaryDescriptor& operator=(const BinaryDescriptor& other) {
    // Clearing previous memory
    if (bits_ != nullptr) {
      delete [] bits_;
    }

    // Allocating new memory
    size_in_bits_ = other.size_in_bits_;
    size_in_bytes_ = other.size_in_bytes_;
    bits_ = new unsigned char[size_in_bytes_];
    memcpy(bits_, other.bits_, sizeof(unsigned char) * size_in_bytes_);

    return *this;
  }

  inline BinaryDescriptor& operator&=(const BinaryDescriptor& other) {
    unsigned size = other.size_in_bytes_;
    for (unsigned i = 0; i < size; i++) {
      bits_[i] = bits_[i] & other.bits_[i];
    }

    return *this;
  }

  inline BinaryDescriptor& operator|=(const BinaryDescriptor& other) {
    unsigned size = other.size_in_bytes_;
    for (unsigned i = 0; i < size; i++) {
      bits_[i] = bits_[i] | other.bits_[i];
    }

    return *this;
  }

  cv::Mat toCvMat();
  std::string toString();

  // For simplicity, we made it public, but you should use the public methods
  unsigned char* bits_;
  unsigned size_in_bytes_;
  unsigned size_in_bits_;
};

typedef std::shared_ptr<BinaryDescriptor> BinaryDescriptorPtr;
typedef std::unordered_set<BinaryDescriptorPtr> BinaryDescriptorSet;
typedef std::shared_ptr<BinaryDescriptorSet> BinaryDescriptorSetPtr;

}  // namespace obindex2

#endif  // LIB_INCLUDE_OBINDEX2_BINARY_DESCRIPTOR_H_
