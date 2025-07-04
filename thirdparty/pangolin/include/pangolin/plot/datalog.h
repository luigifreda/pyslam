/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Steven Lovegrove
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#pragma once

#include <pangolin/platform.h>

#include <algorithm> // std::min, std::max
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(HAVE_EIGEN) && !defined(__CUDACC__) //prevent including Eigen in cuda files
#define USE_EIGEN
#endif

#ifdef USE_EIGEN
#include <Eigen/Core>
#endif

namespace pangolin
{

/// Simple statistics recorded for a logged input dimension.
struct DimensionStats
{
    DimensionStats()
    {
        Reset();
    }

    void Reset()
    {
        isMonotonic = true;
        sum = 0.0f;
        sum_sq = 0.0f;
        min = std::numeric_limits<float>::max();
        max = std::numeric_limits<float>::lowest();
    }

    void Add(const float v)
    {
        isMonotonic = isMonotonic && (v >= max);
        sum += v;
        sum_sq += v*v;
        min = std::min(min, v);
        max = std::max(max, v);
    }

    bool isMonotonic;
    float sum;
    float sum_sq;
    float min;
    float max;
};

class DataLogBlock
{
public:
    /// @param dim: dimension of sample
    /// @param max_samples: maximum number of samples this block can hold
    /// @param start_id: index of first sample (from entire dataset) in this buffer
    DataLogBlock(size_t dim, size_t max_samples, size_t start_id)
        : dim(dim), max_samples(max_samples), samples(0),
          start_id(start_id)
    {
        sample_buffer = std::unique_ptr<float[]>(new float[dim*max_samples]);
//        stats = std::unique_ptr<DimensionStats[]>(new DimensionStats[dim]);
    }

    ~DataLogBlock()
    {
    }

    size_t Samples() const
    {
        return samples;
    }

    size_t MaxSamples() const
    {
        return max_samples;
    }

    /// Return how many more samples can fit in this block
    size_t SampleSpaceLeft() const
    {
        return MaxSamples()- Samples();
    }

    bool IsFull() const
    {
        return Samples() >= MaxSamples();
    }

    /// Add data to block
    void AddSamples(size_t num_samples, size_t dimensions, const float* data_dim_major );

    /// Delete all samples
    void ClearLinked()
    {
        samples = 0;
        nextBlock.reset();
    }

    DataLogBlock* NextBlock() const
    {
        return nextBlock.get();
    }

    size_t StartId() const
    {
        return start_id;
    }

    float* DimData(size_t d) const
    {
        return sample_buffer.get() + d;
    }

    size_t Dimensions() const
    {
        return dim;
    }

    const float* Sample(size_t n) const
    {
        const int id = (int)n - (int)start_id;

        if( 0 <= id && id < (int)samples ) {
            return sample_buffer.get() + dim*id;
        }else{
            if(nextBlock) {
                return nextBlock->Sample(n);
            }else{
                throw std::out_of_range("Index out of range.");
            }
        }
    }

protected:
    size_t dim;
    size_t max_samples;
    size_t samples;
    size_t start_id;
    std::unique_ptr<float[]> sample_buffer;
//    std::unique_ptr<DimensionStats[]> stats;
    std::unique_ptr<DataLogBlock> nextBlock;
};

/// A DataLog can efficiently record floating point sample data of any size.
/// Memory is allocated in blocks is transparent to the user.
class PANGOLIN_EXPORT DataLog
{
public:
    /// @param block_samples_alloc number of samples each memory block can hold.
    DataLog(unsigned int block_samples_alloc = 10000 );

    ~DataLog();

    /// Provide textual labels corresponding to each dimension logged.
    /// This information may be used by graphical interfaces to DataLog.
    void SetLabels(const std::vector<std::string> & labels);
    const std::vector<std::string>& Labels() const;

    void Log(size_t dimension, const float * vals, unsigned int samples = 1);
    void Log(float v);
    void Log(float v1, float v2);
    void Log(float v1, float v2, float v3);
    void Log(float v1, float v2, float v3, float v4);
    void Log(float v1, float v2, float v3, float v4, float v5);
    void Log(float v1, float v2, float v3, float v4, float v5, float v6);
    void Log(float v1, float v2, float v3, float v4, float v5, float v6, float v7);
    void Log(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8);
    void Log(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9);
    void Log(float v1, float v2, float v3, float v4, float v5, float v6, float v7, float v8, float v9, float v10);
    void Log(const std::vector<float> & vals);

#ifdef USE_EIGEN
    template<typename Derived>
    void Log(const Eigen::MatrixBase<Derived>& M)
    {
        Log( M.rows() * M.cols(), M.template cast<float>().eval().data() );
    }
#endif

    void Clear();
    void Save(std::string filename);

    // Return first block of stored data
    const DataLogBlock* FirstBlock() const;

    // Return last block of stored data
    const DataLogBlock* LastBlock() const;

    // Return number of samples stored in this DataLog
    size_t Samples() const;

    // Return pointer to stored sample n
    const float* Sample(int n) const;

    // Return stats computed for each dimension if enabled.
    const DimensionStats& Stats(size_t dim) const;

    std::mutex access_mutex;

protected:
    unsigned int block_samples_alloc;
    std::vector<std::string> labels;
    std::unique_ptr<DataLogBlock> block0;
    DataLogBlock* blockn;
    std::vector<DimensionStats> stats;
    bool record_stats;
};

}
