/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2014 Steven Lovegrove
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

#include <condition_variable>
#include <list>
#include <mutex>
#include <mutex>
#include <thread>

namespace pangolin
{

template<typename BufPType>
class FixSizeBuffersQueue
{

public:
    FixSizeBuffersQueue() {}

    ~FixSizeBuffersQueue() {
    }

    BufPType getNewest() {
        std::lock_guard<std::mutex> vlock(vMtx);
        std::lock_guard<std::mutex> elock(eMtx);
        if(validBuffers.size() == 0) {
            // Empty queue.
            return 0;
        } else {
            // Requeue all but newest buffers.
            while(validBuffers.size() > 1) {
                emptyBuffers.push_back(std::move(validBuffers.front()));
                validBuffers.pop_front();
            }
            // Return newest buffer.
            BufPType bp = std::move(validBuffers.front());
            validBuffers.pop_front();
            return bp;
        }
    }

    BufPType getNext() {
        std::lock_guard<std::mutex> vlock(vMtx);
        if(validBuffers.size() == 0) {
            // Empty queue.
            return 0;
        } else {
            // Return oldest buffer.
            BufPType bp = std::move(validBuffers.front());
            validBuffers.pop_front();
            return bp;
        }
    }

    BufPType getFreeBuffer() {
        std::lock_guard<std::mutex> vlock(vMtx);
        std::lock_guard<std::mutex> elock(eMtx);
        if(emptyBuffers.size() > 0) {
            // Simply get a free buffer from the free buffers list.
            BufPType bp = std::move(emptyBuffers.front());
            emptyBuffers.pop_front();
            return bp;
        } else {
            if(validBuffers.size() == 0) {
                // Queue not yet initialized.
                throw std::runtime_error("Queue not yet initialised.");
            } else {
                std::cerr << "Out of free buffers." << std::endl;
                // No free buffers return oldest among the valid buffers.
                BufPType bp = std::move(validBuffers.front());
                validBuffers.pop_front();
                return bp;
            }
        }
    }

    void addValidBuffer(BufPType bp) {
        // Add buffer to valid buffers queue.
        std::lock_guard<std::mutex> vlock(vMtx);
        validBuffers.push_back(std::move(bp));
    }

    void returnOrAddUsedBuffer(BufPType bp) {
        // Add buffer back to empty buffers queue.
        std::lock_guard<std::mutex> elock(eMtx);
        emptyBuffers.push_back(std::move(bp));
    }

    size_t AvailableFrames() const {
        std::lock_guard<std::mutex> vlock(vMtx);
        return validBuffers.size();
    }

    size_t EmptyBuffers() const {
        std::lock_guard<std::mutex> elock(eMtx);
        return emptyBuffers.size();
    }

    bool DropNFrames(size_t n) {
        std::lock_guard<std::mutex> vlock(vMtx);
        if(validBuffers.size() < n) {
            return false;
        } else {
            std::lock_guard<std::mutex> elock(eMtx);
            // Requeue all but newest buffers.
            for(unsigned int i=0; i<n; ++i) {
                emptyBuffers.push_back(std::move(validBuffers.front()));
                validBuffers.pop_front();
            }
            return true;
        }
    }

//    unsigned int BufferSizeBytes(){
//        return bufferSizeBytes;
//    }

private:
    std::list<BufPType> validBuffers;
    std::list<BufPType> emptyBuffers;
    mutable std::mutex vMtx;
    mutable std::mutex eMtx;
//    unsigned int maxNumBuffers;
//    unsigned int bufferSizeBytes;
};

}
