/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2013 Steven Lovegrove
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

#include <pangolin/pangolin.h>
#include <pangolin/video/video.h>
#include <pangolin/video/iostream_operators.h>

namespace pangolin
{

// Take N streams, and place them into one big buffer.
class PANGOLIN_EXPORT MergeVideo : public VideoInterface, public VideoFilterInterface
{
public:
    MergeVideo(std::unique_ptr<VideoInterface>& src, const std::vector<Point>& stream_pos, size_t w, size_t h);
    ~MergeVideo();
    
    void Start() override;
    
    void Stop() override;

    size_t SizeBytes() const override;

    const std::vector<StreamInfo>& Streams() const override;
    
    bool GrabNext( unsigned char* image, bool wait = true ) override;
    
    bool GrabNewest( unsigned char* image, bool wait = true ) override;

    std::vector<VideoInterface*>& InputStreams() override;
    
protected:
    void CopyBuffer(unsigned char* dst_bytes, unsigned char* src_bytes);

    std::unique_ptr<VideoInterface> src;
    std::vector<VideoInterface*> videoin;
    std::unique_ptr<uint8_t[]> buffer;
    std::vector<Point> stream_pos;

    std::vector<StreamInfo> streams;
    size_t size_bytes;
};

}
