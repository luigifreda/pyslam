/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011-2013 Steven Lovegrove
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

#include <pangolin/video/video.h>
#include <pangolin/video/video_output.h>

#include <pangolin/video/drivers/pango_video_output.h>

#include <pangolin/utils/file_utils.h>

namespace pangolin
{
VideoOutput::VideoOutput()
{
}

VideoOutput::VideoOutput(const std::string& uri)
{
    Open(uri);
}

VideoOutput::~VideoOutput()
{
}

bool VideoOutput::IsOpen() const
{
    return recorder.get() != nullptr;
}

void VideoOutput::Open(const std::string& str_uri)
{
    Close();
    uri = ParseUri(str_uri);
    recorder = OpenVideoOutput(uri);
}

void VideoOutput::Close()
{
    recorder.reset();
}

const std::vector<StreamInfo>& VideoOutput::Streams() const
{
    return recorder->Streams();
}

void VideoOutput::SetStreams(const std::vector<StreamInfo>& streams,
                             const std::string& uri,
                             const picojson::value& properties)
{
    recorder->SetStreams(streams, uri, properties);
}

int VideoOutput::WriteStreams(const unsigned char* data, const picojson::value& frame_properties)
{
    return recorder->WriteStreams(data, frame_properties);
}

bool VideoOutput::IsPipe() const
{
    return recorder->IsPipe();
}

void VideoOutput::AddStream(const PixelFormat& pf, size_t w, size_t h, size_t pitch)
{
    streams.emplace_back(pf, w, h, pitch, nullptr);
}

void VideoOutput::AddStream(const PixelFormat& pf, size_t w, size_t h)
{
    AddStream(pf, w, h, w * pf.bpp / 8);
}

void VideoOutput::SetStreams(const std::string& uri, const picojson::value& properties)
{
    size_t offset = 0;
    for(size_t i = 0; i < streams.size(); i++)
    {
        streams[i] = StreamInfo(streams[i].PixFormat(),
                                streams[i].Width(),
                                streams[i].Height(),
                                streams[i].Pitch(),
                                (unsigned char*)offset);
        offset += streams[i].SizeBytes();
    }
    SetStreams(streams, uri, properties);
}

size_t VideoOutput::SizeBytes(void) const
{
    size_t total = 0;
    for(const StreamInfo& si : recorder->Streams())
        total += si.SizeBytes();
    return total;
}

std::vector<Image<unsigned char>> VideoOutput::GetOutputImages(unsigned char* buffer) const
{
    std::vector<Image<unsigned char>> images;
    for(size_t s = 0; s < recorder->Streams().size(); ++s)
    {
        images.push_back(recorder->Streams()[s].StreamImage(buffer));
    }
    return images;
}

std::vector<Image<unsigned char>> VideoOutput::GetOutputImages(std::vector<unsigned char>& buffer) const
{
    buffer.resize(SizeBytes());
    return GetOutputImages(buffer.data());
}
}
