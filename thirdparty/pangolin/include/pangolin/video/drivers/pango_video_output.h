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

#include <pangolin/log/packetstream_writer.h>
#include <pangolin/video/video_output.h>

#include <pangolin/video/stream_encoder_factory.h>

#include <functional>

namespace pangolin
{

class PANGOLIN_EXPORT PangoVideoOutput : public VideoOutputInterface
{
public:
    PangoVideoOutput(const std::string& filename, size_t buffer_size_bytes, const std::map<size_t, std::string> &stream_encoder_uris);
    ~PangoVideoOutput();

    const std::vector<StreamInfo>& Streams() const override;
    void SetStreams(const std::vector<StreamInfo>& streams, const std::string& uri, const picojson::value& device_properties) override;
    int WriteStreams(const unsigned char* data, const picojson::value& frame_properties) override;
    bool IsPipe() const override;

protected:
//    void WriteHeader();

    std::vector<StreamInfo> streams;
    std::string input_uri;
    const std::string filename;
    picojson::value device_properties;

    PacketStreamWriter packetstream;
    size_t packetstream_buffer_size_bytes;
    int packetstreamsrcid;
    size_t total_frame_size;
    bool is_pipe;

    bool fixed_size;
    std::map<size_t, std::string> stream_encoder_uris;
    std::vector<ImageEncoderFunc> stream_encoders;
};

}
