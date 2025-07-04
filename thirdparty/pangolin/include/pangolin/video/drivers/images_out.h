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

#include <fstream>
#include <pangolin/video/video_output.h>
#include <pangolin/log/packetstream_writer.h>

namespace pangolin
{

class PANGOLIN_EXPORT ImagesVideoOutput : public VideoOutputInterface
{
public:
    ImagesVideoOutput(const std::string& image_folder, const std::string& json_file_out, const std::string &image_file_extension);
    ~ImagesVideoOutput();

    const std::vector<StreamInfo>& Streams() const override;
    void SetStreams(const std::vector<StreamInfo>& streams, const std::string& uri, const picojson::value& device_properties) override;
    int WriteStreams(const unsigned char* data, const picojson::value& frame_properties) override;
    bool IsPipe() const override;

protected:
    std::vector<StreamInfo> streams;
    std::string input_uri;
    picojson::value device_properties;
    picojson::value json_frames;

    size_t image_index;
    std::string image_folder;
    std::string image_file_extension;
    std::ofstream file;
};

}
