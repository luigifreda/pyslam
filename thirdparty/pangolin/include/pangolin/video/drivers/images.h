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

#include <pangolin/image/image_io.h>
#include <pangolin/pangolin.h>
#include <pangolin/video/video.h>

#include <deque>
#include <vector>

namespace pangolin
{

// Video class that outputs test video signal.
class PANGOLIN_EXPORT ImagesVideo : public VideoInterface, public VideoPlaybackInterface, public VideoPropertiesInterface
{
public:
    ImagesVideo(const std::string& wildcard_path);
    ImagesVideo(const std::string& wildcard_path, const PixelFormat& raw_fmt, size_t raw_width, size_t raw_height);

    // Explicitly delete copy ctor and assignment operator.
    // See http://stackoverflow.com/questions/29565299/how-to-use-a-vector-of-unique-pointers-in-a-dll-exported-class-with-visual-studi
    // >> It appears adding __declspec(dllexport) forces the compiler to define the implicitly-declared copy constructor and copy assignment operator
    ImagesVideo(const ImagesVideo&) = delete;
    ImagesVideo& operator=(const ImagesVideo&) = delete;

    ~ImagesVideo();

    ///////////////////////////////////
    // Implement VideoInterface
    
    void Start() override;
    
    void Stop() override;

    size_t SizeBytes() const override;

    const std::vector<StreamInfo>& Streams() const override;
    
    bool GrabNext( unsigned char* image, bool wait = true ) override;
    
    bool GrabNewest( unsigned char* image, bool wait = true ) override;

    ///////////////////////////////////
    // Implement VideoPlaybackInterface

    size_t GetCurrentFrameId() const override;

    size_t GetTotalFrames() const override;

    size_t Seek(size_t frameid) override;

    ///////////////////////////////////
    // Implement VideoPropertiesInterface

    const picojson::value& DeviceProperties() const override;

    const picojson::value& FrameProperties() const override;
    
protected:
    typedef std::vector<TypedImage> Frame;
    
    const std::string& Filename(size_t frameNum, size_t channelNum) {
        return filenames[channelNum][frameNum];
    }
    
    void PopulateFilenames(const std::string& wildcard_path);

    void PopulateFilenamesFromJson(const std::string& filename);

    bool LoadFrame(size_t i);

    void ConfigureStreamSizes();
    
    std::vector<StreamInfo> streams;
    size_t size_bytes;
    
    size_t num_files;
    size_t num_channels;
    size_t next_frame_id;
    std::vector<std::vector<std::string> > filenames;
    std::vector<Frame> loaded;

    bool unknowns_are_raw;
    PixelFormat raw_fmt;
    size_t raw_width;
    size_t raw_height;

    // Load any json properties if they are defined
    picojson::value device_properties;
    picojson::value json_frames;
    picojson::value null_props;
};

}
