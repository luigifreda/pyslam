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

#include <pangolin/video/video.h>
#include <pangolin/video/video_output.h>

namespace pangolin
{

struct PANGOLIN_EXPORT VideoInput
    : public VideoInterface,
      public VideoFilterInterface
{
    /////////////////////////////////////////////////////////////
    // VideoInterface Methods
    /////////////////////////////////////////////////////////////

    size_t SizeBytes() const override;
    const std::vector<StreamInfo>& Streams() const override;
    void Start() override;
    void Stop() override;
    bool GrabNext( unsigned char* image, bool wait = true ) override;
    bool GrabNewest( unsigned char* image, bool wait = true ) override;

    /////////////////////////////////////////////////////////////
    // VideoFilterInterface Methods
    /////////////////////////////////////////////////////////////

    std::vector<VideoInterface*>& InputStreams() override
    {
        return videos;
    }

    /////////////////////////////////////////////////////////////
    // VideoInput Methods
    /////////////////////////////////////////////////////////////

    VideoInput();
    VideoInput(VideoInput&& other) = default;
    VideoInput(const std::string &input_uri, const std::string &output_uri = "pango:[buffer_size_mb=100]//video_log.pango");
    ~VideoInput();

    void Open(const std::string &input_uri, const std::string &output_uri = "pango:[buffer_size_mb=100]//video_log.pango");
    void Close();

    // experimental - not stable
    bool Grab( unsigned char* buffer, std::vector<Image<unsigned char> >& images, bool wait = true, bool newest = false);

    // Return details of first stream
    unsigned int Width() const {
        return (unsigned int)Streams()[0].Width();
    }
    unsigned int Height() const {
        return (unsigned int)Streams()[0].Height();
    }
    PixelFormat PixFormat() const {
        return Streams()[0].PixFormat();
    }
    const Uri& VideoUri() const {
        return uri_input;
    }

    void Reset() {
        Close();
        Open(uri_input.full_uri, uri_output.full_uri);
    }

    // Return pointer to inner video class as VideoType
    template<typename VideoType>
    VideoType* Cast() {
        return dynamic_cast<VideoType*>(video_src.get());
    }

    const std::string& LogFilename() const;
    std::string& LogFilename();

    // Switch to live video and record output to file
    void Record();

    // Switch to live video and record a single frame
    void RecordOneFrame();

    // Specify that one in n frames are logged to file. Default is 1.
    void SetTimelapse(size_t one_in_n_frames);

    // True iff grabbed live frames are being logged to file
    bool IsRecording() const;

protected:
    void InitialiseRecorder();

    Uri uri_input;
    Uri uri_output;

    std::unique_ptr<VideoInterface> video_src;
    std::unique_ptr<VideoOutputInterface> video_recorder;

    // Use to store either video_src or video_file for VideoFilterInterface,
    // depending on which is active
    std::vector<VideoInterface*> videos;

    int buffer_size_bytes;

    int frame_num;
    size_t record_frame_skip;

    bool record_once;
    bool record_continuous;
};

// VideoInput subsumes the previous VideoRecordRepeat class.
typedef VideoInput VideoRecordRepeat;

}
