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

#include <pangolin/log/packetstream_reader.h>
#include <pangolin/log/playback_session.h>
#include <pangolin/video/stream_encoder_factory.h>
#include <pangolin/video/video.h>

namespace pangolin
{

class PANGOLIN_EXPORT PangoVideo
    : public VideoInterface, public VideoPropertiesInterface, public VideoPlaybackInterface
{
public:
    PangoVideo(const std::string& filename, std::shared_ptr<PlaybackSession> playback_session);
    ~PangoVideo();

    // Implement VideoInterface

    size_t SizeBytes() const override;

    const std::vector<StreamInfo>& Streams() const override;

    void Start() override;

    void Stop() override;

    bool GrabNext( unsigned char* image, bool wait = true ) override;

    bool GrabNewest( unsigned char* image, bool wait = true ) override;

    // Implement VideoPropertiesInterface
    const picojson::value& DeviceProperties() const override {
        if (-1 == _src_id) throw std::runtime_error("Not initialised");
        return _device_properties;
    }

    const picojson::value& FrameProperties() const override {
        return _frame_properties;
    }

    // Implement VideoPlaybackInterface

    size_t GetCurrentFrameId() const override;

    size_t GetTotalFrames() const override;

    size_t Seek(size_t frameid) override;

    std::string GetSourceUri();

private:
    void HandlePipeClosed();

protected:
    int FindPacketStreamSource();
    void SetupStreams(const PacketStreamSource& src);

    const std::string _filename;
    std::shared_ptr<PlaybackSession> _playback_session;
    std::shared_ptr<PacketStreamReader> _reader;
    SyncTimeEventPromise _event_promise;
    int _src_id;
    const PacketStreamSource* _source;

    size_t _size_bytes;
    bool _fixed_size;
    std::vector<StreamInfo> _streams;
    std::vector<ImageDecoderFunc> stream_decoder;
    picojson::value _device_properties;
    picojson::value _frame_properties;
    std::string _source_uri;

    Registration<size_t> session_seek;
};

}
