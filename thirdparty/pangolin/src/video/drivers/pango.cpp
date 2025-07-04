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

#include <pangolin/factory/factory_registry.h>
#include <pangolin/log/playback_session.h>
#include <pangolin/utils/file_extension.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/signal_slot.h>
#include <pangolin/video/drivers/pango.h>
#include <pangolin/video/iostream_operators.h>

#include <functional>

namespace pangolin
{

const std::string pango_video_type = "raw_video";

PangoVideo::PangoVideo(const std::string& filename, std::shared_ptr<PlaybackSession> playback_session)
    : _filename(filename),
      _playback_session(playback_session),
      _reader(_playback_session->Open(filename)),
      _event_promise(_playback_session->Time()),
      _src_id(FindPacketStreamSource()),
      _source(nullptr)
{
    PANGO_ENSURE(_src_id != -1, "No appropriate video streams found in log.");

    _source = &_reader->Sources()[_src_id];
    SetupStreams(*_source);

    // Make sure we time-seek with other playback devices
    session_seek = _playback_session->Time().OnSeek.Connect(
        [&](SyncTime::TimePoint t){
            _event_promise.Cancel();
            _reader->Seek(_src_id, t);
            _event_promise.WaitAndRenew(_source->NextPacketTime());
        }
    );

    _event_promise.WaitAndRenew(_source->NextPacketTime());
}

PangoVideo::~PangoVideo()
{
}

size_t PangoVideo::SizeBytes() const
{
    return _size_bytes;
}

const std::vector<StreamInfo>& PangoVideo::Streams() const
{
    return _streams;
}

void PangoVideo::Start()
{

}

void PangoVideo::Stop()
{

}

bool PangoVideo::GrabNext(unsigned char* image, bool /*wait*/)
{
    try
    {
        Packet fi = _reader->NextFrame(_src_id);
        _frame_properties = fi.meta;

        if(_fixed_size) {
            fi.Stream().read(reinterpret_cast<char*>(image), _size_bytes);
        }else{
            for(size_t s=0; s < _streams.size(); ++s) {
                StreamInfo& si = _streams[s];
                pangolin::Image<unsigned char> dst = si.StreamImage(image);

                if(stream_decoder[s]) {
                    pangolin::TypedImage img = stream_decoder[s](fi.Stream());
                    PANGO_ENSURE(img.IsValid());

                    // TODO: We can avoid this copy by decoding directly into img
                    for(size_t row =0; row < dst.h; ++row) {
                        std::memcpy(dst.RowPtr(row), img.RowPtr(row), si.RowBytes());
                    }
                }else{
                    for(size_t row =0; row < dst.h; ++row) {
                        fi.Stream().read((char*)dst.RowPtr(row), si.RowBytes());
                    }
                }
            }
        }

        _event_promise.WaitAndRenew(_source->NextPacketTime());
        return true;
    }
    catch(...)
    {
        _frame_properties = picojson::value();
        return false;
    }
}

bool PangoVideo::GrabNewest( unsigned char* image, bool wait )
{
    return GrabNext(image, wait);
}

size_t PangoVideo::GetCurrentFrameId() const
{
    return (int)(_reader->Sources()[_src_id].next_packet_id) - 1;
}

size_t PangoVideo::GetTotalFrames() const
{
    return _source->index.size();
}

size_t PangoVideo::Seek(size_t next_frame_id)
{
    // Get time for seek
    if(next_frame_id < _source->index.size()) {
        const int64_t capture_time = _source->index[next_frame_id].capture_time;
        _playback_session->Time().Seek(SyncTime::TimePoint(std::chrono::microseconds(capture_time)));
        return next_frame_id;
    }else{
        return _source->next_packet_id;
    }
}

std::string PangoVideo::GetSourceUri()
{
    return _source_uri;
}

int PangoVideo::FindPacketStreamSource()
{
    for(const auto& src : _reader->Sources())
    {
        if (!src.driver.compare(pango_video_type))
        {
            return static_cast<int>(src.id);
        }
    }

    return -1;
}

void PangoVideo::SetupStreams(const PacketStreamSource& src)
{
    // Read sources header
    _fixed_size = src.data_size_bytes != 0;
    _size_bytes = src.data_size_bytes;
    _source_uri = src.uri;

    _device_properties = src.info["device"];
    const picojson::value& json_streams = src.info["streams"];
    const size_t num_streams = json_streams.size();

    for (size_t i = 0; i < num_streams; ++i)
    {
        const picojson::value& json_stream = json_streams[i];

        std::string encoding = json_stream["encoding"].get<std::string>();

        // Check if the stream is compressed
        if(json_stream.contains("decoded")) {
            const std::string compressed_encoding = encoding;
            encoding = json_stream["decoded"].get<std::string>();
            const PixelFormat decoded_fmt = PixelFormatFromString(encoding);
            stream_decoder.push_back(StreamEncoderFactory::I().GetDecoder(compressed_encoding, decoded_fmt));
        }else{
            stream_decoder.push_back(nullptr);
        }

        PixelFormat fmt = PixelFormatFromString(encoding);

        fmt.channel_bit_depth = json_stream.get_value<int64_t>("channel_bit_depth", 0);

        StreamInfo si(
                fmt,
                json_stream["width"].get<int64_t>(),
                json_stream["height"].get<int64_t>(),
                json_stream["pitch"].get<int64_t>(),
                (unsigned char*) 0 + json_stream["offset"].get<int64_t>()
                        );

        if(!_fixed_size) {
            _size_bytes += si.SizeBytes();
        }


        _streams.push_back(si);
    }
}

PANGOLIN_REGISTER_FACTORY(PangoVideo)
{
    struct PangoVideoFactory : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            const std::string path = PathExpand(uri.url);

            if( !uri.scheme.compare("pango") || FileType(uri.url) == ImageFileTypePango ) {
                return std::unique_ptr<VideoInterface>(new PangoVideo(path.c_str(), PlaybackSession::ChooseFromParams(uri)));
            }
            return std::unique_ptr<VideoInterface>();
        }
    };

    auto factory = std::make_shared<PangoVideoFactory>();
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "pango");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory,  5, "file");
}

}
