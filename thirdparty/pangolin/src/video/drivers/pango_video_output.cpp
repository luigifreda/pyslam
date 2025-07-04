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
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/memstreambuf.h>
#include <pangolin/utils/picojson.h>
#include <pangolin/utils/sigstate.h>
#include <pangolin/utils/timer.h>
#include <pangolin/video/drivers/pango_video_output.h>
#include <pangolin/video/iostream_operators.h>
#include <pangolin/video/video_interface.h>
#include <set>
#include <future>

#ifndef _WIN_
#  include <unistd.h>
#endif

namespace pangolin
{

const std::string pango_video_type = "raw_video";

void SigPipeHandler(int sig)
{
    SigState::I().sig_callbacks.at(sig).value = true;
}

PangoVideoOutput::PangoVideoOutput(const std::string& filename, size_t buffer_size_bytes, const std::map<size_t, std::string> &stream_encoder_uris)
    : filename(filename),
      packetstream_buffer_size_bytes(buffer_size_bytes),
      packetstreamsrcid(-1),
      total_frame_size(0),
      is_pipe(pangolin::IsPipe(filename)),
      fixed_size(true),
      stream_encoder_uris(stream_encoder_uris)
{
    if(!is_pipe)
    {
        packetstream.Open(filename, packetstream_buffer_size_bytes);
    }
    else
    {
        RegisterNewSigCallback(&SigPipeHandler, (void*)this, SIGPIPE);
    }

    // Instantiate encoders
}

PangoVideoOutput::~PangoVideoOutput()
{
}

const std::vector<StreamInfo>& PangoVideoOutput::Streams() const
{
    return streams;
}

bool PangoVideoOutput::IsPipe() const
{
    return is_pipe;
}

void PangoVideoOutput::SetStreams(const std::vector<StreamInfo>& st, const std::string& uri, const picojson::value& properties)
{
    std::set<unsigned char*> unique_ptrs;
    for (size_t i = 0; i < st.size(); ++i)
    {
    unique_ptrs.insert(st[i].Offset());
    }

    if (unique_ptrs.size() < st.size())
    throw std::invalid_argument("Each image must have unique offset into buffer.");

    if (packetstreamsrcid == -1)
    {
        input_uri = uri;
        streams = st;
        device_properties = properties;

        picojson::value json_header(picojson::object_type, false);
        picojson::value& json_streams = json_header["streams"];
        json_header["device"] = device_properties;

        stream_encoders.resize(streams.size());

        fixed_size = true;

        total_frame_size = 0;
        for (unsigned int i = 0; i < streams.size(); ++i)
        {
            StreamInfo& si = streams[i];
            total_frame_size = std::max(total_frame_size, (size_t) si.Offset() + si.SizeBytes());

            picojson::value& json_stream = json_streams.push_back();

            std::string encoder_name = si.PixFormat().format;
            if(stream_encoder_uris.find(i) != stream_encoder_uris.end() && !stream_encoder_uris[i].empty() ) {
                // instantiate encoder and write it's name to the stream properties
                json_stream["decoded"] = si.PixFormat().format;
                encoder_name = stream_encoder_uris[i];
                stream_encoders[i] = StreamEncoderFactory::I().GetEncoder(encoder_name, si.PixFormat());
                fixed_size = false;
            }

            json_stream["channel_bit_depth"] = si.PixFormat().channel_bit_depth;
            json_stream["encoding"] = encoder_name;
            json_stream["width"] = si.Width();
            json_stream["height"] = si.Height();
            json_stream["pitch"] = si.Pitch();
            json_stream["offset"] = (size_t) si.Offset();
        }

        PacketStreamSource pss;
        pss.driver = pango_video_type;
        pss.uri = input_uri;
        pss.info = json_header;
        pss.data_size_bytes = fixed_size ? total_frame_size : 0;
        pss.data_definitions = "struct Frame{ uint8 stream_data[" + pangolin::Convert<std::string, size_t>::Do(total_frame_size) + "];};";

        packetstreamsrcid = (int)packetstream.AddSource(pss);
    } else {
        throw std::runtime_error("Unable to add new streams");
    }
}

int PangoVideoOutput::WriteStreams(const unsigned char* data, const picojson::value& frame_properties)
{
    const int64_t host_reception_time_us = frame_properties.get_value(PANGO_HOST_RECEPTION_TIME_US, Time_us(TimeNow()));

#ifndef _WIN_
    if (is_pipe)
    {
        // If there is a reader waiting on the other side of the pipe, open
        // a file descriptor to the file and close it only after the file
        // has been opened by the PacketStreamWriter. This avoids the reader
        // from seeing EOF on its next read because all file descriptors on
        // the write side have been closed.
        //
        // When the stream is already open but the reader has disappeared,
        // opening a file descriptor will fail and errno will be ENXIO.
        int fd = WritablePipeFileDescriptor(filename);

        if (!packetstream.IsOpen())
        {
            if (fd != -1)
            {
                packetstream.Open(filename, packetstream_buffer_size_bytes);
                close(fd);
            }
        }
        else
        {
            if (fd != -1)
            {
                // There's a reader on the other side of the pipe.
                close(fd);
            }
            else
            {
                if (errno == ENXIO)
                {
                    packetstream.ForceClose();
                    SigState::I().sig_callbacks.at(SIGPIPE).value = false;

                    // This should be unnecessary since per the man page,
                    // data should be dropped from the buffer upon closing the
                    // writable file descriptors.
                    pangolin::FlushPipe(filename);
                }
            }
        }

        if (!packetstream.IsOpen())
            return 0;
    }
#endif

    if(!fixed_size) {
        // TODO: Make this more efficient (without so many allocs and memcpy's)

        std::vector<memstreambuf> encoded_stream_data;

        // Create buffers for compressed data: the first will be reused for all the data later
        encoded_stream_data.emplace_back(total_frame_size);
        for(size_t i=1; i < streams.size(); ++i) {
            encoded_stream_data.emplace_back(streams[i].SizeBytes());
        }

        // lambda encodes frame data i to encoded_stream_data[i]
        auto encode_stream = [&](int i){
            encoded_stream_data[i].clear();
            std::ostream encode_stream(&encoded_stream_data[i]);

            const StreamInfo& si = streams[i];
            const Image<unsigned char> stream_image = si.StreamImage(data);

            if(stream_encoders[i]) {
                // Encode to buffer
                stream_encoders[i](encode_stream, stream_image);
            }else{
                if(stream_image.IsContiguous()) {
                    encode_stream.write((char*)stream_image.ptr, streams[i].SizeBytes());
                }else{
                    for(size_t row=0; row < stream_image.h; ++row) {
                        encode_stream.write((char*)stream_image.RowPtr(row), si.RowBytes());
                    }
                }
            }
            return true;
        };

        // Compress each stream (>0 in another thread)
        std::vector<std::future<bool>> encode_finished;
        for(size_t i=1; i < streams.size(); ++i) {
            encode_finished.emplace_back(std::async(std::launch::async, [&,i](){
                return encode_stream(i);
            }));
        }
        // Encode stream 0 in this thread
        encode_stream(0);

        // Reuse our first compression stream for the rest of the data too.
        std::vector<uint8_t>& encoded = encoded_stream_data[0].buffer;

        // Wait on all threads to finish and copy into data packet
        for(size_t i=1; i < streams.size(); ++i) {
            encode_finished[i-1].get();
            encoded.insert(encoded.end(), encoded_stream_data[i].buffer.begin(), encoded_stream_data[i].buffer.end());
        }

        packetstream.WriteSourcePacket(packetstreamsrcid, reinterpret_cast<const char*>(encoded.data()), host_reception_time_us, encoded.size(), frame_properties);
    }else{
        packetstream.WriteSourcePacket(packetstreamsrcid, reinterpret_cast<const char*>(data), host_reception_time_us, total_frame_size, frame_properties);
    }

    return 0;
}

PANGOLIN_REGISTER_FACTORY(PangoVideoOutput)
{
    struct PangoVideoFactory final : public FactoryInterface<VideoOutputInterface> {
        std::unique_ptr<VideoOutputInterface> Open(const Uri& uri) override {
            const size_t mb = 1024*1024;
            const size_t buffer_size_bytes = uri.Get("buffer_size_mb", 100) * mb;
            std::string filename = uri.url;

            if(uri.Contains("unique_filename")) {
                filename = MakeUniqueFilename(filename);
            }

            // Default encoder
            std::string default_encoder = "";

            if(uri.Contains("encoder")) {
                default_encoder = uri.Get<std::string>("encoder","");
            }

            // Encoders for each stream
            std::map<size_t, std::string> stream_encoder_uris;
            for(size_t i=0; i<100; ++i)
            {
                const std::string encoder_key = pangolin::FormatString("encoder%",i+1);
                stream_encoder_uris[i] = uri.Get<std::string>(encoder_key, default_encoder);
            }

            return std::unique_ptr<VideoOutputInterface>(
                new PangoVideoOutput(filename, buffer_size_bytes, stream_encoder_uris)
            );
        }
    };

    auto factory = std::make_shared<PangoVideoFactory>();
    FactoryRegistry<VideoOutputInterface>::I().RegisterFactory(factory, 10, "pango");
    FactoryRegistry<VideoOutputInterface>::I().RegisterFactory(factory, 10, "file");
}

}
