/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
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

#include <ostream>

#include <pangolin/log/packetstream.h>
#include <pangolin/log/packetstream_source.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/threadedfilebuf.h>

namespace pangolin
{

class PANGOLIN_EXPORT PacketStreamWriter
{
public:
    PacketStreamWriter()
        : _stream(&_buffer), _indexable(false), _open(false), _bytes_written(0)
    {
        _stream.exceptions(std::ostream::badbit);
    }

    PacketStreamWriter(const std::string& filename, size_t buffer_size  = 100*1024*1024)
        : _buffer(pangolin::PathExpand(filename), buffer_size), _stream(&_buffer),
          _indexable(!IsPipe(filename)), _open(_stream.good()), _bytes_written(0)
    {
        _stream.exceptions(std::ostream::badbit);
        WriteHeader();
    }

    ~PacketStreamWriter() {
        Close();
    }

    void Open(const std::string& filename, size_t buffer_size = 100 * 1024 * 1024)
    {
        Close();
        _buffer.open(filename, buffer_size);
        _open = _stream.good();
        _bytes_written = 0;
        _indexable = !IsPipe(filename);
        WriteHeader();
    }

    void Close()
    {
        if (_open)
        {
            if (_indexable) {
                WriteEnd();
            }
            _buffer.close();
            _open = false;
        }
    }

    // Does not write footer or index.
    void ForceClose()
    {
        if (_open)
        {
        _buffer.force_close();
        Close();
        }
    }


    // Writes to the stream immediately upon add. Return source id # and writes
    // source id # to argument struct
    PacketStreamSourceId AddSource(PacketStreamSource& source);

    // If constructor is called inline
    PacketStreamSourceId AddSource(const PacketStreamSource& source);

    void WriteSourcePacket(
        PacketStreamSourceId src, const char* source,const int64_t receive_time_us,
        size_t sourcelen, const picojson::value& meta = picojson::value()
    );

    // For stream read/write synchronization. Note that this is NOT the same as
    // time synchronization on playback of iPacketStreams.
    void WriteSync();

    // Writes the end of the stream data, including the index. Does NOT close
    // the underlying ostream.
    void WriteEnd();

    const std::vector<PacketStreamSource>& Sources() const {
        return _sources;
    }

    bool IsOpen() const {
        return _open;
    }

private:
    void WriteHeader();
    void Write(const PacketStreamSource&);
    void WriteMeta(PacketStreamSourceId src, const picojson::value& data);

    threadedfilebuf _buffer;
    std::ostream _stream;
    bool _indexable, _open;

    std::vector<PacketStreamSource> _sources;
    size_t _bytes_written;
    std::recursive_mutex _lock;
};

inline void writeCompressedUnsignedInt(std::ostream& writer, size_t n)
{
    while (n >= 0x80)
    {
    writer.put(0x80 | (n & 0x7F));
    n >>= 7;
    }
    writer.put(static_cast<unsigned char>(n));
}

inline void writeTimestamp(std::ostream& writer, int64_t time_us)
{
    writer.write(reinterpret_cast<const char*>(&time_us), sizeof(decltype(time_us)));
}

inline void writeTag(std::ostream& writer, const pangoTagType tag)
{
    writer.write(reinterpret_cast<const char*>(&tag), TAG_LENGTH);
}

inline picojson::value SourceStats(const std::vector<PacketStreamSource>& srcs)
{
    picojson::value stat;
    stat["num_sources"] = srcs.size();
    stat["src_packet_index"] = picojson::array();
    stat["src_packet_times"] = picojson::array();

    for(auto& src : srcs) {
        picojson::array pkt_index, pkt_times;
        for (const PacketStreamSource::PacketInfo& frame : src.index) {
            pkt_index.emplace_back(frame.pos);
            pkt_times.emplace_back(frame.capture_time);
        }
        stat["src_packet_index"].push_back(std::move(pkt_index));
        stat["src_packet_times"].push_back(std::move(pkt_times));
    }
    return stat;
}

}
