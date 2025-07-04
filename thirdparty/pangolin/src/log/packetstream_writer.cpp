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

#include <pangolin/log/packetstream_writer.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>

using std::ios;
using std::lock_guard;

#define SCOPED_LOCK lock_guard<decltype(_lock)> lg(_lock)

//#define SCOPED_LOCK

namespace pangolin
{

static inline const std::string CurrentTimeStr()
{
    time_t time_now = time(0);
    struct tm time_struct = *localtime(&time_now);
    char buffer[80];
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %X", &time_struct);
    return buffer;
}

void PacketStreamWriter::WriteHeader()
{
    SCOPED_LOCK;
    _stream.write(PANGO_MAGIC.c_str(), PANGO_MAGIC.size());
    picojson::value pango;
    pango["pangolin_version"] = PANGOLIN_VERSION_STRING;
    pango["time_us"] = Time_us(TimeNow());
    pango["date_created"] = CurrentTimeStr();
    pango["endian"] = "little_endian";

    writeTag(_stream, TAG_PANGO_HDR);
    pango.serialize(std::ostream_iterator<char>(_stream), true);

    for (const auto& source : _sources)
        Write(source);
}

void PacketStreamWriter::Write(const PacketStreamSource& source)
{
    SCOPED_LOCK;
    picojson::value serialize;
    serialize[pss_src_driver] = source.driver;
    serialize[pss_src_id] = source.id;
    serialize[pss_src_uri] = source.uri;
    serialize[pss_src_info] = source.info;
    serialize[pss_src_version] = source.version;
    serialize[pss_src_packet][pss_pkt_alignment_bytes] = source.data_alignment_bytes;
    serialize[pss_src_packet][pss_pkt_definitions] = source.data_definitions;
    serialize[pss_src_packet][pss_pkt_size_bytes] = source.data_size_bytes;

    writeTag(_stream, TAG_ADD_SOURCE);
    serialize.serialize(std::ostream_iterator<char>(_stream), true);
}


PacketStreamSourceId PacketStreamWriter::AddSource(PacketStreamSource& source)
{
    SCOPED_LOCK;
    source.id = AddSource(const_cast<const PacketStreamSource&>(source));
    return source.id;
}

PacketStreamSourceId PacketStreamWriter::AddSource(const PacketStreamSource& source)
{
    SCOPED_LOCK;
    PacketStreamSourceId r = _sources.size(); //source id is by vector position, so we must reassign.
    _sources.push_back(source);
    _sources.back().id = r;

    if (_open) //we might be a pipe, in which case we may not be open
        Write(_sources.back());

    return _sources.back().id;
}

void PacketStreamWriter::WriteMeta(PacketStreamSourceId src, const picojson::value& data)
{
    SCOPED_LOCK;
    writeTag(_stream, TAG_SRC_JSON);
    writeCompressedUnsignedInt(_stream, src);
    data.serialize(std::ostream_iterator<char>(_stream), false);
}

void PacketStreamWriter::WriteSourcePacket(PacketStreamSourceId src, const char* source, const int64_t receive_time_us, size_t sourcelen, const picojson::value& meta)
{

    SCOPED_LOCK;
    _sources[src].index.push_back({_stream.tellp(), receive_time_us});

    if (!meta.is<picojson::null>())
        WriteMeta(src, meta);

    writeTag(_stream, TAG_SRC_PACKET);
    writeTimestamp(_stream, receive_time_us);
    writeCompressedUnsignedInt(_stream, src);

    if (_sources[src].data_size_bytes) {
        if (sourcelen != static_cast<size_t>(_sources[src].data_size_bytes))
            throw std::runtime_error("oPacketStream::writePacket --> Tried to write a fixed-size packet with bad size.");
    } else {
        writeCompressedUnsignedInt(_stream, sourcelen);
    }

    _stream.write(source, sourcelen);
    _bytes_written += sourcelen;
}

void PacketStreamWriter::WriteSync()
{
    SCOPED_LOCK;
    for (unsigned i = 0; i < 10; ++i)
    writeTag(_stream, TAG_PANGO_SYNC);
}

void PacketStreamWriter::WriteEnd()
{
    SCOPED_LOCK;
    if (!_indexable)
        return;

    auto indexpos = _stream.tellp();
    writeTag(_stream, TAG_PANGO_STATS);
    SourceStats(_sources).serialize(std::ostream_iterator<char>(_stream), false);
    writeTag(_stream, TAG_PANGO_FOOTER);
    _stream.write(reinterpret_cast<char*>(&indexpos), sizeof(uint64_t));
}

}
