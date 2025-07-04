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

#include <mutex>

#include <pangolin/log/packetstream.h>
#include <pangolin/log/packetstream_source.h>

namespace pangolin {

// Encapsulate serialized reading of Packet from stream.
struct Packet
{
    Packet(PacketStream& s, std::unique_lock<std::recursive_mutex>&& mutex, std::vector<PacketStreamSource>& srcs);
    Packet(const Packet&) = delete;
    Packet(Packet&& o);
    ~Packet();

    size_t BytesRead() const;
    int BytesRemaining() const;

    PacketStream& Stream()
    {
        return _stream;
    }

    PacketStreamSourceId src;
    int64_t time;
    size_t size;
    size_t sequence_num;
    picojson::value meta;
    std::streampos frame_streampos;

private:
    void ParsePacketHeader(PacketStream& s, std::vector<PacketStreamSource>& srcs);
    void ReadRemaining();

    PacketStream& _stream;

    std::unique_lock<std::recursive_mutex> lock;

    std::streampos data_streampos;
    size_t _data_len;
};

}
