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

#include <fstream>
#include <mutex>
#include <thread>

#include <pangolin/log/packet.h>

#include <pangolin/log/sync_time.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/timer.h>

namespace pangolin
{

class PANGOLIN_EXPORT PacketStreamReader
{
public:
    PacketStreamReader();

    PacketStreamReader(const std::string& filename);

    ~PacketStreamReader();

    void Open(const std::string& filename);

    void Close();

    const std::vector<PacketStreamSource>&
    Sources() const
    {
        return _sources;
    }

    // Grab Next available frame packetstream
    Packet NextFrame();

    // Grab Next available frame in packetstream from src, discarding other frames.
    Packet NextFrame(PacketStreamSourceId src);

    bool Good() const
    {
        return _stream.good();
    }

    // Jumps to a particular packet.
    size_t Seek(PacketStreamSourceId src, size_t framenum);

    // Jumps to the first packet with time >= time
    size_t Seek(PacketStreamSourceId src, SyncTime::TimePoint time);

    void FixFileIndex();

private:
    bool GoodToRead();

    bool SetupIndex();

    void ParseHeader();

    void ParseNewSource();

    bool ParseIndex();

    void RebuildIndex();

    void AppendIndex();

    std::streampos ParseFooter();

    void SkipSync();

    void ReSync() {
        _stream.syncToTag();
    }

    std::string _filename;
    std::vector<PacketStreamSource> _sources;
    SyncTime::TimePoint packet_stream_start;

    PacketStream _stream;
    std::recursive_mutex _mutex;

    bool _is_pipe;
    int _pipe_fd;
};








}
