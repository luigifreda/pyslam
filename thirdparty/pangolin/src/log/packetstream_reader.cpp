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

#include <pangolin/log/packetstream_reader.h>
#include <pangolin/log/packetstream_writer.h>

using std::string;
using std::istream;
using std::ios;
using std::lock_guard;
using std::runtime_error;
using std::ios_base;
using std::streampos;
using std::streamoff;

#include <thread>

#ifndef _WIN_
#  include <unistd.h>
#endif

namespace pangolin
{

PacketStreamReader::PacketStreamReader()
    : _pipe_fd(-1)
{
}

PacketStreamReader::PacketStreamReader(const std::string& filename)
    : _pipe_fd(-1)
{
    Open(filename);
}

PacketStreamReader::~PacketStreamReader()
{
    Close();
}

void PacketStreamReader::Open(const std::string& filename)
{
    std::lock_guard<std::recursive_mutex> lg(_mutex);

    Close();

    _filename = filename;
    _is_pipe = IsPipe(filename);
    _stream.open(filename);

    if (!_stream.is_open())
        throw runtime_error("Cannot open stream.");

    for (auto i : PANGO_MAGIC)
    {
        if (_stream.get() != i)
            throw runtime_error("Unrecognised file header.");
        if (!_stream.good())
            throw runtime_error("Bad stream");
    }


    ParseHeader();

    while (_stream.peekTag() == TAG_ADD_SOURCE) {
        ParseNewSource();
    }

    if(!SetupIndex()) {
        FixFileIndex();
    }
}

void PacketStreamReader::Close() {
    std::lock_guard<std::recursive_mutex> lg(_mutex);

    _stream.close();
    _sources.clear();

#ifndef _WIN_
    if (_pipe_fd != -1) {
        close(_pipe_fd);
    }
#endif
}

void PacketStreamReader::ParseHeader()
{
    _stream.readTag(TAG_PANGO_HDR);

    picojson::value json_header;
    picojson::parse(json_header, _stream);

    // File timestamp
    const int64_t start_us = json_header["time_us"].get<int64_t>();
    packet_stream_start = SyncTime::TimePoint() + std::chrono::microseconds(start_us);

    _stream.get(); // consume newline
}

void PacketStreamReader::ParseNewSource()
{
    _stream.readTag(TAG_ADD_SOURCE);
    picojson::value json;
    picojson::parse(json, _stream);
    _stream.get(); // consume newline


    const size_t src_id = json[pss_src_id].get<int64_t>();

    if(_sources.size() <= src_id) {
        _sources.resize(src_id+1);
    }

    PacketStreamSource& pss = _sources[src_id];
    pss.id = src_id;
    pss.driver = json[pss_src_driver].get<string>();
    pss.uri = json[pss_src_uri].get<string>();
    pss.info = json[pss_src_info];
    pss.version = json[pss_src_version].get<int64_t>();
    pss.data_alignment_bytes = json[pss_src_packet][pss_pkt_alignment_bytes].get<int64_t>();
    pss.data_definitions = json[pss_src_packet][pss_pkt_definitions].get<string>();
    pss.data_size_bytes = json[pss_src_packet][pss_pkt_size_bytes].get<int64_t>();
}

bool PacketStreamReader::SetupIndex()
{
    bool index_good = false;

    if (_stream.seekable())
    {
        // Save current position
        std::streampos pos = _stream.tellg();

        // Look for footer at end of file (TAG_PANGO_FOOTER + index position).
        _stream.seekg(-(static_cast<istream::off_type>(sizeof(uint64_t)) + TAG_LENGTH), ios_base::end);
        if (_stream.peekTag() == TAG_PANGO_FOOTER)
        {
            //parsing the footer returns the index position
            _stream.seekg(ParseFooter());
            if (_stream.peekTag() == TAG_PANGO_STATS) {
                // Read the pre-build index from the file
                index_good = ParseIndex();
            }
        }

        // Restore previous location
        _stream.clear();
        _stream.seekg(pos);
    }

    return index_good;
}

streampos PacketStreamReader::ParseFooter() //returns position of index.
{
    _stream.readTag(TAG_PANGO_FOOTER);
    uint64_t index=0;
    size_t bytes_read = _stream.read(reinterpret_cast<char*>(&index), sizeof(index));
    PANGO_ENSURE(bytes_read == sizeof(index));
    return index;
}

bool PacketStreamReader::ParseIndex()
{
    _stream.readTag(TAG_PANGO_STATS);
    picojson::value json;
    picojson::parse(json, _stream);

    const bool index_good = json.contains("src_packet_index") && json.contains("src_packet_times");

    if (index_good)
    {
        // This is a two-dimensional serialized array, [source id][sequence number] ---> packet position in stream
        const auto& json_index = json["src_packet_index"].get<picojson::array>();
        const auto& json_times = json["src_packet_times"].get<picojson::array>();

        // We shouldn't have seen more sources than exist in the index
        PANGO_ENSURE(_sources.size() <= json_index.size());
        PANGO_ENSURE(json_index.size() == json_times.size());

        _sources.resize(json_index.size());

        // Populate index
        for(size_t i=0; i < _sources.size(); ++i) {
            PANGO_ENSURE(json_index[i].size() == json_times[i].size());
            _sources[i].index.resize(json_index[i].size());
            for(size_t f=0; f < json_index[i].size(); ++f) {
                _sources[i].index[f].pos = json_index[i][f].get<int64_t>();
                _sources[i].index[f].capture_time = json_times[i][f].get<int64_t>();
            }
        }
    }

    return index_good;
}

bool PacketStreamReader::GoodToRead()
{
    if(!_stream.good()) {
#ifndef _WIN_
        if (_is_pipe)
        {
            if (_pipe_fd == -1) {
                _pipe_fd = ReadablePipeFileDescriptor(_filename);
            }

            if (_pipe_fd != -1) {
                // Test whether the pipe has data to be read. If so, open the
                // file stream and start reading. After this point, the file
                // descriptor is owned by the reader.
                if (PipeHasDataToRead(_pipe_fd))
                {
                    close(_pipe_fd);
                    _pipe_fd = -1;
                    Open(_filename);
                    return _stream.good();
                }
            }
        }
#endif

        return false;
    }

    return true;

}

Packet PacketStreamReader::NextFrame()
{
    std::unique_lock<std::recursive_mutex> lock(_mutex);

    while (GoodToRead())
    {
        const pangoTagType t = _stream.peekTag();

        switch (t)
        {
        case TAG_PANGO_SYNC:
            SkipSync();
            break;
        case TAG_ADD_SOURCE:
            ParseNewSource();
            break;
        case TAG_SRC_JSON: //frames are sometimes preceded by metadata, but metadata must ALWAYS be followed by a frame from the same source.
        case TAG_SRC_PACKET:
            return Packet(_stream, std::move(lock), _sources);
        case TAG_PANGO_STATS:
            ParseIndex();
            break;
        case TAG_PANGO_FOOTER: //end of frames
        case TAG_END:
            throw std::runtime_error("PacketStreamReader: end of stream");
        case TAG_PANGO_HDR: //shoudln't encounter this
            ParseHeader();
            break;
        case TAG_PANGO_MAGIC: //or this
            SkipSync();
            break;
        default: //or anything else
            pango_print_warn("Unexpected packet type: \"%s\". Resyncing()\n", tagName(t).c_str());
            ReSync();
            break;
        }
    }

    // No frame
    throw std::runtime_error("PacketStreamReader: no frame");
}

Packet PacketStreamReader::NextFrame(PacketStreamSourceId src)
{
    while (1)
    {
        // This will throw if nothing is left.
        auto fi = NextFrame();
        if (fi.src == src) {
            return fi;
        }
    }
}

void PacketStreamReader::RebuildIndex()
{
    lock_guard<decltype(_mutex)> lg(_mutex);

    if(_stream.seekable()) {
        pango_print_warn("Index for '%s' bad / outdated. Rebuilding.\n", _filename.c_str());

        // Save current position
        std::streampos pos = _stream.tellg();

        // Clear existing index
        for(PacketStreamSource& s : _sources) {
            s.index.clear();
            s.next_packet_id = 0;
        }

        // Read through entire file, updating index
        try{
            while (1)
            {
                // This will throw if we've run out of frames
                auto fi = NextFrame();
                PacketStreamSource& s = _sources[fi.src];
                PANGO_ENSURE(s.index.size() == fi.sequence_num);
                s.index.push_back({fi.frame_streampos, fi.time});
            }
        }catch(...){
        }

        // Reset Packet id's
        for(PacketStreamSource& s : _sources) {
            s.next_packet_id = 0;
        }

        // Restore previous location
        _stream.clear();
        _stream.seekg(pos);
    }
}

void PacketStreamReader::AppendIndex()
{
    lock_guard<decltype(_mutex)> lg(_mutex);

    if(_stream.seekable()) {
        // Open file again for append
        std::ofstream of(_filename, std::ios::app | std::ios::binary);
        if(of.is_open()) {
            pango_print_warn("Appending new index to '%s'.\n", _filename.c_str());
            uint64_t indexpos = (uint64_t)of.tellp();
            writeTag(of, TAG_PANGO_STATS);
            SourceStats(_sources).serialize(std::ostream_iterator<char>(of), false);
            writeTag(of, TAG_PANGO_FOOTER);
            of.write(reinterpret_cast<char*>(&indexpos), sizeof(uint64_t));
        }
    }
}

void PacketStreamReader::FixFileIndex()
{
    if(_stream.seekable())
    {
        RebuildIndex();
        AppendIndex();
    }
}

size_t PacketStreamReader::Seek(PacketStreamSourceId src, size_t framenum)
{
    lock_guard<decltype(_mutex)> lg(_mutex);

    PANGO_ASSERT(_stream.seekable());
    PANGO_ASSERT(src < _sources.size());
    PacketStreamSource& source = _sources[src];
    PANGO_ASSERT(framenum < source.index.size());

    if(source.index[framenum].pos > 0) {
        _stream.clear();
        _stream.seekg(source.index[framenum].pos);
        source.next_packet_id = framenum;
    }
    return source.next_packet_id;
}

// Jumps to the first packet with time >= time
size_t PacketStreamReader::Seek(PacketStreamSourceId src, SyncTime::TimePoint time)
{
    PacketStreamSource& source = _sources[src];

    PacketStreamSource::PacketInfo v = {
        0, std::chrono::duration_cast<std::chrono::microseconds>(time.time_since_epoch()).count()
    };

    // Find time in indextime
    auto lb = std::lower_bound(
        source.index.begin(), source.index.end(), v,
        [](const PacketStreamSource::PacketInfo& a, const PacketStreamSource::PacketInfo& b){
            return a.capture_time < b.capture_time;
        }
    );

    if(lb != source.index.end()) {
        const size_t frame_num = lb - source.index.begin();
        return Seek(src, frame_num);
    }else{
        return source.next_packet_id;
    }
}

void PacketStreamReader::SkipSync()
{
    //Assume we have just read PAN, read GO
    if (_stream.get() != 'G' && _stream.get() != 'O')
        throw std::runtime_error("Unknown packet type.");

    while (_stream.peekTag() != TAG_SRC_PACKET && _stream.peekTag() != TAG_END)
        _stream.readTag();
}

}
