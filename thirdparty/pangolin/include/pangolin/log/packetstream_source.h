#pragma once

#include <iostream>
#include <pangolin/platform.h>
#include <pangolin/utils/picojson.h>

namespace pangolin {

using PacketStreamSourceId = size_t;

struct PANGOLIN_EXPORT PacketStreamSource
{
    struct PacketInfo
    {
        std::streampos pos;
        int64_t capture_time;
    };

    PacketStreamSource()
        : id(static_cast<PacketStreamSourceId>(-1)),
          version(0),
          data_alignment_bytes(1),
          data_size_bytes(0),
          next_packet_id(0)
    {
    }

    std::streampos FindSeekLocation(size_t packet_id)
    {
        if(packet_id < index.size()) {
            return index[packet_id].pos;
        }else{
            return std::streampos(-1);
        }

    }

    int64_t NextPacketTime() const
    {
        if(next_packet_id < index.size()) {
            return index[next_packet_id].capture_time;
        }else{
            return 0;
        }
    }

    std::string     driver;
    size_t          id;
    std::string     uri;
    picojson::value info;
    int64_t         version;
    int64_t         data_alignment_bytes;
    std::string     data_definitions;
    int64_t         data_size_bytes;

    // Index keyed by packet_id
    std::vector<PacketInfo> index;

    // Based on current position in stream
    size_t          next_packet_id;
};

}
