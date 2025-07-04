#include <pangolin/log/packet.h>

namespace pangolin {


Packet::Packet(PacketStream& s, std::unique_lock<std::recursive_mutex>&& lock, std::vector<PacketStreamSource>& srcs)
    : _stream(s), lock(std::move(lock))
{
    ParsePacketHeader(s, srcs);
}

Packet::Packet(Packet&& o)
    : src(o.src), time(o.time), size(o.size), sequence_num(o.sequence_num),
      meta(std::move(o.meta)), frame_streampos(o.frame_streampos), _stream(o._stream),
      lock(std::move(o.lock)), data_streampos(o.data_streampos), _data_len(o._data_len)
{
    o._data_len = 0;
}

Packet::~Packet()
{
    ReadRemaining();
}

size_t Packet::BytesRead() const
{
    return _stream.tellg() - data_streampos;
}

int Packet::BytesRemaining() const
{
    if(_data_len) {
        return (int)_data_len - (int)BytesRead();
    }else{
        return 0;
    }
}

void Packet::ParsePacketHeader(PacketStream& s, std::vector<PacketStreamSource>& srcs)
{
    size_t json_src = -1;

    frame_streampos = s.tellg();
    if (s.peekTag() == TAG_SRC_JSON)
    {
        s.readTag(TAG_SRC_JSON);
        json_src = s.readUINT();
        picojson::parse(meta, s);
    }

    s.readTag(TAG_SRC_PACKET);
    time = s.readTimestamp();

    src = s.readUINT();
    PANGO_ENSURE(json_src == size_t(-1) || json_src == src, "Frame preceded by metadata for a mismatched source. Stream may be corrupt.");

    PacketStreamSource& src_packet = srcs[src];

    size = src_packet.data_size_bytes;
    if (!size) {
        size = s.readUINT();
    }
    sequence_num = src_packet.next_packet_id++;

    _data_len = size;
    data_streampos = s.tellg();
}

void Packet::ReadRemaining()
{
    int bytes_left = BytesRemaining();

    while(bytes_left > 0 && Stream().good()) {
        Stream().skip(bytes_left);
        bytes_left = BytesRemaining();
    }
}

}
