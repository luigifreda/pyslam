#include <pangolin/log/packetstream.h>
#include <stdexcept>

namespace pangolin {

size_t PacketStream::readUINT()
{
    size_t n = 0;
    size_t v = get();
    uint32_t shift = 0;
    while (good() && (v & 0x80))
    {
        n |= (v & 0x7F) << shift;
        shift += 7;
        v = get();
    }
    if (!good())
        return static_cast<size_t>(-1);
    return n | (v & 0x7F) << shift;
}

int64_t PacketStream::readTimestamp()
{
    int64_t time_us;
    read(reinterpret_cast<char*>(&time_us), sizeof(int64_t));
    return time_us;
}

pangoTagType PacketStream::readTag()
{
    auto r = peekTag();
    _tag = 0;
    return r;
}

pangoTagType PacketStream::readTag(pangoTagType x)
{
    auto r = readTag();
    if (r != x)
        throw std::runtime_error(("Tag mismatch error: expected tag '" + tagName(r) + "' does not match found tag '" + tagName(x) + "'").c_str());
    return r;
}

pangoTagType PacketStream::peekTag()
{
    if (!_tag)
    {
        _tag = 0;
        Base::read(reinterpret_cast<char*>(&_tag), TAG_LENGTH);
        if (!good())
            _tag = TAG_END;
    }
    return _tag;
}

char PacketStream::get()
{
    _tag = 0;
    return Base::get();
}

size_t PacketStream::read(char* target, size_t len)
{
    _tag = 0;
    Base::read(target, len);
    return gcount();
}

size_t PacketStream::skip(size_t len)
{
    if (seekable()) {
        Base::seekg(len, std::ios_base::cur);
    } else {
        Base::ignore(len);
    }
    cclear();
    return len;
}

std::streampos PacketStream::tellg()
{
    if (_tag) {
        return Base::tellg() - std::streamoff(TAG_LENGTH);
    }else{
        return Base::tellg();
    }
}

void PacketStream::seekg(std::streampos target)
{
    if (seekable()) {
        cclear();
        Base::seekg(target);
    }
}

void PacketStream::seekg(std::streamoff off, std::ios_base::seekdir way)
{
    if (seekable()) {
        cclear();
        Base::seekg(off, way);
    }
}

static bool valid(pangoTagType t)
{
    switch (t)
    {
    case TAG_PANGO_SYNC:
        case TAG_ADD_SOURCE:
        case TAG_SRC_JSON:
        case TAG_SRC_PACKET:
        case TAG_PANGO_STATS:
        case TAG_PANGO_FOOTER:
        case TAG_END:
        case TAG_PANGO_HDR:
        case TAG_PANGO_MAGIC:
        return true;
    default:
        return false;
    }
}

pangoTagType PacketStream::syncToTag() //scan through chars one by one until the last three look like a tag
{
    peekTag();
    char * buffer = reinterpret_cast<char*>(&_tag);

    buffer[3] = 0;

    do
    {
        buffer[0] = buffer[1];
        buffer[1] = buffer[2];
        buffer[2] = get();
    }
    while (good() && !valid(_tag));

    if (!good())
        _tag = TAG_END;

    return _tag;

}

}
