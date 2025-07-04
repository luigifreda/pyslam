#pragma once

#include <string>
#ifdef __linux__
#include <cinttypes>
#endif 

namespace pangolin {

using pangoTagType = uint32_t;

const static std::string PANGO_MAGIC = "PANGO";

const static std::string pss_src_driver = "driver";
const static std::string pss_src_id = "id";
const static std::string pss_src_info = "info";
const static std::string pss_src_uri = "uri";
const static std::string pss_src_packet = "packet";
const static std::string pss_src_version = "version";
const static std::string pss_pkt_alignment_bytes = "alignment_bytes";
const static std::string pss_pkt_definitions = "definitions";
const static std::string pss_pkt_size_bytes = "size_bytes";
const static std::string pss_pkt_format_written = "format_written";

const unsigned int TAG_LENGTH = 3;

#define PANGO_TAG(a,b,c) ( (c<<16) | (b<<8) | a)
const uint32_t TAG_PANGO_HDR    = PANGO_TAG('L', 'I', 'N');
const uint32_t TAG_PANGO_MAGIC  = PANGO_TAG('P', 'A', 'N');
const uint32_t TAG_PANGO_SYNC   = PANGO_TAG('S', 'Y', 'N');
const uint32_t TAG_PANGO_STATS  = PANGO_TAG('S', 'T', 'A');
const uint32_t TAG_PANGO_FOOTER = PANGO_TAG('F', 'T', 'R');
const uint32_t TAG_ADD_SOURCE   = PANGO_TAG('S', 'R', 'C');
const uint32_t TAG_SRC_JSON     = PANGO_TAG('J', 'S', 'N');
const uint32_t TAG_SRC_PACKET   = PANGO_TAG('P', 'K', 'T');
const uint32_t TAG_END          = PANGO_TAG('E', 'N', 'D');
#undef PANGO_TAG

inline std::string tagName(int v)
{
    char b[4];
    b[0] = v&0xff;
    b[1] = (v>>8)&0xff;
    b[2] = (v>>16)&0xff;
    b[3] = 0x00;
    return std::string(b);
}

}
