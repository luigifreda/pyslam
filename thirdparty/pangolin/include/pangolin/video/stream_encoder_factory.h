#pragma once

#include <memory>

#include <pangolin/image/image_io.h>

namespace pangolin {

using ImageEncoderFunc = std::function<void(std::ostream&, const Image<unsigned char>&)>;
using ImageDecoderFunc = std::function<TypedImage(std::istream&)>;

class StreamEncoderFactory
{
public:
    static StreamEncoderFactory& I();

    ImageEncoderFunc GetEncoder(const std::string& encoder_spec, const PixelFormat& fmt);

    ImageDecoderFunc GetDecoder(const std::string& encoder_spec, const PixelFormat& fmt);
};

}
