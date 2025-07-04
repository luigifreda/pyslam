#include <fstream>
#include <memory>

#include <pangolin/image/typed_image.h>

namespace pangolin {

#pragma pack(push, 1)
struct packed12bit_image_header
{
    char magic[4];
    char fmt[16];
    size_t w, h;
};
#pragma pack(pop)

void SavePacked12bit(const Image<uint8_t>& image, const pangolin::PixelFormat& fmt, std::ostream& out, int /*compression_level*/)
{

  if (fmt.bpp != 16) {
    throw std::runtime_error("packed12bit currently only supported with 16bit input image");
  }

  const size_t dest_pitch = (image.w*12)/ 8 + ((image.w*12) % 8 > 0? 1 : 0);
  const size_t dest_size = image.h*dest_pitch;
  std::unique_ptr<uint8_t[]> output_buffer(new uint8_t[dest_size]);

    for(size_t r=0; r<image.h; ++r) {
        uint8_t* pout = output_buffer.get() + r*dest_pitch;
        uint16_t* pin = (uint16_t*)(image.ptr + r*image.pitch);
        const uint16_t* pin_end = (uint16_t*)(image.ptr + (r+1)*image.pitch);
        while(pin < pin_end) {
            uint32_t val = (*(pin++) & 0x00000FFF);
            val |= uint32_t(*(pin++) & 0x00000FFF) << 12;
            *(pout++) = uint8_t( val & 0x000000FF);
            *(pout++) = uint8_t((val & 0x0000FF00) >> 8);
            *(pout++) = uint8_t((val & 0x00FF0000) >> 16);
        }
    }

  packed12bit_image_header header;
  strncpy(header.magic,"P12B",4);
  strncpy(header.fmt, fmt.format.c_str(), sizeof(header.fmt));
  header.w = image.w;
  header.h = image.h;
  out.write((char*)&header, sizeof(header));
  out.write((char*)output_buffer.get(), dest_size);

}

TypedImage LoadPacked12bit(std::istream& in)
{
    // Read in header, uncompressed
    packed12bit_image_header header;
    in.read((char*)&header, sizeof(header));

    TypedImage img(header.w, header.h, PixelFormatFromString(header.fmt));

  if (img.fmt.bpp != 16) {
    throw std::runtime_error("packed12bit currently only supported with 16bit input image");
  }

  const size_t input_pitch = (img.w*12)/ 8 + ((img.w*12) % 8 > 0? 1 : 0);
  const size_t input_size = img.h*input_pitch;
    std::unique_ptr<uint8_t[]> input_buffer(new uint8_t[input_size]);

    in.read((char*)input_buffer.get(), input_size);

    for(size_t r=0; r<img.h; ++r) {
        uint16_t* pout = (uint16_t*)(img.ptr + r*img.pitch);
        uint8_t* pin = input_buffer.get() + r*input_pitch;
        const uint8_t* pin_end = input_buffer.get() + (r+1)*input_pitch;
        while(pin < pin_end) {
            uint32_t val = *(pin++);
            val |= uint32_t(*(pin++)) << 8;
            val |= uint32_t(*(pin++)) << 16;
            *(pout++) = uint16_t( val & 0x000FFF);
            *(pout++) = uint16_t((val & 0xFFF000) >> 12);
        }
    }

    return img;
}

}
