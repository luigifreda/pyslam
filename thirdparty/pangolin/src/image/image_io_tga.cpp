#include <pangolin/platform.h>

#include <pangolin/image/image_io.h>

namespace pangolin {

PixelFormat TgaFormat(int depth, int color_type, int color_map)
{
    if(color_map == 0) {
        if(color_type == 2) {
            // Colour
            if(depth == 24) {
                return PixelFormatFromString("RGB24");
            }else if(depth == 32) {
                return PixelFormatFromString("RGBA32");
            }
        }else if(color_type == 3){
            // Greyscale
            if(depth == 8) {
                return PixelFormatFromString("GRAY8");
            }else if(depth == 16) {
                return PixelFormatFromString("Y400A");
            }
        }
    }
    throw std::runtime_error("Unsupported TGA format");
}

TypedImage LoadTga(std::istream& in)
{
    unsigned char type[4];
    unsigned char info[6];

    in.read((char*)type, 3*sizeof(char));
    in.seekg(12);
    in.read((char*)info,6*sizeof(char));

    const int width  = info[0] + (info[1] * 256);
    const int height = info[2] + (info[3] * 256);

    if(in.good()) {
        TypedImage img(width, height, TgaFormat(info[4], type[2], type[1]) );

        //read in image data
        const size_t data_size = img.h * img.pitch;
        in.read((char*)img.ptr, sizeof(char)*data_size);
        return img;
    }

    throw std::runtime_error("Unable to load TGA file");
}

}
