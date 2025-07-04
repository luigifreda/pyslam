#include <fstream>
#include <pangolin/image/typed_image.h>

namespace pangolin {

TypedImage LoadImage(
    const std::string& filename,
    const PixelFormat& raw_fmt,
    size_t raw_width, size_t raw_height, size_t raw_pitch
) {
    TypedImage img(raw_width, raw_height, raw_fmt, raw_pitch);

    // Read from file, row at a time.
    std::ifstream bFile( filename.c_str(), std::ios::in | std::ios::binary );
    for(size_t r=0; r<img.h; ++r) {
        bFile.read( (char*)img.ptr + r*img.pitch, img.pitch );
        if(bFile.fail()) {
            pango_print_warn("Unable to read raw image file to completion.");
            break;
        }
    }
    return img;
}

}
