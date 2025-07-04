#include <fstream>
#include <pangolin/image/typed_image.h>

namespace pangolin {

PixelFormat PpmFormat(const std::string& strType, int num_colours)
{
    if(strType == "P5") {
        if(num_colours < 256) {
            return PixelFormatFromString("GRAY8");
        } else {
            return PixelFormatFromString("GRAY16LE");
        }
    }else if(strType == "P6") {
        return PixelFormatFromString("RGB24");
    }else{
        throw std::runtime_error("Unsupported PPM/PGM format");
    }
}

void PpmConsumeWhitespaceAndComments(std::istream& in)
{
    // TODO: Make a little more general / more efficient
    while( in.peek() == ' ' )  in.get();
    while( in.peek() == '\n' ) in.get();
    while( in.peek() == '#' )  in.ignore(4096, '\n');
}

TypedImage LoadPpm(std::istream& in)
{
    // Parse header
    std::string ppm_type = "";
    int num_colors = 0;
    int w = 0;
    int h = 0;

    in >> ppm_type;
    PpmConsumeWhitespaceAndComments(in);
    in >> w;
    PpmConsumeWhitespaceAndComments(in);
    in >> h;
    PpmConsumeWhitespaceAndComments(in);
    in >> num_colors;
    in.ignore(1,'\n');

    if(!in.fail() && w > 0 && h > 0) {
        TypedImage img(w, h, PpmFormat(ppm_type, num_colors) );

        // Read in data
        for(size_t r=0; r<img.h; ++r) {
            in.read( (char*)img.ptr + r*img.pitch, img.pitch );
        }
        if(!in.fail()) {
            return img;
        }
    }

    throw std::runtime_error("Unable to load PPM file.");
}

void SavePpm(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, bool top_line_first)
{
    // Setup header variables
    std::string ppm_type = "";
    int num_colors = 0;
    int w = (int)image.w;
    int h = (int)image.h;

    if(fmt.format == "GRAY8") {
        ppm_type = "P5";
        num_colors = 255;
    }else if(fmt.format == "GRAY16LE") {
        ppm_type = "P5";
        num_colors = 65535;
    }else if(fmt.format == "RGB24") {
        ppm_type = "P6";
        num_colors = 255;
    }else{
        throw std::runtime_error("Unsupported pixel format");
    }

    // Write header
    out << ppm_type;
    out << " ";
    out << w;
    out << " ";
    out << h;
    out << " ";
    out << num_colors;
    out << "\n";

    // Write out data
    if(top_line_first) {
        for(size_t r=0; r<image.h; ++r) {
            out.write( (char*)image.ptr + r*image.pitch, image.pitch );
        }
    }else{
        for(size_t r=0; r<image.h; ++r) {
            out.write( (char*)image.ptr + (image.h-1-r)*image.pitch, image.pitch );
        }
    }
}

}
