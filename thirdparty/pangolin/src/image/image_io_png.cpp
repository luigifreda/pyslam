#include <pangolin/platform.h>

#include <fstream>
#include <pangolin/image/image_io.h>
#include <vector>

#ifdef HAVE_PNG
#  include <png.h>
#endif // HAVE_PNG

namespace pangolin {

#ifdef HAVE_PNG

PixelFormat PngFormat(png_structp png_ptr, png_infop info_ptr )
{
    const png_byte colour = png_get_color_type(png_ptr, info_ptr);
    const png_byte depth  = png_get_bit_depth(png_ptr, info_ptr);

    if( depth == 8 ) {
        if( colour == PNG_COLOR_MASK_COLOR ) {
            return PixelFormatFromString("RGB24");
        } else if( colour == (PNG_COLOR_MASK_COLOR | PNG_COLOR_MASK_ALPHA) ) {
            return PixelFormatFromString("RGBA32");
        } else if( colour == PNG_COLOR_MASK_ALPHA ) {
            return PixelFormatFromString("Y400A");
        } else {
            return PixelFormatFromString("GRAY8");
        }
    }else if( depth == 16 ) {
        if( colour == 0 ) {
            return PixelFormatFromString("GRAY16LE");
        }
    }

    throw std::runtime_error("Unsupported PNG format");
}

void PNGAPI PngWarningsCallback(png_structp /*png_ptr*/, png_const_charp /*warning_message*/)
{
    // Override default behaviour - don't do anything.
}

#define PNGSIGSIZE 8
bool pango_png_validate(std::istream& source)
{
    png_byte pngsig[PNGSIGSIZE];
    source.read((char*)pngsig, PNGSIGSIZE);
    return source.good() && png_sig_cmp(pngsig, 0, PNGSIGSIZE) == 0;
}

void pango_png_stream_read(png_structp pngPtr, png_bytep data, png_size_t length) {
    std::istream* s = (std::istream*)png_get_io_ptr(pngPtr);
    PANGO_ASSERT(s);
    s->read((char*)data, length);
}

void pango_png_stream_write(png_structp pngPtr, png_bytep data, png_size_t length) {
    std::ostream* s = (std::ostream*)png_get_io_ptr(pngPtr);
    PANGO_ASSERT(s);
    s->write((char*)data, length);
}

void pango_png_stream_write_flush(png_structp pngPtr)
{
    std::ostream* s = (std::ostream*)png_get_io_ptr(pngPtr);
    PANGO_ASSERT(s);
    s->flush();
}

#endif // HAVE_PNG


TypedImage LoadPng(std::istream& source)
{
#ifdef HAVE_PNG
    //so First, we validate our stream with the validate function I just mentioned
    if (!pango_png_validate(source)) {
        throw std::runtime_error("Not valid PNG header");
    }

    //set up initial png structs
    png_structp png_ptr = png_create_read_struct( PNG_LIBPNG_VER_STRING, (png_voidp)NULL, NULL, &PngWarningsCallback);
    if (!png_ptr) {
        throw std::runtime_error( "PNG Init error 1" );
    }

    png_infop info_ptr = png_create_info_struct(png_ptr);
    if (!info_ptr)  {
        png_destroy_read_struct(&png_ptr, (png_infopp)NULL, (png_infopp)NULL);
        throw std::runtime_error( "PNG Init error 2" );
    }

    png_infop end_info = png_create_info_struct(png_ptr);
    if (!end_info) {
        png_destroy_read_struct(&png_ptr, &info_ptr, (png_infopp)NULL);
        throw std::runtime_error( "PNG Init error 3" );
    }

    png_set_read_fn(png_ptr,(png_voidp)&source, pango_png_stream_read);

    png_set_sig_bytes(png_ptr, PNGSIGSIZE);

    // Setup transformation options
    if( png_get_bit_depth(png_ptr, info_ptr) == 1)  {
        //Unpack bools to bytes to ease loading.
        png_set_packing(png_ptr);
    } else if( png_get_bit_depth(png_ptr, info_ptr) < 8) {
        //Expand nonbool colour depths up to 8bpp
        png_set_expand_gray_1_2_4_to_8(png_ptr);
    }

    //Get rid of palette, by transforming it to RGB
    if(png_get_color_type(png_ptr, info_ptr) == PNG_COLOR_TYPE_PALETTE) {
        png_set_palette_to_rgb(png_ptr);
    }

    //read the file
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_SWAP_ENDIAN, NULL);

    if( png_get_interlace_type(png_ptr,info_ptr) != PNG_INTERLACE_NONE) {
        throw std::runtime_error( "Interlace not yet supported" );
    }

    const size_t w = png_get_image_width(png_ptr,info_ptr);
    const size_t h = png_get_image_height(png_ptr,info_ptr);
    const size_t pitch = png_get_rowbytes(png_ptr, info_ptr);

    TypedImage img(w, h, PngFormat(png_ptr, info_ptr), pitch);

    png_bytepp rows = png_get_rows(png_ptr, info_ptr);
    for( unsigned int r = 0; r < h; r++) {
        memcpy( img.ptr + pitch*r, rows[r], pitch );
    }
    png_destroy_read_struct(&png_ptr, &info_ptr, &end_info);

    return img;
#else
    PANGOLIN_UNUSED(source);
    throw std::runtime_error("Rebuild Pangolin for PNG support.");
#endif // HAVE_PNG
}

TypedImage LoadPng(const std::string& filename)
{
    std::ifstream f(filename);
    return LoadPng(f);
}

void SavePng(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& stream, bool top_line_first, int zlib_compression_level)
{
#ifdef HAVE_PNG
    // Check image has supported bit depth
    for(unsigned int i=1; i < fmt.channels; ++i) {
        if( fmt.channel_bits[i] != fmt.channel_bits[0] ) {
            throw std::runtime_error("PNG Saving only supported for images where each channel has the same bit depth.");
        }
    }

    png_structp png_ptr;
    png_infop info_ptr;

    // Initialize write structure
    png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    if (png_ptr == NULL) {
        throw std::runtime_error( "PNG Error: Could not allocate write struct." );
    }

    // Initialize info structure
    info_ptr = png_create_info_struct(png_ptr);
    if (info_ptr == NULL) {
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        throw std::runtime_error( "PNG Error: Could not allocate info struct." );
    }

    // Setup Exception handling
    if (setjmp(png_jmpbuf(png_ptr))) {
        png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
        png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
        throw std::runtime_error( "PNG Error: Error during png creation." );
    }

    png_set_compression_level(png_ptr, zlib_compression_level);

    png_set_write_fn(png_ptr,(png_voidp)&stream, pango_png_stream_write, pango_png_stream_write_flush);

    const int bit_depth = fmt.channel_bits[0];

    int colour_type;
    switch (fmt.channels) {
    case 1: colour_type = PNG_COLOR_TYPE_GRAY; break;
    case 2: colour_type = PNG_COLOR_TYPE_GRAY_ALPHA; break;
    case 3: colour_type = PNG_COLOR_TYPE_RGB; break;
    case 4: colour_type = PNG_COLOR_TYPE_RGBA; break;
    default:
        throw std::runtime_error( "PNG Error: unexpected image channel number");
    }

    // Write header
    png_set_IHDR(
        png_ptr, info_ptr, (png_uint_32)image.w, (png_uint_32)image.h, bit_depth, colour_type,
        PNG_INTERLACE_NONE, PNG_COMPRESSION_TYPE_DEFAULT, PNG_FILTER_TYPE_DEFAULT
    );

    // Setup rows to write:
    std::vector<png_bytep> rows(image.h);
    if(top_line_first) {
        for (unsigned int y = 0; y< image.h; y++) {
            rows[y] = image.ptr + y*image.pitch;
        }
    }else{
        for (unsigned int y = 0; y< image.h; y++) {
            rows[y] = image.ptr + (image.h-1-y)*image.pitch;
        }
    }
    png_set_rows(png_ptr,info_ptr, &rows[0]);

    // Write image data: switch to little-endian byte order, to match host.
    png_write_png(png_ptr,info_ptr, PNG_TRANSFORM_SWAP_ENDIAN, 0);

    // Free resources
    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
#else
    PANGOLIN_UNUSED(image);
    PANGOLIN_UNUSED(fmt);
    PANGOLIN_UNUSED(stream);
    PANGOLIN_UNUSED(top_line_first);
    PANGOLIN_UNUSED(zlib_compression_level);
    throw std::runtime_error("Rebuild Pangolin for PNG support.");
#endif // HAVE_PNG
}

}
