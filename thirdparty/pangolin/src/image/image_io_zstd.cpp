
#include <fstream>
#include <memory>

#include <pangolin/image/typed_image.h>

#ifdef HAVE_ZSTD
#  include <zstd.h>
#endif

namespace pangolin {

#pragma pack(push, 1)
struct zstd_image_header
{
    char magic[4];
    char fmt[16];
    size_t w, h;
};
#pragma pack(pop)

void SaveZstd(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, int compression_level)
{
#ifdef HAVE_ZSTD
    // Write out header, uncompressed
    zstd_image_header header;
    strncpy(header.magic,"ZSTD",4);
    strncpy(header.fmt, fmt.format.c_str(), sizeof(header.fmt));
    header.w = image.w;
    header.h = image.h;
    out.write((char*)&header, sizeof(header));

    // Write out image data
    const size_t output_buffer_size = ZSTD_CStreamOutSize();
    std::unique_ptr<char[]> output_buffer(new char[output_buffer_size]);

    ZSTD_CStream* const cstream = ZSTD_createCStream();
    if (cstream==nullptr) {
        throw std::runtime_error("ZSTD_createCStream() error");
    }

    size_t const initResult = ZSTD_initCStream(cstream, compression_level);
    if (ZSTD_isError(initResult)) {
        throw std::runtime_error(FormatString("ZSTD_initCStream() error : %", ZSTD_getErrorName(initResult)));
    }

    const size_t row_size_bytes = (fmt.bpp * image.w)/8;

    for(size_t y=0; y < image.h; ++y) {
        ZSTD_inBuffer input = { image.RowPtr(y), row_size_bytes, 0 };

        while (input.pos < input.size) {
            ZSTD_outBuffer output = { output_buffer.get(), output_buffer_size, 0 };
            size_t left_to_read = ZSTD_compressStream(cstream, &output , &input);
            if (ZSTD_isError(left_to_read)) {
                throw std::runtime_error(FormatString("ZSTD_compressStream() error : %", ZSTD_getErrorName(left_to_read)));
            }
            out.write(output_buffer.get(), output.pos);
        }
    }

    ZSTD_outBuffer output = { output_buffer.get(), output_buffer_size, 0 };
    size_t const remainingToFlush = ZSTD_endStream(cstream, &output);   /* close frame */
    if (remainingToFlush) {
        throw std::runtime_error("not fully flushed");
    }
    out.write(output_buffer.get(), output.pos);

    ZSTD_freeCStream(cstream);
#else
    PANGOLIN_UNUSED(image);
    PANGOLIN_UNUSED(fmt);
    PANGOLIN_UNUSED(out);
    PANGOLIN_UNUSED(compression_level);
    throw std::runtime_error("Rebuild Pangolin for ZSTD support.");
#endif // HAVE_ZSTD
}

TypedImage LoadZstd(std::istream& in)
{
#ifdef HAVE_ZSTD
    // Read in header, uncompressed
    zstd_image_header header;
    in.read( (char*)&header, sizeof(header));

    TypedImage img(header.w, header.h, PixelFormatFromString(header.fmt));

    const size_t input_buffer_size = ZSTD_DStreamInSize();
    std::unique_ptr<char[]> input_buffer(new char[input_buffer_size]);

    ZSTD_DStream* dstream = ZSTD_createDStream();
    if(!dstream) {
        throw std::runtime_error("ZSTD_createDStream() error");
    }

    size_t read_size_hint = ZSTD_initDStream(dstream);
    if (ZSTD_isError(read_size_hint)) {
        throw std::runtime_error(FormatString("ZSTD_initDStream() error : % \n", ZSTD_getErrorName(read_size_hint)));
    }

    // Image represents our fixed buffer.
    ZSTD_outBuffer output = { img.ptr, img.SizeBytes(), 0 };

    while(read_size_hint)
    {
        const size_t read = in.readsome(input_buffer.get(), read_size_hint);
        ZSTD_inBuffer input = { input_buffer.get(), read, 0 };
        while (input.pos < input.size) {
            read_size_hint = ZSTD_decompressStream(dstream, &output , &input);
            if (ZSTD_isError(read_size_hint)) {
                throw std::runtime_error(FormatString("ZSTD_decompressStream() error : %", ZSTD_getErrorName(read_size_hint)));
            }
        }
    }

    ZSTD_freeDStream(dstream);

    return img;
#else
    PANGOLIN_UNUSED(in);
    throw std::runtime_error("Rebuild Pangolin for ZSTD support.");
#endif // HAVE_ZSTD
}

}
