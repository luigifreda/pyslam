#include <algorithm>
#include <fstream>
#ifdef __linux__
#include <cinttypes>
#endif 

#include <pangolin/platform.h>

#include <pangolin/image/typed_image.h>

#ifdef HAVE_JPEG
#  include <jpeglib.h>
#  ifdef _WIN_
//   Undef windows Macro polution from jpeglib.h
#    undef LoadImage
#  endif
#endif // HAVE_JPEG


// Inspired by https://cs.stanford.edu/~acoates/jpegAndIOS.txt

namespace pangolin {

#ifdef HAVE_JPEG

void error_handler(j_common_ptr cinfo) {
    char msg[JMSG_LENGTH_MAX];
    (*(cinfo->err->format_message)) (cinfo, msg);
    throw std::runtime_error(msg);
}

const static size_t PANGO_JPEG_BUF_SIZE = 16384;

struct pango_jpeg_source_mgr {
    struct jpeg_source_mgr pub;
    std::istream* is;
    JOCTET*       buffer;
};

static void pango_jpeg_init_source(j_decompress_ptr /*cinfo*/) {
}

static boolean pango_jpeg_fill_input_buffer(j_decompress_ptr cinfo) {
    pango_jpeg_source_mgr* src = (pango_jpeg_source_mgr*)cinfo->src;

    src->is->read((char*)src->buffer, PANGO_JPEG_BUF_SIZE);
    size_t bytes = src->is->gcount();
    if (bytes == 0) {
        /* Insert a fake EOI marker */
        src->buffer[0] = (JOCTET) 0xFF;
        src->buffer[1] = (JOCTET) JPEG_EOI;
        bytes = 2;
    }
    src->pub.next_input_byte = src->buffer;
    src->pub.bytes_in_buffer = bytes;
    return TRUE;
}
static void pango_jpeg_skip_input_data(j_decompress_ptr cinfo, long num_bytes) {
    pango_jpeg_source_mgr* src = (pango_jpeg_source_mgr*)cinfo->src;
    if (num_bytes > 0) {
        while (num_bytes > (long)src->pub.bytes_in_buffer) {
            num_bytes -= (long)src->pub.bytes_in_buffer;
            pango_jpeg_fill_input_buffer(cinfo);
        }
        src->pub.next_input_byte += num_bytes;
        src->pub.bytes_in_buffer -= num_bytes;
    }
}
static void pango_jpeg_term_source(j_decompress_ptr cinfo) {
    // must seek backward so that future reads will start at correct place.
    pango_jpeg_source_mgr* src = (pango_jpeg_source_mgr*)cinfo->src;
    src->is->clear();
    src->is->seekg( src->is->tellg() - (std::streampos)src->pub.bytes_in_buffer );
}

static void pango_jpeg_set_source_mgr(j_decompress_ptr cinfo, std::istream& is) {
    pango_jpeg_source_mgr* src = nullptr;

    if (cinfo->src == 0)
    {
        cinfo->src = (struct jpeg_source_mgr *)(*cinfo->mem->alloc_small)
                ((j_common_ptr) cinfo, JPOOL_PERMANENT, sizeof(pango_jpeg_source_mgr));

        src = (pango_jpeg_source_mgr*) cinfo->src;
        src->buffer = (JOCTET *)(*cinfo->mem->alloc_small)
                ((j_common_ptr) cinfo, JPOOL_PERMANENT, PANGO_JPEG_BUF_SIZE*sizeof(JOCTET));
    }else{
        src = (pango_jpeg_source_mgr*) cinfo->src;
    }

    src->is = &is;
    src->pub.init_source = pango_jpeg_init_source;
    src->pub.fill_input_buffer = pango_jpeg_fill_input_buffer;
    src->pub.skip_input_data = pango_jpeg_skip_input_data;
    src->pub.resync_to_restart = jpeg_resync_to_restart; /* use default method */
    src->pub.term_source = pango_jpeg_term_source;
    src->pub.bytes_in_buffer = 0;
    src->pub.next_input_byte = 0;
}

struct pango_jpeg_destination_mgr {
    struct jpeg_destination_mgr pub; /* public fields */
    std::ostream* os; /* target stream */
    JOCTET * buffer;	   /* start of buffer */
};

void pango_jpeg_init_destination (j_compress_ptr cinfo) {
    pango_jpeg_destination_mgr* dest = (pango_jpeg_destination_mgr*) cinfo->dest;
    dest->buffer = (JOCTET *)(*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_IMAGE,
                                                         PANGO_JPEG_BUF_SIZE * sizeof(JOCTET));
    dest->pub.next_output_byte = dest->buffer;
    dest->pub.free_in_buffer = PANGO_JPEG_BUF_SIZE;
}

boolean pango_jpeg_empty_output_buffer(j_compress_ptr cinfo) {
    pango_jpeg_destination_mgr* dest = (pango_jpeg_destination_mgr*)cinfo->dest;

    dest->os->write((const char*)dest->buffer, PANGO_JPEG_BUF_SIZE);

    if (dest->os->fail()) {
        throw std::runtime_error("Couldn't write entire jpeg buffer to stream.");
    }

    dest->pub.next_output_byte = dest->buffer;
    dest->pub.free_in_buffer = PANGO_JPEG_BUF_SIZE;
    return TRUE;
}

void pango_jpeg_term_destination (j_compress_ptr cinfo) {
    pango_jpeg_destination_mgr* dest = (pango_jpeg_destination_mgr*) cinfo->dest;
    size_t datacount = PANGO_JPEG_BUF_SIZE - dest->pub.free_in_buffer;

    /* Write any data remaining in the buffer */
    if (datacount > 0) {
        dest->os->write((const char*)dest->buffer, datacount);
        if (dest->os->fail()) {
            throw std::runtime_error("Couldn't write remaining jpeg data to stream.");
        }
    }
    dest->os->flush();
}

void pango_jpeg_set_dest_mgr(j_compress_ptr cinfo, std::ostream& os) {
    pango_jpeg_destination_mgr* dest;

    if (cinfo->dest == NULL) {	/* first time for this JPEG object? */
        cinfo->dest = (struct jpeg_destination_mgr *)
                (*cinfo->mem->alloc_small) ((j_common_ptr) cinfo, JPOOL_PERMANENT,
                                            sizeof(pango_jpeg_destination_mgr));
    }

    dest = (pango_jpeg_destination_mgr*)cinfo->dest;
    dest->pub.init_destination = pango_jpeg_init_destination;
    dest->pub.empty_output_buffer = pango_jpeg_empty_output_buffer;
    dest->pub.term_destination = pango_jpeg_term_destination;
    dest->os = &os;
}

#endif // HAVE_JPEG

TypedImage LoadJpg(std::istream& is) {
#ifdef HAVE_JPEG
    TypedImage image;

    struct jpeg_decompress_struct cinfo;
    struct jpeg_error_mgr jerr;

    // Setup decompression structure
    cinfo.err = jpeg_std_error(&jerr);
    jerr.error_exit = error_handler;
    jpeg_create_decompress(&cinfo);
    pango_jpeg_set_source_mgr(&cinfo, is);

    // read info from header.
    int r = jpeg_read_header(&cinfo, TRUE);
    if (r != JPEG_HEADER_OK) {
        throw std::runtime_error("Failed to read JPEG header.");
    } else if (cinfo.num_components != 3 && cinfo.num_components != 1) {
        throw std::runtime_error("Unsupported number of color components");
    } else {
        jpeg_start_decompress(&cinfo);
        // resize storage if necessary
        PixelFormat fmt = PixelFormatFromString(cinfo.output_components == 3 ? "RGB24" : "GRAY8");
        image.Reinitialise(cinfo.output_width, cinfo.output_height, fmt);
        JSAMPARRAY imageBuffer = (*cinfo.mem->alloc_sarray)((j_common_ptr)&cinfo, JPOOL_IMAGE,
                                                            cinfo.output_width*cinfo.output_components, 1);
        for (size_t y = 0; y < cinfo.output_height; y++) {
            jpeg_read_scanlines(&cinfo, imageBuffer, 1);
            uint8_t* dstRow = (uint8_t*)image.RowPtr(y);
            memcpy(dstRow, imageBuffer[0], cinfo.output_width*cinfo.output_components);
        }
        jpeg_finish_decompress(&cinfo);
    }

    // clean up.
    jpeg_destroy_decompress(&cinfo);

    return image;
#else
    PANGOLIN_UNUSED(is);
    throw std::runtime_error("Rebuild Pangolin for JPEG support.");
#endif // HAVE_JPEG

}

TypedImage LoadJpg(const std::string& filename) {
    std::ifstream f(filename);
    return LoadJpg(f);
}

void SaveJpg(const Image<unsigned char>& img, const PixelFormat& fmt, std::ostream& os, float quality) {
#ifdef HAVE_JPEG
    const int iquality = (int)std::max(std::min(quality, 100.0f),0.0f);

    struct jpeg_compress_struct cinfo;
    struct jpeg_error_mgr       jerr;

    if (fmt.channels != 1 && fmt.channels != 3) {
        throw std::runtime_error("Unsupported number of image channels.");
    }
    if (fmt.bpp != 8 &&  fmt.bpp != 24) {
        throw std::runtime_error("Unsupported image depth.");
    }

    // set up compression structure
    cinfo.err = jpeg_std_error(&jerr);
    jpeg_create_compress(&cinfo);
    pango_jpeg_set_dest_mgr(&cinfo, os);

    cinfo.image_width      = img.w;
    cinfo.image_height     = img.h;
    cinfo.input_components = fmt.channels;
    if (fmt.channels == 3) {
        cinfo.in_color_space   = JCS_RGB;
    } else {
        cinfo.in_color_space   = JCS_GRAYSCALE;
    }

    jpeg_set_defaults(&cinfo);
    jpeg_set_quality(&cinfo, iquality, (boolean)true);
    jpeg_start_compress(&cinfo, (boolean)true);

    JSAMPROW row;
    while (cinfo.next_scanline < cinfo.image_height) {
        row = (JSAMPROW)((char*)img.RowPtr(cinfo.next_scanline));
        jpeg_write_scanlines(&cinfo, &row, 1);
    }

    jpeg_finish_compress(&cinfo);
    jpeg_destroy_compress(&cinfo);
#else
    PANGOLIN_UNUSED(img);
    PANGOLIN_UNUSED(fmt);
    PANGOLIN_UNUSED(os);
    PANGOLIN_UNUSED(quality);
    throw std::runtime_error("Rebuild Pangolin for JPEG support.");
#endif // HAVE_JPEG
}

void SaveJpg(const Image<unsigned char>& img, const PixelFormat& fmt, const std::string& filename, float quality) {
    std::ofstream f(filename);
    SaveJpg(img, fmt, f, quality);
}

}
