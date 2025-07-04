/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2013 Steven Lovegrove
 *
 * Permission is hereby granted, free of charge, to any person
 * obtaining a copy of this software and associated documentation
 * files (the "Software"), to deal in the Software without
 * restriction, including without limitation the rights to use,
 * copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the
 * Software is furnished to do so, subject to the following
 * conditions:
 *
 * The above copyright notice and this permission notice shall be
 * included in all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES
 * OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 * NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT
 * HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY,
 * WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING
 * FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR
 * OTHER DEALINGS IN THE SOFTWARE.
 */

#include <pangolin/image/image_io.h>

#include <fstream>

namespace pangolin {

// PNG
TypedImage LoadPng(std::istream& in);
void SavePng(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, bool top_line_first, int zlib_compression_level );

// JPG
TypedImage LoadJpg(std::istream& in);
void SaveJpg(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, float quality);

// PPM
TypedImage LoadPpm(std::istream& in);
void SavePpm(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, bool top_line_first);

// TGA
TypedImage LoadTga(std::istream& in);

// Pango
TypedImage LoadPango(const std::string& filename);
void SavePango(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, const std::string& filename, bool top_line_first);

// EXR
TypedImage LoadExr(std::istream& source);
void SaveExr(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, const std::string& filename, bool top_line_first);

// ZSTD (https://github.com/facebook/zstd)
TypedImage LoadZstd(std::istream& in);
void SaveZstd(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, int compression_level);

// https://github.com/lz4/lz4
TypedImage LoadLz4(std::istream& in);
void SaveLz4(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, int compression_level);

// packed 12 bit image (obtained from unpacked 16bit)
TypedImage LoadPacked12bit(std::istream& in);
void SavePacked12bit(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, int compression_level);

TypedImage LoadImage(std::istream& in, ImageFileType file_type)
{
    switch (file_type) {
    case ImageFileTypePng:
        return LoadPng(in);
    case ImageFileTypeJpg:
        return LoadJpg(in);
    case ImageFileTypePpm:
        return LoadPpm(in);
    case ImageFileTypeTga:
        return LoadTga(in);
    case ImageFileTypeZstd:
        return LoadZstd(in);
    case ImageFileTypeLz4:
        return LoadLz4(in);
    case ImageFileTypeP12b:
        return LoadPacked12bit(in);
    case ImageFileTypeExr:
        return LoadExr(in);
    default:
        throw std::runtime_error("Unable to load image file-type through std::istream");
    }
}

TypedImage LoadImage(const std::string& filename, ImageFileType file_type)
{
    switch (file_type) {
    case ImageFileTypePng:
    case ImageFileTypeJpg:
    case ImageFileTypePpm:
    case ImageFileTypeTga:
    case ImageFileTypeZstd:
    case ImageFileTypeLz4:
    case ImageFileTypeP12b:
    case ImageFileTypeExr:
    {
        std::ifstream ifs(filename, std::ios_base::in|std::ios_base::binary);
        return LoadImage(ifs, file_type);
    }
    case ImageFileTypePango:
        return LoadPango(filename);
    default:
        throw std::runtime_error("Unsupported image file type, '" + filename + "'");
    }
}

TypedImage LoadImage(const std::string& filename)
{
    ImageFileType file_type = FileType(filename);
    return LoadImage( filename, file_type );
}

void SaveImage(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, std::ostream& out, ImageFileType file_type, bool top_line_first, float quality)
{
    switch (file_type) {
    case ImageFileTypePng:
        // map quality [0..100] to PNG compression levels [0..9]
        return SavePng(image, fmt, out, top_line_first, int(quality*0.09));
    case ImageFileTypeJpg:
        return SaveJpg(image, fmt, out, quality);
    case ImageFileTypePpm:
        return SavePpm(image,fmt,out,top_line_first);
    case ImageFileTypeZstd:
        return SaveZstd(image,fmt,out, quality);
    case ImageFileTypeLz4:
        return SaveLz4(image,fmt,out, quality);
    case ImageFileTypeP12b:
        return SavePacked12bit(image,fmt,out, quality);
    default:
        throw std::runtime_error("Unable to save image file-type through std::istream");
    }
}


void SaveImage(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, const std::string& filename, ImageFileType file_type, bool top_line_first, float quality)
{
    switch (file_type) {
    case ImageFileTypePng:
    case ImageFileTypeJpg:
    case ImageFileTypePpm:
    case ImageFileTypeZstd:
    case ImageFileTypeLz4:
    case ImageFileTypeP12b:
    {
        std::ofstream ofs(filename, std::ios_base::binary);
        return SaveImage(image, fmt, ofs, file_type, top_line_first, quality);
    }
    case ImageFileTypeExr:
        return SaveExr(image, fmt, filename, top_line_first);
    case ImageFileTypePango:
        return SavePango(image, fmt, filename, top_line_first);
    default:
        throw std::runtime_error("Unsupported image file type, '" + filename + "'");
    }
}

void SaveImage(const Image<unsigned char>& image, const pangolin::PixelFormat& fmt, const std::string& filename, bool top_line_first, float quality)
{
    const std::string ext = FileLowercaseExtention(filename);
    const ImageFileType file_type = FileTypeExtension(ext);
    SaveImage(image, fmt, filename,file_type, top_line_first, quality);
}

void SaveImage(const TypedImage& image, const std::string& filename, bool top_line_first, float quality)
{
    SaveImage(image, image.fmt, filename, top_line_first, quality);
}

}
