/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Steven Lovegrove
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

#pragma once

#include <pangolin/image/image.h>
#include <pangolin/image/pixel_format.h>

namespace pangolin {

class PANGOLIN_EXPORT StreamInfo
{
public:
    inline StreamInfo()
        : fmt(PixelFormatFromString("GRAY8")) {}

    inline StreamInfo(PixelFormat fmt, const Image<unsigned char> img_offset )
        : fmt(fmt), img_offset(img_offset) {}

    inline StreamInfo(PixelFormat fmt, size_t w, size_t h, size_t pitch, unsigned char* offset = 0)
        : fmt(fmt), img_offset(offset,w,h,pitch) {}

    //! Format representing how image is laid out in memory
    inline const PixelFormat &PixFormat() const { return fmt; }

    //! Image width in pixels
    inline size_t Width() const { return img_offset.w; }

    //! Image height in pixels
    inline size_t Height() const { return img_offset.h; }

    inline double Aspect() const { return (double)Width() / (double)Height(); }

    //! Pitch: Number of bytes between one image row and the next
    inline size_t Pitch() const { return img_offset.pitch; }

    //! Number of contiguous bytes in memory that the image occupies
    inline size_t RowBytes() const {
        // Row size without padding
        return (fmt.bpp*img_offset.w)/8;
    }

    //! Returns true iff image contains padding or stridded access
    //! This implies that the image data is not contiguous in memory.
    inline bool IsPitched() const {
        return Pitch() != RowBytes();
    }

    //! Number of contiguous bytes in memory that the image occupies
    inline size_t SizeBytes() const {
        return (img_offset.h-1) * img_offset.pitch + RowBytes();
    }

    //! Offset in bytes relative to start of frame buffer
    inline unsigned char* Offset() const { return img_offset.ptr; }

    //! Return Image wrapper around raw base pointer
    inline Image<unsigned char> StreamImage(unsigned char* base_ptr) const {
        Image<unsigned char> img = img_offset;
        img.ptr += (size_t)base_ptr;
        return img;
    }

    //! Return Image wrapper around raw base pointer
    inline const Image<unsigned char> StreamImage(const unsigned char* base_ptr) const {
        Image<unsigned char> img = img_offset;
        img.ptr += (size_t)base_ptr;
        return img;
    }

protected:
    PixelFormat fmt;
    Image<unsigned char> img_offset;
};

}
