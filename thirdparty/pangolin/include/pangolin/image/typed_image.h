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

#pragma once

#include <pangolin/image/managed_image.h>
#include <pangolin/image/pixel_format.h>

namespace pangolin {

struct TypedImage : public ManagedImage<unsigned char>
{
    typedef ManagedImage<unsigned char> Base;

    inline TypedImage()
        : Base()
    {
    }

    inline TypedImage(size_t w, size_t h, const PixelFormat& fmt)
        : Base(w,h,w*fmt.bpp/8), fmt(fmt)
    {
    }

    inline TypedImage(size_t w, size_t h, const PixelFormat& fmt, size_t pitch )
        : Base(w,h, pitch), fmt(fmt)
    {
    }

    inline
    void Reinitialise(size_t w, size_t h, const PixelFormat& fmt)
    {
        Base::Reinitialise(w, h, w*fmt.bpp/8);
        this->fmt = fmt;
    }

    inline
    void Reinitialise(size_t w, size_t h, const PixelFormat& fmt, size_t pitch)
    {
        Base::Reinitialise(w, h, pitch);
        this->fmt = fmt;
    }

    // Not copy constructable
    inline
    TypedImage( const TypedImage& other ) = delete;

    // Move constructor
    inline
    TypedImage(TypedImage&& img)
    {
        *this = std::move(img);
    }

    // Move asignment
    inline
    void operator=(TypedImage&& img)
    {
        fmt = img.fmt;
        Base::operator =( std::move(img));
    }


    PixelFormat fmt;
};

}
