/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011-2013 Steven Lovegrove
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

#include <pangolin/platform.h>
#include <string>

namespace pangolin
{

struct PANGOLIN_EXPORT PixelFormat
{
    // Previously, VideoInterface::PixFormat returned a string.
    // For compatibility, make this string convertable
    inline operator std::string() const { return format; }
    
    std::string  format;
    unsigned int channels;
    unsigned int channel_bits[4]; //Of the data type
    unsigned int bpp; //Of the data type
    unsigned int channel_bit_depth; //Of the data
    bool planar;
};


//! Return Pixel Format properties given string specification in
//! FFMPEG notation.
PANGOLIN_EXPORT
PixelFormat PixelFormatFromString(const std::string& format);

////////////////////////////////////////////////////////////////////
/// Deprecated aliases for above

PANGOLIN_DEPRECATED
typedef PixelFormat VideoPixelFormat;
PANGOLIN_DEPRECATED
inline PixelFormat VideoFormatFromString(const std::string& format) {
    return PixelFormatFromString(format);
}

}
