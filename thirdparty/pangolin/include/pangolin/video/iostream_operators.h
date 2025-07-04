/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2015 Steven Lovegrove
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

#include <iostream>
#include <cctype>

#include <pangolin/video/video_exception.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/video/stream_info.h>

namespace pangolin
{

struct PANGOLIN_EXPORT Point
{
    inline Point() : x(0), y(0) {}
    inline Point(size_t x, size_t y) : x(x), y(y) {}
    size_t x;
    size_t y;
};

typedef Point ImageDim;

struct PANGOLIN_EXPORT ImageRoi
{
    inline ImageRoi() : x(0), y(0), w(0), h(0) {}
    inline ImageRoi(size_t x, size_t y, size_t w, size_t h) : x(x), y(y), w(w), h(h) {}
    size_t x; size_t y;
    size_t w; size_t h;
};

inline std::istream& operator>> (std::istream &is, ImageDim &dim)
{
    if(std::isdigit(is.peek()) ) {
        // Expect 640x480, 640*480, ...
        is >> dim.x; is.get(); is >> dim.y;
    }else{
        // Expect 'VGA', 'QVGA', etc
        std::string sdim;
        is >> sdim;
        ToUpper(sdim);

        if( !sdim.compare("QQVGA") ) {
            dim = ImageDim(160,120);
        }else if( !sdim.compare("HQVGA") ) {
            dim = ImageDim(240,160);
        }else if( !sdim.compare("QVGA") ) {
            dim = ImageDim(320,240);
        }else if( !sdim.compare("WQVGA") ) {
            dim = ImageDim(360,240);
        }else if( !sdim.compare("HVGA") ) {
            dim = ImageDim(480,320);
        }else if( !sdim.compare("VGA") ) {
            dim = ImageDim(640,480);
        }else if( !sdim.compare("WVGA") ) {
            dim = ImageDim(720,480);
        }else if( !sdim.compare("SVGA") ) {
            dim = ImageDim(800,600);
        }else if( !sdim.compare("DVGA") ) {
            dim = ImageDim(960,640);
        }else if( !sdim.compare("WSVGA") ) {
            dim = ImageDim(1024,600);
        }else{
            throw VideoException("Unrecognised image-size string.");
        }
    }
    return is;
}

inline std::istream& operator>> (std::istream &is, ImageRoi &roi)
{
    is >> roi.x; is.get(); is >> roi.y; is.get();
    is >> roi.w; is.get(); is >> roi.h;
    return is;
}

inline std::istream& operator>> (std::istream &is, PixelFormat& fmt)
{
    std::string sfmt;
    is >> sfmt;
    fmt = PixelFormatFromString(sfmt);
    return is;
}

inline std::istream& operator>> (std::istream &is, Image<unsigned char>& img)
{
    size_t offset;
    is >> offset; is.get();
    img.ptr = (unsigned char*)(offset);
    is >> img.w; is.get();
    is >> img.h; is.get();
    is >> img.pitch;
    return is;
}

inline std::istream& operator>> (std::istream &is, StreamInfo &stream)
{
    PixelFormat fmt;
    Image<unsigned char> img_offset;
    is >> img_offset; is.get();
    is >> fmt;
    stream = StreamInfo(fmt, img_offset);
    return is;
}

}
