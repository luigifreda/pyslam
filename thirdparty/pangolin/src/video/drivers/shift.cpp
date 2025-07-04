/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2014 Steven Lovegrove
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

#include <pangolin/video/drivers/shift.h>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/iostream_operators.h>

namespace pangolin
{

ShiftVideo::ShiftVideo(std::unique_ptr<VideoInterface> &src_, PixelFormat out_fmt, int shift_right_bits, unsigned int mask)
    : src(std::move(src_)), size_bytes(0), buffer(0), shift_right_bits(shift_right_bits), mask(mask)
{
    if(!src) {
        throw VideoException("ShiftVideo: VideoInterface in must not be null");
    }
    videoin.push_back(src.get());

    for(size_t s=0; s< src->Streams().size(); ++s) {
        const size_t w = src->Streams()[s].Width();
        const size_t h = src->Streams()[s].Height();

        // Check compatibility of formats
        const PixelFormat in_fmt = src->Streams()[s].PixFormat();
        if(in_fmt.channels != out_fmt.channels) {
            throw VideoException("ShiftVideo: output format is not compatible with input format for shifting.");
        }
        if(out_fmt.channels > 1 || out_fmt.bpp != 8 || in_fmt.bpp != 16) {
            // TODO: Remove restriction
            throw VideoException("ShiftVideo: currently only supports one channel input formats of 16 bits.");
        }

        streams.push_back(pangolin::StreamInfo( out_fmt, w, h, w*out_fmt.bpp / 8, (unsigned char*)0 + size_bytes ));
        size_bytes += w*h*out_fmt.bpp / 8;
    }

    buffer = new unsigned char[src->SizeBytes()];
}

ShiftVideo::~ShiftVideo()
{
    delete[] buffer;
}

//! Implement VideoInput::Start()
void ShiftVideo::Start()
{
    videoin[0]->Start();
}

//! Implement VideoInput::Stop()
void ShiftVideo::Stop()
{
    videoin[0]->Stop();
}

//! Implement VideoInput::SizeBytes()
size_t ShiftVideo::SizeBytes() const
{
    return size_bytes;
}

//! Implement VideoInput::Streams()
const std::vector<StreamInfo>& ShiftVideo::Streams() const
{
    return streams;
}

void DoShift16to8(
    Image<unsigned char>& out,
    const Image<unsigned char>& in,
    int shift_right_bits,
    unsigned int mask
) {
    for(size_t y=0; y<out.h; ++y) {
        for(size_t x=0; x<out.w; ++x) {
            const unsigned short vin = ((unsigned short*)(in.ptr + y * in.pitch))[x];
            out.ptr[y*out.pitch+x] = (vin >> shift_right_bits) & mask;
        }
    }
}

//! Implement VideoInput::GrabNext()
bool ShiftVideo::GrabNext( unsigned char* image, bool wait )
{
    if(videoin[0]->GrabNext(buffer,wait)) {
        for(size_t s=0; s<streams.size(); ++s) {
            Image<unsigned char> img_in  = videoin[0]->Streams()[s].StreamImage(buffer);
            Image<unsigned char> img_out = Streams()[s].StreamImage(image);
            DoShift16to8(img_out, img_in, shift_right_bits, mask);
        }
        return true;
    }else{
        return false;
    }
}

//! Implement VideoInput::GrabNewest()
bool ShiftVideo::GrabNewest( unsigned char* image, bool wait )
{
    if(videoin[0]->GrabNewest(buffer,wait)) {
        for(size_t s=0; s<streams.size(); ++s) {
            Image<unsigned char> img_in  = videoin[0]->Streams()[s].StreamImage(buffer);
            Image<unsigned char> img_out = Streams()[s].StreamImage(image);
            DoShift16to8(img_out, img_in, shift_right_bits, mask);
        }
        return true;
    }else{
        return false;
    }
}

std::vector<VideoInterface*>& ShiftVideo::InputStreams()
{
    return videoin;
}

PANGOLIN_REGISTER_FACTORY(ShiftVideo)
{
    struct ShiftVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            const int shift_right = uri.Get<int>("shift", 0);
            const int mask = uri.Get<int>("mask",  0xffff);

            std::unique_ptr<VideoInterface> subvid = pangolin::OpenVideo(uri.url);
            return std::unique_ptr<VideoInterface>(
                new ShiftVideo(subvid, PixelFormatFromString("GRAY8"), shift_right, mask)
            );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<ShiftVideoFactory>(), 10, "shift");
}

}
