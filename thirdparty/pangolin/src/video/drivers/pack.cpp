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

#include <pangolin/video/drivers/pack.h>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/iostream_operators.h>

#ifdef DEBUGUNPACK
  #include <pangolin/utils/timer.h>
  #define TSTART() pangolin::basetime start,last,now; start = pangolin::TimeNow(); last = start;
  #define TGRABANDPRINT(...)  now = pangolin::TimeNow(); fprintf(stderr,"UNPACK: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, " %fms.\n",1000*pangolin::TimeDiff_s(last, now)); last = now;
  #define DBGPRINT(...) fprintf(stderr,"UNPACK: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr,"\n");
#else
  #define TSTART()
  #define TGRABANDPRINT(...)
  #define DBGPRINT(...)
#endif

namespace pangolin
{

PackVideo::PackVideo(std::unique_ptr<VideoInterface> &src_, PixelFormat out_fmt)
    : src(std::move(src_)), size_bytes(0), buffer(0)
{
    if( !src || out_fmt.channels != 1) {
        throw VideoException("PackVideo: Only supports single channel input.");
    }

    videoin.push_back(src.get());

    for(size_t s=0; s< src->Streams().size(); ++s) {
        const size_t w = src->Streams()[s].Width();
        const size_t h = src->Streams()[s].Height();

        // Check compatibility of formats
        const PixelFormat in_fmt = src->Streams()[s].PixFormat();
        if(in_fmt.channels > 1 || in_fmt.bpp > 16) {
            throw VideoException("PackVideo: Only supports one channel input.");
        }

        // round up to ensure enough bytes for packing
        const size_t pitch = (w*out_fmt.bpp)/ 8 + ((w*out_fmt.bpp) % 8 > 0? 1 : 0);
        streams.push_back(pangolin::StreamInfo( out_fmt, w, h, pitch, (unsigned char*)0 + size_bytes ));
        size_bytes += h*pitch;
    }

    buffer = new unsigned char[src->SizeBytes()];
}

PackVideo::~PackVideo()
{
    delete[] buffer;
}

//! Implement VideoInput::Start()
void PackVideo::Start()
{
    videoin[0]->Start();
}

//! Implement VideoInput::Stop()
void PackVideo::Stop()
{
    videoin[0]->Stop();
}

//! Implement VideoInput::SizeBytes()
size_t PackVideo::SizeBytes() const
{
    return size_bytes;
}

//! Implement VideoInput::Streams()
const std::vector<StreamInfo>& PackVideo::Streams() const
{
    return streams;
}

template<typename T>
void ConvertTo8bit(
    Image<unsigned char>& out,
    const Image<unsigned char>& in
) {
    for(size_t r=0; r<out.h; ++r) {
        T* pout = (T*)(out.ptr + r*out.pitch);
        uint8_t* pin = in.ptr + r*in.pitch;
        const uint8_t* pin_end = in.ptr + (r+1)*in.pitch;
        while(pin != pin_end) {
            *(pout++) = *(pin++);
        }
    }
}

template<typename T>
void ConvertTo10bit(
    Image<unsigned char>& out,
    const Image<unsigned char>& in
) {
    for(size_t r=0; r<out.h; ++r) {
        uint8_t* pout = out.ptr + r*out.pitch;
        T* pin = (T*)(in.ptr + r*in.pitch);
        const T* pin_end = (T*)(in.ptr + (r+1)*in.pitch);
        while(pin != pin_end) {
            uint64_t val = (*(pin++) & 0x00000003FF);
            val |= uint64_t(*(pin++) & 0x00000003FF) << 10;
            val |= uint64_t(*(pin++) & 0x00000003FF) << 20;
            val |= uint64_t(*(pin++) & 0x00000003FF) << 30;
            *(pout++) = uint8_t( val & 0x00000000FF);
            *(pout++) = uint8_t((val & 0x000000FF00) >> 8);
            *(pout++) = uint8_t((val & 0x0000FF0000) >> 16);
            *(pout++) = uint8_t((val & 0x00FF000000) >> 24);
            *(pout++) = uint8_t((val & 0xFF00000000) >> 32);
        }
    }
}

template<typename T>
void ConvertTo12bit(
    Image<unsigned char>& out,
    const Image<unsigned char>& in
) {
    for(size_t r=0; r<out.h; ++r) {
        uint8_t* pout = out.ptr + r*out.pitch;
        T* pin = (T*)(in.ptr + r*in.pitch);
        const T* pin_end = (T*)(in.ptr + (r+1)*in.pitch);
        while(pin != pin_end) {
            uint32_t val = (*(pin++) & 0x00000FFF);
            val |= uint32_t(*(pin++) & 0x00000FFF) << 12;
            *(pout++) = uint8_t( val & 0x000000FF);
            *(pout++) = uint8_t((val & 0x0000FF00) >> 8);
            *(pout++) = uint8_t((val & 0x00FF0000) >> 16);
        }
    }
}

void PackVideo::Process(unsigned char* image, const unsigned char* buffer)
{
    TSTART()
    for(size_t s=0; s<streams.size(); ++s) {
        const Image<unsigned char> img_in  = videoin[0]->Streams()[s].StreamImage(buffer);
        Image<unsigned char> img_out = Streams()[s].StreamImage(image);

        const int bits_out = Streams()[s].PixFormat().bpp;

        if(videoin[0]->Streams()[s].PixFormat().format == "GRAY16LE") {
            if(bits_out == 8) {
                ConvertTo8bit<uint16_t>(img_out, img_in);
            }else if( bits_out == 10) {
                ConvertTo10bit<uint16_t>(img_out, img_in);
            }else if( bits_out == 12){
                ConvertTo12bit<uint16_t>(img_out, img_in);
            }else{
                throw pangolin::VideoException("Unsupported bitdepths.");
            }
        }else{
                throw pangolin::VideoException("Unsupported input pix format.");
        }
    }
    TGRABANDPRINT("Packing took ")
}

//! Implement VideoInput::GrabNext()
bool PackVideo::GrabNext( unsigned char* image, bool wait )
{
    if(videoin[0]->GrabNext(buffer,wait)) {
        Process(image,buffer);
        return true;
    }else{
        return false;
    }
}

//! Implement VideoInput::GrabNewest()
bool PackVideo::GrabNewest( unsigned char* image, bool wait )
{
    if(videoin[0]->GrabNewest(buffer,wait)) {
        Process(image,buffer);
        return true;
    }else{
        return false;
    }
}

std::vector<VideoInterface*>& PackVideo::InputStreams()
{
    return videoin;
}

unsigned int PackVideo::AvailableFrames() const
{
    BufferAwareVideoInterface* vpi = dynamic_cast<BufferAwareVideoInterface*>(videoin[0]);
    if(!vpi)
    {
        pango_print_warn("Pack: child interface is not buffer aware.");
        return 0;
    }
    else
    {
        return vpi->AvailableFrames();
    }
}

bool PackVideo::DropNFrames(uint32_t n)
{
    BufferAwareVideoInterface* vpi = dynamic_cast<BufferAwareVideoInterface*>(videoin[0]);
    if(!vpi)
    {
        pango_print_warn("Pack: child interface is not buffer aware.");
        return false;
    }
    else
    {
        return vpi->DropNFrames(n);
    }
}

PANGOLIN_REGISTER_FACTORY(PackVideo)
{
    struct PackVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            std::unique_ptr<VideoInterface> subvid = pangolin::OpenVideo(uri.url);
            const std::string fmt = uri.Get("fmt", std::string("GRAY16LE") );
            return std::unique_ptr<VideoInterface>(
                new PackVideo(subvid, PixelFormatFromString(fmt) )
            );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<PackVideoFactory>(), 10, "pack");
}

}

#undef TSTART
#undef TGRABANDPRINT
#undef DBGPRINT
