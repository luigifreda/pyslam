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

#include <pangolin/video/drivers/mirror.h>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/iostream_operators.h>

namespace pangolin
{


MirrorVideo::MirrorVideo(std::unique_ptr<VideoInterface>& src, const std::vector<MirrorOptions>& flips)
    : videoin(std::move(src)), flips(flips), size_bytes(0),buffer(0)
{
    if(!videoin) {
        throw VideoException("MirrorVideo: VideoInterface in must not be null");
    }

    inputs.push_back(videoin.get());

    for(size_t i=0;i<videoin->Streams().size();i++)
        switch (flips[i]) {
            case MirrorOptionsFlipX:
            case MirrorOptionsFlipY:
            case MirrorOptionsFlipXY:
            case MirrorOptionsNone:
            streams.push_back(videoin->Streams()[i]);
            break;

            case MirrorOptionsTranspose:
            case MirrorOptionsRotateCW:
            case MirrorOptionsRotateCCW:

            unsigned char*ptr=videoin->Streams()[i].Offset();
            size_t w=videoin->Streams()[i].Height();
            size_t h=videoin->Streams()[i].Width();
            size_t Bpp=videoin->Streams()[i].PixFormat().bpp / 8;

            streams.emplace_back(videoin->Streams()[i].PixFormat(),pangolin::Image<unsigned char>(ptr,w,h,w*Bpp));
            break;
        };

    size_bytes = videoin->SizeBytes();
    buffer = new unsigned char[size_bytes];
}

MirrorVideo::~MirrorVideo()
{
    delete[] buffer;
}

//! Implement VideoInput::Start()
void MirrorVideo::Start()
{
    videoin->Start();
}

//! Implement VideoInput::Stop()
void MirrorVideo::Stop()
{
    videoin->Stop();
}

//! Implement VideoInput::SizeBytes()
size_t MirrorVideo::SizeBytes() const
{
    return size_bytes;
}

//! Implement VideoInput::Streams()
const std::vector<StreamInfo>& MirrorVideo::Streams() const
{
    return streams;
}

void PitchedImageCopy(
    Image<unsigned char>& img_out,
    const Image<unsigned char>& img_in,
    size_t bytes_per_pixel
) {
    if( img_out.w != img_in.w || img_out.h != img_in.h ) {
        throw std::runtime_error("PitchedImageCopy: Incompatible image sizes");
    }

    for(size_t y=0; y < img_out.h; ++y) {
        std::memcpy(img_out.RowPtr((int)y), img_in.RowPtr((int)y), bytes_per_pixel * img_in.w);
    }
}

void FlipY(
    Image<unsigned char>& img_out,
    const Image<unsigned char>& img_in,
    size_t bytes_per_pixel
) {
    if( img_out.w != img_in.w || img_out.h != img_in.h ) {
        throw std::runtime_error("FlipY: Incompatible image sizes");
    }

    for(size_t y_out=0; y_out < img_out.h; ++y_out) {
        const size_t y_in = (img_in.h-1) - y_out;
        std::memcpy(img_out.RowPtr((int)y_out), img_in.RowPtr((int)y_in), bytes_per_pixel * img_in.w);
    }
}

template <typename T>
void ChainSwap2(T& a, T& b)
{
    T t = a;
    a = b;
    b = t;
}

template <typename T>
void ChainSwap4(T& a, T& b, T& c, T& d)
{
    T t = a;
    a = b;
    b = c;
    c = d;
    d = t;
}

template <size_t BPP, size_t TSZ>
void TiledFlipX(Image<unsigned char>& img_out, const Image<unsigned char>& img_in)
{
    const size_t w = img_in.w;
    const size_t h = img_in.h;

    typedef struct
    {
        unsigned char d[BPP];
    } T;
    T d[TSZ][TSZ];

    for(size_t xin = 0; xin < w; xin += TSZ)
        for(size_t yin = 0; yin < h; yin += TSZ)
        {
            const size_t xspan = std::min(TSZ, w - xin);
            const size_t yspan = std::min(TSZ, h - yin);
            const size_t xout = w - xin - TSZ;
            const size_t yout = yin;

            for(size_t y = 0; y < yspan; y++)
                memcpy(d[y], img_in.RowPtr(yin + y) + xin * BPP, xspan * BPP);

            for(size_t y = 0; y < TSZ; y++)
                for(size_t x = 0; x < TSZ / 2; x++)
                    ChainSwap2(d[y][x], d[y][TSZ - 1 - x]);

            for(size_t y = 0; y < yspan; y++)
                memcpy(img_out.RowPtr(yout + y) + (xout + TSZ - xspan) * BPP, d[y] + TSZ - xspan, xspan * BPP);
        }
}

template <size_t BPP, size_t TSZ>
void TiledRotate180(Image<unsigned char>& img_out, const Image<unsigned char>& img_in)
{
    static_assert(!(TSZ & 1), "Tilesize must be even.");

    const size_t w = img_in.w;
    const size_t h = img_in.h;

    typedef struct
    {
        unsigned char d[BPP];
    } T;
    T d[TSZ][TSZ];

    for(size_t xin = 0; xin < w; xin += TSZ)
        for(size_t yin = 0; yin < h; yin += TSZ)
        {
            const size_t xspan = std::min(TSZ, w - xin);
            const size_t yspan = std::min(TSZ, h - yin);
            const size_t xout = w - xin - TSZ;
            const size_t yout = h - yin - TSZ;

            for(size_t y = 0; y < yspan; y++)
                memcpy(d[y], img_in.RowPtr(yin + y) + xin * BPP, xspan * BPP);

            for(size_t y = 0; y < TSZ / 2; y++)
                for(size_t x = 0; x < TSZ; x++)
                    ChainSwap2(d[y][x], d[TSZ - 1 - y][TSZ - 1 - x]);

            for(size_t y = TSZ - yspan; y < TSZ; y++)
                memcpy(img_out.RowPtr(yout + y) + (xout + TSZ - xspan) * BPP, d[y] + TSZ - xspan, xspan * BPP);
        }
}

template <size_t BPP, size_t TSZ>
void TiledTranspose(Image<unsigned char>& img_out, const Image<unsigned char>& img_in)
{
    const size_t w = img_in.w;
    const size_t h = img_in.h;

    typedef struct
    {
        unsigned char d[BPP];
    } T;
    T d[TSZ][TSZ];

    for(size_t xin = 0; xin < w; xin += TSZ)
        for(size_t yin = 0; yin < h; yin += TSZ)
        {
            const size_t xspan = std::min(TSZ, w - xin);
            const size_t yspan = std::min(TSZ, h - yin);
            const size_t dmin = std::min(xspan, yspan);
            const size_t dmax = std::max(xspan, yspan);
            const size_t xout = yin;
            const size_t yout = xin;

            for(size_t y = 0; y < yspan; y++)
                memcpy(d[y], img_in.RowPtr(yin + y) + xin * BPP, xspan * BPP);

            for(size_t x = 0; x < dmin; x++)
                for(size_t y = x + 1; y < dmax; y++)
                    ChainSwap2(d[x][y], d[y][x]);

            for(size_t y = 0; y < xspan; y++)
                memcpy(img_out.RowPtr(yout + y) + xout * BPP, d[y], yspan * BPP);
        }
}

template <size_t BPP, size_t TSZ>
void TiledRotateCW(Image<unsigned char>& img_out, const Image<unsigned char>& img_in)
{
    static_assert(!(TSZ & 1), "Tilesize must be even.");

    const size_t w = img_in.w;
    const size_t h = img_in.h;

    typedef struct
    {
        unsigned char d[BPP];
    } T;
    T d[TSZ][TSZ];

    for(size_t xin = 0; xin < w; xin += TSZ)
        for(size_t yin = 0; yin < h; yin += TSZ)
        {
            const size_t xspan = std::min(TSZ, w - xin);
            const size_t yspan = std::min(TSZ, h - yin);
            const size_t xout = h - yin - TSZ;
            const size_t yout = xin;

            for(size_t y = 0; y < yspan; y++)
                memcpy(d[y], img_in.RowPtr(yin + y) + xin * BPP, xspan * BPP);

            for(size_t y = 0; y < TSZ / 2; y++)
                for(size_t x = 0; x < TSZ / 2; x++)
                    ChainSwap4(d[TSZ - 1 - x][y], d[TSZ - 1 - y][TSZ - 1 - x], d[x][TSZ - 1 - y], d[y][x]);

            for(size_t y = 0; y < xspan; y++)
                memcpy(img_out.RowPtr(yout + y) + (xout + TSZ - yspan) * BPP, d[y] + TSZ - yspan, yspan * BPP);
        }
}

template <size_t BPP, size_t TSZ>
void TiledRotateCCW(Image<unsigned char>& img_out, const Image<unsigned char>& img_in)
{
    static_assert(!(TSZ & 1), "Tilesize must be even.");

    const size_t w = img_in.w;
    const size_t h = img_in.h;

    typedef struct
    {
        unsigned char d[BPP];
    } T;
    T d[TSZ][TSZ];

    for(size_t xin = 0; xin < w; xin += TSZ)
        for(size_t yin = 0; yin < h; yin += TSZ)
        {
            const size_t xspan = std::min(TSZ, w - xin);
            const size_t yspan = std::min(TSZ, h - yin);
            const size_t xout = yin;
            const size_t yout = w - xin - TSZ;

            for(size_t y = 0; y < yspan; y++)
                memcpy(d[y], img_in.RowPtr(yin + y) + xin * BPP, xspan * BPP);

            for(size_t y = 0; y < TSZ / 2; y++)
                for(size_t x = 0; x < TSZ / 2; x++)
                    ChainSwap4(d[y][x], d[x][TSZ - 1 - y], d[TSZ - 1 - y][TSZ - 1 - x], d[TSZ - 1 - x][y]);

            for(size_t y = TSZ - xspan; y < TSZ; y++)
                memcpy(img_out.RowPtr(yout + y) + xout * BPP, d[y], yspan * BPP);
        }
}

void FlipX(Image<unsigned char>& img_out, const Image<unsigned char>& img_in, size_t bytes_per_pixel)
{
    if(bytes_per_pixel == 1)
        TiledFlipX<1, 160>(img_out, img_in);
    else if(bytes_per_pixel == 2)
        TiledFlipX<2, 120>(img_out, img_in);
    else if(bytes_per_pixel == 3)
        TiledFlipX<3, 80>(img_out, img_in);
    else if(bytes_per_pixel == 4)
        TiledFlipX<4, 80>(img_out, img_in);
    else if(bytes_per_pixel == 6)
        TiledFlipX<6, 64>(img_out, img_in);
    else {
        for(size_t y = 0; y < img_out.h; ++y) {
            for(size_t x = 0; x < img_out.w; ++x) {
                memcpy(img_out.RowPtr((int)y) + (img_out.w - 1 - x) * bytes_per_pixel,
                       img_in.RowPtr((int)y) + x * bytes_per_pixel,
                       bytes_per_pixel);
            }
        }
    }
}

void FlipXY(Image<unsigned char>& img_out, const Image<unsigned char>& img_in, size_t bytes_per_pixel)
{
    if(bytes_per_pixel == 1)
        TiledRotate180<1, 160>(img_out, img_in);
    else if(bytes_per_pixel == 2)
        TiledRotate180<2, 120>(img_out, img_in);
    else if(bytes_per_pixel == 3)
        TiledRotate180<3, 80>(img_out, img_in);
    else if(bytes_per_pixel == 4)
        TiledRotate180<4, 80>(img_out, img_in);
    else if(bytes_per_pixel == 6)
        TiledRotate180<6, 64>(img_out, img_in);
    else {
        for(size_t y_out = 0; y_out < img_out.h; ++y_out) {
            for(size_t x = 0; x < img_out.w; ++x) {
                const size_t y_in = (img_in.h - 1) - y_out;
                memcpy(img_out.RowPtr((int)y_out) + (img_out.w - 1 - x) * bytes_per_pixel,
                       img_in.RowPtr((int)y_in) + x * bytes_per_pixel,
                       bytes_per_pixel);
            }
        }
    }
}

void RotateCW(Image<unsigned char>& img_out, const Image<unsigned char>& img_in, size_t bytes_per_pixel)
{
    if(bytes_per_pixel == 1)
        TiledRotateCW<1, 160>(img_out, img_in);
    else if(bytes_per_pixel == 2)
        TiledRotateCW<2, 120>(img_out, img_in);
    else if(bytes_per_pixel == 3)
        TiledRotateCW<3, 80>(img_out, img_in);
    else if(bytes_per_pixel == 4)
        TiledRotateCW<4, 80>(img_out, img_in);
    else if(bytes_per_pixel == 6)
        TiledRotateCW<6, 64>(img_out, img_in);
    else {
        for(size_t yout = 0; yout < img_out.h; ++yout)
            for(size_t xout = 0; xout < img_out.w; ++xout) {
                size_t xin = yout;
                size_t yin = img_out.w - 1 - xout;
                memcpy(img_out.RowPtr((int)yout) + xout * bytes_per_pixel,
                       img_in.RowPtr((int)yin) + xin * bytes_per_pixel,
                       bytes_per_pixel);
            }
    }
}

void Transpose(Image<unsigned char>& img_out, const Image<unsigned char>& img_in, size_t bytes_per_pixel)
{
    if(bytes_per_pixel == 1)
        TiledTranspose<1, 160>(img_out, img_in);
    else if(bytes_per_pixel == 2)
        TiledTranspose<2, 120>(img_out, img_in);
    else if(bytes_per_pixel == 3)
        TiledTranspose<3, 80>(img_out, img_in);
    else if(bytes_per_pixel == 4)
        TiledTranspose<4, 80>(img_out, img_in);
    else if(bytes_per_pixel == 6)
        TiledTranspose<6, 64>(img_out, img_in);
    else {
        for(size_t yout = 0; yout < img_out.h; ++yout)
            for(size_t xout = 0; xout < img_out.w; ++xout) {
                size_t xin = yout;
                size_t yin = xout;
                memcpy(img_out.RowPtr((int)yout) + xout * bytes_per_pixel,
                       img_in.RowPtr((int)yin) + xin * bytes_per_pixel,
                       bytes_per_pixel);
            }
    }
}

void RotateCCW(Image<unsigned char>& img_out, const Image<unsigned char>& img_in, size_t bytes_per_pixel)
{
    if(bytes_per_pixel == 1)
        TiledRotateCCW<1, 160>(img_out, img_in);
    else if(bytes_per_pixel == 2)
        TiledRotateCCW<2, 120>(img_out, img_in);
    else if(bytes_per_pixel == 3)
        TiledRotateCCW<3, 80>(img_out, img_in);
    else if(bytes_per_pixel == 4)
        TiledRotateCCW<4, 80>(img_out, img_in);
    else if(bytes_per_pixel == 6)
        TiledRotateCCW<6, 64>(img_out, img_in);
    else {
        for(size_t yout = 0; yout < img_out.h; ++yout)
            for(size_t xout = 0; xout < img_out.w; ++xout) {
                size_t xin = img_out.h - 1 - yout;
                size_t yin = xout;
                memcpy(img_out.RowPtr((int)yout) + xout * bytes_per_pixel,
                       img_in.RowPtr((int)yin) + xin * bytes_per_pixel,
                       bytes_per_pixel);
            }
    }
}


void MirrorVideo::Process(unsigned char* buffer_out, const unsigned char* buffer_in)
{

    for(size_t s=0; s<streams.size(); ++s) {
        Image<unsigned char> img_out = Streams()[s].StreamImage(buffer_out);
        const Image<unsigned char> img_in  = videoin->Streams()[s].StreamImage(buffer_in);
        const size_t bytes_per_pixel = Streams()[s].PixFormat().bpp / 8;

        switch (flips[s]) {
        case MirrorOptionsFlipX:
            FlipX(img_out, img_in, bytes_per_pixel);
            break;
        case MirrorOptionsFlipY:
            FlipY(img_out, img_in, bytes_per_pixel);
            break;
        case MirrorOptionsFlipXY:
            FlipXY(img_out, img_in, bytes_per_pixel);
            break;
        case MirrorOptionsRotateCW:
            RotateCW(img_out, img_in, bytes_per_pixel);
            break;
        case MirrorOptionsRotateCCW:
            RotateCCW(img_out, img_in, bytes_per_pixel);
            break;
        case MirrorOptionsTranspose:
            Transpose(img_out, img_in, bytes_per_pixel);
            break;
        case MirrorOptionsNone:
            PitchedImageCopy(img_out, img_in, bytes_per_pixel);
            break;
        default:
            pango_print_warn("MirrorVideo::Process(): Invalid enum %i.\n", flips[s]);
        }
    }

}

//! Implement VideoInput::GrabNext()
bool MirrorVideo::GrabNext( unsigned char* image, bool wait )
{
    if(videoin->GrabNext(buffer,wait)) {
        Process(image, buffer);
        return true;
    }else{
        return false;
    }
}

//! Implement VideoInput::GrabNewest()
bool MirrorVideo::GrabNewest( unsigned char* image, bool wait )
{
    if(videoin->GrabNewest(buffer,wait)) {
        Process(image, buffer);
        return true;
    }else{
        return false;
    }
}

std::vector<VideoInterface*>& MirrorVideo::InputStreams()
{
    return inputs;
}

unsigned int MirrorVideo::AvailableFrames() const
{
    BufferAwareVideoInterface* vpi = dynamic_cast<BufferAwareVideoInterface*>(videoin.get());
    if(!vpi)
    {
        pango_print_warn("Mirror: child interface is not buffer aware.");
        return 0;
    }
    else
    {
        return vpi->AvailableFrames();
    }
}

bool MirrorVideo::DropNFrames(uint32_t n)
{
    BufferAwareVideoInterface* vpi = dynamic_cast<BufferAwareVideoInterface*>(videoin.get());
    if(!vpi)
    {
        pango_print_warn("Mirror: child interface is not buffer aware.");
        return false;
    }
    else
    {
        return vpi->DropNFrames(n);
    }
}

std::istream& operator>> (std::istream &is, MirrorOptions &mirror)
{
    std::string str_mirror;
    is >> str_mirror;
    std::transform(str_mirror.begin(), str_mirror.end(), str_mirror.begin(), toupper);

    if(!str_mirror.compare("NONE")) {
        mirror = MirrorOptionsNone;
    }else if(!str_mirror.compare("FLIPX") || !str_mirror.compare("MIRROR")) {
        mirror = MirrorOptionsFlipX;
    }else if(!str_mirror.compare("FLIPY") || !str_mirror.compare("FLIP")) {
        mirror = MirrorOptionsFlipY;
    }else if(!str_mirror.compare("ROTATECW")) {
        mirror = MirrorOptionsRotateCW;
    }else if(!str_mirror.compare("ROTATECCW")) {
        mirror = MirrorOptionsRotateCCW;
    }else if(!str_mirror.compare("FLIPXY") || !str_mirror.compare("TRANSPOSE")) {
        mirror = MirrorOptionsFlipXY;
    }else{
        pango_print_warn("Unknown mirror option %s.", str_mirror.c_str());
        mirror = MirrorOptionsNone;
    }

    return is;
}

PANGOLIN_REGISTER_FACTORY(MirrorVideo)
{
    struct MirrorVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            std::unique_ptr<VideoInterface> subvid = pangolin::OpenVideo(uri.url);

            MirrorOptions default_opt = MirrorOptionsNone;
            if(uri.scheme == "flip") default_opt = MirrorOptionsFlipY;
            if(uri.scheme == "rotate") default_opt = MirrorOptionsFlipXY;
            if(uri.scheme == "transpose") default_opt = MirrorOptionsTranspose;
            if(uri.scheme == "rotateCW") default_opt = MirrorOptionsRotateCW;
            if(uri.scheme == "rotateCCW") default_opt = MirrorOptionsRotateCCW;

            std::vector<MirrorOptions> flips;

            for(size_t i=0; i < subvid->Streams().size(); ++i){
                std::stringstream ss;
                ss << "stream" << i;
                const std::string key = ss.str();
                flips.push_back(uri.Get<MirrorOptions>(key, default_opt) );
            }

            return std::unique_ptr<VideoInterface> (new MirrorVideo(subvid, flips));
        }
    };

    auto factory = std::make_shared<MirrorVideoFactory>();
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "mirror");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "flip");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "rotate");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "transpose");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "rotateCW");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "rotateCCW");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "transform");
}

}
