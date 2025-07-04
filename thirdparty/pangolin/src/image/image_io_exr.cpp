#include <pangolin/platform.h>

#include <fstream>
#include <pangolin/image/typed_image.h>

#ifdef HAVE_OPENEXR
#include <ImfChannelList.h>
#include <ImfInputFile.h>
#include <ImfOutputFile.h>
#include <ImfIO.h>
#endif // HAVE_OPENEXR

namespace pangolin {

#ifdef HAVE_OPENEXR
Imf::PixelType OpenEXRPixelType(int channel_bits)
{
    if( channel_bits == 16 ) {
        return Imf::PixelType::HALF;
    }else if( channel_bits == 32 ) {
        return Imf::PixelType::FLOAT;
    }else{
        throw std::runtime_error("Unsupported OpenEXR Pixel Type.");
    }
}

void SetOpenEXRChannels(Imf::ChannelList& ch, const pangolin::PixelFormat& fmt)
{
    const char* CHANNEL_NAMES[] = {"R","G","B","A"};
    for(size_t c=0; c < fmt.channels; ++c) {
        ch.insert( CHANNEL_NAMES[c], Imf::Channel(OpenEXRPixelType(fmt.channel_bits[c])) );
    }
}

class StdIStream: public Imf::IStream
{
  public:
    StdIStream (std::istream &is):
        Imf::IStream ("stream"),
        _is (&is)
    {
    }

    virtual bool read (char c[/*n*/], int n)
    {
        if (!*_is)
            throw std::runtime_error("Unexpected end of file.");
        _is->read (c, n);
        if (_is->gcount() < n)
        {
            throw std::runtime_error("Early end of file");
            return false;
        }
        return true;
    }

    virtual Imf::Int64 tellg ()
    {
        return std::streamoff (_is->tellg());
    }

    virtual void seekg (Imf::Int64 pos)
    {
        _is->seekg (pos);
    }

    virtual void clear ()
    {
        _is->clear();
    }

  private:
    std::istream *	_is;
};

PixelFormat GetPixelFormat(const Imf::Header& header)
{
    const Imf::ChannelList &channels = header.channels();
    size_t count = 0;
    for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i){
        const Imf::Channel& channel = i.channel();
        if (channel.type != Imf::FLOAT){
            throw std::invalid_argument("Currently, only 32-bit float OpenEXR files are supported.");
        }
        count += 1;
    }

    switch (count) {
        case 1: return PixelFormatFromString("GRAY32F");
        case 3: return PixelFormatFromString("RGB96F");
        case 4: return PixelFormatFromString("RGBA128F");
        default: throw std::invalid_argument("Currently, only 1, 3 or 4-channel OpenEXR files are supported.");
    }
}

#endif //HAVE_OPENEXR

TypedImage LoadExr(std::istream& source)
{
#ifdef HAVE_OPENEXR
    StdIStream istream(source);
    Imf::InputFile file(istream);
    PANGO_ENSURE(file.isComplete());

    Imath::Box2i dw = file.header().dataWindow();
    int width = dw.max.x - dw.min.x + 1;
    int height = dw.max.y - dw.min.y + 1;

    PixelFormat format = GetPixelFormat(file.header());
    TypedImage img(width, height, format);

    char *imgBase = (char *) img.ptr - (dw.min.x + dw.min.y * width) * sizeof(float) * format.channels;
    Imf::FrameBuffer fb;

    const Imf::ChannelList &channels = file.header().channels();
    size_t c = 0;
    for (Imf::ChannelList::ConstIterator i = channels.begin(); i != channels.end(); ++i){
        fb.insert(i.name(), Imf::Slice(
                        Imf::FLOAT, imgBase + sizeof(float) * c++,
                        sizeof(float) * format.channels,
                        sizeof(float) * format.channels * size_t(width),
                        1, 1,
                        0.0));
    }

    file.setFrameBuffer(fb);
    file.readPixels(dw.min.y, dw.max.y);

    return img;
#else
    PANGOLIN_UNUSED(source);
    throw std::runtime_error("Rebuild Pangolin for EXR support.");
#endif //HAVE_OPENEXR
}

void SaveExr(const Image<unsigned char>& image_in, const pangolin::PixelFormat& fmt, const std::string& filename, bool top_line_first)
{
#ifdef HAVE_OPENEXR
    ManagedImage<unsigned char> flip_image;
    Image<unsigned char> image;

    if(top_line_first) {
        image = image_in;
    }else{
        flip_image.Reinitialise(image_in.pitch,image_in.h);
        for(size_t y=0; y<image_in.h; ++y) {
            std::memcpy(flip_image.RowPtr(y), image_in.RowPtr(y), image_in.pitch);
        }
        image = flip_image;
    }


    Imf::Header header (image.w, image.h);
    SetOpenEXRChannels(header.channels(), fmt);

    Imf::OutputFile file (filename.c_str(), header);
    Imf::FrameBuffer frameBuffer;

    int ch=0;
    size_t ch_bits = 0;
    for(Imf::ChannelList::Iterator it = header.channels().begin(); it != header.channels().end(); ++it)
    {
        frameBuffer.insert(
            it.name(),
            Imf::Slice(
                it.channel().type,
                (char*)image.ptr + ch_bits/8,
                fmt.channel_bits[ch]/8,
                image.pitch
            )
        );

        ch_bits += fmt.channel_bits[ch++];
    }

    file.setFrameBuffer(frameBuffer);
    file.writePixels(image.h);

#else
    PANGOLIN_UNUSED(image_in);
    PANGOLIN_UNUSED(fmt);
    PANGOLIN_UNUSED(filename);
    PANGOLIN_UNUSED(top_line_first);
    throw std::runtime_error("EXR Support not enabled. Please rebuild Pangolin.");
#endif // HAVE_OPENEXR
}

}
