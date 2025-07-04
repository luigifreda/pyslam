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

#include <pangolin/video/drivers/deinterlace.h>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/iostream_operators.h>
#include <dc1394/conversions.h>

namespace pangolin
{

DeinterlaceVideo::DeinterlaceVideo(std::unique_ptr<VideoInterface> &videoin_)
    : videoin(std::move(videoin_)), buffer(0)
{
    if(videoin->Streams().size() != 1)
        throw VideoException("FirewireDeinterlace input must have exactly one stream");

    const StreamInfo& stmin = videoin->Streams()[0];

    StreamInfo stm1(PixelFormatFromString("GRAY8"), stmin.Width(), stmin.Height(), stmin.Width(), 0);
    StreamInfo stm2(PixelFormatFromString("GRAY8"), stmin.Width(), stmin.Height(), stmin.Width(), (unsigned char*)0 + stmin.Width()*stmin.Height());
    streams.push_back(stm1);
    streams.push_back(stm2);

    buffer = new unsigned char[videoin->SizeBytes()];

    std::cout << videoin->Streams()[0].Width() << ", " << videoin->Streams()[0].Height() << std::endl;
}

DeinterlaceVideo::~DeinterlaceVideo()
{
    delete[] buffer;
}

size_t DeinterlaceVideo::SizeBytes() const
{
    return videoin->SizeBytes();
}

const std::vector<StreamInfo>& DeinterlaceVideo::Streams() const
{
    return streams;
}

void DeinterlaceVideo::Start()
{
    videoin->Start();
}

void DeinterlaceVideo::Stop()
{
    videoin->Stop();
}

bool DeinterlaceVideo::GrabNext( unsigned char* image, bool wait )
{
    if(videoin->GrabNext(buffer, wait)) {
        return ( dc1394_deinterlace_stereo(buffer,image, videoin->Streams()[0].Width(), 2*videoin->Streams()[0].Height() ) == DC1394_SUCCESS );
    }
    return false;
}

bool DeinterlaceVideo::GrabNewest( unsigned char* image, bool wait )
{
    if(videoin->GrabNewest(buffer, wait)) {
        return ( dc1394_deinterlace_stereo(buffer,image, videoin->Streams()[0].Width(), 2*videoin->Streams()[0].Height() ) == DC1394_SUCCESS );
    }
    return false;
}

PANGOLIN_REGISTER_FACTORY(DeinterlaceVideo)
{
    struct DeinterlaceVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            std::unique_ptr<VideoInterface> subvid = pangolin::OpenVideo(uri.url);
            return std::unique_ptr<VideoInterface>( new DeinterlaceVideo(subvid) );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<DeinterlaceVideoFactory>(), 10, "deinterlace");
}

}
