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

#include <pangolin/video/drivers/merge.h>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/iostream_operators.h>
#include <pangolin/plot/range.h>
#include <assert.h> // assert()

#include <assert.h>

namespace pangolin
{

MergeVideo::MergeVideo(std::unique_ptr<VideoInterface>& src_, const std::vector<Point>& stream_pos, size_t w = 0, size_t h = 0 )
    : src( std::move(src_) ), buffer(new uint8_t[src->SizeBytes()]), stream_pos(stream_pos)
{
    videoin.push_back(src.get());

    // Must be a stream_pos for each stream
    // Each stream must have the same format.
    assert(stream_pos.size() == src->Streams().size());
    assert(src->Streams().size() > 0);
    const PixelFormat fmt = src->Streams()[0].PixFormat();
    for(size_t i=1; i < src->Streams().size(); ++i) {
        assert(src->Streams()[i].PixFormat().format == fmt.format);
    }

    // Compute buffer regions for data copying.
    XYRange<size_t> r = XYRange<size_t>::Empty();
    for(size_t i=0; i < src->Streams().size(); ++i) {
        const StreamInfo& si = src->Streams()[i];
        const size_t x = stream_pos[i].x;
        const size_t y = stream_pos[i].y;
        XYRange<size_t> sr(x, x + si.Width(), y, y + si.Height());
        r.Insert(sr);
    }

    // Use implied min / max based on points
    if(!w && !h) {
        w = r.x.max;
        h = r.y.max;
    }

    size_bytes = w*h*fmt.bpp/8;
    streams.emplace_back(fmt,w,h,w*fmt.bpp/8,(unsigned char*)0);
}

MergeVideo::~MergeVideo()
{

}

//! Implement VideoInput::Start()
void MergeVideo::Start()
{
    src->Start();
}

//! Implement VideoInput::Stop()
void MergeVideo::Stop()
{
    src->Stop();
}

//! Implement VideoInput::SizeBytes()
size_t MergeVideo::SizeBytes() const
{
    return size_bytes;
}

//! Implement VideoInput::Streams()
const std::vector<StreamInfo>& MergeVideo::Streams() const
{
    return streams;
}

void MergeVideo::CopyBuffer(unsigned char* dst_bytes, unsigned char* src_bytes)
{
    Image<unsigned char> dst_image = Streams()[0].StreamImage(dst_bytes);
    const size_t dst_pix_bytes = Streams()[0].PixFormat().bpp / 8;

    for(size_t i=0; i < stream_pos.size(); ++i) {
        const StreamInfo& src_stream = src->Streams()[i];
        const Image<unsigned char> src_image = src_stream.StreamImage(src_bytes);
        const Point& p = stream_pos[i];
        for(size_t y=0; y < src_stream.Height(); ++y) {
            // Copy row from src to dst
            std::memcpy(
                dst_image.RowPtr(y + p.y) + p.x * dst_pix_bytes,
                src_image.RowPtr(y), src_stream.RowBytes()
            );
        }
    }
}

//! Implement VideoInput::GrabNext()
bool MergeVideo::GrabNext( unsigned char* image, bool wait )
{
    const bool success = src->GrabNext(buffer.get(), wait);
    if(success) CopyBuffer(image, buffer.get());
    return success;
}

//! Implement VideoInput::GrabNewest()
bool MergeVideo::GrabNewest( unsigned char* image, bool wait )
{
    const bool success = src->GrabNewest(buffer.get(), wait);
    if(success) CopyBuffer(image, buffer.get());
    return success;
}

std::vector<VideoInterface*>& MergeVideo::InputStreams()
{
    return videoin;
}

PANGOLIN_REGISTER_FACTORY(MergeVideo)
{
    struct MergeVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            const ImageDim dim = uri.Get<ImageDim>("size", ImageDim(0,0));

            std::unique_ptr<VideoInterface> subvid = pangolin::OpenVideo(uri.url);
            std::vector<Point> points;
            Point p(0,0);
            for(size_t s=0; s < subvid->Streams().size(); ++s) {
                const StreamInfo& si = subvid->Streams()[s];
                p = uri.Get<Point>("pos"+std::to_string(s+1), p);
                points.push_back(p);
                p.x += si.Width();
            }

            return std::unique_ptr<VideoInterface>(new MergeVideo(subvid, points, dim.x, dim.y));
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<MergeVideoFactory>(), 10, "merge");
}

}
