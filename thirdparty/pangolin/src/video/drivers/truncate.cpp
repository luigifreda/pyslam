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

#include <pangolin/video/drivers/truncate.h>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/iostream_operators.h>

namespace pangolin
{

TruncateVideo::TruncateVideo(std::unique_ptr<VideoInterface> &src_, size_t begin, size_t end)
    : src(std::move(src_)), streams(src->Streams()), begin(begin), end(end), next_frame_to_grab(0)
{
    videoin.push_back(src.get());

    if(VideoPlaybackInterface* v = GetVideoPlaybackInterface()){
        // Guard against the obscure case of nested TruncateVideo filters
        if( !pangolin::FindFirstMatchingVideoInterface<TruncateVideo>(*src_) ) {
            next_frame_to_grab = v->Seek(begin);
        }
    }
}

TruncateVideo::~TruncateVideo()
{
}

size_t TruncateVideo::SizeBytes() const
{
    return videoin[0]->SizeBytes();
}

const std::vector<StreamInfo>& TruncateVideo::Streams() const
{
    return streams;
}

void TruncateVideo::Start()
{
    videoin[0]->Start();
}

void TruncateVideo::Stop()
{
    videoin[0]->Stop();
}

bool TruncateVideo::GrabNext( unsigned char* image, bool wait )
{
    if(next_frame_to_grab < end) {
        bool grab_success = videoin[0]->GrabNext(image, wait);
        return grab_success && (next_frame_to_grab++) >= begin;
    }
    return false;
}

bool TruncateVideo::GrabNewest( unsigned char* image, bool wait )
{
    return videoin[0]->GrabNewest(image, wait);
}

std::vector<VideoInterface*>& TruncateVideo::InputStreams()
{
    return videoin;
}

PANGOLIN_REGISTER_FACTORY(TruncateVideo)
{
    struct TruncateVideoFactory : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            std::unique_ptr<VideoInterface> subvid = pangolin::OpenVideo(uri.url);
            if(subvid->Streams().size() == 0) {
                throw VideoException("VideoTruncater input must have at least one stream");
            }

            const size_t start = uri.Get<size_t>("begin", 0);
            const size_t end = uri.Get<size_t>("end", std::numeric_limits<size_t>::max());

            return std::unique_ptr<VideoInterface>( new TruncateVideo(subvid,start,end) );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<TruncateVideoFactory>(), 10, "truncate");
}

}
