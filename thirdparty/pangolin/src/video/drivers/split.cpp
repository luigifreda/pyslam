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

#include <pangolin/video/drivers/split.h>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/iostream_operators.h>

namespace pangolin
{

SplitVideo::SplitVideo(std::unique_ptr<VideoInterface> &src_, const std::vector<StreamInfo>& streams)
    : src(std::move(src_)), streams(streams)
{
    videoin.push_back(src.get());

    // Warn if stream over-runs input stream
    for(unsigned int i=0; i < streams.size(); ++i) {
        if(src->SizeBytes() < (size_t)streams[i].Offset() + streams[i].SizeBytes() ) {
            pango_print_warn("VideoSplitter: stream extends past end of input.\n");
            break;
        }
    }
}

SplitVideo::~SplitVideo()
{
}

size_t SplitVideo::SizeBytes() const
{
    return videoin[0]->SizeBytes();
}

const std::vector<StreamInfo>& SplitVideo::Streams() const
{
    return streams;
}

void SplitVideo::Start()
{
    videoin[0]->Start();
}

void SplitVideo::Stop()
{
    videoin[0]->Stop();
}

bool SplitVideo::GrabNext( unsigned char* image, bool wait )
{
    return videoin[0]->GrabNext(image, wait);
}

bool SplitVideo::GrabNewest( unsigned char* image, bool wait )
{
    return videoin[0]->GrabNewest(image, wait);
}

std::vector<VideoInterface*>& SplitVideo::InputStreams()
{
    return videoin;
}

PANGOLIN_REGISTER_FACTORY(SplitVideo)
{
    struct SplitVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            std::vector<StreamInfo> streams;

            std::unique_ptr<VideoInterface> subvid = pangolin::OpenVideo(uri.url);
            if(subvid->Streams().size() == 0) {
                throw VideoException("VideoSplitter input must have at least one stream");
            }

            while(true) {
                const size_t n = streams.size() + 1;
                std::string key_roi = std::string("roi") + pangolin::Convert<std::string, size_t>::Do(n);
                std::string key_mem = std::string("mem") + pangolin::Convert<std::string, size_t>::Do(n);
                std::string key_str = std::string("stream") + pangolin::Convert<std::string, size_t>::Do(n);

                if(uri.Contains(key_roi)) {
                    const StreamInfo& st1 = subvid->Streams()[0];
                    const ImageRoi& roi = uri.Get<ImageRoi>(key_roi, ImageRoi() );
                    if(roi.w == 0 || roi.h == 0) {
                        throw VideoException("split: empty ROI.");
                    }
                    const size_t start1 = roi.y * st1.Pitch() + st1.PixFormat().bpp * roi.x / 8;
                    streams.push_back( StreamInfo( st1.PixFormat(), roi.w, roi.h, st1.Pitch(), (unsigned char*)0 + start1 ) );
                }else if(uri.Contains(key_mem)) {
                    const StreamInfo& info = uri.Get<StreamInfo>(key_mem, subvid->Streams()[0] );
                    streams.push_back(info);
                }else if(uri.Contains(key_str)) {
                    const size_t old_stream = uri.Get<size_t>(key_str, 0) -1;
                    if(old_stream >= subvid->Streams().size()) {
                        throw VideoException("split: requesting source stream which does not exist.");
                    }
                    streams.push_back(subvid->Streams()[old_stream]);
                }else{
                    break;
                }
            }

            // Default split if no arguments
            if(streams.size() == 0) {
                const StreamInfo& st1 = subvid->Streams()[0];
                const size_t subw = st1.Width();
                const size_t subh = st1.Height();

                ImageRoi roi1, roi2;

                if(subw > subh) {
                    // split horizontally
                    roi1 = ImageRoi(0,0, subw/2, subh );
                    roi2 = ImageRoi(subw/2,0, subw/2, subh );
                }else{
                    // split vertically
                    roi1 = ImageRoi(0,0, subw, subh/2 );
                    roi2 = ImageRoi(0,subh/2, subw, subh/2 );
                }

                const size_t start1 = roi1.y * st1.Pitch() + st1.PixFormat().bpp * roi1.x / 8;
                const size_t start2 = roi2.y * st1.Pitch() + st1.PixFormat().bpp * roi2.x / 8;
                streams.push_back( StreamInfo( st1.PixFormat(), roi1.w, roi1.h, st1.Pitch(), (unsigned char*)(start1) ) );
                streams.push_back( StreamInfo( st1.PixFormat(), roi2.w, roi2.h, st1.Pitch(), (unsigned char*)(start2) ) );
            }

            return std::unique_ptr<VideoInterface>( new SplitVideo(subvid,streams) );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<SplitVideoFactory>(), 10, "split");
}

}
