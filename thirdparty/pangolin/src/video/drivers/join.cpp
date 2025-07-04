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

#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/drivers/join.h>
#include <pangolin/video/iostream_operators.h>

//#define DEBUGJOIN

#ifdef DEBUGJOIN
  #include <pangolin/utils/timer.h>
  #define TSTART() pangolin::basetime start,last,now; start = pangolin::TimeNow(); last = start;
  #define TGRABANDPRINT(...)  now = pangolin::TimeNow(); fprintf(stderr,"JOIN: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr, " %fms.\n",1000*pangolin::TimeDiff_s(last, now)); last = now;
  #define DBGPRINT(...) fprintf(stderr,"JOIN: "); fprintf(stderr, __VA_ARGS__); fprintf(stderr,"\n");
#else
  #define TSTART()
  #define TGRABANDPRINT(...)
  #define DBGPRINT(...)
#endif

namespace pangolin
{
JoinVideo::JoinVideo(std::vector<std::unique_ptr<VideoInterface> > &src_)
    : storage(std::move(src_)), size_bytes(0), sync_tolerance_us(0)
{
    for(auto& p : storage) {
        src.push_back(p.get());
    }

    // Add individual streams
    for(size_t s=0; s< src.size(); ++s)
    {
        VideoInterface& vid = *src[s];
        for(size_t i=0; i < vid.Streams().size(); ++i)
        {
            const StreamInfo si = vid.Streams()[i];
            const PixelFormat fmt = si.PixFormat();
            const Image<unsigned char> img_offset = si.StreamImage((unsigned char*)size_bytes);
            streams.push_back(StreamInfo(fmt, img_offset));
        }
        size_bytes += src[s]->SizeBytes();
    }
}

JoinVideo::~JoinVideo()
{
    for(size_t s=0; s< src.size(); ++s) {
        src[s]->Stop();
    }
}

size_t JoinVideo::SizeBytes() const
{
    return size_bytes;
}

const std::vector<StreamInfo>& JoinVideo::Streams() const
{
    return streams;
}

void JoinVideo::Start()
{
    for(size_t s=0; s< src.size(); ++s) {
        src[s]->Start();
    }
}

void JoinVideo::Stop()
{
    for(size_t s=0; s< src.size(); ++s) {
        src[s]->Stop();
    }
}

bool JoinVideo::Sync(int64_t tolerance_us, double transfer_bandwidth_gbps)
{
    transfer_bandwidth_bytes_per_us = int64_t((transfer_bandwidth_gbps * 1E3) / 8.0);
//    std::cout << "transfer_bandwidth_gbps: " << transfer_bandwidth_gbps << std::endl;

    for(size_t s=0; s< src.size(); ++s)
    {
       picojson::value props = GetVideoDeviceProperties(src[s]);
       if(!props.get_value(PANGO_HAS_TIMING_DATA, false)) {
           if (props.contains("streams")) {
               picojson::value streams = props["streams"];
               for (size_t i=0; i<streams.size(); ++i) {
                    if(!streams[i].get_value(PANGO_HAS_TIMING_DATA, false)) {
                        sync_tolerance_us = 0;
                        return false;
                    }
               }
           } else {
               sync_tolerance_us = 0;
               return false;
           }
       }
    }

    sync_tolerance_us = tolerance_us;

//    std::cout << "transfer_bandwidth_bytes_per_us: " << transfer_bandwidth_bytes_per_us << std::endl;
    return true;
}

// Assuming that src_index supports VideoPropertiesInterface and has a valid PANGO_HOST_RECEPTION_TIME_US, or PANGO_ESTIMATED_CENTER_CAPTURE_TIME_US
// returns a capture time adjusted for transfer time and when possible also for exposure.
int64_t JoinVideo::GetAdjustedCaptureTime(size_t src_index)
{
    picojson::value props = GetVideoFrameProperties(src[src_index]);
    if(props.contains(PANGO_ESTIMATED_CENTER_CAPTURE_TIME_US)) {
        // great, the driver already gave us an estimated center of capture
        return props[PANGO_ESTIMATED_CENTER_CAPTURE_TIME_US].get<int64_t>();

    } else {
        if(props.contains(PANGO_HOST_RECEPTION_TIME_US)) {
            int64_t transfer_time_us = 0;
            if( transfer_bandwidth_bytes_per_us > 0 ) {
                transfer_time_us = src[src_index]->SizeBytes() / transfer_bandwidth_bytes_per_us;
            }
            return props[PANGO_HOST_RECEPTION_TIME_US].get<int64_t>() - transfer_time_us;
        } else {
            if (props.contains("streams")) {
                picojson::value streams = props["streams"];

                if(streams.size()>0){
                     if(streams[0].contains(PANGO_ESTIMATED_CENTER_CAPTURE_TIME_US)) {
                         // great, the driver already gave us an estimated center of capture
                         return streams[0][PANGO_ESTIMATED_CENTER_CAPTURE_TIME_US].get<int64_t>();
                     }
                     else if( streams[0].contains(PANGO_HOST_RECEPTION_TIME_US)) {
                         int64_t transfer_time_us = 0;
                         if( transfer_bandwidth_bytes_per_us > 0 ) {
                             transfer_time_us = src[src_index]->SizeBytes() / transfer_bandwidth_bytes_per_us;
                         }
                        return streams[0][PANGO_HOST_RECEPTION_TIME_US].get<int64_t>() - transfer_time_us;
                    }
                }
           }
       }

       PANGO_ENSURE(false, "JoinVideo: Stream % does contain any timestamp info.\n", src_index);
       return 0;
   }
}

bool JoinVideo::GrabNext(unsigned char* image, bool wait)
{
    size_t offset = 0;
    std::vector<size_t> offsets(src.size(), 0);
    std::vector<int64_t> capture_us(src.size(), 0);

    TSTART()
    DBGPRINT("Entering GrabNext:")
    for(size_t s=0; s<src.size(); ++s) {
        if( src[s]->GrabNext(image+offset,wait) ) {
            if(sync_tolerance_us > 0) {
                capture_us[s] = GetAdjustedCaptureTime(s);
            }else{
                capture_us[s] = std::numeric_limits<int64_t>::max();
            }
        }
        offsets[s] = offset;
        offset += src[s]->SizeBytes();
        TGRABANDPRINT("Stream %ld grab took ",s);
    }

    // Check if any streams didn't return an image. This means a stream is waiting on data or has finished.
    if( std::any_of(capture_us.begin(), capture_us.end(), [](int64_t v){return v == 0;}) ){
        return false;
    }

    // Check Sync if a tolerence has been specified.
    if(sync_tolerance_us > 0)
    {
        auto range = std::minmax_element(capture_us.begin(), capture_us.end());
        if( (*range.second - *range.first) > sync_tolerance_us)
        {
            pango_print_warn("JoinVideo: Source timestamps span  %lu us, not within %lu us. Ignoring frames, trying to sync...\n", (unsigned long)((*range.second - *range.first)), (unsigned long)sync_tolerance_us);

            // Attempt to resync...
            for(size_t n=0; n<10; ++n){
                for(size_t s=0; s<src.size(); ++s) {
                    // Catch up frames that are behind
                    if(capture_us[s] < (*range.second - sync_tolerance_us))
                    {
                        if(src[s]->GrabNext(image+offsets[s],true)) {
                            capture_us[s] = GetAdjustedCaptureTime(s);
                        }
                    }
                }
            }
        }

        // Check sync again
        range = std::minmax_element(capture_us.begin(), capture_us.end());
        if( (*range.second - *range.first) > sync_tolerance_us) {
            TGRABANDPRINT("NOT IN SYNC oldest:%ld newest:%ld delta:%ld", *range.first, *range.second, (*range.second - *range.first));
            return false;
        } else {
            TGRABANDPRINT("    IN SYNC oldest:%ld newest:%ld delta:%ld", *range.first, *range.second, (*range.second - *range.first));
            return true;
        }
    }
    else
    {
        pango_print_warn("JoinVideo: sync_tolerance_us = 0, frames are not synced!\n");
        return true;
    }
}

bool AllInterfacesAreBufferAware(std::vector<VideoInterface*>& src){
  for(size_t s=0; s<src.size(); ++s) {
      if(!dynamic_cast<BufferAwareVideoInterface*>(src[s])) return false;
  }
  return true;
}

bool JoinVideo::GrabNewest( unsigned char* image, bool wait )
{
  // TODO: Tidy to correspond to GrabNext()
  TSTART()
  DBGPRINT("Entering GrabNewest:");
  if(AllInterfacesAreBufferAware(src)) {
     DBGPRINT("All interfaces are BufferAwareVideoInterface.")
     unsigned int minN = std::numeric_limits<unsigned int>::max();
     //Find smallest number of frames it is safe to drop.
     for(size_t s=0; s<src.size(); ++s) {
         auto bai = dynamic_cast<BufferAwareVideoInterface*>(src[s]);
         unsigned int n = bai->AvailableFrames();
         minN = std::min(n, minN);
         DBGPRINT("Interface %ld has %u frames available.",s ,n)
     }
     TGRABANDPRINT("Quering avalable frames took ")
     DBGPRINT("Safe number of buffers to drop: %d.",((minN > 1) ? (minN-1) : 0));

     //Safely drop minN-1 frames on each interface.
     if(minN > 1) {
         for(size_t s=0; s<src.size(); ++s) {
             auto bai = dynamic_cast<BufferAwareVideoInterface*>(src[s]);
             if(!bai->DropNFrames(minN - 1)) {
                 pango_print_error("Stream %lu did not drop %u frames altough available.\n", (unsigned long)s, (minN-1));
                 return false;
             }
         }
         TGRABANDPRINT("Dropping %u frames on each interface took ",(minN -1));
     }
     return GrabNext(image, wait);
  } else {
      DBGPRINT("NOT all interfaces are BufferAwareVideoInterface.")
      // Simply calling GrabNewest on the child streams might cause loss of sync,
      // instead we perform as many GrabNext as possible on the first stream and
      // then pull the same number of frames from every other stream.
      size_t offset = 0;
      std::vector<size_t> offsets;
      std::vector<int64_t> reception_times;
      int64_t newest = std::numeric_limits<int64_t>::min();
      int64_t oldest = std::numeric_limits<int64_t>::max();
      bool grabbed_any = false;
      int first_stream_backlog = 0;
      int64_t rt = 0;
      bool got_frame = false;

      do {
          got_frame = src[0]->GrabNext(image+offset,false);
          if(got_frame) {
              if(sync_tolerance_us > 0) {
                  rt = GetAdjustedCaptureTime(0);
              }
              first_stream_backlog++;
              grabbed_any = true;
          }
      } while(got_frame);
      offsets.push_back(offset);
      offset += src[0]->SizeBytes();
      if(sync_tolerance_us > 0) {
          reception_times.push_back(rt);
          if(newest < rt) newest = rt;
          if(oldest > rt) oldest = rt;
      }
      TGRABANDPRINT("Stream 0 grab took ");

      for(size_t s=1; s<src.size(); ++s) {
          for (int i=0; i<first_stream_backlog; i++){
              grabbed_any |= src[s]->GrabNext(image+offset,true);
              if(sync_tolerance_us > 0) {
                  rt = GetAdjustedCaptureTime(s);
              }
          }
          offsets.push_back(offset);
          offset += src[s]->SizeBytes();
          if(sync_tolerance_us > 0) {
              reception_times.push_back(rt);
              if(newest < rt) newest = rt;
              if(oldest > rt) oldest = rt;
          }
      }
      TGRABANDPRINT("Stream >=1 grab took ");

      if(sync_tolerance_us > 0) {
          if(std::abs(newest - oldest) > sync_tolerance_us){
              pango_print_warn("Join timestamps not within %lu us trying to sync\n", (unsigned long)sync_tolerance_us);

              for(size_t n=0; n<10; ++n){
                  for(size_t s=0; s<src.size(); ++s) {
                      if(reception_times[s] < (newest - sync_tolerance_us)) {
                          VideoInterface& vid = *src[s];
                          if(vid.GrabNewest(image+offsets[s],false)) {
                              rt = GetAdjustedCaptureTime(s);
                              if(newest < rt) newest = rt;
                              if(oldest > rt) oldest = rt;
                              reception_times[s] = rt;
                          }
                      }
                  }
              }
          }

          if(std::abs(newest - oldest) > sync_tolerance_us ) {
              TGRABANDPRINT("NOT IN SYNC newest:%ld oldest:%ld delta:%ld syncing took ", newest, oldest, (newest - oldest));
              return false;
          } else {
              TGRABANDPRINT("    IN SYNC newest:%ld oldest:%ld delta:%ld syncing took ", newest, oldest, (newest - oldest));
              return true;
          }
      } else {
          return true;
      }
  }

}

std::vector<VideoInterface*>& JoinVideo::InputStreams()
{
    return src;
}

std::vector<std::string> SplitBrackets(const std::string src, char open = '{', char close = '}')
{
    std::vector<std::string> splits;

    int nesting = 0;
    int begin = -1;

    for(size_t i=0; i < src.length(); ++i) {
        if(src[i] == open) {
            if(nesting==0) {
                begin = (int)i;
            }
            nesting++;
        }else if(src[i] == close) {
            nesting--;
            if(nesting == 0) {
                // matching close bracket.
                int str_start = begin+1;
                splits.push_back( src.substr(str_start, i-str_start) );
            }
        }
    }

    return splits;
}

PANGOLIN_REGISTER_FACTORY(JoinVideo)
{
    struct JoinVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {

            std::vector<std::string> uris = SplitBrackets(uri.url);

            // Standard by which we should measure if frames are in sync.
            const unsigned long sync_tol_us = uri.Get<unsigned long>("sync_tolerance_us", 0);

            // Bandwidth used to compute exposure end time from reception time for sync logic
            const double transfer_bandwidth_gbps = uri.Get<double>("transfer_bandwidth_gbps", 0.0);

            if(uris.size() == 0) {
                throw VideoException("No VideoSources found in join URL.", "Specify videos to join with curly braces, e.g. join://{test://}{test://}");
            }

            std::vector<std::unique_ptr<VideoInterface>> src;
            for(size_t i=0; i<uris.size(); ++i) {
                src.push_back( pangolin::OpenVideo(uris[i]) );
            }

            JoinVideo* video_raw = new JoinVideo(src);

            if(sync_tol_us>0) {
                if(!video_raw->Sync(sync_tol_us, transfer_bandwidth_gbps)) {
                    pango_print_error("WARNING: not all streams in join support sync_tolerance_us option. Not using tolerance.\n");
                }
            }

            return std::unique_ptr<VideoInterface>(video_raw);
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<JoinVideoFactory>(), 10, "join");
}

}

#undef TSTART
#undef TGRABANDPRINT
#undef DBGPRINT
