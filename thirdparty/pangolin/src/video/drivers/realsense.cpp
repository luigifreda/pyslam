#include <librealsense/rs.hpp>
#include <pangolin/video/drivers/realsense.h>
#include <pangolin/factory/factory_registry.h>

namespace pangolin {

RealSenseVideo::RealSenseVideo(ImageDim dim, int fps) 
  : dim_(dim), fps_(fps) {
  ctx_ = new rs::context();
  sizeBytes = 0;
  for (int32_t i=0; i<ctx_->get_device_count(); ++i) {
    devs_.push_back(ctx_->get_device(i));

    devs_[i]->enable_stream(rs::stream::depth, dim_.x, dim_.y, rs::format::z16, fps_);
    StreamInfo streamD(PixelFormatFromString("GRAY16LE"), dim_.x, dim_.y, dim_.x*2, 0);
    streams.push_back(streamD);

    sizeBytes += streamD.SizeBytes();
    devs_[i]->enable_stream(rs::stream::color, dim_.x, dim_.y, rs::format::rgb8, fps_);
    StreamInfo streamRGB(PixelFormatFromString("RGB24"), dim_.x, dim_.y, dim_.x*3, (uint8_t*)0+sizeBytes);
    streams.push_back(streamRGB);
    sizeBytes += streamRGB.SizeBytes();

    devs_[i]->start();
  }
  total_frames = std::numeric_limits<int>::max();
}

RealSenseVideo::~RealSenseVideo() {
  delete ctx_;
}

void RealSenseVideo::Start() {
  for (int32_t i=0; i<ctx_->get_device_count(); ++i) {
    devs_[i]->stop();
    devs_[i]->start();
  }
  current_frame_index = 0;
}

void RealSenseVideo::Stop() {
  for (int32_t i=0; i<ctx_->get_device_count(); ++i) {
    devs_[i]->stop();
  }
}

size_t RealSenseVideo::SizeBytes() const {
  return sizeBytes;
}

const std::vector<StreamInfo>& RealSenseVideo::Streams() const {
  return streams;
}

bool RealSenseVideo::GrabNext(unsigned char* image, bool wait) {

  unsigned char* out_img = image;
  for (int32_t i=0; i<ctx_->get_device_count(); ++i) {
    if (wait) {
      devs_[i]->wait_for_frames();
    }
    memcpy(out_img, devs_[i]->get_frame_data(rs::stream::depth), streams[i*2].SizeBytes());
    out_img += streams[i*2].SizeBytes();
    memcpy(out_img, devs_[i]->get_frame_data(rs::stream::color), streams[i*2+1].SizeBytes());
    out_img += streams[i*2+1].SizeBytes();
  }
  return true;
}

bool RealSenseVideo::GrabNewest(unsigned char* image, bool wait) {
  return GrabNext(image, wait);
}

size_t RealSenseVideo::GetCurrentFrameId() const {
  return current_frame_index;
}

size_t RealSenseVideo::GetTotalFrames() const {
  return total_frames;
}

size_t RealSenseVideo::Seek(size_t frameid) {
  // TODO
  return -1;
}

PANGOLIN_REGISTER_FACTORY(RealSenseVideo)
{
    struct RealSenseVideoFactory : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            const ImageDim dim = uri.Get<ImageDim>("size", ImageDim(640,480));
            const unsigned int fps = uri.Get<unsigned int>("fps", 30);
            return std::unique_ptr<VideoInterface>( new RealSenseVideo(dim, fps) );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<RealSenseVideoFactory>(), 10, "realsense");
}

}
