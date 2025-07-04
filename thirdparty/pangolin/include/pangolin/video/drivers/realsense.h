#pragma once

#include <pangolin/pangolin.h>

#include <pangolin/video/video.h>

#include <pangolin/video/iostream_operators.h>

namespace rs {
class context;
class device;
}

namespace pangolin
{

//! Interface to video capture sources
struct RealSenseVideo : public VideoInterface, public VideoPropertiesInterface, public VideoPlaybackInterface
{
public:

    // Open all RGB and Depth streams from all devices
    RealSenseVideo(ImageDim dim=ImageDim(640,480), int fps=30);

    // Open streams specified
    // TODO
    //RealSenseVideo(std::vector<OpenNiStreamMode>& stream_modes);

    ~RealSenseVideo();

    //! Implement VideoInput::Start()
    void Start() override;

    //! Implement VideoInput::Stop()
    void Stop() override;

    //! Implement VideoInput::SizeBytes()
    size_t SizeBytes() const override;

    //! Implement VideoInput::Streams()
    const std::vector<StreamInfo>& Streams() const override;

    //! Implement VideoInput::GrabNext()
    bool GrabNext( unsigned char* image, bool wait = true ) override;

    //! Implement VideoInput::GrabNewest()
    bool GrabNewest( unsigned char* image, bool wait = true ) override;

    //! Implement VideoPropertiesInterface::Properties()
    const picojson::value& DeviceProperties() const override {
        return device_properties;
    }

    //! Implement VideoPropertiesInterface::Properties()
    const picojson::value& FrameProperties() const override {
        return frame_properties;
    }

    //! Implement VideoPlaybackInterface::GetCurrentFrameId
    size_t GetCurrentFrameId() const override;

    //! Implement VideoPlaybackInterface::GetTotalFrames
    size_t GetTotalFrames() const override;

    //! Implement VideoPlaybackInterface::Seek
    size_t Seek(size_t frameid) override;

protected:
    size_t sizeBytes;

    std::vector<StreamInfo> streams;

    picojson::value device_properties;
    picojson::value frame_properties;
    picojson::value* streams_properties;

    size_t current_frame_index;
    size_t total_frames;

    rs::context* ctx_;
    std::vector<rs::device*> devs_;

    ImageDim  dim_;
    size_t fps_;
};

}
