#include <librealsense2/rs.hpp>
#include <pangolin/video/drivers/realsense2.h>
#include <pangolin/factory/factory_registry.h>

namespace pangolin {

RealSense2Video::RealSense2Video(ImageDim dim, int fps)
    : dim_(dim), fps_(fps) {

    sizeBytes = 0;

    // Create RealSense pipeline, encapsulating the actual device and sensors
    pipe = new rs2::pipeline();

    //Configure the pipeline
    cfg = new rs2::config();

    {   //config depth
        cfg->enable_stream(RS2_STREAM_DEPTH, dim_.x, dim_.y, RS2_FORMAT_Z16, fps_);
        StreamInfo streamD(PixelFormatFromString("GRAY16LE"), dim_.x, dim_.y, dim_.x*2, 0);
        streams.push_back(streamD);
        sizeBytes += streamD.SizeBytes();
    }

    {   //config color
        cfg->enable_stream(RS2_STREAM_COLOR, dim_.x, dim_.y, RS2_FORMAT_RGB8, fps_);
        StreamInfo streamRGB(PixelFormatFromString("RGB24"), dim_.x, dim_.y, dim_.x*3, (uint8_t*)0+sizeBytes);
        streams.push_back(streamRGB);
        sizeBytes += streamRGB.SizeBytes();
    }

    // Start streaming with default recommended configuration
    pipe->start(*cfg);
    rs2::pipeline_profile profile = pipe->get_active_profile();
    auto sensor = profile.get_device().first<rs2::depth_sensor>();
    auto scale = sensor.get_depth_scale();
    std::cout << "Depth scale is: " << scale << std::endl;

    total_frames = std::numeric_limits<int>::max();
}

RealSense2Video::~RealSense2Video() {
    delete pipe;
    pipe = nullptr;

    delete cfg;
    cfg = nullptr;
}

void RealSense2Video::Start() {
    pipe->start(*cfg);
    current_frame_index = 0;
}

void RealSense2Video::Stop() {
    pipe->stop();
}

size_t RealSense2Video::SizeBytes() const {
    return sizeBytes;
}

const std::vector<StreamInfo>& RealSense2Video::Streams() const {
    return streams;
}

bool RealSense2Video::GrabNext(unsigned char* image, bool /*wait*/) {

    unsigned char* out_img = image;

    rs2::frameset data = pipe->wait_for_frames(); // Wait for next set of frames from the camera
    rs2::frame depth = data.get_depth_frame(); // Get the depth data
    rs2::frame color = data.get_color_frame(); // Get the color data

    memcpy(out_img, depth.get_data(), streams[0].SizeBytes());
    out_img += streams[0].SizeBytes();

    memcpy(out_img, color.get_data(), streams[1].SizeBytes());
    out_img += streams[1].SizeBytes();

    return true;
}

bool RealSense2Video::GrabNewest(unsigned char* image, bool wait) {
    return GrabNext(image, wait);
}

size_t RealSense2Video::GetCurrentFrameId() const {
    return current_frame_index;
}

size_t RealSense2Video::GetTotalFrames() const {
    return total_frames;
}

size_t RealSense2Video::Seek(size_t /*frameid*/) {
    // TODO
    return -1;
}

PANGOLIN_REGISTER_FACTORY(RealSense2Video)
{
    struct RealSense2VideoFactory : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            const ImageDim dim = uri.Get<ImageDim>("size", ImageDim(640,480));
            const unsigned int fps = uri.Get<unsigned int>("fps", 30);
            return std::unique_ptr<VideoInterface>( new RealSense2Video(dim, fps) );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<RealSense2VideoFactory>(), 10, "realsense2");
}

}
