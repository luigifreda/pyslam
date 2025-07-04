#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/video/video.h>
#include <pangolin/video/video_interface.h>

struct IMFActivate;
struct IMFMediaSource;
struct IMFSourceReader;
struct IBaseFilter;
struct IKsControl;

namespace pangolin
{

class PANGOLIN_EXPORT UvcMediaFoundationVideo
    : public pangolin::VideoInterface, public pangolin::VideoUvcInterface, public pangolin::VideoPropertiesInterface
{
  public:
    UvcMediaFoundationVideo(int vendorId, int productId, int deviceId, size_t width, size_t height, int fps);
    ~UvcMediaFoundationVideo();

    //! Implement VideoInput::Start()
    void Start();

    //! Implement VideoInput::Stop()
    void Stop();

    //! Implement VideoInput::SizeBytes()
    size_t SizeBytes() const;

    //! Implement VideoInput::Streams()
    const std::vector<pangolin::StreamInfo>& Streams() const;

    //! Implement VideoInput::GrabNext()
    bool GrabNext(unsigned char* image, bool wait = true);

    //! Implement VideoInput::GrabNewest()
    bool GrabNewest(unsigned char* image, bool wait = true);

    //! Implement VideoUvcInterface::GetCtrl()
    int IoCtrl(uint8_t unit, uint8_t ctrl, unsigned char* data, int len, pangolin::UvcRequestCode req_code);

    //! Implement VideoUvcInterface::GetExposure()
    bool GetExposure(int& exp_us);

    //! Implement VideoUvcInterface::SetExposure()
    bool SetExposure(int exp_us);

    //! Implement VideoUvcInterface::GetGain()
    bool GetGain(float& gain);

    //! Implement VideoUvcInterface::SetGain()
    bool SetGain(float gain);

    //! Access JSON properties of device
    const picojson::value& DeviceProperties() const;

    //! Access JSON properties of most recently captured frame
    const picojson::value& FrameProperties() const;

  protected:
    bool FindDevice(int vendorId, int productId, int deviceId);
    void InitDevice(size_t width, size_t height, int fps);
    void DeinitDevice();

    static bool DeviceMatches(const std::wstring& symLink, int vendorId, int productId);
    static bool SymLinkIDMatches(const std::wstring& symLink, const wchar_t* idStr, int id);

    std::vector<pangolin::StreamInfo> streams;
    size_t size_bytes;

    IMFMediaSource* mediaSource;
    IMFSourceReader* sourceReader;
    IBaseFilter* baseFilter;
    IKsControl* ksControl;
    DWORD ksControlNodeId;

    picojson::value device_properties;
    picojson::value frame_properties;
};
}
