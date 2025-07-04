/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2015 Steven Lovegrove
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

#pragma once

#include <pangolin/pangolin.h>
#include <pangolin/utils/timer.h>
#include <pangolin/video/video.h>

#include <TeliCamApi.h>

namespace pangolin
{

// Video class that outputs test video signal.
class PANGOLIN_EXPORT TeliVideo : public VideoInterface, public VideoPropertiesInterface,
        public BufferAwareVideoInterface, public GenicamVideoInterface
{
public:
    TeliVideo(const Params &p);
    ~TeliVideo();

    Params OpenCameraAndGetRemainingParameters(Params &params);
    
    //! Implement VideoInput::Start()
    void Start();
    
    //! Implement VideoInput::Stop()
    void Stop();

    //! Implement VideoInput::SizeBytes()
    size_t SizeBytes() const;

    //! Implement VideoInput::Streams()
    const std::vector<StreamInfo>& Streams() const;
    
    //! Implement VideoInput::GrabNext()
    bool GrabNext( unsigned char* image, bool wait = true );
    
    //! Implement VideoInput::GrabNewest()
    bool GrabNewest( unsigned char* image, bool wait = true );

    inline Teli::CAM_HANDLE GetCameraHandle() {
        return cam;
    }

    inline Teli::CAM_STRM_HANDLE GetCameraStreamHandle() {
        return strm;
    }

    bool GetParameter(const std::string& name, std::string& result);

    bool SetParameter(const std::string& name, const std::string& value);

    //! Returns number of available frames
    uint32_t AvailableFrames() const;

    //! Drops N frames in the queue starting from the oldest
    //! returns false if less than n frames arae available
    bool DropNFrames(uint32_t n);

    //! Access JSON properties of device
    const picojson::value& DeviceProperties() const;

    //! Access JSON properties of most recently captured frame
    const picojson::value& FrameProperties() const;

    void PopulateEstimatedCenterCaptureTime(pangolin::basetime host_reception_time);

protected:
    void Initialise();
    void InitPangoDeviceProperties();
    void SetDeviceParams(const Params &p);
    void SetNodeValStr(Teli::CAM_HANDLE cam, Teli::CAM_NODE_HANDLE node, std::string node_str, std::string val_str);

    std::vector<StreamInfo> streams;
    size_t size_bytes;

    Teli::CAM_HANDLE cam;
    Teli::CAM_STRM_HANDLE strm;

#ifdef _WIN_
    HANDLE hStrmCmpEvt;
#endif
#ifdef _LINUX_
    Teli::SIGNAL_HANDLE hStrmCmpEvt;
#endif
    double transfer_bandwidth_gbps;
    int exposure_us;
    picojson::value device_properties;
    picojson::value frame_properties;
};

}
