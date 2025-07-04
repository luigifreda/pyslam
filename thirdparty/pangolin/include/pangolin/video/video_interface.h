/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Steven Lovegrove
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

#include <pangolin/utils/picojson.h>
#include <pangolin/video/stream_info.h>

#include <memory>
#include <vector>

#define PANGO_HAS_TIMING_DATA        "has_timing_data"
#define PANGO_HOST_RECEPTION_TIME_US "host_reception_time_us"
#define PANGO_CAPTURE_TIME_US        "capture_time_us"
#define PANGO_EXPOSURE_US            "exposure_us"
#define PANGO_GAMMA                  "gamma"
// analog gain is in linear scale and not dB
#define PANGO_ANALOG_GAIN            "analog_gain" 
#define PANGO_ANALOG_BLACK_LEVEL     "analog_black_level"
#define PANGO_SENSOR_TEMPERATURE_C   "sensor_temperature_C"
#define PANGO_ESTIMATED_CENTER_CAPTURE_TIME_US "estimated_center_capture_time_us"
#define PANGO_JOIN_OFFSET_US         "join_offset_us"
#define PANGO_FRAME_COUNTER          "frame_counter"

namespace pangolin {

//! Interface to video capture sources
struct PANGOLIN_EXPORT VideoInterface
{
    virtual ~VideoInterface() {}

    //! Required buffer size to store all frames
    virtual size_t SizeBytes() const = 0;

    //! Get format and dimensions of all video streams
    virtual const std::vector<StreamInfo>& Streams() const = 0;

    //! Start Video device
    virtual void Start() = 0;

    //! Stop Video device
    virtual void Stop() = 0;

    //! Copy the next frame from the camera to image.
    //! Optionally wait for a frame if one isn't ready
    //! Returns true iff image was copied
    virtual bool GrabNext( unsigned char* image, bool wait = true ) = 0;

    //! Copy the newest frame from the camera to image
    //! discarding all older frames.
    //! Optionally wait for a frame if one isn't ready
    //! Returns true iff image was copied
    virtual bool GrabNewest( unsigned char* image, bool wait = true ) = 0;
};

//! Interface to GENICAM video capture sources
struct PANGOLIN_EXPORT GenicamVideoInterface
{
    virtual ~GenicamVideoInterface() {}

    virtual bool GetParameter(const std::string& name, std::string& result) = 0;

    virtual bool SetParameter(const std::string& name, const std::string& value) = 0;

    virtual size_t CameraCount() const
    {
        return 1;
    }
};

struct PANGOLIN_EXPORT BufferAwareVideoInterface
{
    virtual ~BufferAwareVideoInterface() {}

    //! Returns number of available frames
    virtual uint32_t AvailableFrames() const = 0;

    //! Drops N frames in the queue starting from the oldest
    //! returns false if less than n frames arae available
    virtual bool DropNFrames(uint32_t n) = 0;
};

struct PANGOLIN_EXPORT VideoPropertiesInterface
{
    virtual ~VideoPropertiesInterface() {}

    //! Access JSON properties of device
    virtual const picojson::value& DeviceProperties() const = 0;

    //! Access JSON properties of most recently captured frame
    virtual const picojson::value& FrameProperties() const = 0;
};

enum UvcRequestCode {
  UVC_RC_UNDEFINED = 0x00,
  UVC_SET_CUR = 0x01,
  UVC_GET_CUR = 0x81,
  UVC_GET_MIN = 0x82,
  UVC_GET_MAX = 0x83,
  UVC_GET_RES = 0x84,
  UVC_GET_LEN = 0x85,
  UVC_GET_INFO = 0x86,
  UVC_GET_DEF = 0x87
};

struct PANGOLIN_EXPORT VideoFilterInterface
{
    virtual ~VideoFilterInterface() {}

    template<typename T>
    std::vector<T*> FindMatchingStreams()
    {
        std::vector<T*> matches;
        std::vector<VideoInterface*> children = InputStreams();
        for(size_t c=0; c < children.size(); ++c) {
            T* concrete_video = dynamic_cast<T*>(children[c]);
            if(concrete_video) {
                matches.push_back(concrete_video);
            }else{
                VideoFilterInterface* filter_video = dynamic_cast<VideoFilterInterface*>(children[c]);
                if(filter_video) {
                    std::vector<T*> child_matches = filter_video->FindMatchingStreams<T>();
                    matches.insert(matches.end(), child_matches.begin(), child_matches.end());
                }
            }
        }
        return matches;
    }

    virtual std::vector<VideoInterface*>& InputStreams() = 0;
};

struct PANGOLIN_EXPORT VideoUvcInterface
{
    virtual ~VideoUvcInterface() {}
    virtual int IoCtrl(uint8_t unit, uint8_t ctrl, unsigned char* data, int len, UvcRequestCode req_code) = 0;
    virtual bool GetExposure(int& exp_us) = 0;
    virtual bool SetExposure(int exp_us) = 0;
    virtual bool GetGain(float& gain) = 0;
    virtual bool SetGain(float gain) = 0;
};

struct PANGOLIN_EXPORT VideoPlaybackInterface
{
    virtual ~VideoPlaybackInterface() {}

    /// Return monotonic id of current frame
    /// The 'current frame' is the frame returned from the last successful call to Grab
    virtual size_t GetCurrentFrameId() const = 0;

    /// Return total number of frames to be captured from device,
    /// or 0 if unknown.
    virtual size_t GetTotalFrames() const = 0;

    /// Return frameid on success, or next frame on failure
    virtual size_t Seek(size_t frameid) = 0;
};

}
