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

#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/drivers/uvc.h>
#include <pangolin/video/iostream_operators.h>

namespace pangolin
{

UvcVideo::UvcVideo(int vendor_id, int product_id, const char* sn, int device_id, int width, int height, int fps)
    : ctx_(NULL),
      dev_(NULL),
      devh_(NULL),
      frame_(NULL),
      is_streaming(false)
{
    uvc_init(&ctx_, NULL);
    if(!ctx_) {
        throw VideoException("Unable to open UVC Context");
    }

    InitDevice(vendor_id, product_id, sn, device_id, width, height, fps);
    InitPangoDeviceProperties();

    // FIX: CRASHING IF WE DON'T START STREAMING STRAIGHT AWAY

    Start();
}

UvcVideo::~UvcVideo()
{
    DeinitDevice();

    if(devh_) uvc_close(devh_);
    if(dev_) uvc_unref_device(dev_);

    if (ctx_) {
        uvc_exit(ctx_);
        ctx_ = 0;
    }
}

uvc_error_t UvcVideo::FindDevice(
    uvc_context_t *ctx, uvc_device_t **dev,
    int vid, int pid, const char *sn, int device_id) {
  uvc_error_t ret = UVC_SUCCESS;

  uvc_device_t **list;
  uvc_device_t *test_dev;

  ret = uvc_get_device_list(ctx, &list);

  int cnt = 0;
  while(list[cnt++]!=NULL);
  pango_print_info("UVC Descriptor list contains %d devices.\n", (cnt-1));

  if (ret != UVC_SUCCESS) {
    return ret;
  }

  int dev_idx = 0;
  int num_found = 0;
  bool found_dev = false;

  while (!found_dev && (test_dev = list[dev_idx++]) != NULL) {
    uvc_device_descriptor_t *desc;

    if (uvc_get_device_descriptor(test_dev, &desc) != UVC_SUCCESS)
      continue;


    const bool matches = (!vid || desc->idVendor == vid)
            && (!pid || desc->idProduct == pid)
            && (!sn || (desc->serialNumber && !strcmp(desc->serialNumber, sn)));

    uvc_free_device_descriptor(desc);

    if (matches) {
        if(device_id == num_found) {
            found_dev = true;
            break;
        }
        num_found++;
    }
  }

  if (found_dev)
    uvc_ref_device(test_dev);

  uvc_free_device_list(list, 1);

  if (found_dev) {
    *dev = test_dev;
    return UVC_SUCCESS;
  } else {
    return UVC_ERROR_NO_DEVICE;
  }
}

void UvcVideo::InitDevice(int vid, int pid, const char* sn, int device_id, int width, int height, int fps)
{
    uvc_error_t find_err = FindDevice(ctx_, &dev_, vid, pid, sn, device_id );
    if (find_err != UVC_SUCCESS) {
        uvc_perror(find_err, "uvc_find_device");
        throw VideoException("Unable to open UVC Device");
    }
    if(!dev_) {
        throw VideoException("Unable to open UVC Device - no pointer returned.");
    }

    uvc_error_t open_err = uvc_open(dev_, &devh_);
    if (open_err != UVC_SUCCESS) {
        uvc_perror(open_err, "uvc_open");
        uvc_unref_device(dev_);
        throw VideoException("Unable to open device");
    }

    //uvc_print_diag(devh_, stderr);

    uvc_error_t mode_err = uvc_get_stream_ctrl_format_size(
                devh_, &ctrl_,
                UVC_FRAME_FORMAT_ANY,
                width, height,
                fps);

    //uvc_print_stream_ctrl(&ctrl_, stderr);

    if (mode_err != UVC_SUCCESS) {
        uvc_perror(mode_err, "uvc_get_stream_ctrl_format_size");
        uvc_close(devh_);
        uvc_unref_device(dev_);
        throw VideoException("Unable to set device mode.");
    }

    uvc_error_t strm_err = uvc_stream_open_ctrl(devh_, &strm_, &ctrl_);
    if(strm_err != UVC_SUCCESS) {
        uvc_perror(strm_err, "uvc_stream_open_ctrl");
        uvc_close(devh_);
        uvc_unref_device(dev_);
        throw VideoException("Unable to open device stream.");
    }

    // Default to greyscale.
    PixelFormat pfmt = PixelFormatFromString("GRAY8");

    const uvc_format_desc_t* uvc_fmt = uvc_get_format_descs(devh_);
    while( uvc_fmt->bFormatIndex != ctrl_.bFormatIndex && uvc_fmt ) {
        uvc_fmt = uvc_fmt->next;
    }

    if(uvc_fmt) {
        // TODO: Use uvc_fmt->fourccFormat
        if( uvc_fmt->bBitsPerPixel == 16 ) {
            pfmt = PixelFormatFromString("GRAY16LE");
        }
    }

    const StreamInfo stream_info(pfmt, width, height, (width*pfmt.bpp)/8, 0);
    streams.push_back(stream_info);
}

void UvcVideo::InitPangoDeviceProperties()
{
    // Store camera details in device properties
    device_properties["BusNumber"] = std::to_string(uvc_get_bus_number(dev_));
    device_properties["DeviceAddress"] = std::to_string(uvc_get_device_address(dev_));
    device_properties[PANGO_HAS_TIMING_DATA] = true;
}

void UvcVideo::DeinitDevice()
{
    Stop();

    if (frame_) {
        uvc_free_frame(frame_);
        frame_ = 0;
    }
}

void UvcVideo::Start()
{
    if(!is_streaming) {
        uvc_error_t stream_err = uvc_stream_start(strm_, NULL, this, 0);

        if (stream_err != UVC_SUCCESS) {
            uvc_perror(stream_err, "uvc_stream_start");
            uvc_close(devh_);
            uvc_unref_device(dev_);
            throw VideoException("Unable to start streaming.");
        }else{
            is_streaming = true;
        }

        if (frame_) {
            uvc_free_frame(frame_);
        }

        size_bytes = ctrl_.dwMaxVideoFrameSize;
        frame_ = uvc_allocate_frame(size_bytes);
        if(!frame_) {
            throw VideoException("Unable to allocate frame.");
        }
    }
}

void UvcVideo::Stop()
{
    if(is_streaming && devh_) {
        uvc_stop_streaming(devh_);
    }
    is_streaming = false;
}

size_t UvcVideo::SizeBytes() const
{
    return size_bytes;
}

const std::vector<StreamInfo>& UvcVideo::Streams() const
{
    return streams;
}

bool UvcVideo::GrabNext( unsigned char* image, bool wait )
{
    uvc_frame_t* frame = NULL;
    uvc_error_t err = uvc_stream_get_frame(strm_, &frame, wait ? 0 : -1);

    if(err!= UVC_SUCCESS) {
        pango_print_error("UvcVideo Error: %s", uvc_strerror(err) );
        return false;
    }else{
        if(frame) {
            memcpy(image, frame->data, frame->data_bytes );
            // This is a hack, this ts sould come from the device.
            frame_properties[PANGO_HOST_RECEPTION_TIME_US] = picojson::value(pangolin::Time_us(pangolin::TimeNow()));
            return true;
        }else{
            if(wait) {
                pango_print_debug("UvcVideo: No frame data");
            }
            return false;
        }
    }
}

bool UvcVideo::GrabNewest( unsigned char* image, bool wait )
{
    return GrabNext(image, wait);
}

int UvcVideo::IoCtrl(uint8_t unit, uint8_t ctrl, unsigned char* data, int len, UvcRequestCode req_code)
{
    if(req_code == UVC_SET_CUR) {
        return uvc_set_ctrl(devh_, unit, ctrl, data, len);
    }else{
        return uvc_get_ctrl(devh_, unit, ctrl, data, len, (uvc_req_code)req_code);
    }
}

bool UvcVideo::SetExposure(int exp_us)
{
    uint32_t e = uint32_t(exp_us);

    if (uvc_set_exposure_abs(devh_, e) < 0) {
        pango_print_warn("UvcVideo::setExposure() ioctl error: %s\n", strerror(errno));
        return false;
    } else {
        return true;
    }
}

bool UvcVideo::GetExposure(int& exp_us)
{
    uint32_t e;
    if (uvc_get_exposure_abs(devh_, &e, uvc_req_code::UVC_GET_CUR) < 0) {
        pango_print_warn("UvcVideo::GetExposure() ioctl error: %s\n", strerror(errno));
        return false;
    } else {
        exp_us = e;
        return true;
    }
}

bool UvcVideo::SetGain(float gain)
{
    uint16_t g = uint16_t(gain);

    if (uvc_set_gain(devh_, g) < 0) {
        pango_print_warn("UvcVideo::setGain() ioctl error: %s\n", strerror(errno));
        return false;
    } else {
        return true;
    }
}

bool UvcVideo::GetGain(float& gain)
{
    uint16_t g;
    if (uvc_get_gain(devh_, &g, uvc_req_code::UVC_GET_CUR) < 0) {
        pango_print_warn("UvcVideo::GetGain() ioctl error: %s\n", strerror(errno));
        return false;
    } else {
        gain = g;
        return true;
    }
}

//! Access JSON properties of device
const picojson::value& UvcVideo::DeviceProperties() const
{
        return device_properties;
}

//! Access JSON properties of most recently captured frame
const picojson::value& UvcVideo::FrameProperties() const
{
    return frame_properties;
}

PANGOLIN_REGISTER_FACTORY(UvcVideo)
{
    struct UvcVideoFactory final : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            int vid = 0;
            int pid = 0;
            std::istringstream(uri.Get<std::string>("vid","0x0000")) >> std::hex >> vid;
            std::istringstream(uri.Get<std::string>("pid","0x0000")) >> std::hex >> pid;
            const unsigned int dev_id = uri.Get<int>("num",0);
            const ImageDim dim = uri.Get<ImageDim>("size", ImageDim(640,480));
            const unsigned int fps = uri.Get<unsigned int>("fps", 0); // 0 means unspecified.
            return std::unique_ptr<VideoInterface>( new UvcVideo(vid,pid,0,dev_id,dim.x,dim.y,fps) );
        }
    };

    FactoryRegistry<VideoInterface>::I().RegisterFactory(std::make_shared<UvcVideoFactory>(), 10, "uvc");
}

}
