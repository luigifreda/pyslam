/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) Andrey Mnatsakanov
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

#include "video.hpp"
#include <pangolin/video/video_interface.h>
#include <pangolin/video/video_input.h>

#include <pybind11/numpy.h>
#include <pybind11/stl.h>

namespace py_pangolin {

    class PyVideoInterface: public pangolin::VideoInterface{
    public:
      using pangolin::VideoInterface::VideoInterface;

      size_t SizeBytes() const override {
        PYBIND11_OVERLOAD_PURE(
                               size_t,
                               pangolin::VideoInterface,
                               SizeBytes);
      }

      const std::vector<pangolin::StreamInfo>& Streams() const override {
        PYBIND11_OVERLOAD_PURE(
                               const std::vector<pangolin::StreamInfo>&,
                               pangolin::VideoInterface,
                               Streams);
      }

      void Start() override {
        PYBIND11_OVERLOAD_PURE(
                               void,
                               pangolin::VideoInterface,
                               Start);
      }

      void Stop() override {
        PYBIND11_OVERLOAD_PURE(
                               void,
                               pangolin::VideoInterface,
                               Stop);
      }

      bool GrabNext(unsigned char* image, bool wait = true) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::VideoInterface,
                               GrabNext,
                               image,
                               wait);
      }

      bool GrabNewest(unsigned char* image, bool wait = true) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::VideoInterface,
                               GrabNewest,
                               image,
                               wait);
      }
    };

    class PyGenicamVideoInterface: public pangolin::GenicamVideoInterface{
    public:
      using pangolin::GenicamVideoInterface::GenicamVideoInterface;

      bool GetParameter(const std::string& name, std::string& result) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::GenicamVideoInterface,
                               GetParameter,
                               name,
                               result);
      }

      bool SetParameter(const std::string& name, const std::string& value) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::GenicamVideoInterface,
                               SetParameter,
                               name,
                               value);
      }
      size_t CameraCount() const override
      {
          PYBIND11_OVERLOAD_PURE(
                                 size_t,
                                 pangolin::GenicamVideoInterface,
                                 CameraCount);
      }



    };

    class PyBufferAwareVideoInterface: public pangolin::BufferAwareVideoInterface{
    public:
      using pangolin::BufferAwareVideoInterface::BufferAwareVideoInterface;

      uint32_t AvailableFrames() const override {
        PYBIND11_OVERLOAD_PURE(
                               uint32_t,
                               pangolin::BufferAwareVideoInterface,
                               AvailableFrames);
      }

      bool DropNFrames(uint32_t n) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::BufferAwareVideoInterface,
                               DropNFrames,
                               n);
      }
    };

    class PyVideoPropertiesInterface: public pangolin::VideoPropertiesInterface{
    public:
      using pangolin::VideoPropertiesInterface::VideoPropertiesInterface;

      const picojson::value& DeviceProperties() const override {
        PYBIND11_OVERLOAD_PURE(
                               const picojson::value&,
                               pangolin::VideoPropertiesInterface,
                               DeviceProperties);
      }

      const picojson::value& FrameProperties() const override {
        PYBIND11_OVERLOAD_PURE(
                               const picojson::value&,
                               pangolin::VideoPropertiesInterface,
                               FrameProperties);
      }
    };  

    class PyVideoFilterInterface: public pangolin::VideoFilterInterface{
    public:
      using pangolin::VideoFilterInterface::VideoFilterInterface;

      // template <typename T>
      // std::vector<T*> FindMatchingStreams() override {
      //   PYBIND11_OVERLOAD(
      //                     std::vector<T*>,
      //                     pangolin::VideoFilterInterface,
      //                     FindMatchingStreams);
      // }

      std::vector<pangolin::VideoInterface*>& InputStreams() override {
        PYBIND11_OVERLOAD_PURE(
                               std::vector<pangolin::VideoInterface*>&,
                               pangolin::VideoFilterInterface,
                               InputStreams);
      }
    };  
    
    class PyVideoUvcInterface: public pangolin::VideoUvcInterface{
    public:
      using pangolin::VideoUvcInterface::VideoUvcInterface;

      int IoCtrl(uint8_t unit, uint8_t ctrl, unsigned char* data, int len, pangolin::UvcRequestCode req_code) override {
        PYBIND11_OVERLOAD_PURE(
                               int,
                               pangolin::VideoUvcInterface,
                               IoCtrl,
                               unit,
                               ctrl,
                               data,
                               len,
                               req_code);
      }

      bool GetExposure(int& exp_us) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::VideoUvcInterface,
                               GetExposure,
                               exp_us);
      }

      bool SetExposure(int exp_us) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::VideoUvcInterface,
                               SetExposure,
                               exp_us);
      }

      bool GetGain(float& gain) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::VideoUvcInterface,
                               GetGain,
                               gain);
      }

      bool SetGain(float gain) override {
        PYBIND11_OVERLOAD_PURE(
                               bool,
                               pangolin::VideoUvcInterface,
                               SetGain,
                               gain);
      }
    };
  
    class PyVideoPlaybackInterface: public pangolin::VideoPlaybackInterface{
    public:
      using pangolin::VideoPlaybackInterface::VideoPlaybackInterface;
    
      size_t GetCurrentFrameId() const override {
        PYBIND11_OVERLOAD_PURE(
                               size_t,
                               pangolin::VideoPlaybackInterface,
                               GetCurrentFrameId);
      }

      size_t GetTotalFrames() const override {
        PYBIND11_OVERLOAD_PURE(
                               size_t,
                               pangolin::VideoPlaybackInterface,
                               GetTotalFrames);
      }

      size_t Seek(size_t frameid) override {
        PYBIND11_OVERLOAD_PURE(
                               size_t,
                               pangolin::VideoPlaybackInterface,
                               Seek,
                               frameid);
      }
    };

  class PyVideoOutputInterface: public pangolin::VideoOutputInterface{
  public:
    using pangolin::VideoOutputInterface::VideoOutputInterface;
    
    const std::vector<pangolin::StreamInfo>& Streams() const override {
      PYBIND11_OVERLOAD_PURE(
                             const std::vector<pangolin::StreamInfo>&,
                             pangolin::VideoOutputInterface,
                             Streams);
    }

    void SetStreams(const std::vector<pangolin::StreamInfo>& streams, const std::string& uri, const picojson::value& properties) override {
      PYBIND11_OVERLOAD_PURE(
                             void,
                             pangolin::VideoOutputInterface,
                             SetStreams,
                             streams,
                             uri,
                             properties);
    }

    int WriteStreams(const unsigned char* data, const picojson::value& frame_properties) override {
      PYBIND11_OVERLOAD_PURE(
                             int,
                             pangolin::VideoOutputInterface,
                             WriteStreams,
                             data,
                             frame_properties);
    }

    bool IsPipe() const override {
      PYBIND11_OVERLOAD_PURE(
                             bool,
                             pangolin::VideoOutputInterface,
                             IsPipe);
    }

  };

  pybind11::list VideoInputGrab(pangolin::VideoInput& vi, bool wait, bool newest){
      unsigned char *buffer = new unsigned char[vi.SizeBytes()];

      std::vector<pangolin::Image<unsigned char>> imgs;
      pybind11::list imgsList;

      if(vi.Grab(buffer,imgs,wait,newest)) {
          for(size_t s=0; s < vi.Streams().size(); ++s) {
              // Let's just return the first stream for the moment
              const pangolin::StreamInfo& si = vi.Streams()[s];
              const pangolin::Image<uint8_t> img = si.StreamImage(buffer);

              const int c = si.PixFormat().channels;
              const std::string fmt = si.PixFormat().format;
              const int Bpp = si.PixFormat().bpp / (8);
              const int Bpc = Bpp / c;
              const int bpc = si.PixFormat().bpp / c;
              PANGO_ASSERT(bpc == 8 || bpc == 16 || bpc == 32, "only support 8, 16, 32 bits channel");

              pangolin::Image<uint8_t> dstImage(
                  new unsigned char[img.h * img.w * Bpp],
                  img.w, img.h, img.w*Bpp
              );

              pangolin::PitchedCopy((char*)dstImage.ptr, dstImage.pitch, (char*)img.ptr, img.pitch, img.w * Bpp, img.h);

              // Create a Python object that will free the allocated memory
              pybind11::capsule free_when_done(dstImage.ptr,[](void* f) {
                unsigned char* buffer = (unsigned char*)f;
                delete[] buffer;
              });

              if (bpc == 8) {
                  imgsList.append(
                      pybind11::array_t<uint8_t>(
                      {(int)dstImage.h, (int)dstImage.w, c },
                      {(int)dstImage.pitch, Bpp, Bpc},
                      (uint8_t*)dstImage.ptr,
                      free_when_done)
                      );
              }
              else if (bpc == 16){
                  imgsList.append(
                      pybind11::array_t<uint16_t>(
                      {(int)dstImage.h, (int)dstImage.w, c },
                      {(int)dstImage.pitch, Bpp, Bpc},
                      (uint16_t*)dstImage.ptr,
                      free_when_done)
                      );
              }
              else if (bpc == 32){
                  if (fmt == "GRAY32")
                  {
                      imgsList.append(
                          pybind11::array_t<uint32_t>(
                          {(int)dstImage.h, (int)dstImage.w, c },
                          {(int)dstImage.pitch, Bpp, Bpc},
                          (uint32_t*)dstImage.ptr,
                          free_when_done)
                          );
                  }
                  else if (fmt == "GRAY32F" ||
                           fmt == "RGB96F" ||
                           fmt == "RGBA128F")
                  {
                      imgsList.append(
                          pybind11::array_t<float>(
                          {(int)dstImage.h, (int)dstImage.w, c },
                          {(int)dstImage.pitch, Bpp, Bpc},
                          (float*)dstImage.ptr,
                          free_when_done)
                          );
                  }
                  else{
                      PANGO_ASSERT(false, "unsupported 32 bpc format");
                  }
              }
              else{
                  PANGO_ASSERT(false, "incompatible bpc");
              }
          }
      }
      delete[] buffer;
      return imgsList;
  }

  picojson::value PicojsonFromPyObject(pybind11::object& obj)
  {
      // convert frame_properties to std::string via json.dumps(...)
      pybind11::module pymodjson = pybind11::module::import("json");
      auto pydumps = pymodjson.attr("dumps");
      const std::string json = pydumps(obj).cast<pybind11::str>();
      std::stringstream ss(json);
      picojson::value pjson;
      picojson::parse(pjson, ss);
      return pjson;
  }

  void bind_video(pybind11::module& m){
        pybind11::class_<pangolin::VideoInterface, PyVideoInterface > video_interface(m, "VideoInterface");
    video_interface
      .def(pybind11::init<>())
      .def("SizeBytes", &pangolin::VideoInterface::SizeBytes)
      .def("Streams", &pangolin::VideoInterface::Streams)
      .def("Start", &pangolin::VideoInterface::Start)
      .def("Stop", &pangolin::VideoInterface::Stop)
      .def("GrabNext", &pangolin::VideoInterface::GrabNext)
      .def("GrabNewest", &pangolin::VideoInterface::GrabNewest);

    pybind11::class_<pangolin::GenicamVideoInterface, PyGenicamVideoInterface > genicam_video_interface(m, "GenicamVideoInterface");
    genicam_video_interface
      .def(pybind11::init<>())
      .def("GetParameter", &pangolin::GenicamVideoInterface::GetParameter)
      .def("SetParameter", &pangolin::GenicamVideoInterface::SetParameter);

    pybind11::class_<pangolin::BufferAwareVideoInterface, PyBufferAwareVideoInterface > buffer_aware_video_interface(m, "BufferAwareVideoInterface");
    buffer_aware_video_interface
      .def(pybind11::init<>())
      .def("AvailableFrames", &pangolin::BufferAwareVideoInterface::AvailableFrames)
      .def("DropNFrames", &pangolin::BufferAwareVideoInterface::DropNFrames);

    pybind11::class_<pangolin::VideoPropertiesInterface, PyVideoPropertiesInterface > video_properties_interface(m, "VideoPropertiesInterface");
    video_properties_interface
      .def(pybind11::init<>())
      .def("DeviceProperties", &pangolin::VideoPropertiesInterface::DeviceProperties)
      .def("FrameProperties", &pangolin::VideoPropertiesInterface::FrameProperties);

    pybind11::class_<pangolin::VideoFilterInterface, PyVideoFilterInterface > video_filter_interface(m, "VideoFilterInterface");
    video_filter_interface
      .def(pybind11::init<>())
      //      .def("FindMatchingStreams", &pangolin::VideoFilterInterface::FindMatchingStreams)
      .def("InputStreams", &pangolin::VideoFilterInterface::InputStreams);
    
    pybind11::class_<pangolin::VideoUvcInterface, PyVideoUvcInterface > video_uvc_interface(m, "VideoUvcInterface");
    video_uvc_interface
      .def(pybind11::init<>())
      .def("IoCtrl", &pangolin::VideoUvcInterface::IoCtrl)
      .def("GetExposure", &pangolin::VideoUvcInterface::GetExposure)
      .def("SetExposure", &pangolin::VideoUvcInterface::SetExposure)
      .def("GetGain", &pangolin::VideoUvcInterface::GetGain)
      .def("SetGain", &pangolin::VideoUvcInterface::SetGain);

    pybind11::class_<pangolin::VideoPlaybackInterface, PyVideoPlaybackInterface > video_playback_interface(m, "VideoPlaybackInterface");
    video_playback_interface
      .def(pybind11::init<>())
      .def("GetCurrentFrameId", &pangolin::VideoPlaybackInterface::GetCurrentFrameId)
      .def("GetTotalFrames", &pangolin::VideoPlaybackInterface::GetTotalFrames)
      .def("Seek", &pangolin::VideoPlaybackInterface::Seek);

    pybind11::class_<pangolin::VideoOutputInterface, PyVideoOutputInterface > video_output_interface(m, "VideoOutputInterface");
    video_output_interface
      .def(pybind11::init<>())
      .def("Streams", &pangolin::VideoOutputInterface::Streams)
      .def("SetStreams", &pangolin::VideoOutputInterface::SetStreams) 
      .def("WriteStreams", &pangolin::VideoOutputInterface::WriteStreams)
      .def("IsPipe", &pangolin::VideoOutputInterface::IsPipe);

    pybind11::enum_<pangolin::UvcRequestCode>(m, "UvcRequestCode")
      .value("UVC_RC_UNDEFINED", pangolin::UvcRequestCode::UVC_RC_UNDEFINED)
      .value("UVC_SET_CUR", pangolin::UvcRequestCode::UVC_SET_CUR)
      .value("UVC_GET_CUR", pangolin::UvcRequestCode::UVC_GET_CUR)
      .value("UVC_GET_MIN", pangolin::UvcRequestCode::UVC_GET_MIN)
      .value("UVC_GET_MAX", pangolin::UvcRequestCode::UVC_GET_MAX)
      .value("UVC_GET_RES", pangolin::UvcRequestCode::UVC_GET_RES)
      .value("UVC_GET_LEN", pangolin::UvcRequestCode::UVC_GET_LEN)
      .value("UVC_GET_INFO", pangolin::UvcRequestCode::UVC_GET_INFO)
      .value("UVC_GET_DEF", pangolin::UvcRequestCode::UVC_GET_DEF)
      .export_values();

    /// This iterator enable pythonic video iterations a la `for frame in video_input: ...`
    struct VideoInputIterator {
        VideoInputIterator(pangolin::VideoInput& vi, pybind11::object ref) : vi(vi), ref(ref) { }

        pybind11::list next() {
            pybind11::list result = VideoInputGrab(vi, true, false);
            if(result.size()==0)
                throw pybind11::stop_iteration();
            return result;
        }
        pangolin::VideoInput& vi;
        pybind11::object ref; // keep a reference
    };
    pybind11::class_<VideoInputIterator>(m, "Iterator")
      .def("__iter__", [](VideoInputIterator &it) -> VideoInputIterator& { return it; })
      .def("__next__", &VideoInputIterator::next);

    pybind11::class_<pangolin::VideoInput>(m, "VideoInput", video_interface)
      .def(pybind11::init<>())
      .def(pybind11::init<const std::string&, const std::string&>(), pybind11::arg("input_uri"), pybind11::arg("output_uri")="pango:[buffer_size_mb=100]//video_log.pango")      
      .def("SizeBytes", &pangolin::VideoInput::SizeBytes)
      .def("Streams", &pangolin::VideoInput::Streams)
      .def("Start", &pangolin::VideoInput::Start)
      .def("Stop", &pangolin::VideoInput::Stop)
      .def("InputStreams", &pangolin::VideoInput::InputStreams)
      .def("Open", &pangolin::VideoInput::Open, pybind11::arg("input_uri"), pybind11::arg("output_uri")="pango:[buffer_size_mb=100]//video_log.pango")      
      .def("Close", &pangolin::VideoInput::Close)
      .def("Grab", VideoInputGrab, pybind11::arg("wait")=true, pybind11::arg("newest")=false )
      .def("GetStreamsBitDepth", [](pangolin::VideoInput& vi){
        std::vector<int> bitDepthList;
        for(size_t s=0; s < vi.Streams().size(); ++s) {
            bitDepthList.push_back(vi.Streams()[s].PixFormat().channel_bit_depth);
        }
        return bitDepthList;
       })
      .def("GetNumStreams", [](pangolin::VideoInput& vi){
        return (int) vi.Streams().size();
        })
      .def("GetCurrentFrameId", [](pangolin::VideoInput& vi){
        return (int) vi.Cast<pangolin::VideoPlaybackInterface>()->GetCurrentFrameId();
        })
      .def("GetTotalFrames", [](pangolin::VideoInput& vi){
        return (int) vi.Cast<pangolin::VideoPlaybackInterface>()->GetTotalFrames();
        })
      .def("Seek", [](pangolin::VideoInput& vi, size_t frameid){
        vi.Cast<pangolin::VideoPlaybackInterface>()->Seek(frameid);
        return;
        })
      .def("DeviceProperties", [](pangolin::VideoInput& vi) -> pybind11::object {
            // Use std::string as an intermediate representation
            const std::string props = vi.template Cast<pangolin::VideoPropertiesInterface>()->DeviceProperties().serialize();
            pybind11::module pymodjson = pybind11::module::import("json");
            auto pyloads = pymodjson.attr("loads");
            auto json = pyloads(pybind11::str(props));
            return json;
            })
      .def("FrameProperties", [](pangolin::VideoInput& vi) -> pybind11::object {
            // Use std::string as an intermediate representation
            const std::string props = vi.template Cast<pangolin::VideoPropertiesInterface>()->FrameProperties().serialize();
            pybind11::module pymodjson = pybind11::module::import("json");
            auto pyloads = pymodjson.attr("loads");
            auto json = pyloads(pybind11::str(props));
            return json;
            })
      .def("Width", &pangolin::VideoInput::Width)
      .def("Height", &pangolin::VideoInput::Height)
      .def("PixFormat", &pangolin::VideoInput::PixFormat)
      .def("VideoUri", &pangolin::VideoInput::VideoUri)
      .def("Reset", &pangolin::VideoInput::Reset)
      .def("LogFilename", (const std::string& (pangolin::VideoInput::*)() const)&pangolin::VideoInput::LogFilename)
      .def("LogFilename", (std::string& (pangolin::VideoInput::*)())&pangolin::VideoInput::LogFilename)
      .def("Record", &pangolin::VideoInput::Record)
      .def("RecordOneFrame", &pangolin::VideoInput::RecordOneFrame)
      .def("SetTimelapse", &pangolin::VideoInput::SetTimelapse)
      .def("IsRecording", &pangolin::VideoInput::IsRecording)
      .def("__iter__", [](pybind11::object s) { return VideoInputIterator(s.cast<pangolin::VideoInput&>(), s);});

    pybind11::class_<pangolin::VideoOutput>(m, "VideoOutput", video_output_interface)
      .def(pybind11::init<>())
      .def(pybind11::init<const std::string&>())
      .def("IsOpen", &pangolin::VideoOutput::IsOpen)
      .def("Open", &pangolin::VideoOutput::Open)
      .def("Close", &pangolin::VideoOutput::Close)
      .def("Streams", &pangolin::VideoOutput::Streams)
      .def("WriteStreams", [](pangolin::VideoOutput& vo, pybind11::list images, const std::vector<int> &streamsBitDepth, pybind11::object frame_properties, pybind11::object device_properties, const std::string& descriptive_uri){
        if(vo.SizeBytes()==0) {
            PANGO_ASSERT(streamsBitDepth.size() == images.size() || streamsBitDepth.size() == 0);

            // Setup stream info
            for(size_t i = 0; i < images.size(); ++i){
                // num bits per channel
                auto arr = pybind11::array::ensure(images[i]);

                // num channels
                PANGO_ASSERT(arr.ndim() == 2 || arr.ndim() == 3, "Method only accepts ndarrays of 2 or 3 dimensions.");
                const size_t channels = (arr.ndim() == 3) ? arr.shape(2) : 1;

                std::string fmtStr;
                if(pybind11::isinstance<pybind11::array_t<std::uint8_t>>(arr)){
                   if(channels == 1) fmtStr = "GRAY8";
                   else if(channels == 3) fmtStr = "RGB24";
                   else if(channels == 4) fmtStr = "RGBA32";
                   else PANGO_ASSERT(false, "Only 1, 3 and 4 channel uint8_t formats are supported.");
                } else if (pybind11::isinstance<pybind11::array_t<std::uint16_t>>(arr)){
                   if(channels == 1) fmtStr = "GRAY16LE";
                   else if(channels == 3) fmtStr = "RGB48";
                   else if(channels == 4) fmtStr = "RGBA64";
                   else PANGO_ASSERT(false, "Only 1, 3 and 4 channel uint16_t formats are supported.");
                } else if (pybind11::isinstance<pybind11::array_t<std::float_t>>(arr)){
                    if(channels == 1) fmtStr = "GRAY32F";
                    else if(channels == 3) fmtStr = "RGB96F";
                    else if(channels == 4) fmtStr = "RGBA128F";
                    else PANGO_ASSERT(false, "Only 1, 3 and 4 channel float_t formats are supported.");
                } else if (pybind11::isinstance<pybind11::array_t<std::double_t>>(arr)){
                    if(channels == 1) fmtStr = "GRAY64F";
                    else PANGO_ASSERT(false, "Only 1 channel double_t format is supported.");
                } else {
                    PANGO_ASSERT(false, "numpy dtype must be either uint8_t, uint16_t, float_t or double_t");
                }

                pangolin::PixelFormat pf = pangolin::PixelFormatFromString(fmtStr);
                if(streamsBitDepth.size())
                    pf.channel_bit_depth = (unsigned int) streamsBitDepth[i];

                vo.AddStream(pf, arr.shape(1), arr.shape(0));
            }

            picojson::value json_device_properties;
            if(device_properties) {
                json_device_properties = PicojsonFromPyObject(device_properties);
            }

            vo.SetStreams(descriptive_uri, json_device_properties);
        }

        picojson::value json_frame_properties;
        if(frame_properties) {
            json_frame_properties = PicojsonFromPyObject(frame_properties);
        }

        std::unique_ptr<uint8_t[]> buffer(new uint8_t[vo.SizeBytes()]);
        std::vector<pangolin::Image<unsigned char>> bimgs = vo.GetOutputImages(buffer.get());
        PANGO_ASSERT(bimgs.size() == images.size(), "length of input streams not consistent");
        for(size_t i=0; i < images.size(); ++i) {
            const pangolin::StreamInfo& so = vo.Streams()[i];
            const unsigned int bpc = so.PixFormat().channel_bit_depth;
            const unsigned int Bpp = so.PixFormat().bpp / (8);
            if (bpc == 8){
                pybind11::array_t<uint8_t> arr = images[i].cast<pybind11::array_t<unsigned char, pybind11::array::c_style | pybind11::array::forcecast>>();
                pangolin::Image<uint8_t> srcImg(arr.mutable_data(0,0), bimgs[i].w, bimgs[i].h, bimgs[i].w * Bpp);
                pangolin::PitchedCopy((char*)bimgs[i].ptr, bimgs[i].pitch, (char*)srcImg.ptr,
                                      srcImg.pitch, srcImg.pitch, srcImg.h);
            }else if (bpc == 12 || bpc == 16){
                pybind11::array_t<uint16_t> arr = images[i].cast<pybind11::array_t<uint16_t, pybind11::array::c_style | pybind11::array::forcecast>>();
                pangolin::Image<uint16_t> srcImg(arr.mutable_data(0,0), bimgs[i].w, bimgs[i].h, bimgs[i].w * Bpp);
                pangolin::PitchedCopy((char*)bimgs[i].ptr, bimgs[i].pitch, (char*)srcImg.ptr,
                                      srcImg.pitch, srcImg.pitch, srcImg.h);
            }else if (bpc == 32){
                pybind11::array_t<float_t> arr = images[i].cast<pybind11::array_t<float_t, pybind11::array::c_style | pybind11::array::forcecast>>();
                pangolin::Image<float_t> srcImg(arr.mutable_data(0,0), bimgs[i].w, bimgs[i].h, bimgs[i].w * Bpp);
                pangolin::PitchedCopy((char*)bimgs[i].ptr, bimgs[i].pitch, (char*)srcImg.ptr,
                                      srcImg.pitch, srcImg.pitch, srcImg.h);
            }else if (bpc == 64){
                pybind11::array_t<double_t> arr = images[i].cast<pybind11::array_t<double_t, pybind11::array::c_style | pybind11::array::forcecast>>();
                pangolin::Image<double_t> srcImg(arr.mutable_data(0,0), bimgs[i].w, bimgs[i].h, bimgs[i].w * Bpp);
                pangolin::PitchedCopy((char*)bimgs[i].ptr, bimgs[i].pitch, (char*)srcImg.ptr,
                                      srcImg.pitch, srcImg.pitch, srcImg.h);
            }else{
                PANGO_ASSERT(false, "format must have 8, 12, 16, 32 or 64 bit depth");
            }
        }
        vo.WriteStreams(buffer.get(), json_frame_properties);
      }, pybind11::arg("images"), pybind11::arg("streamsBitDepth") = std::vector<int>(), pybind11::arg("frame_properties") = pybind11::none(), pybind11::arg("device_properties") = pybind11::none(), pybind11::arg("descriptive_uri") = "python://")
      .def("IsPipe", &pangolin::VideoOutput::IsPipe)
      .def("AddStream", (void (pangolin::VideoOutput::*)(const pangolin::PixelFormat&, size_t,size_t,size_t))&pangolin::VideoOutput::AddStream)
      .def("AddStream", (void (pangolin::VideoOutput::*)(const pangolin::PixelFormat&, size_t,size_t))&pangolin::VideoOutput::AddStream)
      .def("SizeBytes", &pangolin::VideoOutput::SizeBytes)
      .def("GetOutputImages", (std::vector<pangolin::Image<unsigned char>> (pangolin::VideoOutput::*)(unsigned char*) const)&pangolin::VideoOutput::GetOutputImages)    
      .def("GetOutputImages", (std::vector<pangolin::Image<unsigned char>> (pangolin::VideoOutput::*)(std::vector<unsigned char>&) const)&pangolin::VideoOutput::GetOutputImages);   
    
  }
}  // py_pangolin
