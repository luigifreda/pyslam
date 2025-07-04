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

// Pangolin video supports various cameras and file formats through
// different 3rd party libraries.
//
// Video URI's take the following form:
//  scheme:[param1=value1,param2=value2,...]//device
//
// scheme = file | files | pango | shmem | dc1394 | uvc | v4l | openni2 |
//          openni | depthsense | pleora | teli | mjpeg | test |
//          thread | convert | debayer | split | join | shift | mirror | unpack
//
// file/files - read one or more streams from image file(s) / video
//  e.g. "files://~/data/dataset/img_*.jpg"
//  e.g. "files://~/data/dataset/img_[left,right]_*.pgm"
//  e.g. "files:///home/user/sequence/foo%03d.jpeg"
//
//  e.g. "file:[fmt=GRAY8,size=640x480]///home/user/raw_image.bin"
//  e.g. "file:[realtime=1]///home/user/video/movie.pango"
//  e.g. "file:[stream=1]///home/user/video/movie.avi"
//
// dc1394 - capture video through a firewire camera
//  e.g. "dc1394:[fmt=RGB24,size=640x480,fps=30,iso=400,dma=10]//0"
//  e.g. "dc1394:[fmt=FORMAT7_1,size=640x480,pos=2+2,iso=400,dma=10]//0"
//  e.g. "dc1394:[fmt=FORMAT7_3,deinterlace=1]//0"
//
// v4l - capture video from a Video4Linux (USB) camera (normally YUVY422 format)
//           method=mmap|read|userptr
//  e.g. "v4l:///dev/video0"
//  e.g. "v4l[method=mmap]:///dev/video0"
//
// openni2 - capture video / depth from OpenNI2 SDK  (Kinect / Xtrion etc)
//           imgN=grey|rgb|ir|ir8|ir24|depth|reg_depth
//  e.g. "openni2://'
//  e.g. "openni2:[img1=rgb,img2=depth,coloursync=true]//"
//  e.g. "openni2:[img1=depth,close=closerange,holefilter=true]//"
//  e.g. "openni2:[size=320x240,fps=60,img1=ir]//"
//
// openni - capture video / depth from OpenNI 1.0 SDK (Kinect / Xtrion etc)
//           Sensor modes containing '8' will truncate to 8-bits.
//           Sensor modes containing '+' explicitly enable IR illuminator
//           imgN=rgb|ir|ir8|ir+|ir8+|depth|reg_depth
//           autoexposure=true|false
//  e.g. "openni://'
//  e.g. "openni:[img1=rgb,img2=depth]//"
//  e.g. "openni:[size=320x240,fps=60,img1=ir]//"
//
// depthsense - capture video / depth from DepthSense SDK.
//              DepthSenseViewer can be used to alter capture settings.
//              imgN=depth|rgb
//              sizeN=QVGA|320x240|...
//              fpsN=25|30|60|...
//  e.g. "depthsense://"
//  e.g. "depthsense:[img1=depth,img2=rgb]//"
//
// pleora - USB 3 vision cameras accepts any option in the same format reported by eBUSPlayer
//  e.g. for lightwise cameras: "pleora:[size=512x256,pos=712x512,sn=00000274,ExposureTime=10000,PixelFormat=Mono12p,AcquisitionMode=SingleFrame,TriggerSource=Line0,TriggerMode=On]//"
//  e.g. for toshiba cameras: "pleora:[size=512x256,pos=712x512,sn=0300056,PixelSize=Bpp12,ExposureTime=10000,ImageFormatSelector=Format1,BinningHorizontal=2,BinningVertical=2]//"
//  e.g. toshiba alternated "pleora:[UserSetSelector=UserSet1,ExposureTime=10000,PixelSize=Bpp12,Width=1400,OffsetX=0,Height=1800,OffsetY=124,LineSelector=Line1,LineSource=ExposureActive,LineSelector=Line2,LineSource=Off,LineModeAll=6,LineInverterAll=6,UserSetSave=Execute,
//                                   UserSetSelector=UserSet2,PixelSize=Bpp12,Width=1400,OffsetX=1048,Height=1800,OffsetY=124,ExposureTime=10000,LineSelector=Line1,LineSource=Off,LineSelector=Line2,LineSource=ExposureActive,LineModeAll=6,LineInverterAll=6,UserSetSave=Execute,
//                                   SequentialShutterIndex=1,SequentialShutterEntry=1,SequentialShutterIndex=2,SequentialShutterEntry=2,SequentialShutterTerminateAt=2,SequentialShutterEnable=On,,AcquisitionFrameRateControl=Manual,AcquisitionFrameRate=70]//"
//
// thread - thread that continuously pulls from the child streams so that data in, unpacking, debayering etc can be decoupled from the main application thread
//  e.g. thread://pleora://
//  e.g. thread://unpack://pleora:[PixelFormat=Mono12p]//
//
// convert - use FFMPEG to convert between video pixel formats
//  e.g. "convert:[fmt=RGB24]//v4l:///dev/video0"
//  e.g. "convert:[fmt=GRAY8]//v4l:///dev/video0"
//
// mjpeg - capture from (possibly networked) motion jpeg stream using FFMPEG
//  e.g. "mjpeg://http://127.0.0.1/?action=stream"
//
// debayer - debayer an input video stream
//  e.g.  "debayer:[tile="BGGR",method="downsample"]//v4l:///dev/video0
//
// split - split an input video into a one or more streams based on Region of Interest / memory specification
//           roiN=X+Y+WxH
//           memN=Offset:WxH:PitchBytes:Format
//  e.g. "split:[roi1=0+0+640x480,roi2=640+0+640x480]//files:///home/user/sequence/foo%03d.jpeg"
//  e.g. "split:[mem1=307200:640x480:1280:GRAY8,roi2=640+0+640x480]//files:///home/user/sequence/foo%03d.jpeg"
//  e.g. "split:[stream1=2,stream2=1]//pango://video.pango"
//
// truncate - select a subregion of a video based on start and end (last index+1) index
//  e.g. Generate 30 random frames: "truncate:[end=30]//test://"
//  e.g. "truncate:[begin=100,end=120]"
//
// join - join streams
//  e.g. "join:[sync_tolerance_us=100, sync_continuously=true]//{pleora:[sn=00000274]//}{pleora:[sn=00000275]//}"
//
// test - output test video sequence
//  e.g. "test://"
//  e.g. "test:[size=640x480,fmt=RGB24]//"

#include <pangolin/utils/uri.h>
#include <pangolin/video/video_exception.h>
#include <pangolin/video/video_interface.h>
#include <pangolin/video/video_output_interface.h>

namespace pangolin
{

//! Open Video Interface from string specification (as described in this files header)
PANGOLIN_EXPORT
std::unique_ptr<VideoInterface> OpenVideo(const std::string& uri);

//! Open Video Interface from Uri specification
PANGOLIN_EXPORT
std::unique_ptr<VideoInterface> OpenVideo(const Uri& uri);

//! Open VideoOutput Interface from string specification (as described in this files header)
PANGOLIN_EXPORT
std::unique_ptr<VideoOutputInterface> OpenVideoOutput(const std::string& str_uri);

//! Open VideoOutput Interface from Uri specification
PANGOLIN_EXPORT
std::unique_ptr<VideoOutputInterface> OpenVideoOutput(const Uri& uri);

//! Create vector of matching interfaces either through direct cast or filter interface.
template<typename T>
std::vector<T*> FindMatchingVideoInterfaces( VideoInterface& video )
{
    std::vector<T*> matches;

    T* vid = dynamic_cast<T*>(&video);
    if(vid) {
        matches.push_back(vid);
    }

    VideoFilterInterface* vidf = dynamic_cast<VideoFilterInterface*>(&video);
    if(vidf) {
        std::vector<T*> fmatches = vidf->FindMatchingStreams<T>();
        matches.insert(matches.begin(), fmatches.begin(), fmatches.end());
    }

    return matches;
}

template<typename T>
T* FindFirstMatchingVideoInterface( VideoInterface& video )
{
    T* vid = dynamic_cast<T*>(&video);
    if(vid) {
        return vid;
    }

    VideoFilterInterface* vidf = dynamic_cast<VideoFilterInterface*>(&video);
    if(vidf) {
        std::vector<T*> fmatches = vidf->FindMatchingStreams<T>();
        if(fmatches.size()) {
            return fmatches[0];
        }
    }

    return 0;
}

inline
picojson::value GetVideoFrameProperties(VideoInterface* video)
{
    VideoPropertiesInterface* pi = dynamic_cast<VideoPropertiesInterface*>(video);
    VideoFilterInterface* fi = dynamic_cast<VideoFilterInterface*>(video);

    if(pi) {
        return pi->FrameProperties();
    }else if(fi){
        if(fi->InputStreams().size() == 1) {
            return GetVideoFrameProperties(fi->InputStreams()[0]);
        }else if(fi->InputStreams().size() > 0){
            picojson::value streams;

            for(size_t i=0; i< fi->InputStreams().size(); ++i) {
                const picojson::value dev_props = GetVideoFrameProperties(fi->InputStreams()[i]);
                if(dev_props.contains("streams")) {
                    const picojson::value& dev_streams = dev_props["streams"];
                    for(size_t j=0; j < dev_streams.size(); ++j) {
                        streams.push_back(dev_streams[j]);
                    }
                }else{
                    streams.push_back(dev_props);
                }
            }

            if(streams.size() > 1) {
                picojson::value json = streams[0];
                json["streams"] = streams;
                return json;
            }else{
                return streams[0];
            }
        }
    }
    return picojson::value();
}

inline
picojson::value GetVideoDeviceProperties(VideoInterface* video)
{
    VideoPropertiesInterface* pi = dynamic_cast<VideoPropertiesInterface*>(video);
    VideoFilterInterface* fi = dynamic_cast<VideoFilterInterface*>(video);

    if(pi) {
        return pi->DeviceProperties();
    }else if(fi){
        if(fi->InputStreams().size() == 1) {
            return GetVideoDeviceProperties(fi->InputStreams()[0]);
        }else if(fi->InputStreams().size() > 0){
            picojson::value streams;

            for(size_t i=0; i< fi->InputStreams().size(); ++i) {
                const picojson::value dev_props = GetVideoDeviceProperties(fi->InputStreams()[i]);
                if(dev_props.contains("streams")) {
                    const picojson::value& dev_streams = dev_props["streams"];
                    for(size_t j=0; j < dev_streams.size(); ++j) {
                        streams.push_back(dev_streams[j]);
                    }
                }else{
                    streams.push_back(dev_props);
                }
            }

            if(streams.size() > 1) {
                picojson::value json = streams[0];
                json["streams"] = streams;
                return json;
            }else{
                return streams[0];
            }
        }
    }
    return picojson::value();
}

}
