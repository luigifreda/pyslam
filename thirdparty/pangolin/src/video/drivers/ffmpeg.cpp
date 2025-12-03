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

#include <array>
#include <pangolin/factory/factory_registry.h>
#include <pangolin/video/iostream_operators.h>
#include <pangolin/utils/file_utils.h>
#include <pangolin/video/drivers/ffmpeg.h>

// Some versions of FFMPEG define this horrid macro in global scope.
#undef PixelFormat

// It is impossible to keep up with ffmpeg deprecations, so ignore these warnings.
#if defined(_GCC_) || defined(_CLANG_)
#  pragma GCC diagnostic ignored "-Wdeprecated"
#endif


extern "C"
{
#include <libavformat/avio.h>
#include <libavutil/mathematics.h>
#include <libavutil/imgutils.h>
#include <libavdevice/avdevice.h>
}

#define CODEC_FLAG_GLOBAL_HEADER AV_CODEC_FLAG_GLOBAL_HEADER

namespace pangolin
{

#ifdef HAVE_FFMPEG_AVPIXELFORMAT
#  define TEST_PIX_FMT_RETURN(fmt) case AV_PIX_FMT_##fmt: return #fmt;
#else
#  define AV_PIX_FMT_NONE  PIX_FMT_NONE
#  define AV_PIX_FMT_GRAY8 PIX_FMT_GRAY8
#  define TEST_PIX_FMT_RETURN(fmt) case PIX_FMT_##fmt: return #fmt;
#endif

AVPixelFormat FfmpegFmtFromString(const std::string fmt)
{
    const std::string lfmt = ToLowerCopy(fmt);
    if(!lfmt.compare("gray8") || !lfmt.compare("grey8") || !lfmt.compare("grey")) {
        return AV_PIX_FMT_GRAY8;
    }
    return av_get_pix_fmt(lfmt.c_str());
}


std::string FfmpegFmtToString(const AVPixelFormat fmt)
{
    switch( fmt )
    {
    TEST_PIX_FMT_RETURN(YUV420P);
    TEST_PIX_FMT_RETURN(YUYV422);
    TEST_PIX_FMT_RETURN(RGB24);
    TEST_PIX_FMT_RETURN(BGR24);
    TEST_PIX_FMT_RETURN(YUV422P);
    TEST_PIX_FMT_RETURN(YUV444P);
    TEST_PIX_FMT_RETURN(YUV410P);
    TEST_PIX_FMT_RETURN(YUV411P);
    TEST_PIX_FMT_RETURN(GRAY8);
    TEST_PIX_FMT_RETURN(MONOWHITE);
    TEST_PIX_FMT_RETURN(MONOBLACK);
    TEST_PIX_FMT_RETURN(PAL8);
    TEST_PIX_FMT_RETURN(YUVJ420P);
    TEST_PIX_FMT_RETURN(YUVJ422P);
    TEST_PIX_FMT_RETURN(YUVJ444P);
#ifdef FF_API_XVMC
    TEST_PIX_FMT_RETURN(XVMC_MPEG2_MC);
    TEST_PIX_FMT_RETURN(XVMC_MPEG2_IDCT);
#endif
    TEST_PIX_FMT_RETURN(UYVY422);
    TEST_PIX_FMT_RETURN(UYYVYY411);
    TEST_PIX_FMT_RETURN(BGR8);
    TEST_PIX_FMT_RETURN(BGR4);
    TEST_PIX_FMT_RETURN(BGR4_BYTE);
    TEST_PIX_FMT_RETURN(RGB8);
    TEST_PIX_FMT_RETURN(RGB4);
    TEST_PIX_FMT_RETURN(RGB4_BYTE);
    TEST_PIX_FMT_RETURN(NV12);
    TEST_PIX_FMT_RETURN(NV21);
    TEST_PIX_FMT_RETURN(ARGB);
    TEST_PIX_FMT_RETURN(RGBA);
    TEST_PIX_FMT_RETURN(ABGR);
    TEST_PIX_FMT_RETURN(BGRA);
    TEST_PIX_FMT_RETURN(GRAY16BE);
    TEST_PIX_FMT_RETURN(GRAY16LE);
    TEST_PIX_FMT_RETURN(YUV440P);
    TEST_PIX_FMT_RETURN(YUVJ440P);
    TEST_PIX_FMT_RETURN(YUVA420P);
#ifdef FF_API_VDPAU
    TEST_PIX_FMT_RETURN(VDPAU_H264);
    TEST_PIX_FMT_RETURN(VDPAU_MPEG1);
    TEST_PIX_FMT_RETURN(VDPAU_MPEG2);
    TEST_PIX_FMT_RETURN(VDPAU_WMV3);
    TEST_PIX_FMT_RETURN(VDPAU_VC1);
#endif
    TEST_PIX_FMT_RETURN(RGB48BE );
    TEST_PIX_FMT_RETURN(RGB48LE );
    TEST_PIX_FMT_RETURN(RGB565BE);
    TEST_PIX_FMT_RETURN(RGB565LE);
    TEST_PIX_FMT_RETURN(RGB555BE);
    TEST_PIX_FMT_RETURN(RGB555LE);
    TEST_PIX_FMT_RETURN(BGR565BE);
    TEST_PIX_FMT_RETURN(BGR565LE);
    TEST_PIX_FMT_RETURN(BGR555BE);
    TEST_PIX_FMT_RETURN(BGR555LE);
#if LIBAVUTIL_VERSION_MAJOR < 57
    // These pixel formats were removed in newer FFmpeg versions
    TEST_PIX_FMT_RETURN(VAAPI_MOCO);
    TEST_PIX_FMT_RETURN(VAAPI_IDCT);
    TEST_PIX_FMT_RETURN(VAAPI_VLD);
#endif
    TEST_PIX_FMT_RETURN(YUV420P16LE);
    TEST_PIX_FMT_RETURN(YUV420P16BE);
    TEST_PIX_FMT_RETURN(YUV422P16LE);
    TEST_PIX_FMT_RETURN(YUV422P16BE);
    TEST_PIX_FMT_RETURN(YUV444P16LE);
    TEST_PIX_FMT_RETURN(YUV444P16BE);
#ifdef FF_API_VDPAU
    TEST_PIX_FMT_RETURN(VDPAU_MPEG4);
#endif
    TEST_PIX_FMT_RETURN(DXVA2_VLD);
    TEST_PIX_FMT_RETURN(RGB444BE);
    TEST_PIX_FMT_RETURN(RGB444LE);
    TEST_PIX_FMT_RETURN(BGR444BE);
    TEST_PIX_FMT_RETURN(BGR444LE);
    TEST_PIX_FMT_RETURN(Y400A   );
    TEST_PIX_FMT_RETURN(NB      );
    default: return "";
    }
}

#undef TEST_PIX_FMT_RETURN

FfmpegVideo::FfmpegVideo(const std::string filename, const std::string strfmtout, const std::string codec_hint, bool dump_info, int user_video_stream, ImageDim size)
    :pFormatCtx(0)
{
    InitUrl(filename, strfmtout, codec_hint, dump_info, user_video_stream, size);
}

void FfmpegVideo::InitUrl(const std::string url, const std::string strfmtout, const std::string codec_hint, bool dump_info, int user_video_stream, ImageDim size)
{
    if( url.find('*') != url.npos )
        throw VideoException("Wildcards not supported. Please use ffmpegs printf style formatting for image sequences. e.g. img-000000%04d.ppm");

    // Register all formats and codecs (deprecated in FFmpeg 4.0+, removed in 4.1+)
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    av_register_all();
    // Register all devices
    avdevice_register_all();
#endif

    const AVInputFormat* fmt = NULL;

    if( !codec_hint.empty() ) {
        fmt = av_find_input_format(codec_hint.c_str());
    }

#if (LIBAVFORMAT_VERSION_MAJOR >= 53)
    AVDictionary* options = nullptr;
    if(size.x != 0 && size.y != 0) {
        std::string s = std::to_string(size.x) + "x" + std::to_string(size.y);
        av_dict_set(&options, "video_size", s.c_str(), 0);
    }
    if( avformat_open_input(&pFormatCtx, url.c_str(), fmt, &options) )
#else
    // Deprecated - can't use with mjpeg
    if( av_open_input_file(&pFormatCtx, url.c_str(), fmt, 0, NULL) )
#endif
        throw VideoException("Couldn't open stream");

    if( !ToLowerCopy(codec_hint).compare("mjpeg") )
#ifdef HAVE_FFMPEG_MAX_ANALYZE_DURATION2
        pFormatCtx->max_analyze_duration2 = AV_TIME_BASE * 0.0;
#else
        pFormatCtx->max_analyze_duration = AV_TIME_BASE * 0.0;
#endif

    // Retrieve stream information
#if (LIBAVFORMAT_VERSION_MAJOR >= 53)
    if(avformat_find_stream_info(pFormatCtx, 0)<0)
#else
    // Deprecated
    if(av_find_stream_info(pFormatCtx)<0)
#endif
        throw VideoException("Couldn't find stream information");

    if(dump_info) {
        // Dump information about file onto standard error
#if (LIBAVFORMAT_VERSION_MAJOR >= 53)
        av_dump_format(pFormatCtx, 0, url.c_str(), false);
#else
        // Deprecated
        dump_format(pFormatCtx, 0, url.c_str(), false);
#endif
    }

    // Find the first video stream
    videoStream=-1;
    audioStream=-1;

    std::vector<int> videoStreams;
    std::vector<int> audioStreams;

    for(unsigned i=0; i<pFormatCtx->nb_streams; i++)
    {
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(57, 40, 101)
        if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_VIDEO)
        {
            videoStreams.push_back(i);
        }else if(pFormatCtx->streams[i]->codec->codec_type==AVMEDIA_TYPE_AUDIO)
        {
            audioStreams.push_back(i);
        }
#else
        if(pFormatCtx->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_VIDEO)
        {
            videoStreams.push_back(i);
        }else if(pFormatCtx->streams[i]->codecpar->codec_type==AVMEDIA_TYPE_AUDIO)
        {
            audioStreams.push_back(i);
        }
#endif
    }

    if(videoStreams.size()==0)
        throw VideoException("Couldn't find a video stream");

    if(0 <= user_video_stream && user_video_stream < (int)videoStreams.size() ) {
        videoStream = videoStreams[user_video_stream];
    }else{
        videoStream = videoStreams[0];
    }

    // Get a pointer to the codec context for the video stream
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(57, 40, 101)
    pVidCodecCtx = pFormatCtx->streams[videoStream]->codec;
#else
    // Allocate codec context for newer FFmpeg
    pVidCodecCtx = avcodec_alloc_context3(NULL);
    if (!pVidCodecCtx)
        throw VideoException("Could not allocate codec context");
    
    // Copy codec parameters to context
    if (avcodec_parameters_to_context(pVidCodecCtx, pFormatCtx->streams[videoStream]->codecpar) < 0)
        throw VideoException("Could not copy codec parameters");
#endif

    // Find the decoder for the video stream
    const AVCodec* codec = avcodec_find_decoder(pVidCodecCtx->codec_id);
    pVidCodec = const_cast<AVCodec*>(codec);
    if(pVidCodec==0)
        throw VideoException("Codec not found");

    // Open video codec
#if LIBAVCODEC_VERSION_MAJOR > 52
    if(avcodec_open2(pVidCodecCtx, pVidCodec,0)<0)
#else
    if(avcodec_open(pVidCodecCtx, pVidCodec)<0)
#endif
        throw VideoException("Could not open codec");

    // Hack to correct wrong frame rates that seem to be generated by some codecs
    if(pVidCodecCtx->time_base.num>1000 && pVidCodecCtx->time_base.den==1)
        pVidCodecCtx->time_base.den=1000;


    // Allocate video frames
#if LIBAVUTIL_VERSION_MAJOR >= 54
    pFrame = av_frame_alloc();
    pFrameOut = av_frame_alloc();
#else
    // deprecated
    pFrame = avcodec_alloc_frame();
    pFrameOut = avcodec_alloc_frame();
#endif
    if(!pFrame || !pFrameOut)
        throw VideoException("Couldn't allocate frames");

    fmtout = FfmpegFmtFromString(strfmtout);
    if(fmtout == AV_PIX_FMT_NONE )
        throw VideoException("Output format not recognised",strfmtout);

    // Image dimensions
    const int w = pVidCodecCtx->width;
    const int h = pVidCodecCtx->height;

    // Determine required buffer size and allocate buffer
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
    numBytesOut=avpicture_get_size(fmtout, w, h);
    buffer= new uint8_t[numBytesOut];
    // Assign appropriate parts of buffer to image planes in pFrameRGB
    avpicture_fill((AVPicture *)pFrameOut, buffer, fmtout, w, h);
#else
    numBytesOut = av_image_get_buffer_size(fmtout, w, h, 1);
    buffer= new uint8_t[numBytesOut];
    // Assign appropriate parts of buffer to image planes in pFrameRGB
    av_image_fill_arrays(pFrameOut->data, pFrameOut->linesize, buffer, fmtout, w, h, 1);
#endif

    // Allocate SWS for converting pixel formats
    img_convert_ctx = sws_getContext(w, h,
                                     pVidCodecCtx->pix_fmt,
                                     w, h, fmtout, FFMPEG_POINT,
                                     NULL, NULL, NULL);
    if(img_convert_ctx == NULL) {
        throw VideoException("Cannot initialize the conversion context");
    }

    // Populate stream info for users to query
    const PixelFormat strm_fmt = PixelFormatFromString(FfmpegFmtToString(fmtout));
    const StreamInfo stream(strm_fmt, w, h, (w*strm_fmt.bpp)/8, 0);
    streams.push_back(stream);
}

FfmpegVideo::~FfmpegVideo()
{
    // Free the RGB image
    delete[] buffer;
    av_free(pFrameOut);

    // Free the YUV frame
    av_free(pFrame);

    // Close the codec
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 12, 0)
    avcodec_close(pVidCodecCtx);
#else
    avcodec_free_context(&pVidCodecCtx);
#endif

    // Close the video file
#if (LIBAVFORMAT_VERSION_MAJOR >= 54 || (LIBAVFORMAT_VERSION_MAJOR >= 53 && LIBAVFORMAT_VERSION_MINOR >= 21) )
    avformat_close_input(&pFormatCtx);
#else
    // Deprecated
    av_close_input_file(pFormatCtx);
#endif

    // Free pixel conversion context
    sws_freeContext(img_convert_ctx);
}

const std::vector<StreamInfo>& FfmpegVideo::Streams() const
{
    return streams;
}

size_t FfmpegVideo::SizeBytes() const
{
    return numBytesOut;
}

void FfmpegVideo::Start()
{
}

void FfmpegVideo::Stop()
{
}

bool FfmpegVideo::GrabNext(unsigned char* image, bool /*wait*/)
{
    int gotFrame = 0;

    while(!gotFrame && av_read_frame(pFormatCtx, &packet)>=0)
    {
        // Is this a packet from the video stream?
        if(packet.stream_index==videoStream)
        {
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 37, 100)
            // Decode video frame (old API)
            avcodec_decode_video2(pVidCodecCtx, pFrame, &gotFrame, &packet);
#else
            // Decode video frame (new API)
            int ret = avcodec_send_packet(pVidCodecCtx, &packet);
            if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
                av_packet_unref(&packet);
                continue;
            }
            
            ret = avcodec_receive_frame(pVidCodecCtx, pFrame);
            if (ret == 0) {
                gotFrame = 1;
            } else if (ret == AVERROR(EAGAIN)) {
                // Need more input
            } else if (ret == AVERROR_EOF) {
                // End of stream
                av_packet_unref(&packet);
                break;
            }
#endif
        }

        // Did we get a video frame?
        if(gotFrame) {
            sws_scale(img_convert_ctx, pFrame->data, pFrame->linesize, 0, pVidCodecCtx->height, pFrameOut->data, pFrameOut->linesize);
            memcpy(image,pFrameOut->data[0],numBytesOut);
        }

        // Free the packet that was allocated by av_read_frame
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 12, 0)
        av_free_packet(&packet);
#else
        av_packet_unref(&packet);
#endif
    }

    return gotFrame;
}

bool FfmpegVideo::GrabNewest(unsigned char *image, bool wait)
{
    return GrabNext(image,wait);
}

void FfmpegConverter::ConvertContext::convert(const unsigned char* src, unsigned char* dst)
{
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
    // avpicture_fill expects uint8_t* w/o const as the second parameter in earlier versions
    avpicture_fill((AVPicture*)avsrc, const_cast<unsigned char*>(src + src_buffer_offset), fmtsrc, w, h);
    avpicture_fill((AVPicture*)avdst, dst + dst_buffer_offset, fmtdst, w, h);
#else
    av_image_fill_arrays(avsrc->data, avsrc->linesize, const_cast<unsigned char*>(src + src_buffer_offset), fmtsrc, w, h, 1);
    av_image_fill_arrays(avdst->data, avdst->linesize, dst + dst_buffer_offset, fmtdst, w, h, 1);
#endif
    sws_scale(  img_convert_ctx,
                avsrc->data, avsrc->linesize, 0, h,
                avdst->data, avdst->linesize         );
}

FfmpegConverter::FfmpegConverter(std::unique_ptr<VideoInterface> &videoin_, const std::string sfmtdst, FfmpegMethod method )
    :videoin(std::move(videoin_))
{
    if( !videoin )
        throw VideoException("Source video interface not specified");

    input_buffer = std::unique_ptr<unsigned char[]>(new unsigned char[videoin->SizeBytes()]);

    converters.resize(videoin->Streams().size());

    dst_buffer_size = 0;

    for(size_t i=0; i < videoin->Streams().size(); ++i) {
        const StreamInfo instrm = videoin->Streams()[i];

        converters[i].w=instrm.Width();
        converters[i].h=instrm.Height();

        converters[i].fmtdst = FfmpegFmtFromString(sfmtdst);
        converters[i].fmtsrc = FfmpegFmtFromString(instrm.PixFormat());
        converters[i].img_convert_ctx = sws_getContext(
            instrm.Width(), instrm.Height(), converters[i].fmtsrc,
            instrm.Width(), instrm.Height(), converters[i].fmtdst,
            method, NULL, NULL, NULL
        );
        if(!converters[i].img_convert_ctx)
            throw VideoException("Could not create SwScale context for pixel conversion");

        converters[i].dst_buffer_offset=dst_buffer_size;
        converters[i].src_buffer_offset=instrm.Offset() - (unsigned char*)0;
        //converters[i].src_buffer_offset=src_buffer_size;

        #if LIBAVUTIL_VERSION_MAJOR >= 54
            converters[i].avsrc = av_frame_alloc();
            converters[i].avdst = av_frame_alloc();
        #else
            // deprecated
            converters[i].avsrc = avcodec_alloc_frame();
            converters[i].avdst = avcodec_alloc_frame();
        #endif

        const PixelFormat pxfmtdst = PixelFormatFromString(sfmtdst);
        const StreamInfo sdst( pxfmtdst, instrm.Width(), instrm.Height(), (instrm.Width()*pxfmtdst.bpp)/8, (unsigned char*)0 + converters[i].dst_buffer_offset );
        streams.push_back(sdst);


        //src_buffer_size += instrm.SizeBytes();
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
        dst_buffer_size += avpicture_get_size(converters[i].fmtdst, instrm.Width(), instrm.Height());
#else
        dst_buffer_size += av_image_get_buffer_size(converters[i].fmtdst, instrm.Width(), instrm.Height(), 1);
#endif
    }

}

FfmpegConverter::~FfmpegConverter()
{
    for(ConvertContext&c:converters)
    {
        av_free(c.avsrc);
        av_free(c.avdst);
    }
}

void FfmpegConverter::Start()
{
    // No-Op
}

void FfmpegConverter::Stop()
{
    // No-Op
}

size_t FfmpegConverter::SizeBytes() const
{
    return dst_buffer_size;
}

const std::vector<StreamInfo>& FfmpegConverter::Streams() const
{
    return streams;
}

bool FfmpegConverter::GrabNext( unsigned char* image, bool wait )
{
    if( videoin->GrabNext(input_buffer.get(),wait) )
    {
        for(ConvertContext&c:converters) {
            c.convert(input_buffer.get(),image);
        }
        return true;
    }
    return false;
}

bool FfmpegConverter::GrabNewest( unsigned char* image, bool wait )
{
    if( videoin->GrabNewest(input_buffer.get(),wait) )
    {
        for(ConvertContext&c:converters) {
            c.convert(input_buffer.get(),image);
        }
        return true;
    }
    return false;
}

// Based on this example
// http://cekirdek.pardus.org.tr/~ismail/ffmpeg-docs/output-example_8c-source.html
static AVStream* CreateStream(AVFormatContext *oc, CodecID codec_id, uint64_t frame_rate, int bit_rate, AVPixelFormat EncoderFormat, int width, int height, AVCodecContext** out_codec_ctx)
{
    const AVCodec* codec = avcodec_find_encoder(codec_id);
    if (!(codec)) throw
        VideoException("Could not find encoder");

#if (LIBAVFORMAT_VERSION_MAJOR >= 54 || (LIBAVFORMAT_VERSION_MAJOR >= 53 && LIBAVFORMAT_VERSION_MINOR >= 21) )
    AVStream* stream = avformat_new_stream(oc, codec);
#else
    AVStream* stream = av_new_stream(oc, codec_id);
#endif

    if (!stream) throw VideoException("Could not allocate stream");

    stream->id = oc->nb_streams-1;

#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(57, 40, 101)
    // Old API - stream->codec is directly accessible
    AVCodecContext* codec_ctx = stream->codec;
#else
    // New API - allocate codec context
    AVCodecContext* codec_ctx = avcodec_alloc_context3(codec);
    if (!codec_ctx) throw VideoException("Could not allocate codec context");
    stream->codecpar->codec_type = codec->type;
    stream->codecpar->codec_id = codec_id;
#endif

    switch (codec->type) {
//    case AVMEDIA_TYPE_AUDIO:
//        stream->id = 1;
//        codec_ctx->sample_fmt  = AV_SAMPLE_FMT_S16;
//        codec_ctx->bit_rate    = 64000;
//        codec_ctx->sample_rate = 44100;
//        codec_ctx->channels    = 2;
//        break;
    case AVMEDIA_TYPE_VIDEO:
        codec_ctx->codec_id = codec_id;
        codec_ctx->bit_rate = bit_rate;
        codec_ctx->width    = width;
        codec_ctx->height   = height;
        codec_ctx->time_base.num = 1;
        codec_ctx->time_base.den = frame_rate;
        codec_ctx->gop_size      = 12;
        codec_ctx->pix_fmt       = EncoderFormat;
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(57, 40, 101)
        // Copy codec parameters to stream
        avcodec_parameters_from_context(stream->codecpar, codec_ctx);
#endif
        break;
    default:
        break;
    }

    /* Some formats want stream headers to be separate. */
    if (oc->oformat->flags & AVFMT_GLOBALHEADER)
        codec_ctx->flags |= CODEC_FLAG_GLOBAL_HEADER;

    /* open the codec */
    int ret = avcodec_open2(codec_ctx, codec, NULL);
    if (ret < 0)  throw VideoException("Could not open video codec");

#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(57, 40, 101)
    // Copy codec parameters to stream
    avcodec_parameters_from_context(stream->codecpar, codec_ctx);
    // Return codec context to caller since stream->codec doesn't exist
    *out_codec_ctx = codec_ctx;
#else
    // Return codec context from stream
    *out_codec_ctx = codec_ctx;
#endif

    return stream;
}

class FfmpegVideoOutputStream
{
public:
    FfmpegVideoOutputStream(FfmpegVideoOutput& recorder, CodecID codec_id, uint64_t frame_rate, int bit_rate, const StreamInfo& input_info, bool flip );
    ~FfmpegVideoOutputStream();

    const StreamInfo& GetStreamInfo() const;

    void WriteImage(const uint8_t* img, int w, int h, double time);
    void Flush();

protected:
    void WriteAvPacket(AVPacket* pkt);
    void WriteFrame(AVFrame* frame);
    double BaseFrameTime();

    FfmpegVideoOutput& recorder;

    StreamInfo input_info;
    AVPixelFormat input_format;
    AVPixelFormat output_format;

#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
    AVPicture src_picture;
    AVPicture dst_picture;
#else
    AVFrame* src_frame;
    AVFrame* dst_frame;
#endif
    int64_t last_pts;

    // These pointers are owned by class
    AVStream* stream;
    AVCodecContext* codec_ctx;  // Store codec context separately for newer FFmpeg
    SwsContext *sws_ctx;
    AVFrame* frame;

    bool flip;
};

void FfmpegVideoOutputStream::WriteAvPacket(AVPacket* pkt)
{
    if (pkt->size) {
        pkt->stream_index = stream->index;
        int64_t pts = pkt->pts;
        /* convert unit from CODEC's timestamp to stream's one */
#define C2S(field)                                              \
        do {                                                    \
          if (pkt->field != (int64_t) AV_NOPTS_VALUE)           \
            pkt->field = av_rescale_q(pkt->field,               \
                                      codec_ctx->time_base, \
                                      stream->time_base);       \
        } while (0)

        C2S(pts);
        C2S(dts);
        C2S(duration);
#undef C2S
        int ret = av_interleaved_write_frame(recorder.oc, pkt);
        if (ret < 0) throw VideoException("Error writing video frame");
        if(pkt->pts != (int64_t)AV_NOPTS_VALUE) last_pts = pts;
    }
}

void FfmpegVideoOutputStream::WriteFrame(AVFrame* frame)
{
    AVPacket pkt;
    pkt.data = NULL;
    pkt.size = 0;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 12, 0)
    av_init_packet(&pkt);
#endif

    int ret;
    int got_packet = 1;

#if FF_API_LAVF_FMT_RAWPICTURE
    // Setup AVPacket
    if (recorder.oc->oformat->flags & AVFMT_RAWPICTURE) {
        /* Raw video case - directly store the picture in the packet */
        pkt.flags        |= AV_PKT_FLAG_KEY;
        pkt.data          = frame->data[0];
        pkt.size          = sizeof(AVPicture);
        pkt.pts           = frame->pts;
        ret = 0;
    } else {
#else
    {
#endif
        /* encode the image */
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
        // New encoding API
        ret = avcodec_send_frame(codec_ctx, frame);
        if (ret < 0 && ret != AVERROR(EAGAIN) && ret != AVERROR_EOF) {
            throw VideoException("Error sending frame to encoder");
        }
        ret = avcodec_receive_packet(codec_ctx, &pkt);
        if (ret == 0) {
            got_packet = 1;
        } else if (ret == AVERROR(EAGAIN)) {
            got_packet = 0;
        } else if (ret == AVERROR_EOF) {
            got_packet = 0;
        } else {
            throw VideoException("Error receiving packet from encoder");
        }
#elif (LIBAVFORMAT_VERSION_MAJOR >= 54)
        ret = avcodec_encode_video2(codec_ctx, &pkt, frame, &got_packet);
#else
        // TODO: Why is ffmpeg so fussy about this buffer size?
        //       Making this too big results in garbled output.
        //       Too small and it will fail entirely.
        pkt.size = 50* FF_MIN_BUFFER_SIZE; //std::max(FF_MIN_BUFFER_SIZE, frame->width * frame->height * 4 );
        // TODO: Make sure this is being freed by av_free_packet
        pkt.data = (uint8_t*) malloc(pkt.size);
        pkt.pts = frame->pts;
        ret = avcodec_encode_video(codec_ctx, pkt.data, pkt.size, frame);
        got_packet = ret > 0;
#endif
        if (ret < 0) throw VideoException("Error encoding video frame");
    }

    if (ret >= 0 && got_packet) {
        WriteAvPacket(&pkt);
    }

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 12, 0)
    av_free_packet(&pkt);
#else
    av_packet_unref(&pkt);
#endif
}

void FfmpegVideoOutputStream::WriteImage(const uint8_t* img, int w, int h, double time=-1.0)
{
    const int64_t pts = (time >= 0) ? time / BaseFrameTime() : ++last_pts;

    recorder.StartStream();

    AVCodecContext *c = codec_ctx;

    if(flip) {
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
        // Earlier versions of ffmpeg do not annotate img as const, although it is
        avpicture_fill(&src_picture,const_cast<uint8_t*>(img),input_format,w,h);
        for(int i=0; i<4; ++i) {
            src_picture.data[i] += (h-1) * src_picture.linesize[i];
            src_picture.linesize[i] *= -1;
        }
#else
        av_image_fill_arrays(src_frame->data, src_frame->linesize, const_cast<uint8_t*>(img), input_format, w, h, 1);
        for(int i=0; i<4; ++i) {
            src_frame->data[i] += (h-1) * src_frame->linesize[i];
            src_frame->linesize[i] *= -1;
        }
#endif
    }else{
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
        // Earlier versions of ffmpeg do not annotate img as const, although it is
        avpicture_fill(&src_picture,const_cast<uint8_t*>(img),input_format,w,h);
#else
        av_image_fill_arrays(src_frame->data, src_frame->linesize, const_cast<uint8_t*>(img), input_format, w, h, 1);
#endif
    }

    if (c->pix_fmt != input_format || c->width != w || c->height != h) {
        if(!sws_ctx) {
            sws_ctx = sws_getCachedContext( sws_ctx,
                w, h, input_format,
                c->width, c->height, c->pix_fmt,
                SWS_BICUBIC, NULL, NULL, NULL
            );
            if (!sws_ctx) throw VideoException("Could not initialize the conversion context");
        }
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
        sws_scale(sws_ctx,
            src_picture.data, src_picture.linesize, 0, h,
            dst_picture.data, dst_picture.linesize
        );
        *((AVPicture *)frame) = dst_picture;
#else
        sws_scale(sws_ctx,
            src_frame->data, src_frame->linesize, 0, h,
            dst_frame->data, dst_frame->linesize
        );
        // Copy frame properties and data pointers
        av_frame_copy_props(frame, dst_frame);
        for (int i = 0; i < AV_NUM_DATA_POINTERS; i++) {
            frame->data[i] = dst_frame->data[i];
            frame->linesize[i] = dst_frame->linesize[i];
        }
#endif
    } else {
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
        *((AVPicture *)frame) = src_picture;
#else
        // Copy frame properties and data pointers
        av_frame_copy_props(frame, src_frame);
        for (int i = 0; i < AV_NUM_DATA_POINTERS; i++) {
            frame->data[i] = src_frame->data[i];
            frame->linesize[i] = src_frame->linesize[i];
        }
#endif
    }

    frame->pts = pts;
    frame->width =  w;
    frame->height = h;
    WriteFrame(frame);
}

void FfmpegVideoOutputStream::Flush()
{
#if (LIBAVFORMAT_VERSION_MAJOR >= 54)
    if (codec_ctx->codec->capabilities & AV_CODEC_CAP_DELAY) {
        /* some CODECs like H.264 needs flushing buffered frames by encoding NULL frames. */
        /* cf. https://www.ffmpeg.org/doxygen/trunk/group__lavc__encoding.html#ga2c08a4729f72f9bdac41b5533c4f2642 */

        AVPacket pkt;
        pkt.data = NULL;
        pkt.size = 0;
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 12, 0)
        av_init_packet(&pkt);
#endif

        int got_packet = 1;
        while (got_packet) {
            int ret;
#if LIBAVCODEC_VERSION_INT >= AV_VERSION_INT(57, 37, 100)
            // New encoding API
            ret = avcodec_send_frame(codec_ctx, NULL);
            if (ret < 0 && ret != AVERROR_EOF) {
                throw VideoException("Error sending flush frame to encoder");
            }
            ret = avcodec_receive_packet(codec_ctx, &pkt);
            if (ret == 0) {
                got_packet = 1;
            } else if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) {
                got_packet = 0;
            } else {
                throw VideoException("Error receiving packet from encoder");
            }
#else
            ret = avcodec_encode_video2(codec_ctx, &pkt, NULL, &got_packet);
            if (ret < 0) throw VideoException("Error encoding video frame");
#endif
            if (got_packet) {
                WriteAvPacket(&pkt);
            }
        }

#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 12, 0)
        av_free_packet(&pkt);
#else
        av_packet_unref(&pkt);
#endif
    }
#endif
}

const StreamInfo& FfmpegVideoOutputStream::GetStreamInfo() const
{
    return input_info;
}

double FfmpegVideoOutputStream::BaseFrameTime()
{
    return (double)codec_ctx->time_base.num / (double)codec_ctx->time_base.den;
}

FfmpegVideoOutputStream::FfmpegVideoOutputStream(
    FfmpegVideoOutput& recorder, CodecID codec_id, uint64_t frame_rate,
    int bit_rate, const StreamInfo& input_info, bool flip_image
)
    : recorder(recorder), input_info(input_info),
      input_format(FfmpegFmtFromString(input_info.PixFormat())),
      output_format( FfmpegFmtFromString("YUV420P") ),
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
      src_picture(), dst_picture(),
#else
      src_frame(NULL), dst_frame(NULL),
#endif
      last_pts(-1), sws_ctx(NULL), frame(NULL), flip(flip_image)
{
    AVCodecContext* temp_codec_ctx = NULL;
    stream = CreateStream(recorder.oc, codec_id, frame_rate, bit_rate, output_format, input_info.Width(), input_info.Height(), &temp_codec_ctx);
    codec_ctx = temp_codec_ctx;

    // Allocate the encoded raw picture.
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
    int ret = avpicture_alloc(&dst_picture, codec_ctx->pix_fmt, codec_ctx->width, codec_ctx->height);
    if (ret < 0) throw VideoException("Could not allocate picture");
#else
    dst_frame = av_frame_alloc();
    if (!dst_frame) throw VideoException("Could not allocate frame");
    int ret = av_image_alloc(dst_frame->data, dst_frame->linesize, codec_ctx->width, codec_ctx->height, codec_ctx->pix_fmt, 1);
    if (ret < 0) throw VideoException("Could not allocate picture");
    src_frame = av_frame_alloc();
    if (!src_frame) throw VideoException("Could not allocate frame");
#endif

    // Allocate frame
#if LIBAVUTIL_VERSION_MAJOR >= 54
    frame = av_frame_alloc();
#else
    // Deprecated
    frame = avcodec_alloc_frame();
#endif
}

FfmpegVideoOutputStream::~FfmpegVideoOutputStream()
{
    Flush();

    if(sws_ctx) {
        sws_freeContext(sws_ctx);
    }

    av_free(frame);
#if LIBAVUTIL_VERSION_INT < AV_VERSION_INT(52, 8, 0)
    av_free(dst_picture.data[0]);
#else
    if (dst_frame) {
        av_freep(&dst_frame->data[0]);
        av_frame_free(&dst_frame);
    }
    if (src_frame) {
        av_frame_free(&src_frame);
    }
#endif
#if LIBAVCODEC_VERSION_INT < AV_VERSION_INT(57, 12, 0)
    if (codec_ctx) {
        avcodec_close(codec_ctx);
    }
#else
    // For output streams, the codec context cleanup depends on FFmpeg version
    if (codec_ctx) {
#if LIBAVFORMAT_VERSION_INT >= AV_VERSION_INT(57, 40, 101)
        // In newer FFmpeg, we allocated it ourselves, so free it
        avcodec_free_context(&codec_ctx);
#else
        // In older FFmpeg, it's managed by the stream
        avcodec_close(codec_ctx);
#endif
    }
#endif
}

FfmpegVideoOutput::FfmpegVideoOutput(const std::string& filename, int base_frame_rate, int bit_rate, bool flip_image)
    : filename(filename), started(false), oc(NULL),
      frame_count(0), base_frame_rate(base_frame_rate), bit_rate(bit_rate), is_pipe(pangolin::IsPipe(filename)), flip(flip_image)
{
    Initialise(filename);
}

FfmpegVideoOutput::~FfmpegVideoOutput()
{
    Close();
}

bool FfmpegVideoOutput::IsPipe() const
{
    return is_pipe;
}

void FfmpegVideoOutput::Initialise(std::string filename)
{
#if LIBAVFORMAT_VERSION_INT < AV_VERSION_INT(58, 9, 100)
    av_register_all();
#endif

#ifdef HAVE_FFMPEG_AVFORMAT_ALLOC_OUTPUT_CONTEXT2
    int ret = avformat_alloc_output_context2(&oc, NULL, NULL, filename.c_str());
#else
    oc = avformat_alloc_context();
    oc->oformat = av_guess_format(NULL, filename.c_str(), NULL);
    int ret = oc->oformat ? 0 : -1;
#endif

    if (ret < 0 || !oc) {
        pango_print_error("Could not deduce output format from file extension: using MPEG.\n");
#ifdef HAVE_FFMPEG_AVFORMAT_ALLOC_OUTPUT_CONTEXT2
        ret = avformat_alloc_output_context2(&oc, NULL, "mpeg", filename.c_str());
#else
        oc->oformat = av_guess_format("mpeg", filename.c_str(), NULL);
#endif
        if (ret < 0 || !oc) throw VideoException("Couldn't create AVFormatContext");
    }

    /* open the output file, if needed */
    if (!(oc->oformat->flags & AVFMT_NOFILE)) {
        ret = avio_open(&oc->pb, filename.c_str(), AVIO_FLAG_WRITE);
        if (ret < 0) throw VideoException("Could not open '%s'\n", filename);
    }
}

void FfmpegVideoOutput::StartStream()
{
    if(!started) {
#if (LIBAVFORMAT_VERSION_MAJOR >= 53)
        av_dump_format(oc, 0, filename.c_str(), 1);
#else
        // Deprecated
        dump_format(oc, 0, filename.c_str(), 1);
#endif

        /* Write the stream header, if any. */
        int ret = avformat_write_header(oc, NULL);
        if (ret < 0) throw VideoException("Error occurred when opening output file");

        started = true;
    }
}

void FfmpegVideoOutput::Close()
{
    for(std::vector<FfmpegVideoOutputStream*>::iterator i = streams.begin(); i!=streams.end(); ++i)
    {
        (*i)->Flush();
        delete *i;
    }

    av_write_trailer(oc);

    if (!(oc->oformat->flags & AVFMT_NOFILE)) avio_close(oc->pb);

    avformat_free_context(oc);
}

const std::vector<StreamInfo>& FfmpegVideoOutput::Streams() const
{
    return strs;
}

void FfmpegVideoOutput::SetStreams(const std::vector<StreamInfo>& str, const std::string& /*uri*/, const picojson::value& properties)
{
    strs.insert(strs.end(), str.begin(), str.end());

    for(std::vector<StreamInfo>::const_iterator i = str.begin(); i!= str.end(); ++i)
    {
        streams.push_back( new FfmpegVideoOutputStream(
            *this, oc->oformat->video_codec, base_frame_rate, bit_rate, *i, flip
        ) );
    }

    if(!properties.is<picojson::null>()) {
        pango_print_warn("Ignoring attached video properties.");
    }
}

int FfmpegVideoOutput::WriteStreams(const unsigned char* data, const picojson::value& /*frame_properties*/)
{
    for(std::vector<FfmpegVideoOutputStream*>::iterator i = streams.begin(); i!= streams.end(); ++i)
    {
        FfmpegVideoOutputStream& s = **i;
        Image<unsigned char> img = s.GetStreamInfo().StreamImage(data);
        s.WriteImage(img.ptr, img.w, img.h);
    }
    return frame_count++;
}

PANGOLIN_REGISTER_FACTORY(FfmpegVideo)
{
    struct FfmpegVideoFactory : public FactoryInterface<VideoInterface> {
        std::unique_ptr<VideoInterface> Open(const Uri& uri) override {
            const std::array<std::string,43> ffmpeg_ext = {{
                ".3g2",".3gp", ".amv", ".asf", ".avi", ".drc", ".flv", ".f4v",
                ".f4p", ".f4a", ".f4b", ".gif", ".gifv", ".m4v", ".mkv", ".mng", ".mov", ".qt",
                ".mp4", ".m4p", ".m4v", ".mpg", ".mp2", ".mpeg", ".mpe", ".mpv", ".mpg", ".mpeg",
                ".m2v", ".mxf", ".nsv",  ".ogv", ".ogg", ".rm", ".rmvb", ".roq", ".svi", ".vob",
                ".webm", ".wmv", ".yuv", ".h264", ".h265"
            }};

            if(!uri.scheme.compare("ffmpeg") || !uri.scheme.compare("file") || !uri.scheme.compare("files") )
            {
                if(!uri.scheme.compare("file") || !uri.scheme.compare("files")) {
                    const std::string ext = FileLowercaseExtention(uri.url);
                    if(std::find(ffmpeg_ext.begin(), ffmpeg_ext.end(), ext) == ffmpeg_ext.end()) {
                        // Don't try to load unknown files without the ffmpeg:// scheme.
                        return std::unique_ptr<VideoInterface>();
                    }
                }
                std::string outfmt = uri.Get<std::string>("fmt","RGB24");
                ToUpper(outfmt);
                const int video_stream = uri.Get<int>("stream",-1);
                return std::unique_ptr<VideoInterface>( new FfmpegVideo(uri.url.c_str(), outfmt, "", false, video_stream) );
            }else if( !uri.scheme.compare("v4lmjpeg")) {
                const int video_stream = uri.Get<int>("stream",-1);
                const ImageDim size = uri.Get<ImageDim>("size",ImageDim(0,0));
                return std::unique_ptr<VideoInterface>( new FfmpegVideo(uri.url.c_str(),"RGB24", "video4linux", false, video_stream, size) );
            } else if( !uri.scheme.compare("mjpeg")) {
                return std::unique_ptr<VideoInterface>( new FfmpegVideo(uri.url.c_str(),"RGB24", "MJPEG" ) );
            }else if( !uri.scheme.compare("convert") ) {
                std::string outfmt = uri.Get<std::string>("fmt","RGB24");
                ToUpper(outfmt);
                std::unique_ptr<VideoInterface> subvid = pangolin::OpenVideo(uri.url);
                return std::unique_ptr<VideoInterface>( new FfmpegConverter(subvid,outfmt,FFMPEG_POINT) );
            }else{
                return std::unique_ptr<VideoInterface>();
            }
        }
    };

    auto factory = std::make_shared<FfmpegVideoFactory>();
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "ffmpeg");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "v4lmjpeg");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 10, "mjpeg");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 20, "convert");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 15, "file");
    FactoryRegistry<VideoInterface>::I().RegisterFactory(factory, 15, "files");
}

PANGOLIN_REGISTER_FACTORY(FfmpegVideoOutput)
{
    struct FfmpegVideoFactory final : public FactoryInterface<VideoOutputInterface> {
        std::unique_ptr<VideoOutputInterface> Open(const Uri& uri) override {
            const int desired_frame_rate = uri.Get("fps", 60);
            const int desired_bit_rate = uri.Get("bps", 20000*1024);
            const bool flip = uri.Get("flip", false);
            std::string filename = uri.url;

            if(uri.Contains("unique_filename")) {
                filename = MakeUniqueFilename(filename);
            }

            return std::unique_ptr<VideoOutputInterface>(
                new FfmpegVideoOutput(filename, desired_frame_rate, desired_bit_rate, flip)
            );
        }
    };

    auto factory = std::make_shared<FfmpegVideoFactory>();
    FactoryRegistry<VideoOutputInterface>::I().RegisterFactory(factory, 10, "ffmpeg");
}

}
