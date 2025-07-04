#include <pangolin/video/stream_encoder_factory.h>

#include <cctype>
#include <pangolin/utils/file_utils.h>
#include <pangolin/utils/type_convert.h>

namespace pangolin {

StreamEncoderFactory& StreamEncoderFactory::I()
{
    static StreamEncoderFactory instance;
    return instance;
}

struct EncoderDetails
{
    std::string encoder_name;
    ImageFileType file_type;
    float quality;
};

inline EncoderDetails EncoderDetailsFromString(const std::string& encoder_spec)
{
    std::string::const_reverse_iterator rit = encoder_spec.rbegin();
    for(; std::isdigit(*rit) && rit != encoder_spec.rend(); ++rit );

    // png, tga, ...
    std::string encoder_name(encoder_spec.begin(), rit.base());
    ToLower(encoder_name);

    // Quality of encoding for lossy encoders [0..100]
    float quality = 100.0;
    if(rit != encoder_spec.rbegin()) {
        quality = pangolin::Convert<float,std::string>::Do(std::string(rit.base(),encoder_spec.end()));
    }


    return { encoder_name, NameToImageFileType(encoder_name), quality};
}

ImageEncoderFunc StreamEncoderFactory::GetEncoder(const std::string& encoder_spec, const PixelFormat& fmt)
{
    const EncoderDetails encdet = EncoderDetailsFromString(encoder_spec);
    if(encdet.file_type == ImageFileTypeUnknown)
        throw std::invalid_argument("Unsupported encoder format: " + encoder_spec);

    return [fmt,encdet](std::ostream& os, const Image<unsigned char>& img){
        SaveImage(img,fmt,os,encdet.file_type,true,encdet.quality);
    };
}

ImageDecoderFunc StreamEncoderFactory::GetDecoder(const std::string& encoder_spec, const PixelFormat& fmt)
{
    const EncoderDetails encdet = EncoderDetailsFromString(encoder_spec);
    PANGO_ENSURE(encdet.file_type != ImageFileTypeUnknown);

    return [fmt,encdet](std::istream& is){
        return LoadImage(is,encdet.file_type);
    };
}

}
