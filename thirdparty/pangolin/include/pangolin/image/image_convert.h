#pragma once

#include <pangolin/image/managed_image.h>
#include <pangolin/utils/compontent_cast.h>

namespace pangolin
{

template <typename To, typename T>
void ImageConvert(Image<To>& dst, const Image<T>& src, To scale = 1.0)
{
    for(unsigned int y = 0; y < dst.h; ++y)
    {
        const T* prs = src.RowPtr(y);
        To* prd = dst.RowPtr(y);
        for(unsigned int x = 0; x < dst.w; ++x)
        {
            *(prd++) = scale * ComponentCast<To, T>::cast(*(prs++));
        }
    }
}

template <typename To, typename T>
ManagedImage<To> ImageConvert(const Image<T>& src, To scale = 1.0)
{
    ManagedImage<To> dst(src.w, src.h);
    ImageConvert<To,T>(dst,src,scale);
    return dst;
}

}
