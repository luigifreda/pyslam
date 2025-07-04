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

#include <pangolin/platform.h>

#include <cstring>

#ifdef HAVE_CUDA
#  include <cuda_runtime.h>
#endif

namespace pangolin {

template<typename T>
PANGO_HOST_DEVICE inline
bool IsDevicePtr(T* ptr)
{
#ifdef HAVE_CUDA
    cudaPointerAttributes attributes;
    cudaError_t res = cudaPointerGetAttributes(&attributes,ptr);

    //Flushing the error flag for future CUDA error checks
    if(res != cudaSuccess)
    {
        cudaGetLastError();
        return false;
    }

    return attributes.memoryType == cudaMemoryTypeDevice;
#else
    PANGOLIN_UNUSED(ptr);
    return false;
#endif
}

PANGO_HOST_DEVICE inline
void MemCopy(void *dst, const void *src, size_t size_bytes)
{
#ifdef HAVE_CUDA
    cudaMemcpy(dst,src, size_bytes, cudaMemcpyDefault);
#else
    std::memcpy(dst, src, size_bytes);
#endif
}

inline
void PitchedCopy(char* dst, unsigned int dst_pitch_bytes, const char* src, unsigned int src_pitch_bytes, unsigned int width_bytes, unsigned int height)
{
#ifdef HAVE_CUDA
    cudaMemcpy2D(dst, dst_pitch_bytes, src, src_pitch_bytes, width_bytes, height, cudaMemcpyDefault);
#else
    if(dst_pitch_bytes == width_bytes && src_pitch_bytes == width_bytes ) {
        std::memcpy(dst, src, height * width_bytes);
    }else{
        for(unsigned int row=0; row < height; ++row) {
            std::memcpy(dst, src, width_bytes);
            dst += dst_pitch_bytes;
            src += src_pitch_bytes;
        }
    }
#endif
}

PANGO_HOST_DEVICE inline
void Memset(char* ptr, unsigned char v, size_t size_bytes)
{
#ifdef __CUDA_ARCH__
    // Called in kernel
    char* end = ptr + size_bytes;
    for(char* p=ptr; p != end; ++p) *p = v;
#else
#  ifdef HAVE_CUDA
        if(IsDevicePtr(ptr))
        {
            cudaMemset(ptr, v, size_bytes);
        }else
#  endif // HAVE_CUDA
        {
            std::memset(ptr, v, size_bytes);
        }
#endif // __CUDA_ARCH__
}

}
