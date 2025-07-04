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

#include <pangolin/image/image.h>
#include <pangolin/image/copy.h>

namespace pangolin {

template<class T> using DefaultImageAllocator = std::allocator<T>;

// Image that manages it's own memory, storing a strong pointer to it's memory
template<typename T, class Allocator = DefaultImageAllocator<T> >
class ManagedImage : public Image<T>
{
public:
    // Destructor
    inline
    ~ManagedImage()
    {
        Deallocate();
    }

    // Null image
    inline
    ManagedImage()
    {
    }

    // Row image
    inline
    ManagedImage(size_t w)
        : Image<T>(
              Allocator().allocate(w),
               w, 1, w*sizeof(T)
              )
    {
    }

    inline
    ManagedImage(size_t w, size_t h)
        : Image<T>(
              Allocator().allocate(w*h),
               w, h, w*sizeof(T)
              )
    {
    }

    inline
    ManagedImage(size_t w, size_t h, size_t pitch_bytes)
        : Image<T>(
              Allocator().allocate( (h*pitch_bytes) / sizeof(T) + 1),
               w, h, pitch_bytes
              )
    {
    }

    // Not copy constructable
    inline
    ManagedImage( const ManagedImage<T>& other ) = delete;

    // Move constructor
    inline
    ManagedImage(ManagedImage<T,Allocator>&& img)
    {
        *this = std::move(img);
    }

    // Move asignment
    inline
    void operator=(ManagedImage<T,Allocator>&& img)
    {
        Deallocate();
        Image<T>::pitch = img.pitch;
        Image<T>::ptr   = img.ptr;
        Image<T>::w     = img.w;
        Image<T>::h     = img.h;
        img.ptr = nullptr;
    }

    // Explicit copy constructor
    template<typename TOther>
    ManagedImage( const CopyObject<TOther>& other )
    {
        CopyFrom(other.obj);
    }

    // Explicit copy assignment
    template<typename TOther>
    void operator=(const CopyObject<TOther>& other)
    {
        CopyFrom(other.obj);
    }

    inline
    void Swap(ManagedImage<T>& img)
    {
        std::swap(img.pitch, Image<T>::pitch);
        std::swap(img.ptr, Image<T>::ptr);
        std::swap(img.w, Image<T>::w);
        std::swap(img.h, Image<T>::h);
    }

    inline
    void CopyFrom(const Image<T>& img)
    {
        if(!Image<T>::IsValid() || Image<T>::w != img.w || Image<T>::h != img.h) {
            Reinitialise(img.w,img.h);
        }
        Image<T>::CopyFrom(img);
    }

    inline
    void Reinitialise(size_t w, size_t h)
    {
        if(!Image<T>::ptr || Image<T>::w != w || Image<T>::h != h) {
            *this = ManagedImage<T,Allocator>(w,h);
        }
    }

    inline
    void Reinitialise(size_t w, size_t h, size_t pitch)
    {
        if(!Image<T>::ptr || Image<T>::w != w || Image<T>::h != h || Image<T>::pitch != pitch) {
            *this = ManagedImage<T,Allocator>(w,h,pitch);
        }
    }

    inline void Deallocate()
    {
        if (Image<T>::ptr) {
            Allocator().deallocate(Image<T>::ptr, (Image<T>::h * Image<T>::pitch) / sizeof(T) );
            Image<T>::ptr = nullptr;
        }
    }

    // Move asignment
    template<typename TOther, typename AllocOther> inline
    void OwnAndReinterpret(ManagedImage<TOther,AllocOther>&& img)
    {
        Deallocate();
        Image<T>::pitch = img.pitch;
        Image<T>::ptr   = (T*)img.ptr;
        Image<T>::w     = img.w;
        Image<T>::h     = img.h;
        img.ptr = nullptr;
    }
};

}
