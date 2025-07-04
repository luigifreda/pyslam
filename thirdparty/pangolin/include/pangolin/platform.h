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

#include <pangolin/config.h>

// Include portable printf-style format macros
#define __STDC_FORMAT_MACROS

#ifdef _GCC_
#  define PANGOLIN_DEPRECATED __attribute__((deprecated))
#elif defined _MSVC_
#  define PANGOLIN_DEPRECATED __declspec(deprecated)
#else
#  define PANGOLIN_DEPRECATED
#endif

#ifdef _MSVC_
#   define __thread __declspec(thread)
#   include <pangolin/pangolin_export.h>
#else
#   define PANGOLIN_EXPORT
#endif //_MSVC_

#define PANGOLIN_UNUSED(x) (void)(x)

#ifdef _APPLE_IOS_
// Not supported on this platform.
#define __thread
#endif // _APPLE_IOS_

// HOST / DEVICE Annotations
#ifdef __CUDACC__
#  include <cuda_runtime.h>
#  define PANGO_HOST_DEVICE __host__ __device__
#else
#  define PANGO_HOST_DEVICE
#endif

// Non-standard check that header exists (Clang, GCC 5.X)
// Useful for
#if defined(__has_include)
#  define PANGO_HEADER_EXISTS(x) __has_include(x)
#else
#  define PANGO_HEADER_EXISTS(x) 0
#endif

// Workaround for Apple-Clangs lack of thread_local support
#if defined(_CLANG_) && defined(_OSX_)
#  if !__has_feature(cxx_thread_local)
#     define PANGO_NO_THREADLOCAL
#  endif
#endif

#include <pangolin/utils/assert.h>
#include <pangolin/utils/log.h>
