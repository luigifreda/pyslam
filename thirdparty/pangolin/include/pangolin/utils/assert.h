/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) 2011 Hauke Strasdat, Steven Lovegrove
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
#include <pangolin/utils/format_string.h>
#include <iostream>

#ifdef __GNUC__
#  define PANGO_FUNCTION __PRETTY_FUNCTION__
#elif (_MSC_VER >= 1310)
#  define PANGO_FUNCTION __FUNCTION__
#else
#  define PANGO_FUNCTION "unknown"
#endif

namespace pangolin {

template <typename... Args> PANGO_HOST_DEVICE
void abort(const char* function, const char* file, int line, Args&&... args)
{
  std::printf("pangolin::abort() in function '%s', file '%s', line %d.\n", function, file, line);
#ifndef __CUDACC__
  std::cout << FormatString(std::forward<Args>(args)...) << std::endl;
  std::abort();
#endif
}

}

// Always check, even in debug
#define PANGO_ENSURE(expr, ...) ((expr) ? ((void)0) : pangolin::abort(PANGO_FUNCTION, __FILE__, __LINE__, ##__VA_ARGS__))

// May be disabled for optimisation
#define PANGO_ASSERT(expr, ...) ((expr) ? ((void)0) : pangolin::abort(PANGO_FUNCTION, __FILE__, __LINE__, ##__VA_ARGS__))
