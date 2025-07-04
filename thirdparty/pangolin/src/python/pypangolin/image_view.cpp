/* This file is part of the Pangolin Project.
 * http://github.com/stevenlovegrove/Pangolin
 *
 * Copyright (c) Steven Lovegrove
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

#include "image_view.hpp"
#include <pangolin/display/image_view.h>
#include <pybind11/numpy.h>

namespace py_pangolin {

  void bind_image_view(pybind11::module &m) {
    using namespace pangolin;
    pybind11::class_<ImageView, View>(m, "ImageView")
      .def(pybind11::init<>())
      .def("SetImage", [](ImageView& view, pybind11::array_t<float>& img) -> ImageView&{
        if(img.ndim() == 2) {
            Image<uint8_t> wrapper((uint8_t*)img.mutable_data(0,0), img.shape(1), img.shape(0), img.shape(1) * sizeof(float));
            return view.SetImage(wrapper, GlPixFormat(PixelFormatFromString("GRAY32F")) );
        }else if(img.ndim() == 3 && img.shape()[2] == 3) {
            Image<uint8_t> wrapper((uint8_t*)img.mutable_data(0,0), img.shape(1), img.shape(0), 3 * img.shape(1) * sizeof(float));
            return view.SetImage(wrapper, GlPixFormat(PixelFormatFromString("RGB96F")) );
        }else{
            throw std::runtime_error("Unsupported format for now.");
        }
      });
  }
}  // py_pangolin
