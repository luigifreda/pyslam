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

#include "attach.hpp"
#include <pangolin/gl/glinclude.h>
#include <pangolin/display/attach.h>

namespace py_pangolin {
  void bind_attach(pybind11::module &m){
    pybind11::enum_<pangolin::Unit>(m, "Unit")
      .value("Fraction", pangolin::Unit::Fraction)
      .value("Pixel", pangolin::Unit::Pixel)
      .value("ReversePixel", pangolin::Unit::ReversePixel)
      .export_values();

    pybind11::class_<pangolin::Attach>(m, "Attach")
      .def(pybind11::init<>())
      .def(pybind11::init<pangolin::Unit, GLfloat>())
      .def(pybind11::init<GLfloat>())
      .def_static("Pix", &pangolin::Attach::Pix)
      .def_static("ReversePix", &pangolin::Attach::ReversePix)
      .def_static("Frac", &pangolin::Attach::Frac);
  }
}
