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

#include "colour.hpp"
#include <pangolin/gl/colour.h>

namespace py_pangolin {

  void bind_colour(pybind11::module& m){
    pybind11::class_<pangolin::Colour> (m, "Colour")
      .def(pybind11::init<>())
      .def(pybind11::init<const float, const float, const float, const float>(), pybind11::arg("red"), pybind11::arg("green"), pybind11::arg("blue"), pybind11::arg("alpha")=1.0f)
      //      .def(pybind11::init<float[4]>())
      .def("Get", &pangolin::Colour::Get)
      .def_static("White", &pangolin::Colour::White)
      .def_static("Black", &pangolin::Colour::Black)
      .def_static("Red", &pangolin::Colour::Red)
      .def_static("Green", &pangolin::Colour::Green)
      .def_static("Blue", &pangolin::Colour::Blue)
      .def_static("Hsv", &pangolin::Colour::Hsv)
      .def("WithAlpha", &pangolin::Colour::WithAlpha);
  }

}  // py_pangolin
