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

#include "viewport.hpp"
#include <pangolin/display/viewport.h>

namespace py_pangolin {

  void bind_viewport(pybind11::module &m) {
    pybind11::class_<pangolin::Viewport>(m, "Viewport")
      .def(pybind11::init<>())
      .def(pybind11::init<int, int, int, int>())
      .def("Activate", &pangolin::Viewport::Activate)
      .def("ActivateIdentity", &pangolin::Viewport::ActivateIdentity)
      .def("ActivatePixelOrthographic", &pangolin::Viewport::ActivatePixelOrthographic)
      .def("Scissor", &pangolin::Viewport::Scissor)
      .def("ActivateAndScissor", &pangolin::Viewport::ActivateAndScissor)
      .def("Contains", &pangolin::Viewport::Contains)
      .def("Inset", (pangolin::Viewport (pangolin::Viewport::*)(int)const)&pangolin::Viewport::Inset)
      .def("Inset", (pangolin::Viewport (pangolin::Viewport::*)(int, int)const)&pangolin::Viewport::Inset)
      .def("Insetsect", &pangolin::Viewport::Intersect)
      .def("DisableScissor", &pangolin::Viewport::DisableScissor)
      .def("r", &pangolin::Viewport::r)
      .def("t", &pangolin::Viewport::t)
      .def("aspect", &pangolin::Viewport::aspect)
      .def_readwrite("l", &pangolin::Viewport::l)
      .def_readwrite("b", &pangolin::Viewport::b)
      .def_readwrite("w", &pangolin::Viewport::w)
      .def_readwrite("h", &pangolin::Viewport::h);
  }

}  // py_pangolin
