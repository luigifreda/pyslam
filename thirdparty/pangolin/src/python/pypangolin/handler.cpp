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

#include "handler.hpp"
#include <pangolin/handler/handler.h>
#include <pangolin/display/opengl_render_state.h>
#include <pangolin/plot/plotter.h>


namespace py_pangolin {
  
  void bind_handler(pybind11::module &m) {
    
    pybind11::class_<pangolin::Handler, PyHandler> handler(m, "Handler");
    handler
      .def(pybind11::init<>())
      .def("Keyboard", &pangolin::Handler::Keyboard)
      .def("Mouse", &pangolin::Handler::Mouse)
      .def("MouseMotion", &pangolin::Handler::MouseMotion)
      .def("PassiveMouseMotion", &pangolin::Handler::PassiveMouseMotion)
      .def("Special", &pangolin::Handler::Special);

    pybind11::class_<pangolin::Handler3D>(m, "Handler3D", handler)
      .def(pybind11::init<pangolin::OpenGlRenderState&, pangolin::AxisDirection, float, float>(), pybind11::arg("state"), pybind11::arg("enforce_up") = pangolin::AxisNone, pybind11::arg("trans_scale")=0.01f, pybind11::arg("zoom_factor")=PANGO_DFLT_HANDLER3D_ZF)
      .def("ValidWinDepth", &pangolin::Handler3D::ValidWinDepth)
      .def("PixelUnproject", &pangolin::Handler3D::PixelUnproject)
      .def("GetPosNormal", &pangolin::Handler3D::GetPosNormal)
      .def("Keyboard", &pangolin::Handler3D::Keyboard)
      .def("Mouse", &pangolin::Handler3D::Mouse)
      .def("MouseMotion", &pangolin::Handler3D::MouseMotion)
      .def("Special", &pangolin::Handler3D::Special);

  }
}  // py_pangolin
