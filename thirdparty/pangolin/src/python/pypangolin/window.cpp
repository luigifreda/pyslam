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

#include "window.hpp"
#include <pangolin/display/window.h>


namespace py_pangolin {

  class PyWindowInterface: public pangolin::WindowInterface{
    public:
    using pangolin::WindowInterface::WindowInterface;

    void ToggleFullscreen() override {
      PYBIND11_OVERLOAD_PURE(
                             void,
                             pangolin::WindowInterface,
                             ToggleFullscreen);
    }

    void Move(int x, int y) override {
      PYBIND11_OVERLOAD_PURE(
                             void,
                             pangolin::WindowInterface,
                             Move,
                             x,
                             y);
    }

    void Resize(unsigned int w, unsigned int h) override {
      PYBIND11_OVERLOAD_PURE(
                             void,
                             pangolin::WindowInterface,
                             Resize,
                             w,
                             h);
    }

    void MakeCurrent() override {
      PYBIND11_OVERLOAD_PURE(
                             void,
                             pangolin::WindowInterface,
                             MakeCurrent);
    }

    void RemoveCurrent() override {
      PYBIND11_OVERLOAD_PURE(
                             void,
                             pangolin::WindowInterface,
                             RemoveCurrent);
    }

    void ProcessEvents() override {
      PYBIND11_OVERLOAD_PURE(
                             void,
                             pangolin::WindowInterface,
                             ProcessEvents);
    }


    void SwapBuffers() override {
      PYBIND11_OVERLOAD_PURE(
                             void,
                             pangolin::WindowInterface,
                             SwapBuffers);
    }

  };
  
  void bind_window(pybind11::module &m) {
    pybind11::class_<pangolin::WindowInterface, PyWindowInterface > windows_interface(m, "WindowsInterface");
    windows_interface
      .def(pybind11::init<>())
      .def("ToggleFullscreen", &pangolin::WindowInterface::ToggleFullscreen)
      .def("Move", &pangolin::WindowInterface::Move)
      .def("Resize", &pangolin::WindowInterface::Resize)
      .def("MakeCurrent", &pangolin::WindowInterface::MakeCurrent)
      .def("ProcessEvents", &pangolin::WindowInterface::ProcessEvents)
      .def("SwapBuffers", &pangolin::WindowInterface::SwapBuffers);
  }
}
