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

#include "display.hpp"
#include <pangolin/display/display.h>
#include <pangolin/display/view.h>
#include <pybind11/functional.h>

namespace py_pangolin {
  
  void bind_display(pybind11::module &m) {
    m.def("CreateWindowAndBind",
          &pangolin::CreateWindowAndBind,
          pybind11::return_value_policy::reference,
          pybind11::arg("window_title"),
          pybind11::arg("w") = 640,
          pybind11::arg("h") = 480,
          pybind11::arg("params") = pangolin::Params());
	
    m.def("DestroyWindow",
          &pangolin::DestroyWindow,
          pybind11::arg("window_title"));

    m.def("CreateDisplay",
          &pangolin::CreateDisplay,
          pybind11::return_value_policy::reference);

    m.def("ShouldQuit",
          &pangolin::ShouldQuit);

    m.def("FinishFrame",
          &pangolin::FinishFrame);

    m.def("ToggleFullscreen",
          &pangolin::ToggleFullscreen);

    m.def("ToggleConsole",
          &pangolin::ToggleConsole);

    m.def("RegisterKeyPressCallback",
          [](int v, const std::function<void()>& f){
            pangolin::RegisterKeyPressCallback(v, f);
          }
    );
    
    m.def("DisplayBase",
          &pangolin::DisplayBase,
          pybind11::return_value_policy::reference);

    m.def("Display",
          &pangolin::Display,
          pybind11::return_value_policy::reference,
          pybind11::arg("name"));

  }

}  // Py_pangolin
