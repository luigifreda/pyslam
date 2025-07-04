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

#pragma once

#include <pybind11/pybind11.h>
#include <pangolin/handler/handler.h>
#include <pangolin/display/view.h>

namespace py_pangolin {
  
  class PyHandler : public pangolin::Handler {
  public:
    using pangolin::Handler::Handler;

    void Keyboard(pangolin::View& v, unsigned char key, int x, int y, bool pressed) override {
      PYBIND11_OVERLOAD(void, pangolin::Handler, Keyboard, v, key, x, y, pressed);
    }
    void Mouse(pangolin::View& v, pangolin::MouseButton button, int x, int y, bool pressed, int button_state) override {
      PYBIND11_OVERLOAD(void, pangolin::Handler, Mouse, v, button, x, y, pressed, button_state);
    }

    void MouseMotion(pangolin::View& v, int x, int y, int button_state) override {
      PYBIND11_OVERLOAD(void, pangolin::Handler, MouseMotion, v, x, y, button_state);
    }
    void PassiveMouseMotion(pangolin::View& v, int x, int y, int button_state) override {
      PYBIND11_OVERLOAD(void, pangolin::Handler, PassiveMouseMotion, v, x, y, button_state);
    }
      
    void Special(pangolin::View& v, pangolin::InputSpecial inType, float x, float y, float p1, float p2, float p3, float p4, int button_state) override{
      PYBIND11_OVERLOAD(void, pangolin::Handler, Special, v, inType, x, y, p1, p2, p3, p4, button_state);
    }
  };

  void bind_handler(pybind11::module &m);

}  // py_pangolin

