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

#pragma once

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/embed.h>

#include "attach.hpp"
#include "colour.hpp"
#include "datalog.hpp"
#include "display.hpp"
#include "gl.hpp"
#include "glsl.hpp"
#include "gl_draw.hpp"
#include "handler.hpp"
#include "image.hpp"
#include "opengl_render_state.hpp"
#include "params.hpp"
#include "pixel_format.hpp"
#include "plotter.hpp"
#include "var.hpp"
#include "video.hpp"
#include "view.hpp"
#include "viewport.hpp"
#include "widget.hpp"
#include "window.hpp"
#include "image_view.hpp"

namespace pypangolin {

inline void PopulateModule(pybind11::module& m)
{
    m.doc() = "pypangolin python wrapper for Pangolin rapid prototyping graphics and video library.";

    py_pangolin::bind_var(m);
    py_pangolin::bind_params(m);
    py_pangolin::bind_viewport(m);
    py_pangolin::bind_view(m);
    py_pangolin::bind_window(m);
    py_pangolin::bind_display(m);
    py_pangolin::bind_opengl_render_state(m);
    py_pangolin::bind_attach(m);
    py_pangolin::bind_colour(m);
    py_pangolin::bind_datalog(m);
    py_pangolin::bind_plotter(m);
    py_pangolin::bind_handler(m);
    py_pangolin::bind_gl(m);
    py_pangolin::bind_glsl(m);
    py_pangolin::bind_gl_draw(m);
    py_pangolin::bind_widget(m);
    py_pangolin::bind_pixel_format(m);
    py_pangolin::bind_image<unsigned char>(m, "Image");
    py_pangolin::bind_video(m);
    py_pangolin::bind_image_view(m);
}

}
