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

#include "gl_draw.hpp"
#include <pangolin/gl/gldraw.h>
#include <pybind11/eigen.h>

namespace py_pangolin {

  void bind_gl_draw(pybind11::module &m){

    m.def("glDrawAxis",
          (void (*)(float))&pangolin::glDrawAxis);

    m.def("glDrawColouredCube",
          &pangolin::glDrawColouredCube,
          pybind11::arg("axis_min") = -0.5f,
          pybind11::arg("axis_max") = +0.5f);

    m.def("glDraw_x0",
          &pangolin::glDraw_x0);
    m.def("glDraw_y0",
          &pangolin::glDraw_y0);
    m.def("glDraw_z0",
          &pangolin::glDraw_z0);

    m.def("glDrawFrustum", (void (*)(GLfloat, GLfloat, GLfloat, GLfloat, int, int, GLfloat)) &pangolin::glDrawFrustum);
    m.def("glDrawFrustum", (void (*)(const Eigen::Matrix3f &, int, int, GLfloat)) &pangolin::glDrawFrustum<float>);
    m.def("glDrawFrustum", (void (*)(const Eigen::Matrix3d &, int, int, GLfloat)) &pangolin::glDrawFrustum<double>);
    m.def("glDrawFrustum", (void (*)(const Eigen::Matrix3f &, int, int, const Eigen::Matrix4f &, float)) &pangolin::glDrawFrustum<float>);
    m.def("glDrawFrustum", (void (*)(const Eigen::Matrix3d &, int, int, const Eigen::Matrix4d &, double)) &pangolin::glDrawFrustum<double>);

    m.def("glDrawAxis", (void (*)(float)) &pangolin::glDrawAxis);
    m.def("glDrawAxis", (void (*)(const Eigen::Matrix4f &, float)) &pangolin::glDrawAxis<Eigen::Matrix4f, float>);
    m.def("glDrawAxis", (void (*)(const Eigen::Matrix4d &, float)) &pangolin::glDrawAxis<Eigen::Matrix4d, float>);

    m.def("glSetFrameOfReference", (void (*)(const Eigen::Matrix4f &)) &pangolin::glSetFrameOfReference);
    m.def("glSetFrameOfReference", (void (*)(const Eigen::Matrix4d &)) &pangolin::glSetFrameOfReference);

    m.def("glUnsetFrameOfReference", &pangolin::glUnsetFrameOfReference);

    m.def("glDrawAlignedBox", (void (*)(const Eigen::AlignedBox2f &, GLenum)) &pangolin::glDrawAlignedBox<float>, pybind11::arg("box"), pybind11::arg("mode") = GL_TRIANGLE_FAN);

  }


}  // py_pangolin
