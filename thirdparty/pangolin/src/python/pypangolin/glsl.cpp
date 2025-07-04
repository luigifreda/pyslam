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

#include "glsl.hpp"
#include <pangolin/gl/glsl.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>

namespace py_pangolin {

  void bind_glsl(pybind11::module &m) {

    pybind11::enum_<pangolin::GlSlShaderType>(m, "GlSlShaderType")
      .value("GlSlAnnotatedShader", pangolin::GlSlShaderType::GlSlAnnotatedShader)
      .value("GlSlFragmentShader", pangolin::GlSlShaderType::GlSlFragmentShader)
      .value("GlSlVertexShader", pangolin::GlSlShaderType::GlSlVertexShader)
      .value("GlSlGeometryShader", pangolin::GlSlShaderType::GlSlGeometryShader)
      .value("GlSlComputeShader", pangolin::GlSlShaderType::GlSlComputeShader)
      .export_values();

    pybind11::class_<pangolin::GlSlProgram>(m, "GlSlProgram")
      .def(pybind11::init<>())
      .def("AddShader", &pangolin::GlSlProgram::AddShader, pybind11::arg("shader_type"), pybind11::arg("filename"), pybind11::arg("program_defines")=std::map<std::string,std::string>(), pybind11::arg("search_path")=std::vector<std::string>())
      .def("AddShaderFromFile", &pangolin::GlSlProgram::AddShaderFromFile)
      .def("Link", &pangolin::GlSlProgram::Link)
      .def("GetAttributeHandle", &pangolin::GlSlProgram::GetAttributeHandle)
      .def("GetUniformHandle", &pangolin::GlSlProgram::GetUniformHandle)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, int))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, int, int))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, int, int, int))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, int, int, int, int))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, float))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, float, float))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, float, float, float))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, float, float, float, float))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, const pangolin::OpenGlMatrix &))&pangolin::GlSlProgram::SetUniform)
#ifdef HAVE_EIGEN
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, const Eigen::Matrix3f &))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, const Eigen::Matrix4f &))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, const Eigen::Matrix3d &))&pangolin::GlSlProgram::SetUniform)
      .def("SetUniform", (void (pangolin::GlSlProgram::*)(const std::string &, const Eigen::Matrix4d &))&pangolin::GlSlProgram::SetUniform)
#endif
      .def("Bind", &pangolin::GlSlProgram::Bind)
      .def("Unbind", &pangolin::GlSlProgram::Unbind);

  }

}
