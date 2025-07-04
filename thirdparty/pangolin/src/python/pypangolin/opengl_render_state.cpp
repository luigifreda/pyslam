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

#include "opengl_render_state.hpp"
#include <pangolin/display/opengl_render_state.h>
#include <pybind11/eigen.h>
#include <pybind11/operators.h>
#include <pybind11/numpy.h>

namespace py_pangolin {

  void bind_opengl_render_state(pybind11::module &m){
    pybind11::enum_<pangolin::AxisDirection>(m, "AxisDirection")
      .value("AxisNone", pangolin::AxisDirection::AxisNone)
      .value("AxisNegX", pangolin::AxisDirection::AxisNegX)
      .value("AxisNegY", pangolin::AxisDirection::AxisNegY)
      .value("AxisNegZ", pangolin::AxisDirection::AxisNegZ)
      .value("AxisX", pangolin::AxisDirection::AxisX)
      .value("AxisY", pangolin::AxisDirection::AxisY)
      .value("AxisZ", pangolin::AxisDirection::AxisZ)
      .export_values();

    pybind11::enum_<pangolin::OpenGlStack>(m, "OpenGlStack")
      .value("GlModelViewStack", pangolin::OpenGlStack::GlModelViewStack)
      .value("GlProjectionStack", pangolin::OpenGlStack::GlProjectionStack)
      .value("GlTextureStack", pangolin::OpenGlStack::GlTextureStack)
      .export_values();

    pybind11::class_<pangolin::OpenGlMatrix>(m, "OpenGlMatrix")
      .def("Translate", &pangolin::OpenGlMatrix::Translate)
      .def("Scale", &pangolin::OpenGlMatrix::Scale)
      .def("RotateX", &pangolin::OpenGlMatrix::RotateX)
      .def("RotateY", &pangolin::OpenGlMatrix::RotateY)
      .def("RotateZ", &pangolin::OpenGlMatrix::RotateZ)
      .def(pybind11::init<>())
      .def(pybind11::init<const Eigen::Matrix<float, 4, 4> >())
      .def(pybind11::init<const Eigen::Matrix<double, 4, 4> >())
      .def("Load", &pangolin::OpenGlMatrix::Load)
      .def("Multiply", &pangolin::OpenGlMatrix::Multiply)
      .def("SetIdentity", &pangolin::OpenGlMatrix::SetIdentity)
      .def("Transpose", &pangolin::OpenGlMatrix::Transpose)
      .def("Inverse", &pangolin::OpenGlMatrix::Inverse)
      .def("Matrix", [](pangolin::OpenGlMatrix& mat){
            using T = pangolin::GLprecision;
            return pybind11::array_t<T>( {4, 4 }, {1*sizeof(T), 4*sizeof(T)}, mat.m );
      })
      .def(pybind11::self * pybind11::self);

    pybind11::class_<pangolin::OpenGlMatrixSpec, pangolin::OpenGlMatrix>(m, "OpenGlMatrixSpec")
      .def(pybind11::init<>());

    m.def("ProjectionMatrixRUB_BottomLeft", &pangolin::ProjectionMatrixRUB_BottomLeft);
    m.def("ProjectionMatrixRUB_TopLeft", &pangolin::ProjectionMatrixRUB_TopLeft);
    m.def("ProjectionMatrixRDF_BottomLeft", &pangolin::ProjectionMatrixRDF_BottomLeft);
    m.def("ProjectionMatrixRDF_TopLeft", &pangolin::ProjectionMatrixRDF_TopLeft);
    m.def("ProjectionMatrix", &pangolin::ProjectionMatrix);
    m.def("ProjectionMatrixOrthographic", &pangolin::ProjectionMatrixOrthographic);
    m.def("ModelViewLookAtRUB", &pangolin::ModelViewLookAtRUB);
    m.def("ModelViewLookAtRDF", &pangolin::ModelViewLookAtRDF);
    m.def("ModelViewLookAt", (pangolin::OpenGlMatrix (*)(pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::AxisDirection))&pangolin::ModelViewLookAt);
    m.def("ModelViewLookAt", (pangolin::OpenGlMatrix (*)(pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision, pangolin::GLprecision))&pangolin::ModelViewLookAt);
    m.def("IdentityMatrix", (pangolin::OpenGlMatrix (*)())&pangolin::IdentityMatrix);
    m.def("IdentityMatrix", (pangolin::OpenGlMatrixSpec (*)(pangolin::OpenGlStack))&pangolin::IdentityMatrix);
    m.def("negIdentityMatrix", &pangolin::negIdentityMatrix);

    pybind11::class_<pangolin::OpenGlRenderState>(m, "OpenGlRenderState")
      .def(pybind11::init<const pangolin::OpenGlMatrix&>())
      .def(pybind11::init<const pangolin::OpenGlMatrix&, const pangolin::OpenGlMatrix&>())
      .def(pybind11::init<>())
      .def("ApplyIdentity", &pangolin::OpenGlRenderState::ApplyIdentity)
      .def("Apply", &pangolin::OpenGlRenderState::Apply)
      .def("SetProjectionMatrix", &pangolin::OpenGlRenderState::SetProjectionMatrix)
      .def("SetModelViewMatrix", &pangolin::OpenGlRenderState::SetModelViewMatrix)
      .def("GetProjectionMatrix", (pangolin::OpenGlMatrix& (pangolin::OpenGlRenderState::*)())&pangolin::OpenGlRenderState::GetProjectionMatrix)
      .def("GetProjectionMatrix", (pangolin::OpenGlMatrix (pangolin::OpenGlRenderState::*)() const)&pangolin::OpenGlRenderState::GetProjectionMatrix)
      .def("GetModelViewMatrix", (pangolin::OpenGlMatrix& (pangolin::OpenGlRenderState::*)())&pangolin::OpenGlRenderState::GetModelViewMatrix)
      .def("GetModelViewMatrix", (pangolin::OpenGlMatrix (pangolin::OpenGlRenderState::*)() const)&pangolin::OpenGlRenderState::GetModelViewMatrix)
      .def("GetProjectionModelViewMatrix", &pangolin::OpenGlRenderState::GetProjectionModelViewMatrix)
      .def("GetProjectiveTextureMatrix", &pangolin::OpenGlRenderState::GetProjectiveTextureMatrix)
      .def("EnableProjectiveTexturing", &pangolin::OpenGlRenderState::EnableProjectiveTexturing)
      .def("DisableProjectiveTexturing", &pangolin::OpenGlRenderState::DisableProjectiveTexturing)
      .def("Follow", &pangolin::OpenGlRenderState::Follow, pybind11::arg("T_wc"), pybind11::arg("follow")=true)
      .def("Unfollow", &pangolin::OpenGlRenderState::Unfollow)
      .def("GetProjectionMatrix", (pangolin::OpenGlMatrix& (pangolin::OpenGlRenderState::*)(unsigned int))&pangolin::OpenGlRenderState::GetProjectionMatrix)
      .def("GetProjectionMatrix", (pangolin::OpenGlMatrix (pangolin::OpenGlRenderState::*)(unsigned int) const)&pangolin::OpenGlRenderState::GetProjectionMatrix)
      .def("GetViewOffset", (pangolin::OpenGlMatrix& (pangolin::OpenGlRenderState::*)(unsigned int))&pangolin::OpenGlRenderState::GetViewOffset)
      .def("GetViewOffset", (pangolin::OpenGlMatrix (pangolin::OpenGlRenderState::*)(unsigned int) const)&pangolin::OpenGlRenderState::GetViewOffset)
      .def("GetModelViewMatrix", (pangolin::OpenGlMatrix (pangolin::OpenGlRenderState::*)(int) const)&pangolin::OpenGlRenderState::GetModelViewMatrix)
      .def("ApplyNView", &pangolin::OpenGlRenderState::ApplyNView);
  }

}  // py_pangolin
