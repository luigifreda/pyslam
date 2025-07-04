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

#include "gl.hpp"
#include <pangolin/gl/gl.h>
#include <pybind11/numpy.h>

namespace py_pangolin {

  bool is_packed(const pybind11::buffer_info & info) {
    int next_expected_stride = info.itemsize;
    for (int i = info.ndim-1; i >= 0; --i) {
      if (!(info.strides[i] == next_expected_stride)) {
        return false;
      }
      next_expected_stride *= info.shape[i];
    }
    return true;
  }

  void bind_gl(pybind11::module &m) {

    pybind11::class_<pangolin::GlTexture>(m, "GlTexture")
      .def(pybind11::init<>())
      .def(pybind11::init<GLint, GLint, GLint, bool, int, GLenum, GLenum>(), pybind11::arg("width"), pybind11::arg("height"), pybind11::arg("internal_format") = GL_RGBA8, pybind11::arg("sampling_linear") = true, pybind11::arg("border") = 0, pybind11::arg("glformat") = GL_RGBA, pybind11::arg("gltype") = GL_UNSIGNED_BYTE)
      .def("Upload", [](pangolin::GlTexture & texture, pybind11::buffer b, GLenum data_format, GLenum type){
        pybind11::buffer_info info = b.request();
        texture.Upload(info.ptr, data_format, type);
      })
      .def("Download", [](pangolin::GlTexture & texture, pybind11::buffer b, GLenum data_layout, GLenum data_type) {
        pybind11::buffer_info info = b.request();
        if (!is_packed(info)) {
          throw std::runtime_error("cannot Download into non-packed buffer");
        }
        texture.Download(info.ptr, data_layout, data_type);
      })
      .def("Save", &pangolin::GlTexture::Save, pybind11::arg("filename"), pybind11::arg("top_line_first")=true)
      .def("RenderToViewport", (void (pangolin::GlTexture::*)() const)&pangolin::GlTexture::RenderToViewport)
      .def("RenderToViewportFlipY", &pangolin::GlTexture::RenderToViewportFlipY)
      .def("SetNearestNeighbour", &pangolin::GlTexture::SetNearestNeighbour);

    pybind11::class_<pangolin::GlRenderBuffer>(m, "GlRenderBuffer")
      .def(pybind11::init<GLint, GLint, GLint>(), pybind11::arg("width")=0, pybind11::arg("height")=0, pybind11::arg("internal_format") = GL_DEPTH_COMPONENT24)
      .def("Reinitialise", &pangolin::GlRenderBuffer::Reinitialise);

    pybind11::class_<pangolin::GlFramebuffer>(m, "GlFramebuffer")
      .def(pybind11::init<pangolin::GlTexture &, pangolin::GlRenderBuffer &>())
      .def("Bind", &pangolin::GlFramebuffer::Bind)
      .def("Unbind", &pangolin::GlFramebuffer::Unbind);

    pybind11::enum_<pangolin::GlBufferType>(m, "GlBufferType")
      .value("GlUndefined", pangolin::GlBufferType::GlUndefined)
      .value("GlArrayBuffer", pangolin::GlBufferType::GlArrayBuffer)
      .value("GlElementArrayBuffer", pangolin::GlBufferType::GlElementArrayBuffer)
#ifndef HAVE_GLES
      .value("GlPixelPackBuffer", pangolin::GlBufferType::GlPixelPackBuffer)
      .value("GlPixelUnpackBuffer", pangolin::GlBufferType::GlPixelUnpackBuffer)
      .value("GlShaderStorageBuffer", pangolin::GlBufferType::GlShaderStorageBuffer)
#endif
      .export_values();

    pybind11::class_<pangolin::GlBufferData>(m, "GlBufferData")
      .def(pybind11::init<>())
      .def("Reinitialise", [](pangolin::GlBufferData & gl_buffer, pangolin::GlBufferType buffer_type, GLuint size_bytes, GLenum gl_use) {
        gl_buffer.Reinitialise(buffer_type, size_bytes, gl_use);
      })
      .def("Bind", &pangolin::GlBufferData::Bind)
      .def("Unbind", &pangolin::GlBufferData::Unbind)
      .def("Upload", [](pangolin::GlBufferData & gl_buffer, pybind11::buffer b, GLsizeiptr size_bytes, GLintptr offset) {
        pybind11::buffer_info info = b.request();
        gl_buffer.Upload(info.ptr, size_bytes, offset);
      }, pybind11::arg("data"), pybind11::arg("size_bytes"), pybind11::arg("offset")=0)
      .def_readwrite("size_bytes", &pangolin::GlBufferData::size_bytes);

  }


} // namespace py_pangolin
