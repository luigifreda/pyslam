#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/gl/gl.h>
#include "colour.hpp"



namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareGL(py::module & m) {

    declareColour(m);



    py::class_<GlTexture, std::shared_ptr<GlTexture>>(m, "GlTexture")
        .def(py::init<>(), "Default constructor, represents 'no texture'")
        .def(py::init<GLint, GLint, GLint, bool, int, GLenum, GLenum, GLvoid*>(),
            "width"_a, "height"_a, "internal_format"_a = GL_RGBA8, "sampling_linear"_a = true, "border"_a = 0,
            "glformat"_a = GL_RGBA, "gltype"_a = GL_UNSIGNED_BYTE, "data"_a = nullptr,
            "internal_format normally one of GL_RGBA8, GL_LUMINANCE8, GL_INTENSITY16")

        .def("Reinitialise", &GlTexture::Reinitialise, 
            "width"_a, "height"_a, "internal_format"_a = GL_RGBA8, "sampling_linear"_a = true, "border"_a = 0, 
            "glformat"_a = GL_RGBA, "gltype"_a = GL_UNSIGNED_BYTE, "data"_a = nullptr, 
            "Reinitialise teture width / height / format")  // virtual function

        .def("IsValid", &GlTexture::IsValid)
        .def("Delete", &GlTexture::Delete, "Delete OpenGL resources and fall back to representing 'no texture'")
        .def("Bind", &GlTexture::Bind)
        .def("Unbind", &GlTexture::Unbind)

        .def("Upload", (void (GlTexture::*) (const void*, GLenum, GLenum)) &GlTexture::Upload,
            "image"_a, "data_format"_a = GL_LUMINANCE, "data_type"_a = GL_FLOAT,
            "data_layout normally one of GL_LUMINANCE, GL_RGB, ...\n"
            "data_type normally one of GL_UNSIGNED_BYTE, GL_UNSIGNED_SHORT, GL_FLOAT")
        .def("Upload", (void (GlTexture::*) (const void*, GLsizei, GLsizei, GLsizei, GLsizei, GLenum, GLenum))
            &GlTexture::Upload, "data"_a, "tex_x_offset"_a,  "tex_y_offset"_a, "data_w"_a, "data_h"_a,
            "data_format"_a, "data_type"_a, "Upload data to texture, overwriting a sub-region of it.\n"
            "data ptr contains packed data_w x data_h of pixel data.")
        .def("Download", (void (GlTexture::*) (void*, GLenum, GLenum) const) &GlTexture::Download, 
            "image"_a, "data_layout"_a = GL_LUMINANCE, "data_type"_a = GL_FLOAT)
        .def("Download", (void (GlTexture::*) (TypedImage&) const) &GlTexture::Download, "image"_a)

        .def("Upload", [](GlTexture &t, py::array_t<unsigned char> img, GLenum data_format, GLenum data_type) {
                auto buf = img.request();
                unsigned char* data = (unsigned char*) buf.ptr;
                t.Upload(data, data_format, data_type);
            },
            "image"_a, "data_format"_a = GL_RGB, "data_type"_a = GL_UNSIGNED_BYTE)

        .def("Load", &GlTexture::Load, "image"_a, "sampling_linear"_a = true)
        .def("LoadFromFile", &GlTexture::LoadFromFile, "filename"_a, "sampling_linear"_a)
        .def("Save", &GlTexture::Save, "filename"_a, "top_line_first"_a = true)
        .def("SetLinear", &GlTexture::SetLinear)
        .def("SetNearestNeighbour", &GlTexture::SetNearestNeighbour)

        .def("RenderToViewport", (void (GlTexture::*) () const) &GlTexture::RenderToViewport)
        .def("RenderToViewport", (void (GlTexture::*) (Viewport, bool, bool) const) &GlTexture::RenderToViewport,
            "tex_vp"_a, "flipx"_a = false, "flipy"_a = false)
        .def("RenderToViewport", (void (GlTexture::*) (const bool) const) &GlTexture::RenderToViewport, "flip"_a)
            
        .def("RenderToViewportFlipY", &GlTexture::RenderToViewportFlipY)
        .def("RenderToViewportFlipXFlipY", &GlTexture::RenderToViewportFlipXFlipY)

        .def_readwrite("internal_format", &GlTexture::internal_format)
        .def_readwrite("tid", &GlTexture::tid)
        .def_readwrite("width", &GlTexture::width)
        .def_readwrite("height", &GlTexture::height)

        // move constructor
        // move assignment
        // private copy constructor
    
    ;

    // GlRenderBuffer
    // GlFramebuffer
    // GlBufferType
    // GlBuffer
    // GlSizeableBuffer
    // 


}

}  // namespace pangolin::