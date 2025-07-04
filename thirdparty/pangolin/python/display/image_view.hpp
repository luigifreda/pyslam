#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/display/image_view.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareImageView(py::module & m) {

    py::class_<ImageView, std::shared_ptr<ImageView>, View, ImageViewHandler>(m, "ImageView")
        .def(py::init<>())

        .def("Render", &ImageView::Render)
        .def("Mouse", &ImageView::Mouse, "view"_a, "button"_a, "x"_a, "y"_a, "pressed"_a, "button_state"_a)
        .def("Keyboard", &ImageView::Keyboard, "view"_a, "key"_a, "x"_a, "y"_a, "pressed"_a)
        .def("Tex", &ImageView::Tex)

        .def("SetImage", (ImageView& (ImageView::*) (void*, size_t, size_t, size_t, GlPixFormat, bool)) 
            &ImageView::SetImage, "ptr"_a, "w"_a, "h"_a, "pitch"_a, "img_fmt"_a, "delayed_upload"_a = false)
        .def("SetImage", (ImageView& (ImageView::*) (const Image<unsigned char>&, const GlPixFormat&, bool)) 
            &ImageView::SetImage, "img"_a, "glfmt"_a, "delayed_upload"_a = false)
        .def("SetImage", (ImageView& (ImageView::*) (const TypedImage&, bool)) &ImageView::SetImage,
            "img"_a, "delayed_upload"_a = false)
        .def("SetImage", (ImageView& (ImageView::*) (const GlTexture&)) &ImageView::SetImage, "texture"_a)

        .def("LoadPending", &ImageView::LoadPending)
        .def("Clear", &ImageView::Clear)
        .def("GetOffsetScale", &ImageView::GetOffsetScale)
        .def("MouseReleased", &ImageView::MouseReleased)
        .def("MousePressed", &ImageView::MousePressed)
        .def("SetRenderOverlay", &ImageView::SetRenderOverlay, "val"_a)

    ;
    //py::class_<Params, std::shared_ptr<Params>> cls(m, "Params");

    //cls.def(py::init<>());

}

}  // namespace pangolin::