#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/handler/handler.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareHandler(py::module & m) {

    py::class_<Handler, std::shared_ptr<Handler>>(m, "Handler");

    py::class_<HandlerScroll, std::shared_ptr<HandlerScroll>, Handler>(m, "HandlerScroll")
        .def("Mouse", &HandlerScroll::Mouse,
            "view"_a, "botton"_a, "x"_a, "y"_a, "pressed"_a, "button_state"_a)
        .def("Special", &HandlerScroll::Special,
            "view"_a, "inType"_a, "x"_a, "y"_a, "p1"_a, "p2"_a, "p3"_a, "p4"_a, "button_state"_a)
    ;


    py::class_<Handler3D, std::shared_ptr<Handler3D>, Handler>(m, "Handler3D")
        .def(py::init<OpenGlRenderState&, AxisDirection, float, float>(),
            "cam_state"_a, "enforce_up"_a = AxisDirection::AxisNone, "trans_scale"_a = 0.01f,
            "zoom_fraction"_a = PANGO_DFLT_HANDLER3D_ZF)
        .def(py::init<Handler3D&>())   // copy constructor

        .def("Keyboard", &Handler3D::Keyboard, "view"_a, "key"_a, "x"_a, "y"_a, "pressed"_a)
        .def("Mouse", &Handler3D::Mouse,
            "view"_a, "botton"_a, "x"_a, "y"_a, "pressed"_a, "button_state"_a)
        .def("MouseMotion", &Handler3D::MouseMotion, "view"_a, "x"_a, "y"_a, "button_state"_a)
        .def("Special", &Handler3D::Special,
            "view"_a, "inType"_a, "x"_a, "y"_a, "p1"_a, "p2"_a, "p3"_a, "p4"_a, "button_state"_a)

#ifdef USE_EIGEN
        .def("Selected_P_w", &Handler3D::Selected_P_w)   // inline
#endif

    ;

}

}  // namespace pangolin::