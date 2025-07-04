#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/gl/colour.h>



namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareColour(py::module & m) {

    // py::class_<Colour>(m, "Colour")
    //     .def_static("White", &Colour::White)
    //     .def_static("Black", &Colour::Black)
    //     .def_static("Red", &Colour::Red)
    //     .def_static("Green", &Colour::Green)
    //     .def_static("Blue", &Colour::Blue)
    //     .def_static("Unspecified", &Colour::Unspecified)

    //     .def(py::init<>())
    //     .def(py::init<float, float, float, float>(),
    //         "red"_a, "green"_a, "blue"_a, "alpha"_a=1.0)
    //     // .def(py::init<float[4]>(), "rgba")

    //     .def("Get", &Colour::Get)
    //     .def("WithAlpha", &Colour::WithAlpha)
    //     .def_static("Hsv", &Colour::Hsv,
    //         "hue"_a, "sat"_a=1.0, "val"_a=1.0, "alpha"_a=1.0)
    // ;


}

}