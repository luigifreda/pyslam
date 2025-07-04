#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/gl/glinclude.h>
#include <pangolin/display/attach.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareAttach(py::module & m) {

    py::enum_<Unit>(m, "Unit")
        .value("Fraction", Unit::Fraction)
        .value("Pixel", Unit::Pixel)
        .value("ReversePixel", Unit::ReversePixel)
        .export_values();


    py::class_<Attach, std::shared_ptr<Attach>>(m, "Attach")
        .def(py::init<>(), "Attach to Left/Bottom edge")
        .def(py::init<Unit, GLfloat>(), "General constructor")
        .def(py::init<GLfloat>(), "Specify relative position in range [0,1].")

        .def_static("Pix", &Attach::Pix, "p"_a,
            "Specify absolute position from leftmost / bottom-most edge.")

        .def_static("ReversePix", &Attach::ReversePix, "p"_a,
            "Specify absolute position from rightmost / topmost edge.")

        .def_static("Frac", &Attach::Frac, "frac"_a,
            "Specify relative position in range [0,1].")

        .def_readwrite("unit", &Attach::unit)
        .def_readwrite("p", &Attach::p)
    ;

}

}  // namespace pangolin::