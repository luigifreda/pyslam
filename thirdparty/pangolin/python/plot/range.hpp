#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/operators.h>

#include <pangolin/plot/range.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

namespace {

    template<typename T>
    void templatedRange(py::module& m, const std::string& suffix) {
        using CLS = Range<T>;

        py::class_<CLS>(m, ("Range" + suffix).c_str())
            .def(py::init<>())
            .def(py::init<T, T>())

            .def_static("Open", &CLS::Open)
            .def_static("Empty", &CLS::Empty)
            .def_static("Containing", &CLS::Containing)

            // .def(py::self + T())
            // .def(py::self - T())
            // .def(py::self += T())
            // .def(py::self -= T())
            // .def(py::self *= T())
            // .def(py::self /= T())
            // .def(py::self + CLS())
            // .def(py::self - CLS())
            // .def(py::self += CLS())
            // .def(py::self -= CLS())
            // .def(py::self * float())
            .def("Size", &CLS::Size)
            .def("AbsSize", &CLS::AbsSize)
            .def("Mid", &CLS::Mid)
            .def("Scale", &CLS::Scale)
            .def("Insert", (void (CLS::*) (T)) &CLS::Insert)
            .def("Insert", (void (CLS::*) (const Range<T>&)) &CLS::Insert)
            .def("Clamp", (void (CLS::*) (T, T)) &CLS::Clamp)
            .def("Clamp", (void (CLS::*) (const CLS&)) &CLS::Clamp)
            .def("Clear", &CLS::Clear)
            .def("Contains", &CLS::Contains)
            .def("ContainsWeak", &CLS::ContainsWeak)
        ;

        m.def("Round", (Rangei (*) (const Range<T>&)) &Round);
    }

    template<typename T>
    void templatedXYRange(py::module& m, const std::string& suffix) {
        using CLS = XYRange<T>;

        py::class_<CLS>(m, ("XYRange" + suffix).c_str())
            .def(py::init<>())
            .def(py::init<T, T, T, T>())
            .def(py::init<const Range<T>&, const Range<T>&>())

            .def_static("Open", &CLS::Open)
            .def_static("Empty", &CLS::Empty)
            .def_static("Containing", &CLS::Containing)

            // .def(py::self + T())
            .def(py::self - CLS())
            .def(py::self * float())
            .def(py::self += CLS())

            .def("Area", &CLS::Area)
            // .def("AbsSize", &CLS::AbsSize)
            // .def("Mid", &CLS::Mid)
            .def("Scale", &CLS::Scale)
            .def("Insert", (void (CLS::*) (T, T)) &CLS::Insert)
            .def("Insert", (void (CLS::*) (CLS)) &CLS::Insert)
            .def("Clamp", (void (CLS::*) (T, T, T, T)) &CLS::Clamp)
            .def("Clamp", (void (CLS::*) (const CLS&)) &CLS::Clamp)
            .def("Clear", &CLS::Clear)
            .def("Contains", &CLS::Contains)
            .def("ContainsWeak", &CLS::ContainsWeak)

            // .def_readwrite("x", CLS::x)
            // .def_readwrite("y", CLS::y)
        ;

        m.def("Round", (XYRangei (*) (const XYRange<T>&)) &Round);

    }



}



void declareRange(py::module & m) {

    templatedRange<int>(m, "Int");
    templatedRange<double>(m, "Float");

    templatedXYRange<int>(m, "Int");
    templatedXYRange<double>(m, "Float");



}

}