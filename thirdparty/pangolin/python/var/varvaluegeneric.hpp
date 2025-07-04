#include <pybind11/pybind11.h>

#include <pangolin/var/varvaluegeneric.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {


namespace {

    class PyVarValueGeneric : public VarValueGeneric {
    public:
        using VarValueGeneric::VarValueGeneric;

        const char* TypeId() const override { PYBIND11_OVERLOAD_PURE(const char*, VarValueGeneric, TypeId,); }
        void Reset() override { PYBIND11_OVERLOAD_PURE(void, VarValueGeneric, Reset,); }
        VarMeta& Meta() override { PYBIND11_OVERLOAD_PURE( VarMeta&, VarValueGeneric, Meta,); }
    };
}



void declareVarValueGeneric(py::module & m) {

    py::class_<VarMeta>(m, "VarMeta")
        .def(py::init<>())
        .def_readwrite("full_name", &VarMeta::full_name)
        .def_readwrite("friendly", &VarMeta::friendly)
        .def_readwrite("increment", &VarMeta::increment)
        .def_readwrite("flags", &VarMeta::flags)
        .def_readwrite("gui_changed", &VarMeta::gui_changed)
        .def_readwrite("logscale", &VarMeta::logscale)
        .def_readwrite("generic", &VarMeta::generic)
        // double range[2];
    ;

    // abstract class
    py::class_<VarValueGeneric, PyVarValueGeneric>(m, "VarValueGeneric")
        .def(py::init<>())
        .def("TypeId", &VarValueGeneric::TypeId)
        .def("Reset", &VarValueGeneric::Reset)
        .def("Meta", &VarValueGeneric::Meta)
    ;

}

}