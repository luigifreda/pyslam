#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/display/window.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareWindow(py::module & m) {

    py::class_<GlContextInterface, std::shared_ptr<GlContextInterface>>(m, "GlContextInterface");
    py::class_<WindowInterface, std::shared_ptr<WindowInterface>>(m, "WindowInterface");

}

}