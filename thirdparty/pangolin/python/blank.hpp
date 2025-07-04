#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin//.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareParams(py::module & m) {

    py::class_<Params, std::shared_ptr<Params>> cls(m, "Params");

    cls.def(py::init<>());

}

}  // namespace pangolin::