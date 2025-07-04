#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/utils/params.h>


namespace py = pybind11;
using namespace pybind11::literals;

namespace pangolin {

void declareParams(py::module & m) {

    py::class_<Params, std::shared_ptr<Params>> cls(m, "Params");

    cls.def(py::init<>());
    cls.def(py::init<std::initializer_list<std::pair<std::string,std::string>>>(), "l"_a);
    cls.def("Contains", &Params::Contains, "key"_a);
    //cls.def("Get", &Params::Get<??>, "key"_a, "default_val"_a);
    //cls.def("Set", &Params::Set<??>, "key"_a, "default_val"_a);
    cls.def_readwrite("params", &Params::params);

}

}  // namespace pangolin::