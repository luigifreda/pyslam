#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/var/varvalue.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {


namespace {
    
    template<typename T>
    void templatedVarValue(py::module & m, std::string const & suffix) {

        typedef typename std::remove_reference<T>::type VarT;
        using Class = VarValue<T>;

        py::class_<Class, VarValueT<typename std::remove_reference<T>::type>>(m, ("VarValue" + suffix).c_str())
            .def(py::init<>())
            .def(py::init<const T&>(), "value"_a)
            .def(py::init<const T&, const VarT&>(), "value"_a, "default_value"_a)
            // destructor: delete str_ptr

            .def("TypeId", &Class::TypeId)
            .def("Reset", &Class::Reset)
            .def("Meta", &Class::Meta)
            .def("Get", (const VarT& (Class::*) () const) &Class::Get, py::return_value_policy::reference)
            .def("Get", (VarT& (Class::*) ()) &Class::Get)
            .def("Set", &Class::Set);
    }
}




void declareVarValue(py::module & m) {
    //templateVarValue<py::object>(m, "Bool");
    templatedVarValue<bool>(m, "Bool");
    templatedVarValue<int>(m, "Int");
    templatedVarValue<double>(m, "Float");
    templatedVarValue<std::string>(m, "String");
    templatedVarValue<std::function<void(void)>>(m, "Func");    

}

}