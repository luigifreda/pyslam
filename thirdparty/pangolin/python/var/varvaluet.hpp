#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/functional.h>

#include <pangolin/var/varvaluet.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {


namespace {

    // class PyVarValueT

    template<typename T>
    void templatedVarValueT(py::module & m, std::string const & suffix) {
        py::class_<VarValueT<T>, VarValueGeneric>(m, ("VarValueT" + suffix).c_str());
    }

}


void declareVarValueT(py::module & m) {

    templatedVarValueT<bool>(m, "Bool");
    templatedVarValueT<int>(m, "Int");
    templatedVarValueT<double>(m, "Float");
    templatedVarValueT<std::string>(m, "String");
    templatedVarValueT<std::function<void(void)>>(m, "Func");  
    //templatedVarValueT<py::object>(m, "Custom");  // T should define  "operator>>(std::istream&, T&)" and 
                                                    //                  "operator<<(std::istream&, T&)" operators
}

}