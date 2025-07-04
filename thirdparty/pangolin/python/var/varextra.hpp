#include <pybind11/pybind11.h>

#include <pangolin/var/varextra.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {


namespace {
    template<typename T>
    void templatedSetVarFunctor(py::module & m, const std::string & suffix) {
        using Class = SetVarFunctor<T>;
        py::class_<Class, std::shared_ptr<Class>>(m, ("SetVarFunctor" + suffix).c_str())
            .def(py::init<const std::string&, T>(), "name"_a,  "val"_a)
            .def("__call__", &Class::operator())
            .def_readwrite("varName", &Class::varName)
            .def_readwrite("setVal", &Class::setVal);
    }
}



void declareVarExtra(py::module & m) {

    m.def("ParseVarsFile", &ParseVarsFile, "filename"_a);

    m.def("LoadJsonFile", &LoadJsonFile, "filename"_a, "prefix"_a = "");

    m.def("SaveJsonFile", &SaveJsonFile, "filename"_a, "prefix"_a = "");

    m.def("ProcessHistoricCallbacks", &ProcessHistoricCallbacks, "callback"_a, "data"_a, "filter"_a = "");

    m.def("RegisterNewVarCallback", &RegisterNewVarCallback, "callback"_a, "data"_a, "filter"_a = "");

    m.def("RegisterGuiVarChangedCallback", &RegisterGuiVarChangedCallback, "callback"_a, "data"_a, "filter"_a = "");

    templatedSetVarFunctor<bool>(m, "Bool");
    templatedSetVarFunctor<int>(m, "Int");
    templatedSetVarFunctor<double>(m, "Float");
    templatedSetVarFunctor<std::string>(m, "String");
    templatedSetVarFunctor<std::function<void(void)>>(m, "Func");

    py::class_<ToggleVarFunctor, std::shared_ptr<ToggleVarFunctor>>(m, "ToggleVarFunctor")
        .def(py::init<const std::string&>(), "name"_a)
        .def("__call__", &ToggleVarFunctor::operator())
        .def_readwrite("varName", &ToggleVarFunctor::varName);

    m.def("Pushed", (bool (*) (Var<bool>&)) &Pushed, "button"_a);
    m.def("Pushed", (bool (*) (bool&)) &Pushed, "button");

}

}  // namespace pangolin::