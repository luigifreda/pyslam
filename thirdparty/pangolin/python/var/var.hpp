#include <pybind11/pybind11.h>

#include <pangolin/var/var.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

namespace {

    template<typename T>
    void templatedVar(py::module & m, std::string const & suffix) {
        using Class = Var<T>;

        py::class_<Class, std::shared_ptr<Class>> cls(m, ("Var" + suffix).c_str());

            cls.def_static("Attach", [](const std::string& name, T& variable, bool toggle) {
                    return Class::Attach(name, variable, toggle);
                }, "name"_a, "variable"_a, "toggle"_a = false, py::return_value_policy::reference_internal);
            cls.def_static("Attach", [](const std::string& name, T& variable, double min, double max, bool logscale) {
                    return Class::Attach(name, variable, min, max, logscale);
                }, "name"_a, "variable"_a, "min"_a, "max"_a, "logscale"_a = false, py::return_value_policy::reference_internal);

            cls.def(py::init<VarValueGeneric&>(), "v"_a);
            cls.def(py::init<const std::string&>(), "name"_a);
            cls.def(py::init<const std::string&, const T&, bool>(), "name"_a, "value"_a, "toggle"_a = false, py::keep_alive<0, 3>());
            cls.def(py::init<const std::string&, const T&, double, double, bool>(),
                "name"_a, "value"_a, "min"_a, "max"_a, "logscale"_a = false);
            
            cls.def("Reset", &Class::Reset);
            cls.def("Get", &Class::Get, py::return_value_policy::reference);
            cls.def("SetVal", (void (Class::*) (const T&)) &Class::operator=, "val"_a);
            cls.def("SetVal", (void (Class::*) (const Var<T>&)) &Class::operator=, "v"_a);
            cls.def("Meta", &Class::Meta, py::return_value_policy::reference_internal);
            cls.def("GuiChanged", &Class::GuiChanged);
            cls.def("Ref", &Class::Ref, py::return_value_policy::reference_internal);

        //m.def(("_Var" + suffix).c_str(), [](const std::string& name, T value, bool toggle) {return Class(name, value, toggle);});
        //m.def(("_Var" + suffix).c_str(), [](const std::string& name, T value, double min, double max, bool logscale) {
        //        return Class(name, value, min, max, logscale);
        //    });

    }


    template<typename T>
    void templatedVarNum(py::module & m, std::string const & suffix) {
        using Class = Var<T>;

        py::class_<Class, std::shared_ptr<Class>> cls(m, ("Var" + suffix).c_str());

            cls.def_static("Attach", [](const std::string& name, T& variable, bool toggle) {
                    return Class::Attach(name, variable, toggle);
                }, "name"_a, "variable"_a, "toggle"_a = false, py::return_value_policy::reference_internal);
            cls.def_static("Attach", [](const std::string& name, T& variable, double min, double max, bool logscale) {
                    return Class::Attach(name, variable, min, max, logscale);
                }, "name"_a, "variable"_a, "min"_a, "max"_a, "logscale"_a = false, py::return_value_policy::reference_internal);

            cls.def(py::init<VarValueGeneric&>(), "v"_a);
            cls.def(py::init<const std::string&>(), "name"_a);
            cls.def(py::init<const std::string&, const T&, bool>(), "name"_a, "value"_a, "toggle"_a = false);
            cls.def(py::init<const std::string&, const T&, double, double, bool>(),
                "name"_a, "value"_a, "min"_a, "max"_a, "logscale"_a = false);
            
            cls.def("Reset", &Class::Reset);
            cls.def("Get", &Class::Get, py::return_value_policy::reference);
            cls.def("SetVal", (void (Class::*) (const T&)) &Class::operator=, "val"_a);
            cls.def("SetVal", (void (Class::*) (const Var<T>&)) &Class::operator=, "v"_a);
            cls.def("Meta", &Class::Meta, py::return_value_policy::reference_internal);
            cls.def("GuiChanged", &Class::GuiChanged);
            cls.def("Ref", &Class::Ref, py::return_value_policy::reference_internal);

            cls.def("__int__", [](Var<T>& v){return (int)v.Get();});
            cls.def("__float__", [](Var<T>& v){return (double)v.Get();});


    }

}


    
void declareVar(py::module & m) {
    templatedVar<bool>(m, "Bool");
    templatedVar<std::string>(m, "String");
    templatedVar<std::function<void(void)>>(m, "Func");

    templatedVarNum<int>(m, "Int");
    templatedVarNum<double>(m, "Float");
}

}