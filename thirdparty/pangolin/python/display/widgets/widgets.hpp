#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/display/widgets/widgets.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {


namespace {
    template<typename T>
    void templatedWidget(py::module & m, const std::string & suffix) {
        py::class_<Widget<T>, std::shared_ptr<Widget<T>>, View, Handler, Var<T>>(m, ("Widget" + suffix).c_str())
            .def(py::init<std::string, VarValueGeneric&>(), "title"_a, "tv"_a)
            .def_readwrite("title", &Widget<T>::title);
    }
}



void declareWidgets(py::module & m) {

    m.def("CreatePanel", &CreatePanel, py::return_value_policy::reference);


    py::class_<Panel, std::shared_ptr<Panel>, View> (m, "Panel")
        .def(py::init<>())
        .def(py::init<const std::string&>(), "auto_register_var_prefix"_a)
        .def("Render", &Panel::Render)
        .def("ResizeChildren", &Panel::ResizeChildren)
        .def_static("AddVariable", &Panel::AddVariable);


    templatedWidget<bool>(m, "Bool");
    templatedWidget<int>(m, "Int");
    templatedWidget<double>(m, "Float");
    templatedWidget<std::string>(m, "String");
    templatedWidget<std::function<void(void)>>(m, "Func");


    py::class_<Button, std::shared_ptr<Button>, Widget<bool>> (m, "Button")
        .def(py::init<std::string, VarValueGeneric&>(), "title"_a, "tv"_a)
        .def("Mouse", &Button::Mouse)
        .def("Render", &Button::Render)
        .def("ResizeChildren", &Button::ResizeChildren)
        // GLfloat raster[2]
        .def_readwrite("gltext", &Button::gltext)
        .def_readwrite("down", &Button::down);


    py::class_<FunctionButton, std::shared_ptr<FunctionButton>, Widget<std::function<void(void)>>> (m, "FunctionButton")
        .def(py::init<std::string, VarValueGeneric&>(), "title"_a, "tv"_a)
        .def("Mouse", &FunctionButton::Mouse)
        .def("Render", &FunctionButton::Render)
        .def("ResizeChildren", &FunctionButton::ResizeChildren)
        // GLfloat raster[2]
        .def_readwrite("gltext", &FunctionButton::gltext)
        .def_readwrite("down", &FunctionButton::down);
        

    py::class_<Checkbox, std::shared_ptr<Checkbox>, Widget<bool>> (m, "Checkbox")
        .def(py::init<std::string, VarValueGeneric&>(), "title"_a, "tv"_a)
        .def("Mouse", &Checkbox::Mouse)
        .def("Render", &Checkbox::Render)
        .def("ResizeChildren", &Checkbox::ResizeChildren)
        // GLfloat raster[2]
        .def_readwrite("gltext", &Checkbox::gltext)
        .def_readwrite("vcb", &Checkbox::vcb);


    py::class_<Slider, std::shared_ptr<Slider>, Widget<double>> (m, "Slider")
        .def(py::init<std::string, VarValueGeneric&>())
        .def("Mouse", &Slider::Mouse, "view"_a, "button"_a, "x"_a, "y"_a, "pressed"_a, "mouse_state"_a)
        .def("MouseMotion", &Slider::MouseMotion, "view"_a, "x"_a, "y"_a, "mouse_state"_a)
        .def("Keyboard", &Slider::Keyboard, "view"_a, "key"_a, "x"_a, "y"_a, "pressed"_a)
        .def("Render", &Slider::Render)
        .def("ResizeChildren", &Slider::ResizeChildren)
        // GLfloat raster[2]
        .def_readwrite("gltext", &Slider::gltext)
        .def_readwrite("lock_bounds", &Slider::lock_bounds)
        .def_readwrite("logscale", &Slider::logscale)
        .def_readwrite("is_integral_type", &Slider::is_integral_type);


    py::class_<TextInput, std::shared_ptr<TextInput>, Widget<std::string>> (m, "TextInput")
        .def(py::init<std::string, VarValueGeneric&>())
        .def("Mouse", &TextInput::Mouse, "view"_a, "button"_a, "x"_a, "y"_a, "pressed"_a, "mouse_state"_a)
        .def("MouseMotion", &TextInput::MouseMotion, "view"_a, "x"_a, "y"_a, "mouse_state"_a)
        .def("Keyboard", &TextInput::Keyboard, "view"_a, "key"_a, "x"_a, "y"_a, "pressed"_a)
        .def("Render", &TextInput::Render)
        .def("ResizeChildren", &TextInput::ResizeChildren)
        .def_readwrite("edit", &TextInput::edit)
        .def_readwrite("gledit", &TextInput::gledit)
        // GLfloat raster[2]
        // int sel[2]
        .def_readwrite("gltext", &TextInput::gltext)
        .def_readwrite("do_edit", &TextInput::do_edit);

}

}  // namespace pangolin::