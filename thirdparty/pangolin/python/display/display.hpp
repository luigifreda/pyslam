#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/platform.h>
#include <pangolin/display/display.h>
//#include <pangolin/display/user_app.h>
#include <pangolin/display/view.h>
#include <pangolin/display/viewport.h>
#include <pangolin/handler/handler_enums.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareDisplay(py::module & m) {

    //m.def_readonly("PARAM_DISPLAYNAME", &PARAM_DISPLAYNAME)
    //m.def_readonly("PARAM_DOUBLEBUFFER", &PARAM_DOUBLEBUFFER)
    //m.def_readonly("PARAM_SAMPLE_BUFFERS", &PARAM_SAMPLE_BUFFERS)
    //m.def_readonly("PARAM_SAMPLES", &PARAM_SAMPLES)
    //m.def_readonly("PARAM_HIGHRES", &PARAM_HIGHRES)

    m.def("BindToContext", &BindToContext, "name"_a,
        py::return_value_policy::reference,
        "Initialise OpenGL window (determined by platform) and bind context.");

    m.def("CreateWindowAndBind", &CreateWindowAndBind, "window_title"_a, "w"_a = 640, "h"_a = 480, "params"_a = Params(), 
        py::return_value_policy::reference,
        "Initialise OpenGL window (determined by platform) and bind context.");

    m.def("GetBoundWindow", &GetBoundWindow, "Return pointer to current Pangolin Window context, or nullptr if none bound.");

    m.def("DestroyWindow", &DestroyWindow, "window_title"_a);

    m.def("FinishFrame", &FinishFrame, "Perform any post rendering, event processing and frame swapping.");

    m.def("Quit", &Quit, "Request that the window close.");

    m.def("QuitAll", &QuitAll, "Request that all windows close.");

    m.def("ShouldQuit", &ShouldQuit, "Returns true if user has requested to close OpenGL window.");
    
    m.def("HadInput", &HadInput, "Returns true if user has interacted with the window since this was last called.");

    m.def("HasResized", &HasResized, "Returns true if user has resized the window.");

    m.def("RenderViews", &RenderViews, "Renders any views with default draw methods.");

    m.def("PostRender", &PostRender, "Perform any post render events, such as screen recording.");

    m.def("RegisterKeyPressCallback", &RegisterKeyPressCallback, "key"_a, "func"_a,
        "Request to be notified via functor when key is pressed.");    // py::keep_alive<1, 2>()

    m.def("SaveWindowOnRender", &SaveWindowOnRender, "filename_prefix"_a, "Save window contents to image.");

    m.def("SaveFramebuffer", &SaveFramebuffer, "prefix"_a, "v"_a);  //v : Viewport


    // namespace process
    m.def_submodule("process")
    
        .def("Keyboard", &process::Keyboard, "key"_a, "x"_a, "y"_a)

        .def("KeyboardUp", &process::KeyboardUp, "key"_a, "x"_a, "y"_a)

        .def("SpecialFunc", &process::SpecialFunc, "key"_a, "x"_a, "y"_a)

        .def("SpecialFuncUp", &process::SpecialFuncUp, "key"_a, "x"_a, "y"_a)

        .def("Resize", &process::Resize, "width"_a, "height"_a, "Tell pangolin base window size has changed")

        .def("Display", &process::Display)

        .def("Mouse", &process::Mouse, "button"_a, "state"_a, "x"_a, "y"_a)

        .def("MouseMotion", &process::MouseMotion, "x"_a, "y"_a)

        .def("PassiveMouseMotion", &process::PassiveMouseMotion, "x"_a, "y"_a)

        .def("Scroll", &process::Scroll, "x"_a, "y"_a)

        .def("Zoom", &process::Zoom, "m"_a)

        .def("Rotate", &process::Rotate, "r"_a)

        .def("SubpixMotion", &process::SubpixMotion, "x"_a, "y"_a, "pressure"_a, "rotation"_a, "tiltx"_a, "tilty"_a)

        .def("SpecialInput", &process::SpecialInput, "inType"_a, "x"_a, "y"_a, "p1"_a, "p2"_a, "p3"_a, "p4"_a)
    ;
        

    m.def("DisplayBase", &DisplayBase, py::return_value_policy::reference,
        "Retrieve 'base' display, corresponding to entire window.");

    m.def("Display", &Display, "name"_a, py::return_value_policy::reference,
        "Create or retrieve named display managed by pangolin (automatically deleted).");

    m.def("CreateDisplay", &CreateDisplay, py::return_value_policy::reference,
        "Create unnamed display managed by pangolin (automatically deleted).");
        

    m.def("ToggleFullscreen", &ToggleFullscreen, "Switch between windowed and fullscreen mode.");

    m.def("SetFullscreen", &SetFullscreen, "fullscreen"_a = true,  "Switch windows/fullscreenmode = fullscreen.");

    m.def("ToggleConsole", &ToggleConsole, "Toggle display of Pangolin console");

    // not implemented
    //m.def("LaunchUserApp", &LaunchUserApp, "app"_a, "Launch users derived UserApp, controlling OpenGL event loop.");
    //py::class_<ToggleViewFunctor, std::shared_ptr<ToggleViewFunctor>> cls(m, "ToggleViewFunctor");


}

}  // namespace pangolin::