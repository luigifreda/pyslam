#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/display/user_app.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

namespace {
    class PyUserApp : public UserApp {
    public:
        /* Inherit the constructors */
        using UserApp::UserApp;
    
        /* Trampoline (need one for each virtual function) */
        void Init() override {
            PYBIND11_OVERLOAD_PURE(
                void, /* Return type */
                UserApp,      /* Parent class */
                Init,          /* Name of function in C++ (must match Python name) */
                      /* Argument(s) */
            );
        }
        void Render() override { PYBIND11_OVERLOAD_PURE(void, UserApp, Render,) }
    };
}


void declareUserApp(py::module & m) {

    py::class_<UserApp, PyUserApp>(m, "UserApp")
        .def(py::init<>())
        .def("Init", &UserApp::Init)
        .def("Render", &UserApp::Render);

}

}  // namespace pangolin::