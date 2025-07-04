#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <pangolin/display/viewport.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareViewport(py::module & m) {

    py::class_<Viewport, std::shared_ptr<Viewport>> cls(m, "Viewport");

    cls.def(py::init<>());
    cls.def(py::init<GLint, GLint, GLint, GLint>());

    cls.def("Activate", &Viewport::Activate);
    cls.def("ActivateIdentity", &Viewport::ActivateIdentity);
    cls.def("ActivatePixelOrthographic", &Viewport::ActivatePixelOrthographic);

    cls.def("Scissor", &Viewport::Scissor);
    cls.def("ActivateAndScissor", &Viewport::ActivateAndScissor);

    cls.def("Contains", &Viewport::Contains);

    cls.def("Inset", (Viewport (Viewport::*) (int) const) &Viewport::Inset);
    cls.def("Inset", (Viewport (Viewport::*) (int, int) const) &Viewport::Inset);
    cls.def("Intersect", &Viewport::Intersect);

    cls.def_static("DisableScissor", &Viewport::DisableScissor);

    cls.def("r", &Viewport::r);
    cls.def("t", &Viewport::t);
    cls.def("aspect", &Viewport::aspect);

    cls.def_readwrite("l", &Viewport::l);
    cls.def_readwrite("b", &Viewport::b);
    cls.def_readwrite("w", &Viewport::w);
    cls.def_readwrite("h", &Viewport::h);

}

}  // namespace pangolin::