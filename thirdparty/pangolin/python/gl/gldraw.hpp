#include <pybind11/pybind11.h>

#include <pangolin/gl/gldraw.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace pangolin {

void declareGLDraw(py::module & m) {

    m.def("glDrawColouredCube", &glDrawColouredCube,
        "axis_min"_a = -0.5f, "axis_max"_a = 0.5f);

    //py::class_<Params, std::shared_ptr<Params>> cls(m, "Params");

    //cls.def(py::init<>());

}

}  // namespace pangolin::