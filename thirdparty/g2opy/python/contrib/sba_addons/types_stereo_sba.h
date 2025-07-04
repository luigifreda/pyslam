#include <pybind11/pybind11.h>

#include "contrib/sba_addons/types_stereo_sba.hpp"


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareTypesStereoSBA(py::module & m) {

    py::class_<EdgeProjectP2MCRight, BaseBinaryEdge<2, Vector2d, VertexSBAPointXYZ, VertexCam>>(m, "EdgeProjectP2MCRight")
        .def(py::init<>())
        .def("compute_error", &EdgeProjectP2MCRight::computeError)    // () -> void
        .def("linearize_oplus", &EdgeProjectP2MCRight::linearizeOplus)
    ;

}

}