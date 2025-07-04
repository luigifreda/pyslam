#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <g2o/types/sclam2d/vertex_odom_differential_params.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareVertexOdomDifferentialParams(py::module & m) {

    py::class_<VertexOdomDifferentialParams, BaseVertex <3, Vector3D>>(m, "VertexOdomDifferentialParams")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexOdomDifferentialParams::setToOriginImpl)
        .def("oplus_impl", &VertexOdomDifferentialParams::oplusImpl)
    ;

}

}