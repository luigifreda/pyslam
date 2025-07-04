#pragma once

#include <pybind11/pybind11.h>

#include <g2o/core/base_binary_edge.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

template <int D, typename E, typename VertexXi, typename VertexXj>
void templatedBaseBinaryEdge(py::module & m, const std::string & suffix) {
    using CLS = BaseBinaryEdge<D, E, VertexXi, VertexXj>;   

    py::class_<CLS, BaseEdge<D, E>>(m, ("BaseBinaryEdge" + suffix).c_str())
        //.def(py::init<>())    // lead to "error: invalid new-expression of abstract class type ..."
        .def("create_from", &CLS::createFrom)   // -> OptimizableGraph::Vertex*
        .def("create_to", &CLS::createTo)   // -> OptimizableGraph::Vertex*
        .def("create_vertex", &CLS::createVertex, "i"_a)   // -> OptimizableGraph::Vertex*

        .def("resize", &CLS::resize)
        .def("all_vertices_fixed", &CLS::allVerticesFixed)
        .def("linearize_oplus", (void (CLS::*) (JacobianWorkspace&)) &CLS::linearizeOplus)
        .def("linearize_oplus", (void (CLS::*) ()) &CLS::linearizeOplus)
        .def("jacobian_oplus_xi", &CLS::jacobianOplusXi)   // -> JacobianXiOplusType&
        .def("jacobian_oplus_xj", &CLS::jacobianOplusXj)   // -> JacobianXjOplusType&
        .def("construct_quadratic_form", &CLS::constructQuadraticForm)
        .def("map_hessian_memory", &CLS::mapHessianMemory,
                "d"_a, "i"_a, "j"_a, "row_mayor"_a)

    ;
}




void declareBaseBinaryEdge(py::module & m) {
    // common types
}


}  // end namespace g2o