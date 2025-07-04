#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <g2o/types/sim3/types_seven_dof_expmap.h>

#include "python/core/base_binary_edge.h"



namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareTypesSevenDofExpmap(py::module & m) {

    py::class_<VertexSim3Expmap, BaseVertex<7, Sim3>>(m, "VertexSim3Expmap")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexSim3Expmap::setToOriginImpl)
        .def("oplus_impl", &VertexSim3Expmap::oplusImpl)
        .def("cam_map1", &VertexSim3Expmap::cam_map1)
        .def("cam_map2", &VertexSim3Expmap::cam_map2)
        .def_readwrite("_principle_point1", &VertexSim3Expmap::_principle_point1)
        .def_readwrite("_principle_point2", &VertexSim3Expmap::_principle_point2)
        .def_readwrite("_focal_length1", &VertexSim3Expmap::_focal_length1)
        .def_readwrite("_focal_length2", &VertexSim3Expmap::_focal_length2)
        .def_readwrite("_fix_scale", &VertexSim3Expmap::_fix_scale)
    ;



    templatedBaseBinaryEdge<7, Sim3, VertexSim3Expmap, VertexSim3Expmap>(m, 
        "_7_Sim3_VertexSim3Expmap_VertexSim3Expmap");

    py::class_<EdgeSim3, BaseBinaryEdge<7, Sim3, VertexSim3Expmap, VertexSim3Expmap>>(m, "EdgeSim3")
        .def(py::init<>())
        .def("compute_error", &EdgeSim3::computeError)
        .def("initial_estimate_possible", &EdgeSim3::initialEstimatePossible)
        .def("initial_estimate", &EdgeSim3::initialEstimate)
    ;



    templatedBaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSim3Expmap>(m, 
        "_2_Vector2D_VertexSBAPointXYZ_VertexSim3Expmap");

    py::class_<EdgeSim3ProjectXYZ, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSim3Expmap>>(m, 
        "EdgeSim3ProjectXYZ")
        .def(py::init<>())
        .def("compute_error", &EdgeSim3ProjectXYZ::computeError)
        .def("is_depth_positive", &EdgeSim3ProjectXYZ::isDepthPositive)
    ;



    py::class_<EdgeInverseSim3ProjectXYZ, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSim3Expmap>>(m, 
        "EdgeInverseSim3ProjectXYZ")
        .def(py::init<>())
        .def("compute_error", &EdgeInverseSim3ProjectXYZ::computeError)
        .def("is_depth_positive", &EdgeInverseSim3ProjectXYZ::isDepthPositive)
    ;

}

}