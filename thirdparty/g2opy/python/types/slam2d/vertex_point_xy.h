#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <g2o/types/slam2d/vertex_point_xy.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareVertexPointXY(py::module & m) {

    py::class_<VertexPointXY, BaseVertex<2, Vector2D>>(m, "VertexPointXY")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexPointXY::setToOriginImpl)
        .def("oplus_impl", &VertexPointXY::oplusImpl)
        .def("set_estimate_data_impl", &VertexPointXY::setEstimateDataImpl)
        .def("get_estimate_data", &VertexPointXY::getEstimateData)
        .def("estimate_dimension", &VertexPointXY::estimateDimension)
        .def("set_minimal_estimate_data_impl", &VertexPointXY::setMinimalEstimateDataImpl)
        .def("get_minimal_estimate_data", &VertexPointXY::getMinimalEstimateData)
        .def("minimal_estimate_dimension", &VertexPointXY::minimalEstimateDimension)
    ;


    //class G2O_TYPES_SLAM2D_API VertexPointXYWriteGnuplotAction: public WriteGnuplotAction
    //class G2O_TYPES_SLAM2D_API VertexPointXYDrawAction: public DrawAction

}

}  // end namespace g2o