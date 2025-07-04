#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <g2o/types/slam2d/vertex_se2.h>


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareVertexSE2(py::module & m) {

    py::class_<VertexSE2, BaseVertex<3, SE2>>(m, "VertexSE2")
        .def(py::init<>())
        .def("set_to_origin_impl", &VertexSE2::setToOriginImpl)
        .def("oplus_impl", &VertexSE2::oplusImpl)
        .def("set_estimate_data_impl", &VertexSE2::setEstimateDataImpl)
        .def("get_estimate_data", &VertexSE2::getEstimateData)
        .def("estimate_dimension", &VertexSE2::estimateDimension)
        .def("set_minimal_estimate_data_impl", &VertexSE2::setMinimalEstimateDataImpl)
        .def("get_minimal_estimate_data", &VertexSE2::getMinimalEstimateData)
        .def("minimal_estimate_dimension", &VertexSE2::minimalEstimateDimension)
    ;


    //class G2O_TYPES_SLAM2D_API VertexSE2WriteGnuplotAction: public WriteGnuplotAction
    //class G2O_TYPES_SLAM2D_API VertexSE2DrawAction: public DrawAction

}

}  // end namespace g2o