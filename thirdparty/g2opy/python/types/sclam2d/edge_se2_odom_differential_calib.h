#include <pybind11/pybind11.h>

#include <g2o/types/sclam2d/edge_se2_odom_differential_calib.h>



namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareEdgeSE2OdomDifferentialCalib(py::module& m) {

    py::class_<EdgeSE2OdomDifferentialCalib, BaseMultiEdge<3, VelocityMeasurement>>(m, "EdgeSE2OdomDifferentialCalib")
        .def(py::init<>())
        .def("compute_error", &EdgeSE2OdomDifferentialCalib::computeError)
    ;

    // class G2O_TYPES_SCLAM2D_API EdgeSE2OdomDifferentialCalibDrawAction: public DrawAction

}

}