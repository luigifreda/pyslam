#include <pybind11/pybind11.h>

#include <g2o/types/sclam2d/edge_se2_sensor_calib.h>



namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareEdgeSE2SensorCalib(py::module& m) {

    py::class_<EdgeSE2SensorCalib, BaseMultiEdge<3, SE2>>(m, "EdgeSE2SensorCalib")
        .def(py::init<>())
        .def("compute_error", &EdgeSE2SensorCalib::computeError)
        .def("set_measurement", &EdgeSE2SensorCalib::setMeasurement)
        .def("initial_estimate_possible", &EdgeSE2SensorCalib::initialEstimatePossible)
        .def("initial_estimate", &EdgeSE2SensorCalib::initialEstimate)
    ;

    // class EdgeSE2SensorCalibDrawAction: public DrawAction

}

}