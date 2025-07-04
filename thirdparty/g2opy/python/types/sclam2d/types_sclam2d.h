#include <pybind11/pybind11.h>

#include <g2o/types/sclam2d/types_sclam2d.h>
#include <g2o/core/factory.h>
#include <g2o/stuff/macros.h>

#include "odometry_measurement.h"
#include "vertex_odom_differential_params.h"
#include "edge_se2_sensor_calib.h"
#include "edge_se2_odom_differential_calib.h"




namespace g2o {

G2O_USE_TYPE_GROUP(slam2d);

G2O_REGISTER_TYPE_GROUP(sclam);
G2O_REGISTER_TYPE(VERTEX_ODOM_DIFFERENTIAL, VertexOdomDifferentialParams);
G2O_REGISTER_TYPE(EDGE_SE2_CALIB, EdgeSE2SensorCalib);
G2O_REGISTER_TYPE(EDGE_SE2_ODOM_DIFFERENTIAL_CALIB, EdgeSE2OdomDifferentialCalib);


    
void declareTypesSclam2d(py::module & m) {

    declareOdometryMeasurement(m);
    declareVertexOdomDifferentialParams(m);
    declareEdgeSE2SensorCalib(m);
    declareEdgeSE2OdomDifferentialCalib(m);
    

}

}