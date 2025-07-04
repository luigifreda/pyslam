#include <pybind11/pybind11.h>

#include <g2o/types/slam2d/types_slam2d.h>
#include <g2o/core/factory.h>
#include <g2o/stuff/macros.h>

#include "parameter_se2_offset.h"

#include "se2.h"
#include "vertex_point_xy.h"
#include "vertex_se2.h"

#include "edge_pointxy.h"
#include "edge_se2.h"
#include "edge_se2_pointxy.h"




namespace g2o {

G2O_REGISTER_TYPE_GROUP(slam2d);

G2O_REGISTER_TYPE(VERTEX_SE2, VertexSE2);
G2O_REGISTER_TYPE(VERTEX_XY, VertexPointXY);
G2O_REGISTER_TYPE(PARAMS_SE2OFFSET, ParameterSE2Offset);
G2O_REGISTER_TYPE(CACHE_SE2_OFFSET, CacheSE2Offset);
G2O_REGISTER_TYPE(EDGE_PRIOR_SE2, EdgeSE2Prior);
G2O_REGISTER_TYPE(EDGE_PRIOR_SE2_XY, EdgeSE2XYPrior);
G2O_REGISTER_TYPE(EDGE_SE2, EdgeSE2);
G2O_REGISTER_TYPE(EDGE_SE2_XY, EdgeSE2PointXY);
G2O_REGISTER_TYPE(EDGE_BEARING_SE2_XY, EdgeSE2PointXYBearing);
G2O_REGISTER_TYPE(EDGE_SE2_XY_CALIB, EdgeSE2PointXYCalib);
G2O_REGISTER_TYPE(EDGE_SE2_OFFSET, EdgeSE2Offset);
G2O_REGISTER_TYPE(EDGE_SE2_POINTXY_OFFSET, EdgeSE2PointXYOffset);
G2O_REGISTER_TYPE(EDGE_POINTXY, EdgePointXY);
G2O_REGISTER_TYPE(EDGE_SE2_TWOPOINTSXY, EdgeSE2TwoPointsXY);
G2O_REGISTER_TYPE(EDGE_SE2_LOTSOFXY, EdgeSE2LotsOfXY);


    
void declareTypesSlam2d(py::module & m) {

    declareParameterSE2Offset(m);

    declareSE2(m);
    declareVertexPointXY(m);
    declareVertexSE2(m);

    declareEdgePointXY(m);
    declareEdgeSE2(m);
    declareEdgeSE2PointXY(m);

}

}