#include <pybind11/pybind11.h>

#include <g2o/types/slam3d/types_slam3d.h>
#include <g2o/core/factory.h>
#include <g2o/stuff/macros.h>

#include "parameter.h"
#include "se3quat.h"
#include "vertex_se3.h"
#include "vertex_pointxyz.h"

#include "edge_pointxyz.h"
#include "edge_se3.h"
#include "edge_se3_pointxyz.h"



namespace g2o {


// register types
// slam3d
G2O_REGISTER_TYPE_GROUP(slam3d);
G2O_REGISTER_TYPE(VERTEX_SE3:QUAT, VertexSE3);
G2O_REGISTER_TYPE(EDGE_SE3:QUAT, EdgeSE3);
G2O_REGISTER_TYPE(VERTEX_TRACKXYZ, VertexPointXYZ);

G2O_REGISTER_TYPE(PARAMS_SE3OFFSET, ParameterSE3Offset);
G2O_REGISTER_TYPE(EDGE_SE3_TRACKXYZ, EdgeSE3PointXYZ);
G2O_REGISTER_TYPE(EDGE_SE3_PRIOR, EdgeSE3Prior);
G2O_REGISTER_TYPE(CACHE_SE3_OFFSET, CacheSE3Offset);
G2O_REGISTER_TYPE(EDGE_SE3_OFFSET, EdgeSE3Offset);

G2O_REGISTER_TYPE(PARAMS_CAMERACALIB, ParameterCamera);
G2O_REGISTER_TYPE(PARAMS_STEREOCAMERACALIB, ParameterStereoCamera);
G2O_REGISTER_TYPE(CACHE_CAMERA, CacheCamera);
G2O_REGISTER_TYPE(EDGE_PROJECT_DISPARITY, EdgeSE3PointXYZDisparity);
G2O_REGISTER_TYPE(EDGE_PROJECT_DEPTH, EdgeSE3PointXYZDepth);

G2O_REGISTER_TYPE(EDGE_POINTXYZ, EdgePointXYZ);

G2O_REGISTER_TYPE(EDGE_SE3_LOTSOF_XYZ, EdgeSE3LotsOfXYZ);

    
void declareTypesSlam3d(py::module & m) {

    declareSalm3dParameter(m);

    declareSE3Quat(m);
    declareVertexSE3(m);
    declareVertexPointXYZ(m);

    declareEdgePointXYZ(m);
    declareEdgeSE3(m);
    declareEdgeSE3PointXYZ(m);


}

}