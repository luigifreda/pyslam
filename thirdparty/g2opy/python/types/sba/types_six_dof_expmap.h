#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>

#include <g2o/types/sba/types_six_dof_expmap.h>

#include <g2o/types/slam3d/se3quat.h>
//#include "python/core/base_vertex.h"
#include "python/core/base_unary_edge.h"
#include "python/core/base_binary_edge.h"
#include "python/core/base_multi_edge.h"


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareTypesSixDofExpmap(py::module & m) {

    py::class_<CameraParameters, Parameter>(m, "CameraParameters")
        .def(py::init<>())
        .def(py::init<double, const Vector2D &, double>(),
                "focal_length"_a, "principle_point"_a, "baseline"_a)

        .def("cam_map", &CameraParameters::cam_map,
                "trans_xyz")
        .def("stereocam_uvu_map", &CameraParameters::stereocam_uvu_map,
                "trans_xyz")
        .def_readwrite("focal_length", &CameraParameters::focal_length)
        .def_readwrite("principal_point", &CameraParameters::principle_point)
        .def_readwrite("principle_point", &CameraParameters::principle_point)
        .def_readwrite("baseline", &CameraParameters::baseline)
        // read
        // write
    ;

    
    py::class_<VertexSE3Expmap, BaseVertex<6, SE3Quat>>(m, "VertexSE3Expmap")
        .def(py::init<>())
        //.def(py::init([]() {return new VertexSE3Expmap();}))
        .def("set_to_origin_impl", &VertexSE3Expmap::setToOriginImpl)
        .def("oplus_impl", &VertexSE3Expmap::oplusImpl)    // double* -> void
        // read
        // write
    ;


    templatedBaseBinaryEdge<6, SE3Quat, VertexSE3Expmap, VertexSE3Expmap>(m, "_6_SE3Quat_VertexSE3Expmap_VertexSE3Expmap");
    py::class_<EdgeSE3Expmap, BaseBinaryEdge<6, SE3Quat, VertexSE3Expmap, VertexSE3Expmap>>(m, "EdgeSE3Expmap")
        .def(py::init<>())
        .def("compute_error", &EdgeSE3Expmap::computeError)
        .def("linearize_oplus", &EdgeSE3Expmap::linearizeOplus)
    ;


    templatedBaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>(m, "_2_Vector2D_VertexSBAPointXYZ_VertexSE3Expmap");
    py::class_<EdgeProjectXYZ2UV, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeProjectXYZ2UV")
        .def(py::init<>())
        .def("compute_error", &EdgeProjectXYZ2UV::computeError)
        .def("linearize_oplus", &EdgeProjectXYZ2UV::linearizeOplus)
    ;


    py::class_<EdgeProjectPSI2UV, BaseMultiEdge<2, Vector2D>>(m, "EdgeProjectPSI2UV")
        .def(py::init())
        .def("compute_error", &EdgeProjectPSI2UV::computeError)
        .def("linearize_oplus", &EdgeProjectPSI2UV::linearizeOplus)
    ;


    //Stereo Observations
    templatedBaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap>(m, "_3_Vector3D_VertexSBAPointXYZ_VertexSE3Expmap");
    py::class_<EdgeProjectXYZ2UVU, BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeProjectXYZ2UVU")
        .def(py::init<>())
        .def("compute_error", &EdgeProjectXYZ2UVU::computeError)
    ;


    // Projection using focal_length in x and y directions
    py::class_<EdgeSE3ProjectXYZ, BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeSE3ProjectXYZ")
        .def(py::init<>())
        .def("compute_error", &EdgeSE3ProjectXYZ::computeError)
        .def("is_depth_positive", &EdgeSE3ProjectXYZ::isDepthPositive)
        .def("linearize_oplus", &EdgeSE3ProjectXYZ::linearizeOplus)
        .def("cam_project", &EdgeSE3ProjectXYZ::cam_project)
        .def_readwrite("fx", &EdgeSE3ProjectXYZ::fx)
        .def_readwrite("fy", &EdgeSE3ProjectXYZ::fy)
        .def_readwrite("cx", &EdgeSE3ProjectXYZ::cx)
        .def_readwrite("cy", &EdgeSE3ProjectXYZ::cy)
    ;


    // Edge to optimize only the camera pose
    templatedBaseUnaryEdge<2, Vector2D, VertexSE3Expmap>(m, "BaseUnaryEdge_2_Vector2D_VertexSE3Expmap");
    py::class_<EdgeSE3ProjectXYZOnlyPose, BaseUnaryEdge<2, Vector2D, VertexSE3Expmap>>(m, "EdgeSE3ProjectXYZOnlyPose")
        .def(py::init<>())
        .def("compute_error", &EdgeSE3ProjectXYZOnlyPose::computeError)
        .def("is_depth_positive", &EdgeSE3ProjectXYZOnlyPose::isDepthPositive)
        .def("linearize_oplus", &EdgeSE3ProjectXYZOnlyPose::linearizeOplus)
        .def("cam_project", &EdgeSE3ProjectXYZOnlyPose::cam_project)
        .def_readwrite("fx", &EdgeSE3ProjectXYZOnlyPose::fx)
        .def_readwrite("fy", &EdgeSE3ProjectXYZOnlyPose::fy)
        .def_readwrite("cx", &EdgeSE3ProjectXYZOnlyPose::cx)
        .def_readwrite("cy", &EdgeSE3ProjectXYZOnlyPose::cy)
        .def_readwrite("Xw", &EdgeSE3ProjectXYZOnlyPose::Xw)
    ;


    // Projection using focal_length in x and y directions stereo
    py::class_<EdgeStereoSE3ProjectXYZ, BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap>>(m, "EdgeStereoSE3ProjectXYZ")
        .def(py::init<>())
        .def("compute_error", &EdgeStereoSE3ProjectXYZ::computeError)
        .def("is_depth_positive", &EdgeStereoSE3ProjectXYZ::isDepthPositive)
        .def("linearize_oplus", &EdgeStereoSE3ProjectXYZ::linearizeOplus)
        .def("cam_project", &EdgeStereoSE3ProjectXYZ::cam_project)   
        .def_readwrite("fx", &EdgeStereoSE3ProjectXYZ::fx)
        .def_readwrite("fy", &EdgeStereoSE3ProjectXYZ::fy)
        .def_readwrite("cx", &EdgeStereoSE3ProjectXYZ::cx)
        .def_readwrite("cy", &EdgeStereoSE3ProjectXYZ::cy)      
        .def_readwrite("bf", &EdgeStereoSE3ProjectXYZ::bf)              
    ;


    // Edge to optimize only the camera pose stereo
    templatedBaseUnaryEdge<3, Vector3D, VertexSE3Expmap>(m, "BaseUnaryEdge_3_Vector3D_VertexSE3Expmap");
    py::class_<EdgeStereoSE3ProjectXYZOnlyPose, BaseUnaryEdge<3, Vector3D, VertexSE3Expmap>>(m, "EdgeStereoSE3ProjectXYZOnlyPose")
        .def(py::init<>())
        .def("compute_error", &EdgeStereoSE3ProjectXYZOnlyPose::computeError)
        .def("is_depth_positive", &EdgeStereoSE3ProjectXYZOnlyPose::isDepthPositive)
        .def("linearize_oplus", &EdgeStereoSE3ProjectXYZOnlyPose::linearizeOplus)
        .def("cam_project", &EdgeStereoSE3ProjectXYZOnlyPose::cam_project)
        .def_readwrite("fx", &EdgeStereoSE3ProjectXYZOnlyPose::fx)
        .def_readwrite("fy", &EdgeStereoSE3ProjectXYZOnlyPose::fy)
        .def_readwrite("cx", &EdgeStereoSE3ProjectXYZOnlyPose::cx)
        .def_readwrite("cy", &EdgeStereoSE3ProjectXYZOnlyPose::cy)
        .def_readwrite("bf", &EdgeStereoSE3ProjectXYZOnlyPose::bf)        
        .def_readwrite("Xw", &EdgeStereoSE3ProjectXYZOnlyPose::Xw)             
    ;


    // class G2O_TYPES_SBA_API EdgeProjectPSI2UV : public g2o::BaseMultiEdge<2, Vector2D>
    // class G2O_TYPES_SBA_API EdgeProjectXYZ2UVU : public BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap>
    // class EdgeSE3ProjectXYZ : public BaseBinaryEdge<2, Vector2D, VertexSBAPointXYZ, VertexSE3Expmap>
    // class EdgeSE3ProjectXYZOnlyPose : public BaseUnaryEdge<2, Vector2D, VertexSE3Expmap>
    // class EdgeStereoSE3ProjectXYZ : public BaseBinaryEdge<3, Vector3D, VertexSBAPointXYZ, VertexSE3Expmap>
    // class EdgeStereoSE3ProjectXYZOnlyPose : public BaseUnaryEdge<3, Vector3D, VertexSE3Expmap>

}

}  // end namespace g2o