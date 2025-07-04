#include <pybind11/pybind11.h>
//#include <pybind11/eigen.h>
#include <pybind11/operators.h>

#include <g2o/types/slam3d/se3quat.h>
#include "python/core/base_vertex.h"
#include "python/core/base_edge.h"


namespace py = pybind11;
using namespace pybind11::literals;


namespace g2o {

void declareSE3Quat(py::module & m) {

    py::class_<SE3Quat>(m, "SE3Quat")
        .def(py::init<>())
        .def(py::init<const Matrix3D&, const Vector3D&>(),
                "R"_a, "t"_a)
        .def(py::init<const Eigen::Quaterniond&, const Vector3D&>(),
                "q"_a, "t"_a)
        .def(py::init<const Vector6d&>(),
                "v"_a)
        .def(py::init<const Vector7d&>(),
                "v"_a)

        .def("translation", &SE3Quat::translation)
        .def("position", &SE3Quat::translation)
        .def("set_translation", &SE3Quat::setTranslation, 
                "t"_a)
        .def("rotation", &SE3Quat::rotation)
        .def("orientation", &SE3Quat::rotation)
        .def("Quaternion", &SE3Quat::rotation)
        .def("set_rotation", &SE3Quat::setRotation, 
                "q"_a)

        .def(py::self * py::self)
        .def(py::self *= py::self)
        .def(py::self * Vector3D())

        .def("inverse", &SE3Quat::inverse)
        .def("__getitem__", &SE3Quat::operator[],
                "i"_a)

        .def("to_vector", &SE3Quat::toVector)
        .def("vector", &SE3Quat::toVector)
        .def("from_vector", &SE3Quat::fromVector,
                "v"_a)
        .def("to_minimal_vector", &SE3Quat::toMinimalVector)
        .def("from_minimal_vector", &SE3Quat::fromMinimalVector,
                "v"_a)

        .def("log", &SE3Quat::log)   // -> Vector6d
        .def("map", &SE3Quat::map,
                "xyz"_a)   // Vector3D -> vector3D
        .def_static("exp", &SE3Quat::exp,
                "update"_a)   // Vector6d -> SE3Quat

        .def("adj", &SE3Quat::adj)
        .def("to_homogeneous_matrix", &SE3Quat::to_homogeneous_matrix)
        .def("matrix", &SE3Quat::to_homogeneous_matrix)
        .def("normalize_rotation", &SE3Quat::normalizeRotation)

        // operator Isometry3D() const
        .def("Isometry3d", [](const SE3Quat& p) {
                Isometry3D result = (Isometry3D) p.rotation();
                result.translation() = p.translation();
                return result;
        })
    ;

    
    templatedBaseVertex<6, SE3Quat>(m, "_6_SE3Quat");
    templatedBaseEdge<6, SE3Quat>(m, "_6_SE3Quat");
    templatedBaseMultiEdge<6, SE3Quat>(m, "_6_SE3Quat");

}

}  // end namespace g2o